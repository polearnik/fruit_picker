"""Этап 4. Hand-eye калибровка: камера -> рука, с учётом подъёмника.

Идея: на кончик руки (точку gripper_frame_link, между пальцами) крепим яркую
цветную МЕТКУ и снимаем её в разных положениях. В каждом снимке:
  - стерео меряет метку -> XYZ в системе камеры;
  - углы суставов + FK -> XYZ того же кончика в системе основания;
  - подъёмник -> высота каретки z.
Решатель (arm/handeye.py, fit_with_lift) находит преобразование камера ->
мир и вектор up — куда смещается основание при подъёме каретки. Сохраняет в
arm/handeye.npz.

Рука висит на стойке БОКОМ: с выключенным моментом она падает под своим
весом. Поэтому она по умолчанию ДЕРЖИТ позу, а мягкой становится только по
клавише t: придержали рукой, поставили кончик куда надо, снова t —
зафиксировалась. Отпустили и снимаете.

Как снимать (лучше 12-20 точек):
  - на одной высоте каретки — 4-6 поз руки, разнесённых по зоне и по глубине;
  - потом u/d — каретка на другую высоту, и снова несколько поз;
  - минимум 2 высоты (лучше 3), разница между крайними 200+ мм;
  - полезно: ту же позу руки снять на двух высотах — это прямо меряет up.
Метка должна быть видна ОБЕИМ камерам.

Подготовка: ЖЁЛТАЯ метка (шарик/колпачок/изолента) ровно на
кончике руки. Диапазон в HSV задан ниже (MARKER_HSV_RANGES). При старте —
хоминг подъёмника: каретка медленно опустится до упора, рука в безопасной
позе (не достаёт до пола и ящика).

Управление (в окне видео):
  ПРОБЕЛ — снять точку
  t      — рука мягкая / зафиксировать (ДЕРЖИТЕ руку перед тем, как сделать мягкой)
  u / d  — каретка вверх / вниз на 50 мм;  U / D — на 200 мм
  c      — посчитать и сохранить калибровку
  x      — выбросить худшую точку и пересчитать/сохранить
  z      — убрать последнюю снятую точку
  q      — выход (рука остаётся зафиксированной)

Запуск:  python handeye_calibrate.py
"""

import sys

import cv2
import numpy as np

from camera import FRAME_W, FRAME_H, open_pair, overlay_scale, preview

sys.path.insert(0, "arm")
from stereo_core import StereoRig
from kinematics import ArmKinematics, JOINT_NAMES
from handeye import fit_with_lift, residuals_lift_mm, HandEye
from lift import Lift, LiftError, confirm
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

LEFT_CAM, RIGHT_CAM = 0, 2
# Разрешение, формат кадра и заморозка автоматики — в camera.py.
PORT, ROBOT_ID = "/dev/ttyACM0", "fruit_arm"

# Суставы, которые делаются мягкими по t. Клешня не трогается.
ARM_JOINTS = [j for j in JOINT_NAMES if j != "gripper"]
LIFT_STEP_MM, LIFT_BIG_STEP_MM = 50.0, 200.0

# Цвет метки — ЖЁЛТЫЙ, насыщенный (шарик/колпачок/изолента). В HSV (шкала
# OpenCV, H 0..179) жёлтый — H≈30. Диапазон H 20-35 не берёт оранжевый (<20)
# и салатовый (>35); нижний порог S отсекает белое, серое, кремовое и кожу;
# порог V — тени. Если метку не видно (на кадре нет красного кружка) —
# при тёплом/тусклом свете попробуйте снизить S или V; если хватает чужое —
# поднимите S.
MARKER_HSV_RANGES = [
    (np.array([20, 110, 110]), np.array([35, 255, 255])),
]
MIN_BLOB_AREA = 60  # пикселей, чтобы отсеять шум


def find_marker(frame):
    """Центр яркой цветной метки (u, v) или None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = None
    for low, high in MARKER_HSV_RANGES:
        m = cv2.inRange(hsv, low, high)
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_BLOB_AREA:
        return None
    M = cv2.moments(c)
    return (M["m10"] / M["m00"], M["m01"] / M["m00"])


def read_joints(arm, keys):
    obs = arm.get_observation()
    return np.array([obs[keys[j]] for j in JOINT_NAMES], dtype=float)


def lock_arm(arm):
    """Зафиксировать руку в ТЕКУЩЕЙ позе.

    Сначала цель = текущее положение, потом момент: иначе при включении
    сервы рванулись бы к старой цели, оставшейся с прошлой команды.
    """
    present = arm.bus.sync_read("Present_Position", ARM_JOINTS)
    arm.bus.sync_write("Goal_Position", present)
    arm.bus.enable_torque(ARM_JOINTS)


def solve(cam_pts, arm_pts, lift_zs, drop_worst=False):
    if len(cam_pts) < 6:
        print(f"  мало точек ({len(cam_pts)}), нужно >= 6 (лучше 12-20)")
        return
    heights = sorted({round(z) for z in lift_zs})
    if max(lift_zs) - min(lift_zs) < 100:
        print(f"  все точки на высотах {heights} — нужна ещё высота, отличающаяся "
              "на 100+ мм (u/d), иначе направление подъёмника не определить")
        return
    if drop_worst and len(cam_pts) > 6:
        R, t, up, zr = fit_with_lift(cam_pts, arm_pts, lift_zs)
        res0 = residuals_lift_mm(R, t, up, zr, cam_pts, arm_pts, lift_zs)
        w = int(np.argmax(res0))
        print(f"  выбрасываю худшую точку #{w+1} ({res0[w]:.0f} мм)")
        for lst in (cam_pts, arm_pts, lift_zs):
            lst.pop(w)

    R, t, up, zr = fit_with_lift(cam_pts, arm_pts, lift_zs)
    res = residuals_lift_mm(R, t, up, zr, cam_pts, arm_pts, lift_zs)
    print(f"\nКалибровка по {len(cam_pts)} точкам на высотах {heights}: "
          f"ошибка средняя {res.mean():.1f} мм, макс {res.max():.1f} мм")
    worst = np.argsort(res)[::-1]
    print("  по точкам (мм):", ", ".join(f"#{i+1}={res[i]:.0f}" for i in worst))
    mean = res.mean()
    outliers = [i + 1 for i in worst if res[i] > 2 * mean and res[i] > 15]
    if outliers:
        print(f"  вероятные выбросы (промах детекции?): точки {outliers} — "
              "уберите их (x) или переснимите")

    scale = float(np.linalg.norm(up))
    u = up / scale
    print(f"  «вверх» в осях основания: {np.round(u, 3)}")
    print(f"  масштаб подъёмника |up| = {scale:.3f}", end="")
    if abs(scale - 1) > 0.03:
        print(f"  <-- ВНИМАНИЕ: каретка проходит {scale:.3f} мм на 1 мм по счёту.")
        print(f"      Если это не ошибка калибровки — в arm/lift.py MM_PER_REV "
              f"= {40.0 * scale:.2f}, и переснимите.")
    else:
        print("  (норма, MM_PER_REV верен)")
    if mean > 15:
        print("  ВНИМАНИЕ: большая ошибка. Чаще всего причина — крутили запястье")
        print("  при сборе (метка гуляет), мало разброса по глубине, или метка")
        print("  далеко от кончика. Переснимите.")
    HandEye(R, t, up, zr).save()
    print(f"  Сохранено в arm/handeye.npz (мировая система = основание при z={zr:.0f} мм)")


def main():
    rig = StereoRig("stereo_calib.npz")
    rig.assert_frame_size(FRAME_W, FRAME_H)
    kin = ArmKinematics()

    # disable_torque_on_disconnect=False: рука висит боком, при выходе она
    # должна остаться в позе, а не упасть.
    arm = SO101Follower(SO101FollowerConfig(port=PORT, id=ROBOT_ID, use_degrees=True,
                                            disable_torque_on_disconnect=False))
    arm.connect(calibrate=False)
    obs = arm.get_observation()
    keys = {j: next(k for k in obs if j in k and isinstance(obs[k], (int, float)))
            for j in JOINT_NAMES}
    lock_arm(arm)

    lift = Lift(arm.bus.port_handler, arm.bus.packet_handler)
    lift.setup()
    if not confirm("Хоминг подъёмника: каретка медленно опустится до упора. Рука "
                   "в безопасной позе?"):
        print("Без хоминга высота каретки неизвестна — калибровка невозможна.")
        lift.close()
        arm.disconnect()
        return
    lift.home()

    cap_l, cap_r = open_pair(LEFT_CAM, RIGHT_CAM)
    cam_pts, arm_pts, lift_zs = [], [], []
    limp = False
    print("Рука зафиксирована. t — сделать мягкой (ДЕРЖИТЕ её), ПРОБЕЛ — снять точку.")

    try:
        while True:
            ok_l, fl = cap_l.read()
            ok_r, fr = cap_r.read()
            if not (ok_l and ok_r):
                break
            ml, mr = find_marker(fl), find_marker(fr)

            s = overlay_scale(fl, side_by_side=2)
            for f, m in ((fl, ml), (fr, mr)):
                if m:
                    cv2.circle(f, (int(m[0]), int(m[1])), int(8 * s), (0, 0, 255), int(2 * s))
            both = preview(fl, fr)
            seen = "L+R" if (ml and mr) else ("tolko L" if ml else ("tolko R" if mr else "net"))
            heights = len({round(z) for z in lift_zs})
            cv2.putText(both, f"tochek: {len(cam_pts)} vysot: {heights}  metka: {seen}  "
                              f"z={lift.z():.0f}  {'MYAGKAYA' if limp else 'fiks'}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("hand-eye  (SPACE snap, t limp/lock, u/d lift, c solve, q quit)", both)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("t"):
                if limp:
                    lock_arm(arm)
                    print("  рука зафиксирована")
                else:
                    arm.bus.disable_torque(ARM_JOINTS)
                    print("  рука мягкая — держите и ставьте кончик, потом снова t")
                limp = not limp
            if key in (ord("u"), ord("d"), ord("U"), ord("D")):
                if limp:
                    print("  сначала зафиксируйте руку (t), потом двигайте каретку")
                    continue
                step = LIFT_BIG_STEP_MM if key in (ord("U"), ord("D")) else LIFT_STEP_MM
                dz = step if key in (ord("u"), ord("U")) else -step
                try:
                    print(f"  каретка {'вверх' if dz > 0 else 'вниз'} на {abs(dz):.0f} мм...")
                    print(f"  z = {lift.move_by(dz):.0f} мм")
                except LiftError as e:
                    print(f"  [подъёмник] {e}")
                # кадры, скопившиеся за время переезда, — старые: выбросить
                for _ in range(5):
                    cap_l.grab(); cap_r.grab()
            if key == ord(" "):
                if limp:
                    print("  рука мягкая — зафиксируйте (t) и отпустите, потом снимайте")
                elif ml and mr and rig.is_valid_pair(ml, mr):
                    p_cam = rig.triangulate(ml, mr)
                    p_arm = kin.fk(read_joints(arm, keys))
                    z = lift.z()
                    cam_pts.append(p_cam); arm_pts.append(p_arm); lift_zs.append(z)
                    print(f"  точка {len(cam_pts)}: z={z:.0f} cam={np.round(p_cam,1)} "
                          f"arm={np.round(p_arm,1)}")
                elif ml and mr:
                    # метка найдена в обоих кадрах, но пара не прошла проверку —
                    # печатаем, какую именно, иначе не понять, что чинить
                    rl, rr = rig.rectified(ml, mr)
                    dy = rl[1] - rr[1]
                    disp = (rl[0] - rr[0]) * rig.disp_sign
                    print(f"  пара отбракована: расхождение по строке {dy:+.0f} px "
                          f"(допуск ±20), сдвиг {disp:+.0f} px (должен быть > 0)")
                    if disp <= 0:
                        print("    сдвиг не того знака — похоже, левая и правая камеры "
                              "перепутаны (LEFT_CAM/RIGHT_CAM) или номера камер "
                              "поменялись после перезагрузки")
                    if abs(dy) >= 20:
                        print("    строки не совпадают — камеры сдвинулись после "
                              "стереокалибровки, переснимите её")
                else:
                    print("  метка видна не в обеих камерах — не снял")
            if key == ord("z"):
                if cam_pts:
                    cam_pts.pop(); arm_pts.pop(); lift_zs.pop()
                    print(f"  убрал последнюю точку, осталось {len(cam_pts)}")
            if key in (ord("c"), ord("x")):
                try:
                    solve(cam_pts, arm_pts, lift_zs, drop_worst=(key == ord("x")))
                except ValueError as e:
                    print(f"  {e}")
    finally:
        if limp:
            print("Фиксирую руку перед выходом.")
            lock_arm(arm)
        lift.close()
        arm.disconnect()   # момент НЕ снимается: рука остаётся в позе
        cap_l.release(); cap_r.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

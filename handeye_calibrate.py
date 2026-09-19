"""Этап 4. Hand-eye калибровка: находим преобразование камера -> рука.

Идея: на кончик руки (точку gripper_frame_link, между пальцами) крепим яркую
цветную МЕТКУ. Момент выключен — вы рукой водите кончик по разным точкам в поле
зрения камер. В каждой точке:
  - стерео меряет метку -> XYZ в системе камеры;
  - углы суставов + FK -> XYZ того же кончика в системе руки.
По ~6-10 таким парам решатель находит фиксированное преобразование и сохраняет
его в arm/handeye.npz.

Подготовка: светлая КРЕМОВАЯ метка (шарик/колпачок) цвета #f8ebb9 ровно на
кончике руки. Диапазон в HSV задан ниже (MARKER_HSV_RANGES).

Управление:
  ПРОБЕЛ — снять точку (метка должна быть видна ОБЕИМ камерам)
  c      — посчитать и сохранить калибровку (нужно >= 4 точек, лучше 8-12)
  x      — выбросить худшую точку и пересчитать/сохранить
  z      — убрать последнюю снятую точку
  q      — выход

Запуск:  python handeye_calibrate.py
"""

import sys

import cv2
import numpy as np

sys.path.insert(0, "arm")
from stereo_core import StereoRig
from kinematics import ArmKinematics, JOINT_NAMES
from handeye import rigid_transform, residuals_mm, HandEye
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

LEFT_CAM, RIGHT_CAM = 0, 2
FRAME_W, FRAME_H = 1280, 960
PORT, ROBOT_ID = "/dev/ttyACM0", "fruit_arm"

# Цвет метки — КРЕМОВЫЙ #f8ebb9. В HSV (шкала OpenCV) это H≈24, S≈65, V≈248:
# пастельный оттенок, поэтому ловим его в первую очередь яркостью при
# умеренной насыщенности. Нижний порог S отсекает белое и серое, верхний —
# насыщенные жёлто-оранжевые предметы в кадре.
MARKER_HSV_RANGES = [
    (np.array([14, 25, 170]), np.array([34, 150, 255])),
]
MIN_BLOB_AREA = 60  # пикселей, чтобы отсеять шум


def open_cam(i):
    cap = cv2.VideoCapture(i)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        raise RuntimeError(f"Камера {i} не открылась")
    return cap


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


def read_arm_xyz(arm, kin, keys):
    obs = arm.get_observation()
    joints = np.array([obs[keys[j]] for j in JOINT_NAMES], dtype=float)
    return kin.fk(joints)


def main():
    rig = StereoRig("stereo_calib.npz")
    kin = ArmKinematics()
    cap_l, cap_r = open_cam(LEFT_CAM), open_cam(RIGHT_CAM)

    arm = SO101Follower(SO101FollowerConfig(port=PORT, id=ROBOT_ID, use_degrees=True))
    arm.connect(calibrate=False)
    arm.bus.disable_torque()  # рука свободна, водите рукой
    obs = arm.get_observation()
    keys = {j: next(k for k in obs if j in k and isinstance(obs[k], (int, float)))
            for j in JOINT_NAMES}

    cam_pts, arm_pts = [], []
    print("Момент выключен — водите кончик руки с меткой по точкам.")
    print("ПРОБЕЛ — снять точку, c — посчитать/сохранить, q — выход.")

    while True:
        ok_l, fl = cap_l.read()
        ok_r, fr = cap_r.read()
        if not (ok_l and ok_r):
            break
        ml, mr = find_marker(fl), find_marker(fr)

        for f, m in ((fl, ml), (fr, mr)):
            if m:
                cv2.circle(f, (int(m[0]), int(m[1])), 8, (0, 0, 255), 2)
        both = cv2.hconcat([cv2.resize(fl, (640, 480)), cv2.resize(fr, (640, 480))])
        seen = "L+R" if (ml and mr) else ("только L" if ml else ("только R" if mr else "нет"))
        cv2.putText(both, f"tochek: {len(cam_pts)}  metka: {seen}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("hand-eye  (SPACE=snap, c=solve, q=quit)", both)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            if ml and mr and rig.is_valid_pair(ml, mr):
                p_cam = rig.triangulate(ml, mr)
                p_arm = read_arm_xyz(arm, kin, keys)
                cam_pts.append(p_cam)
                arm_pts.append(p_arm)
                print(f"  точка {len(cam_pts)}: cam={np.round(p_cam,1)} arm={np.round(p_arm,1)}")
            else:
                print("  метка видна не в обеих камерах / плохая пара — не снял")
        if key == ord("z"):
            if cam_pts:
                cam_pts.pop(); arm_pts.pop()
                print(f"  убрал последнюю точку, осталось {len(cam_pts)}")
            continue
        if key in (ord("c"), ord("x")):
            if len(cam_pts) < 4:
                print(f"  мало точек ({len(cam_pts)}), нужно >= 4")
                continue
            if key == ord("x") and len(cam_pts) > 4:
                res0 = residuals_mm(*rigid_transform(cam_pts, arm_pts), cam_pts, arm_pts)
                w = int(np.argmax(res0))
                print(f"  выбрасываю худшую точку #{w+1} ({res0[w]:.0f} мм)")
                cam_pts.pop(w); arm_pts.pop(w)
            R, t = rigid_transform(cam_pts, arm_pts)
            res = residuals_mm(R, t, cam_pts, arm_pts)
            print(f"\nКалибровка по {len(cam_pts)} точкам: "
                  f"ошибка средняя {res.mean():.1f} мм, макс {res.max():.1f} мм")
            # ошибка по каждой точке — выбросы видно сразу
            worst = np.argsort(res)[::-1]
            print("  по точкам (мм):",
                  ", ".join(f"#{i+1}={res[i]:.0f}" for i in worst))
            mean = res.mean()
            outliers = [i + 1 for i in worst if res[i] > 2 * mean and res[i] > 15]
            if outliers:
                print(f"  вероятные выбросы (промах детекции?): точки {outliers} — "
                      "стоит перекалиброваться без них")
            if mean > 15:
                print("  ВНИМАНИЕ: большая ошибка. Чаще всего причина — крутили запястье")
                print("  при сборе (метка гуляет), мало разброса по глубине, или метка")
                print("  далеко от кончика. См. советы и переснимите.")
            HandEye(R, t).save()
            print("  Сохранено в arm/handeye.npz")

    arm.bus.enable_torque()
    arm.disconnect()
    cap_l.release(); cap_r.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

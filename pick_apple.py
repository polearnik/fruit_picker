"""Полный цикл: увидеть яблоко двумя камерами, взять клешнёй, поднять.

Требует: stereo_calib.npz (стерео), arm/handeye.npz (hand-eye), калибровку руки.
Порядок запуска в проекте:
  1) stereo_calibrate.py   -> stereo_calib.npz
  2) handeye_calibrate.py  -> arm/handeye.npz
  3) pick_apple.py         (этот файл)

Пайплайн:
  YOLO находит яблоко в левом+правом кадре -> триангуляция -> XYZ в системе камеры
  -> hand-eye -> XYZ в системе руки -> IK -> движение:
  открыть клешню -> подвести НАД яблоком -> опуститься -> сжать -> поднять.

ВНИМАНИЕ: первый реальный захват требует подстройки (GRASP_OFFSET, углы клешни).
Рука в свободном пространстве, выключатель под рукой, смотрите на движение.

Запуск:  python pick_apple.py
"""

import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, "arm")
from stereo_core import StereoRig
from kinematics import ArmKinematics, JOINT_NAMES
from handeye import HandEye
from gripper import close_until_contact
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

LEFT_CAM, RIGHT_CAM = 0, 2
FRAME_W, FRAME_H = 1280, 960
PORT, ROBOT_ID = "/dev/ttyACM0", "fruit_arm"

APPLE_CLASS = 47      # класс "apple" в COCO
CONF = 0.4

# --- параметры захвата (ПОДСТРАИВАЮТСЯ под вашу руку и клешню) ---
# Высота, на которой рука ПРОЛЕТАЕТ над яблоком, прежде чем опускаться.
# Должна быть заметно больше радиуса плода, иначе кончик заденет его боком.
TRAVEL_CLEARANCE_MM = 90.0
GRASP_OFFSET = np.array([-40., 0., -50.]) # x вперед y вбок z вверх поправка к точке захвата (калибровочный сдвиг)
LIFT_MM = 100.0                       # на сколько поднять после захвата
GRIPPER_OPEN_DEG = 45.0               # угол раскрытой клешни
GRIPPER_CLOSED_DEG = 20.0              # ЖЁСТКИЙ ПРЕДЕЛ сжатия: сильнее не сожмём
                                       # никогда. Реальная остановка — по контакту
                                       # (см. arm/gripper.py), обычно раньше.
# Исходная («домашняя») поза: рука вытянута вперёд, все суставы с запасом ~55°
# до пределов. Из сложенной позы, где суставы упёрты в механические концы,
# планировать нельзя — сначала приводим руку сюда.
HOME_JOINTS = np.array([0., -30., 40., -20., 0., 45.])

# --- параметры движения ---
STEP_DEG, DT, MAX_REL, TOL_MM = 1.0, 0.03, 15.0, 10.0
CART_STEP_MM = 10.0   # шаг разбиения прямой в пространстве (мельче = точнее путь)
GRIP_IDX = JOINT_NAMES.index("gripper")


def detect_apple_cam_xyz(model, fl, fr, rig):
    """XYZ ближайшего согласованного яблока в системе камеры (мм) или None."""
    def apples(frame):
        res = model(frame, conf=CONF, verbose=False)[0]
        out = []
        for b in res.boxes:
            if int(b.cls) == APPLE_CLASS:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                out.append(((x1 + x2) / 2, (y1 + y2) / 2, float(b.conf)))
        return out

    left, right = apples(fl), apples(fr)
    best, best_conf = None, 0
    for lx, ly, lc in left:
        for rx, ry, rc in right:
            if rig.is_valid_pair((lx, ly), (rx, ry)) and lc + rc > best_conf:
                best_conf = lc + rc
                best = rig.triangulate((lx, ly), (rx, ry))
    return best


def segment_joints(kin, q_start, xyz_from, xyz_to, name, step_mm=CART_STEP_MM):
    """Прямой отрезок в пространстве -> список углов через каждые step_mm.

    Кончик едет по ПРЯМОЙ, а не по дуге: иначе при интерполяции в углах он
    может проехать сквозь яблоко, даже если конечная точка была над ним.
    """
    xyz_from = np.asarray(xyz_from, float)
    xyz_to = np.asarray(xyz_to, float)
    dist = float(np.linalg.norm(xyz_to - xyz_from))
    n = max(1, int(np.ceil(dist / step_mm)))
    q = np.asarray(q_start, float).copy()
    out = []
    for i in range(1, n + 1):
        p = xyz_from + (xyz_to - xyz_from) * i / n
        q, err, ok = kin.ik(p, q, tol_mm=TOL_MM)
        if not ok:
            return None, f"[СТОП] '{name}': точка {np.round(p,0)} недостижима ({err:.0f} мм)"
        bad = kin.within_limits(q)
        if bad:
            return None, f"[СТОП] '{name}': сустав за пределом: {[b[0] for b in bad]}"
        out.append(q.copy())
    return out, "ok"


def plan_grasp(kin, handeye, cam_xyz, current_joints):
    """cam XYZ -> план траектории: подъём, проход над яблоком, спуск, подъём с ним.

    Путь строится по трём прямым отрезкам, чтобы заходить СВЕРХУ:
      1) от текущей точки вертикально вверх до безопасной высоты;
      2) горизонтально до точки прямо над яблоком;
      3) вертикально вниз к яблоку.
    """
    target = handeye.cam_to_arm(cam_xyz)
    grasp_xyz = target + GRASP_OFFSET
    lift_xyz = grasp_xyz + np.array([0, 0, LIFT_MM])

    # Физический ход суставов шире модельного (URDF), поэтому рука может стоять
    # в позе, которой в модели «не существует». Планируем от зажатой в пределы
    # позы — первым движением сустав вернётся в допустимый диапазон.
    current_joints = np.asarray(current_joints, float)
    clamped = kin.clamp_to_limits(current_joints)
    moved = [n for n, a, b in zip(JOINT_NAMES, current_joints, clamped)
             if abs(a - b) > 0.1]
    if moved:
        print(f"  (стартовая поза вне пределов модели по {moved} — "
              "сначала верну сустав(ы) в рабочий диапазон)")
    current_joints = clamped

    start_xyz = kin.fk(current_joints)
    # высота пролёта: заведомо выше яблока, и не ниже текущего положения кончика
    safe_z = max(grasp_xyz[2] + TRAVEL_CLEARANCE_MM, start_xyz[2])
    up_xyz = np.array([start_xyz[0], start_xyz[1], safe_z])
    over_xyz = np.array([grasp_xyz[0], grasp_xyz[1], safe_z])

    segments = [("подъём на безопасную высоту", start_xyz, up_xyz),
                ("проход над яблоком", up_xyz, over_xyz),
                ("спуск к яблоку", over_xyz, grasp_xyz)]

    q = np.asarray(current_joints, float).copy()
    approach = []
    for name, a, b in segments:
        part, msg = segment_joints(kin, q, a, b, name)
        if part is None:
            return None, msg
        approach.extend(part)
        q = part[-1]

    # путь подъёма с яблоком — тоже прямой, вертикально вверх
    lift_path, msg = segment_joints(kin, q, grasp_xyz, lift_xyz, "подъём с яблоком")
    if lift_path is None:
        return None, msg

    return {"target": target, "grasp_xyz": grasp_xyz,
            "approach": approach, "lift": lift_path}, "ok"


def smooth_to(arm, keys, current, goal):
    n = max(1, int(np.max(np.abs(goal - current)) / STEP_DEG))
    for i in range(1, n + 1):
        q = current + (goal - current) * i / n
        arm.send_action({keys[j]: q[k] for k, j in enumerate(JOINT_NAMES)})
        time.sleep(DT)
    return goal.copy()


def follow_path(arm, keys, current, path, grip_deg=None):
    """Проходит список углов траектории, плавно между соседними точками."""
    cur = np.asarray(current, float).copy()
    for q in path:
        goal = q.copy()
        if grip_deg is not None:
            goal[GRIP_IDX] = grip_deg   # клешню держим на своём угле
        cur = smooth_to(arm, keys, cur, goal)
    return cur


def read_joints(arm, keys):
    obs = arm.get_observation()
    return np.array([obs[keys[j]] for j in JOINT_NAMES], dtype=float)


def go_home(arm, keys, current, slow=0.06):
    """Приводит руку в исходную позу. Медленно: из сложенной позы ход большой."""
    delta = np.max(np.abs(HOME_JOINTS - current))
    print(f"Иду в исходную позу (максимальный ход сустава {delta:.0f}°)...")
    n = max(1, int(delta / STEP_DEG))
    for i in range(1, n + 1):
        q = current + (HOME_JOINTS - current) * i / n
        arm.send_action({keys[j]: q[k] for k, j in enumerate(JOINT_NAMES)})
        time.sleep(slow)
    return HOME_JOINTS.copy()


def main():
    rig = StereoRig("stereo_calib.npz")
    kin = ArmKinematics()
    handeye = HandEye.load()
    model = YOLO("yolov8n.pt")

    cap_l = cv2.VideoCapture(LEFT_CAM); cap_l.set(3, FRAME_W); cap_l.set(4, FRAME_H)
    cap_r = cv2.VideoCapture(RIGHT_CAM); cap_r.set(3, FRAME_W); cap_r.set(4, FRAME_H)

    arm = SO101Follower(SO101FollowerConfig(port=PORT, id=ROBOT_ID,
                                            use_degrees=True, max_relative_target=MAX_REL))
    arm.connect(calibrate=False)
    try:
        keys = {j: next(k for k in arm.get_observation()
                        if j in k and isinstance(arm.get_observation()[k], (int, float)))
                for j in JOINT_NAMES}

        # Из сложенной позы (суставы в механическом упоре) планировать нельзя —
        # сначала разворачиваем руку в известное исходное положение.
        start_pose = read_joints(arm, keys)
        out_of_model = kin.within_limits(start_pose)
        if out_of_model:
            print(f"Рука сложена: суставы {[b[0] for b in out_of_model]} в упоре.")
        go_home(arm, keys, start_pose)

        print("Ищу яблоко... покажите его обеим камерам.")
        cam_xyz = None
        while cam_xyz is None:
            ok_l, fl = cap_l.read(); ok_r, fr = cap_r.read()
            if not (ok_l and ok_r):
                raise RuntimeError("нет кадра с камеры")
            cam_xyz = detect_apple_cam_xyz(model, fl, fr, rig)
            cv2.imshow("pick (нажмите q для отмены)", cv2.hconcat(
                [cv2.resize(fl, (640, 480)), cv2.resize(fr, (640, 480))]))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return
        print(f"Яблоко в системе камеры: {np.round(cam_xyz,1)} мм")
        target_arm = handeye.cam_to_arm(cam_xyz)
        dist = float(np.linalg.norm(target_arm))
        print(f"Яблоко в системе руки: {np.round(target_arm,1)} мм, "
              f"расстояние от основания {dist:.0f} мм")

        current = read_joints(arm, keys)
        plan, msg = plan_grasp(kin, handeye, cam_xyz, current)
        if plan is None:
            print(msg)
            if dist > 420:
                print(f"  Яблоко слишком далеко ({dist:.0f} мм). Рука достаёт ~350-400 мм")
                print("  от основания — придвиньте яблоко ближе к руке (оно всё ещё")
                print("  должно быть видно обеим камерам).")
            else:
                print("  Точка в пределах вылета, но неудобна по высоте/направлению —")
                print("  попробуйте сдвинуть яблоко или уменьшить TRAVEL_CLEARANCE_MM")
                print("  (рука может не дотягиваться на высоте пролёта).")
            return
        # ans = input("Выполнить захват? Смотрите на руку. [y/N] ").strip().lower()
        # if ans != "y":
        #     print("Отменено.")
        #     return

        # последовательность захвата
        cur = current.copy()

        print("1/3 иду к яблоку сверху (вверх -> над яблоком -> вниз), клешня открыта")
        cur = follow_path(arm, keys, cur, plan["approach"], grip_deg=GRIPPER_OPEN_DEG)

        print("2/3 сжимаю клешню до контакта с яблоком")
        # команду шлём только суставу клешни, остальные держатся на месте
        def send_gripper(angle):
            arm.send_action({keys["gripper"]: angle})

        grip_deg, reason = close_until_contact(
            send_gripper, arm.bus,
            start_deg=GRIPPER_OPEN_DEG, closed_deg=GRIPPER_CLOSED_DEG)
        cur[GRIP_IDX] = grip_deg
        if reason == "closed":
            print("  Клешня сомкнулась вхолостую — яблоко не поймано, не поднимаю.")
            return

        time.sleep(0.3)
        print("3/3 поднимаю")
        cur = follow_path(arm, keys, cur, plan["lift"], grip_deg=grip_deg)
        time.sleep(5)
        print(f"Готово — яблоко поднято (клешня держит на {grip_deg:.1f}°).")
    finally:
        arm.disconnect()
        cap_l.release(); cap_r.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

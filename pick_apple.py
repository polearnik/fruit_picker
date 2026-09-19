"""Полный цикл: увидеть яблоко двумя камерами, взять клешнёй, поднять.

Требует: stereo_calib.npz (стерео), arm/handeye.npz (hand-eye), калибровку руки.
Порядок запуска в проекте:
  1) stereo_calibrate.py   -> stereo_calib.npz
  2) handeye_calibrate.py  -> arm/handeye.npz
  3) pick_apple.py         (этот файл)

Пайплайн:
  YOLO находит яблоко в левом+правом кадре -> триангуляция -> XYZ в системе камеры
  -> hand-eye -> XYZ в системе руки -> IK -> движение:
  открыть клешню -> подвести НАД яблоком -> опуститься -> сжать -> поднять
  -> перенести над ящиком -> разжать (яблоко падает в ящик) -> уйти вверх -> домой.

Здесь — зрение, планирование захвата и сценарий целиком. Соседние модули:
  motion.py        — как рука едет (разбиение пути, темп, запись тиков)
  place_in_box.py  — перенос в ящик и точка сброса

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
sys.path.insert(0, ".")
from episode_logger import EpisodeLogger
from motion import (CART_STEP_MM, CART_STEP_TRAVEL_MM, GRIP_IDX, STEP_DEG,
                    follow_path, grab_frames, read_joints, segment_joints)
from place_in_box import CARRY_CLEARANCE_MM, RELEASE_XYZ, place_in_box
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
# Поправка к точке захвата: куда встать КОНЧИКУ относительно яблока.
# Оси — НЕ основания, а направления вылета на яблоко (см. reach_frame):
#   X: вдоль вылета   + дальше от основания / - ближе к основанию
#   Y: поперёк вылета + влево от направления на яблоко / - вправо
#   Z: вверх          + выше яблока / - ниже
# Так и надо: смещение компенсирует геометрию клешни (центр хвата вынесен
# от gripper_frame_link примерно на 45 мм), а она поворачивается вместе с
# рукой. В осях основания то же смещение пришлось бы перенастраивать под
# каждое положение яблока.
GRASP_OFFSET = np.array([0., -20., 25.])
LIFT_MM = 100.0                       # на сколько поднять после захвата
GRIPPER_OPEN_DEG = 95.0               # угол раскрытой клешни
GRIPPER_CLOSED_DEG = 10.0              # ЖЁСТКИЙ ПРЕДЕЛ сжатия: сильнее не сожмём
                                       # никогда. Реальная остановка — по контакту
                                       # (см. arm/gripper.py), обычно раньше.

# Точка сброса в ящик и весь перенос — в place_in_box.py.
# Исходная («домашняя») поза: рука вытянута вперёд, все суставы с запасом ~55°
# до пределов. Из сложенной позы, где суставы упёрты в механические концы,
# планировать нельзя — сначала приводим руку сюда.
HOME_JOINTS = np.array([0., -30., 40., -20., 0., 45.])

# Шаг/пауза движения и разбиение пути — в motion.py (там же крутится скорость).
MAX_REL = 15.0   # аппаратный предел на одну команду серву


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


def reach_frame(target_xyz):
    """Оси «вдоль вылета / поперёк / вверх» для точки target (в системе основания).

    Рука тянется к яблоку под азимутом yaw, и клешня развёрнута туда же. Значит
    поправка на геометрию клешни должна поворачиваться вместе с ней, иначе она
    верна только для одного положения плода.
    """
    yaw = np.arctan2(target_xyz[1], target_xyz[0])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.],
                     [s,  c, 0.],
                     [0., 0., 1.]])


def plan_grasp(kin, handeye, cam_xyz, current_joints):
    """cam XYZ -> план траектории: подъём, проход над яблоком, спуск, подъём с ним.

    Путь строится по трём прямым отрезкам, чтобы заходить СВЕРХУ:
      1) от текущей точки вертикально вверх до безопасной высоты;
      2) горизонтально до точки прямо над яблоком;
      3) вертикально вниз к яблоку.
    """
    target = handeye.cam_to_arm(cam_xyz)
    offset_base = reach_frame(target) @ GRASP_OFFSET
    grasp_xyz = target + offset_base
    print(f"  поправка захвата {GRASP_OFFSET} (вылет/поперёк/вверх) "
          f"-> {np.round(offset_base, 1)} в осях основания")
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

    # в свободном пространстве едем грубыми шагами, у самого яблока — мелкими
    segments = [("подъём на безопасную высоту", start_xyz, up_xyz, CART_STEP_TRAVEL_MM),
                ("проход над яблоком", up_xyz, over_xyz, CART_STEP_TRAVEL_MM),
                ("спуск к яблоку", over_xyz, grasp_xyz, CART_STEP_MM)]

    q = np.asarray(current_joints, float).copy()
    approach = []
    for name, a, b, step in segments:
        part, msg = segment_joints(kin, q, a, b, name, step_mm=step)
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

    cams = {"left": cap_l, "right": cap_r}
    # Запись эпизода: кадры, углы, телеметрия, свои датчики. Чтобы добавить свой
    # датчик — передайте extra_sensors=функция, возвращающая словарь значений.
    log = EpisodeLogger(extra_sensors=None)
    log.set_meta(grasp_offset=GRASP_OFFSET, travel_clearance_mm=TRAVEL_CLEARANCE_MM,
                 lift_mm=LIFT_MM, gripper_open_deg=GRIPPER_OPEN_DEG,
                 gripper_limit_deg=GRIPPER_CLOSED_DEG, model="yolov8n.pt",
                 release_xyz=RELEASE_XYZ, carry_clearance_mm=CARRY_CLEARANCE_MM)
    success, note = False, "прервано"
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
                note = "отменено пользователем"
                return
        print(f"Яблоко в системе камеры: {np.round(cam_xyz,1)} мм")
        target_arm = handeye.cam_to_arm(cam_xyz)
        dist = float(np.linalg.norm(target_arm))
        print(f"Яблоко в системе руки: {np.round(target_arm,1)} мм, "
              f"расстояние от основания {dist:.0f} мм")
        # кадры момента обнаружения — самые ценные для обучения детектора
        log.set_meta(apple_cam_xyz=cam_xyz, apple_arm_xyz=target_arm, distance_mm=dist)
        log.tick("detected", joints=read_joints(arm, keys), tip_xyz=None,
                 frames={"left": fl, "right": fr}, bus=arm.bus, force=True)

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
            log.event("plan_failed", reason=msg, distance_mm=dist)
            note = f"планирование не прошло: {msg}"
            return
        # ans = input("Выполнить захват? Смотрите на руку. [y/N] ").strip().lower()
        # if ans != "y":
        #     print("Отменено.")
        #     return

        # последовательность захвата
        cur = current.copy()

        print("1/5 иду к яблоку сверху (вверх -> над яблоком -> вниз), клешня открыта")
        cur = follow_path(arm, keys, cur, plan["approach"], grip_deg=GRIPPER_OPEN_DEG,
                          log=log, phase="approach", kin=kin, cams=cams)

        print("2/5 сжимаю клешню до контакта с яблоком")
        # команду шлём только суставу клешни, остальные держатся на месте
        def send_gripper(angle):
            arm.send_action({keys["gripper"]: angle})

        grip_deg, reason = close_until_contact(
            send_gripper, arm.bus,
            start_deg=GRIPPER_OPEN_DEG, closed_deg=GRIPPER_CLOSED_DEG)
        cur[GRIP_IDX] = grip_deg
        log.event("grasp", reason=reason, gripper_deg=grip_deg)
        log.tick("grasped", joints=cur, tip_xyz=kin.fk(cur),
                 frames=grab_frames(cams), bus=arm.bus, force=True)
        if reason == "closed":
            print("  Клешня сомкнулась вхолостую — яблоко не поймано, не поднимаю.")
            note = "клешня сомкнулась вхолостую"
            return

        time.sleep(0.3)
        print("3/5 поднимаю")
        cur = follow_path(arm, keys, cur, plan["lift"], grip_deg=grip_deg,
                          log=log, phase="lift", kin=kin, cams=cams)
        log.tick("lifted", joints=cur, tip_xyz=kin.fk(cur),
                 frames=grab_frames(cams), bus=arm.bus, force=True)

        print("4/5 несу к ящику и разжимаю клешню")
        cur, placed, note = place_in_box(arm, keys, kin, cur, grip_deg,
                                         GRIPPER_OPEN_DEG, log=log, cams=cams,
                                         grasp_xyz=plan["grasp_xyz"])
        if not placed:
            return

        print("5/5 возвращаюсь в исходную позу")
        cur = go_home(arm, keys, cur)
        print("Готово — яблоко в ящике, рука в исходной позе.")
        success = True
    finally:
        log.finish(success, note)
        arm.disconnect()
        cap_l.release(); cap_r.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

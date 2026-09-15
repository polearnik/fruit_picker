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
GRASP_OFFSET = np.array([0., -10., -15.])
LIFT_MM = 100.0                       # на сколько поднять после захвата
GRIPPER_OPEN_DEG = 95.0               # угол раскрытой клешни
GRIPPER_CLOSED_DEG = 10.0              # ЖЁСТКИЙ ПРЕДЕЛ сжатия: сильнее не сожмём
                                       # никогда. Реальная остановка — по контакту
                                       # (см. arm/gripper.py), обычно раньше.

# --- точка сброса в ящик (СНИМАЕТСЯ РУКОЙ, не рулеткой) ---
# Положение КОНЧИКА (кадр gripper_frame_link), при котором разжатая клешня
# роняет яблоко в ящик. Это НЕ «край ящика» и НЕ «центр ящика»: яблоко висит в
# клешне ниже и позади кончика (см. GRASP_OFFSET), поэтому пересчитывать высоту
# в уме бесполезно — точку надо снять той же FK, которой считает планировщик:
#     python arm/teach_point.py     (вложить яблоко, подвести рукой, Enter)
# Снимайте так, чтобы яблоко было по центру ящика и лишь чуть выше его края:
# каждый лишний сантиметр высоты — это отскок и промах мимо ящика.
# Значение ниже — точка, промеренная рулеткой в первом заезде; переснимите её.
RELEASE_XYZ = np.array([189.5, 315.3, -128.2])
CARRY_CLEARANCE_MM = 120.0  # высота пролёта над ящиком по пути к нему
RETREAT_MM = 70.0           # уйти вверх после того, как отпустили яблоко
# Исходная («домашняя») поза: рука вытянута вперёд, все суставы с запасом ~55°
# до пределов. Из сложенной позы, где суставы упёрты в механические концы,
# планировать нельзя — сначала приводим руку сюда.
HOME_JOINTS = np.array([0., -30., 40., -20., 0., 45.])

# --- параметры движения (ЗДЕСЬ КРУТИТСЯ СКОРОСТЬ) ---
# Темп задаёт пара «шаг/пауза»: STEP_DEG градусов за DT секунд, то есть
# STEP_DEG/DT градусов в секунду. Но на траектории соседние точки отстоят на
# CART_STEP_MM, и поворот сустава между ними обычно МЕНЬШЕ градуса — тогда шаг
# всего один, и STEP_DEG ни на что не влияет. Реальный пол скорости на
# траектории — это DT на каждую точку, поэтому ускоряют так:
#   DT меньше              — чаще шлём команды (следите за отставанием серв);
#   CART_STEP_TRAVEL больше — меньше точек на том же пути.
# STEP_DEG работает там, где ход большой: go_home и первый заход в позу.
STEP_DEG, DT, MAX_REL, TOL_MM = 5.0, 0.03, 15.0, 10.0
CART_STEP_MM = 10.0          # рядом с плодом и в ящике: точность важнее скорости
CART_STEP_TRAVEL_MM = 25.0   # в свободном пространстве: грубее и быстрее
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


def plan_place(kin, q_start, start_xyz, release_xyz=RELEASE_XYZ):
    """План переноса яблока в ящик и отхода после сброса.

    Тот же принцип, что и у захвата: тремя прямыми отрезками сверху, чтобы
    яблоко в клешне не задевало край ящика по дороге.
      1) вертикально вверх до высоты пролёта над ящиком;
      2) горизонтально до точки прямо над ящиком;
      3) вертикально вниз до точки сброса.
    """
    release_xyz = np.asarray(release_xyz, float)
    retreat_xyz = release_xyz + np.array([0, 0, RETREAT_MM])

    start_xyz = np.asarray(start_xyz, float)
    safe_z = max(release_xyz[2] + CARRY_CLEARANCE_MM, start_xyz[2])
    up_xyz = np.array([start_xyz[0], start_xyz[1], safe_z])
    over_xyz = np.array([release_xyz[0], release_xyz[1], safe_z])

    segments = [("подъём на высоту переноса", start_xyz, up_xyz, CART_STEP_TRAVEL_MM),
                ("перенос к ящику", up_xyz, over_xyz, CART_STEP_TRAVEL_MM),
                ("спуск в ящик", over_xyz, release_xyz, CART_STEP_MM)]

    q = np.asarray(q_start, float).copy()
    carry = []
    for name, a, b, step in segments:
        part, msg = segment_joints(kin, q, a, b, name, step_mm=step)
        if part is None:
            return None, msg
        carry.extend(part)
        q = part[-1]

    # отход вверх уже с пустой клешнёй — чтобы не зацепить сброшенный плод
    retreat, msg = segment_joints(kin, q, release_xyz, retreat_xyz, "отход от ящика",
                                  step_mm=CART_STEP_TRAVEL_MM)
    if retreat is None:
        return None, msg

    return {"release_xyz": release_xyz, "carry": carry, "retreat": retreat}, "ok"


def smooth_to(arm, keys, current, goal):
    n = max(1, int(np.max(np.abs(goal - current)) / STEP_DEG))
    for i in range(1, n + 1):
        q = current + (goal - current) * i / n
        arm.send_action({keys[j]: q[k] for k, j in enumerate(JOINT_NAMES)})
        time.sleep(DT)
    return goal.copy()


def follow_path(arm, keys, current, path, grip_deg=None,
                log=None, phase="", kin=None, cams=None):
    """Проходит список углов траектории, плавно между соседними точками."""
    cur = np.asarray(current, float).copy()
    for q in path:
        goal = q.copy()
        if grip_deg is not None:
            goal[GRIP_IDX] = grip_deg   # клешню держим на своём угле
        cur = smooth_to(arm, keys, cur, goal)
        if log is not None:
            # кадр читаем, только если логгер его действительно запишет:
            # два cap.read() стоят ~100 мс и раньше платились на КАЖДОЙ точке
            frames = grab_frames(cams) if log.wants_frame() else None
            log.tick(phase, joints=cur, tip_xyz=kin.fk(cur) if kin else None,
                     frames=frames, bus=arm.bus)
    return cur


def grab_frames(cams):
    """Кадры с обеих камер для записи эпизода (None, если камеры не переданы)."""
    if not cams:
        return None
    out = {}
    for name, cap in cams.items():
        ok, img = cap.read()
        out[name] = img if ok else None
    return out


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

    cams = {"left": cap_l, "right": cap_r}
    # Запись эпизода: кадры, углы, телеметрия, свои датчики. Чтобы добавить свой
    # датчик — передайте extra_sensors=функция, возвращающая словарь значений.
    log = EpisodeLogger(extra_sensors=None)
    log.set_meta(grasp_offset=GRASP_OFFSET, travel_clearance_mm=TRAVEL_CLEARANCE_MM,
                 lift_mm=LIFT_MM, gripper_open_deg=GRIPPER_OPEN_DEG,
                 gripper_limit_deg=GRIPPER_CLOSED_DEG, model="yolov8n.pt",
                 release_xyz=RELEASE_XYZ,
                 carry_clearance_mm=CARRY_CLEARANCE_MM)
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

        print("4/5 несу к ящику")
        # Высота сброса относительно высоты захвата — то, что видно глазом при
        # промахе «яблоко выпало слишком высоко». Печатаем ДО движения.
        drop_above_grasp = RELEASE_XYZ[2] - plan["grasp_xyz"][2]
        print(f"  точка сброса {np.round(RELEASE_XYZ, 1)} мм — по высоте это "
              f"{drop_above_grasp:+.0f} мм")
        print("  относительно точки захвата (переснять: python arm/teach_point.py)")
        place, msg = plan_place(kin, cur, kin.fk(cur))
        if place is None:
            print(msg)
            print("  До точки сброса рука не дотягивается: подвиньте ящик ближе")
            print("  или переснимите точку (arm/teach_point.py).")
            print("  Яблоко осталось в клешне.")
            log.event("place_failed", reason=msg, release_xyz=RELEASE_XYZ)
            note = f"взято, но до точки сброса не дотянуться: {msg}"
            return
        cur = follow_path(arm, keys, cur, place["carry"], grip_deg=grip_deg,
                          log=log, phase="carry", kin=kin, cams=cams)
        log.tick("over_box", joints=cur, tip_xyz=kin.fk(cur),
                 frames=grab_frames(cams), bus=arm.bus, force=True)

        print("5/5 разжимаю клешню — яблоко падает в ящик")
        # клешню открываем плавно, остальные суставы стоят на месте
        goal = cur.copy()
        goal[GRIP_IDX] = GRIPPER_OPEN_DEG
        cur = smooth_to(arm, keys, cur, goal)
        log.event("release", gripper_deg=GRIPPER_OPEN_DEG,
                  release_xyz=place["release_xyz"])
        time.sleep(0.5)
        log.tick("released", joints=cur, tip_xyz=kin.fk(cur),
                 frames=grab_frames(cams), bus=arm.bus, force=True)

        # отходим вверх с открытой клешнёй, чтобы не задеть сброшенный плод
        cur = follow_path(arm, keys, cur, place["retreat"],
                          grip_deg=GRIPPER_OPEN_DEG,
                          log=log, phase="retreat", kin=kin, cams=cams)
        cur = go_home(arm, keys, cur)
        print("Готово — яблоко в ящике, рука в исходной позе.")
        success, note = True, "яблоко сложено в ящик"
    finally:
        log.finish(success, note)
        arm.disconnect()
        cap_l.release(); cap_r.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

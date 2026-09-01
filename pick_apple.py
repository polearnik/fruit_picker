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
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

LEFT_CAM, RIGHT_CAM = 0, 2
FRAME_W, FRAME_H = 1280, 960
PORT, ROBOT_ID = "/dev/ttyACM0", "fruit_arm"

APPLE_CLASS = 47      # класс "apple" в COCO
CONF = 0.4

# --- параметры захвата (ПОДСТРАИВАЮТСЯ под вашу руку и клешню) ---
APPROACH_UP_MM = 40.0                 # насколько подводить ВЫШЕ яблока перед спуском
GRASP_OFFSET = np.array([-40., 0., -50.]) # x вперед y вбок z вверх поправка к точке захвата (калибровочный сдвиг)
LIFT_MM = 100.0                       # на сколько поднять после захвата
GRIPPER_OPEN_DEG = 45.0               # угол раскрытой клешни
GRIPPER_CLOSED_DEG = 20.0              # угол сжатой клешни (под размер яблока)
# --- параметры движения ---
STEP_DEG, DT, MAX_REL, TOL_MM = 1.0, 0.03, 15.0, 10.0
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


def plan_grasp(kin, handeye, cam_xyz, current_joints):
    """cam XYZ -> план (углы pre-grasp, grasp, точки, флаг успеха, сообщение)."""
    target = handeye.cam_to_arm(cam_xyz)
    grasp_xyz = target + GRASP_OFFSET
    pre_xyz = grasp_xyz + np.array([0, 0, APPROACH_UP_MM])
    lift_xyz = grasp_xyz + np.array([0, 0, LIFT_MM])

    q_pre, e_pre, ok_pre = kin.ik(pre_xyz, current_joints, tol_mm=TOL_MM)
    q_grasp, e_grasp, ok_grasp = kin.ik(grasp_xyz, q_pre, tol_mm=TOL_MM)
    q_lift, e_lift, ok_lift = kin.ik(lift_xyz, q_grasp, tol_mm=TOL_MM)

    for name, q, ok, e in [("подвод", q_pre, ok_pre, e_pre),
                           ("захват", q_grasp, ok_grasp, e_grasp),
                           ("подъём", q_lift, ok_lift, e_lift)]:
        if not ok:
            return None, f"[СТОП] точка '{name}' недостижима (ошибка {e:.0f} мм)"
        bad = kin.within_limits(q)
        if bad:
            return None, f"[СТОП] '{name}': сустав за пределом: {[b[0] for b in bad]}"
    return {"target": target, "pre": q_pre, "grasp": q_grasp, "lift": q_lift}, "ok"


def smooth_to(arm, keys, current, goal):
    n = max(1, int(np.max(np.abs(goal - current)) / STEP_DEG))
    for i in range(1, n + 1):
        q = current + (goal - current) * i / n
        arm.send_action({keys[j]: q[k] for k, j in enumerate(JOINT_NAMES)})
        time.sleep(DT)
    return goal.copy()


def read_joints(arm, keys):
    obs = arm.get_observation()
    return np.array([obs[keys[j]] for j in JOINT_NAMES], dtype=float)


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
                print("  попробуйте сдвинуть яблоко или уменьшить APPROACH_UP_MM.")
            return
        # ans = input("Выполнить захват? Смотрите на руку. [y/N] ").strip().lower()
        # if ans != "y":
        #     print("Отменено.")
        #     return

        # последовательность захвата
        cur = current.copy()
        open_pre = plan["pre"].copy();   open_pre[GRIP_IDX] = GRIPPER_OPEN_DEG
        open_grasp = plan["grasp"].copy(); open_grasp[GRIP_IDX] = GRIPPER_OPEN_DEG
        closed_grasp = plan["grasp"].copy(); closed_grasp[GRIP_IDX] = GRIPPER_CLOSED_DEG
        closed_lift = plan["lift"].copy(); closed_lift[GRIP_IDX] = GRIPPER_CLOSED_DEG

        print("1/4 подвожу над яблоком, клешня открыта")
        cur = smooth_to(arm, keys, cur, open_pre)
        print("2/4 опускаюсь к яблоку")
        cur = smooth_to(arm, keys, cur, open_grasp)
        print("3/4 сжимаю клешню")
        cur = smooth_to(arm, keys, cur, closed_grasp)
        time.sleep(0.5)
        print("4/4 поднимаю")
        cur = smooth_to(arm, keys, cur, closed_lift)
        print("Готово — яблоко поднято.")
    finally:
        arm.disconnect()
        cap_l.release(); cap_r.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

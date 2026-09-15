"""Перенос сорванного яблока в ящик: донести, разжать клешню, отойти.

Отдельно от захвата, потому что это самостоятельное действие со своей
калибровкой: точка сброса снимается рукой (arm/teach_point.py) и от параметров
захвата не зависит.

Путь к ящику строится теми же тремя прямыми отрезками, что и заход к плоду —
вверх, горизонтально, вниз, — чтобы яблоко в клешне не задевало край ящика.

  plan_place    — построить траекторию (ничем не двигает, только считает)
  place_in_box  — выполнить перенос целиком, с записью в эпизод
"""

import time

import numpy as np

from motion import (CART_STEP_MM, CART_STEP_TRAVEL_MM, GRIP_IDX,
                    follow_path, grab_frames, segment_joints, smooth_to)

# --- точка сброса в ящик (СНИМАЕТСЯ РУКОЙ, не рулеткой) ---
# Положение КОНЧИКА (кадр gripper_frame_link), при котором разжатая клешня
# роняет яблоко в ящик. Это НЕ «край ящика» и НЕ «центр ящика»: яблоко висит в
# клешне ниже и позади кончика (см. GRASP_OFFSET в pick_apple.py), поэтому
# пересчитывать высоту в уме бесполезно — точку надо снять той же FK, которой
# считает планировщик:
#     python arm/teach_point.py     (вложить яблоко, подвести рукой, Enter)
# Снимайте так, чтобы яблоко было по центру ящика и лишь чуть выше его края:
# каждый лишний сантиметр высоты — это отскок и промах мимо ящика.
RELEASE_XYZ = np.array([189.5, 315.3, -128.2])
CARRY_CLEARANCE_MM = 120.0  # высота пролёта над ящиком по пути к нему
RETREAT_MM = 70.0           # уйти вверх после того, как отпустили яблоко
SETTLE_S = 0.5              # пауза после разжатия, чтобы плод успел выпасть


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


def place_in_box(arm, keys, kin, current, grip_deg, open_deg,
                 log=None, cams=None, release_xyz=RELEASE_XYZ, grasp_xyz=None):
    """Несёт яблоко к ящику, разжимает клешню, отходит вверх.

    grip_deg — на каком угле клешня держит плод (её нельзя разжимать по дороге);
    open_deg — угол, на котором она выпускает плод;
    grasp_xyz — точка захвата, если известна: только чтобы напечатать, насколько
    сброс выше неё (именно это видно глазом, когда яблоко падает мимо ящика).

    Возвращает (углы после отхода, получилось ли, пояснение для лога эпизода).
    Если до точки сброса не дотянуться — рука не двигается, яблоко остаётся в
    клешне: уронить его на полпути хуже, чем честно остановиться.
    """
    cur = np.asarray(current, float).copy()
    release_xyz = np.asarray(release_xyz, float)

    msg_height = ""
    if grasp_xyz is not None:
        msg_height = (f" — по высоте это {release_xyz[2] - grasp_xyz[2]:+.0f} мм "
                      "относительно точки захвата")
    print(f"  точка сброса {np.round(release_xyz, 1)} мм{msg_height}")
    print("  (переснять: python arm/teach_point.py)")

    plan, msg = plan_place(kin, cur, kin.fk(cur), release_xyz)
    if plan is None:
        print(msg)
        print("  До точки сброса рука не дотягивается: подвиньте ящик ближе")
        print("  или переснимите точку (arm/teach_point.py).")
        print("  Яблоко осталось в клешне.")
        if log is not None:
            log.event("place_failed", reason=msg, release_xyz=release_xyz)
        return cur, False, f"взято, но до точки сброса не дотянуться: {msg}"

    cur = follow_path(arm, keys, cur, plan["carry"], grip_deg=grip_deg,
                      log=log, phase="carry", kin=kin, cams=cams)
    if log is not None:
        log.tick("over_box", joints=cur, tip_xyz=kin.fk(cur),
                 frames=grab_frames(cams), bus=arm.bus, force=True)

    print("  разжимаю клешню — яблоко падает в ящик")
    # клешню открываем плавно, остальные суставы стоят на месте
    goal = cur.copy()
    goal[GRIP_IDX] = open_deg
    cur = smooth_to(arm, keys, cur, goal)
    if log is not None:
        log.event("release", gripper_deg=open_deg, release_xyz=plan["release_xyz"])
    time.sleep(SETTLE_S)
    if log is not None:
        log.tick("released", joints=cur, tip_xyz=kin.fk(cur),
                 frames=grab_frames(cams), bus=arm.bus, force=True)

    # отходим вверх с открытой клешнёй, чтобы не задеть сброшенный плод
    cur = follow_path(arm, keys, cur, plan["retreat"], grip_deg=open_deg,
                      log=log, phase="retreat", kin=kin, cams=cams)
    return cur, True, "яблоко сложено в ящик"

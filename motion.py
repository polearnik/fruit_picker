"""Как рука ЕДЕТ: разбиение пути на точки и проход по ним с записью эпизода.

Общая часть для всех действий руки (захват плода, перенос в ящик, возврат):
и планировщику захвата, и планировщику переноса нужен один и тот же приём —
провести кончик по ПРЯМОЙ, проверяя достижимость каждой точки.

Здесь же живут регуляторы скорости, потому что скорость задаётся именно тем,
как часто мы шлём команды и на сколько точек разбит путь.

  segment_joints  — прямой отрезок в пространстве -> список углов
  follow_path     — проход по списку углов, с тиками в лог эпизода
  smooth_to       — плавный переход между двумя позами
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "arm"))

import numpy as np

from kinematics import JOINT_NAMES

# --- ЗДЕСЬ КРУТИТСЯ СКОРОСТЬ ---
# Темп задаёт пара «шаг/пауза»: STEP_DEG градусов за DT секунд, то есть
# STEP_DEG/DT градусов в секунду. Но на траектории соседние точки отстоят на
# CART_STEP_MM, и поворот сустава между ними обычно МЕНЬШЕ градуса — тогда шаг
# всего один, и STEP_DEG ни на что не влияет. Реальный пол скорости на
# траектории — это DT на каждую точку, поэтому ускоряют так:
#   DT меньше              — чаще шлём команды (следите за отставанием серв);
#   CART_STEP_TRAVEL больше — меньше точек на том же пути.
# STEP_DEG работает там, где ход большой: go_home и первый заход в позу.
STEP_DEG, DT, TOL_MM = 5.0, 0.03, 10.0
CART_STEP_MM = 10.0          # рядом с плодом и в ящике: точность важнее скорости
CART_STEP_TRAVEL_MM = 25.0   # в свободном пространстве: грубее и быстрее

GRIP_IDX = JOINT_NAMES.index("gripper")


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

"""Кинематика руки SO-101: углы суставов <-> положение кончика в пространстве.

Обёртка над placo-решателем из lerobot. Работает в МИЛЛИМЕТРАХ (как наше
стереозрение), суставы — в градусах, в порядке нашей руки.

- fk(углы)            -> XYZ кончика (мм)
- ik(XYZ)             -> углы суставов, ошибка попадания, достижима ли точка

placo-решатель дифференциальный: один его шаг лишь чуть двигает к цели, поэтому
ik() вызывает его в цикле до сходимости.

Запуск напрямую — самопроверка (IK round-trip):  python arm/kinematics.py
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from lerobot.model.kinematics import RobotKinematics

# Порядок суставов — как в наблюдениях руки и в калибровке fruit_arm.
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]
TARGET_FRAME = "gripper_frame_link"
URDF_PATH = Path(__file__).parent / "urdf" / "so101_new_calib.urdf"

# Шаг времени решателя. Нужен, чтобы placo соблюдал пределы суставов;
# на результат IK не влияет, задаёт лишь темп сходимости за итерацию.
SOLVER_DT = 0.05


class ArmKinematics:
    def __init__(self, urdf_path=URDF_PATH):
        self._kin = RobotKinematics(str(urdf_path), TARGET_FRAME, JOINT_NAMES)
        self.joint_names = JOINT_NAMES
        self.limits_deg = self._read_limits(urdf_path)  # {сустав: (мин, макс)} в градусах

        # Задача «попади в точку» задаёт 3 координаты, а свободных суставов 4 —
        # лишняя степень свободы позволяла решателю уводить wrist_flex за предел.
        # Просим placo самому соблюдать пределы из URDF (нужен заданный dt).
        self._kin.solver.dt = SOLVER_DT
        self._kin.solver.enable_joint_limits(True)

    @staticmethod
    def _read_limits(urdf_path):
        root = ET.parse(str(urdf_path)).getroot()
        limits = {}
        for j in root.findall("joint"):
            lim = j.find("limit")
            if j.get("type") == "revolute" and lim is not None:
                limits[j.get("name")] = (np.rad2deg(float(lim.get("lower"))),
                                         np.rad2deg(float(lim.get("upper"))))
        return limits

    def clamp_to_limits(self, joints_deg, margin_deg=2.5):
        """Загоняет углы в пределы URDF.

        Физический диапазон руки ШИРЕ модельного (калибровка lerobot считает
        градусы от середины реального хода). Если сустав припаркован за пределом
        модели, планировать от такой позы нельзя — зажимаем её в допустимую.
        """
        q = np.asarray(joints_deg, dtype=float).copy()
        for i, name in enumerate(self.joint_names):
            if name in self.limits_deg:
                lo, hi = self.limits_deg[name]
                q[i] = float(np.clip(q[i], lo + margin_deg, hi - margin_deg))
        return q

    def within_limits(self, joints_deg, margin_deg=0.0):
        """Список суставов, вышедших за пределы URDF (с запасом margin). Пусто = всё ок.

        Запас по умолчанию нулевой: пределы соблюдает сам решатель (см.
        enable_joint_limits в __init__), причём он законно прижимает сустав
        вплотную к пределу. Это лишь контрольная проверка; ненулевой запас
        забраковал бы такие корректные траектории.
        """
        bad = []
        for name, val in zip(self.joint_names, np.asarray(joints_deg, dtype=float)):
            if name in self.limits_deg:
                lo, hi = self.limits_deg[name]
                if val < lo + margin_deg or val > hi - margin_deg:
                    bad.append((name, float(val), lo, hi))
        return bad

    def fk(self, joints_deg):
        """Углы суставов (градусы) -> XYZ кончика в мм."""
        joints_deg = np.asarray(joints_deg, dtype=float)
        T = self._kin.forward_kinematics(joints_deg)
        return T[:3, 3] * 1000.0  # метры -> мм

    def ik(self, target_xyz_mm, current_joints_deg=None, iters=100,
           position_only=True, tol_mm=2.0, fix_wrist_roll=0.0):
        """XYZ цели (мм) -> (углы суставов, ошибка попадания мм, достижима ли).

        position_only=True: держим только позицию, ориентацию кончика оставляем
        свободной (для 5-суставной руки этого достаточно, чтобы дотянуться до точки).

        fix_wrist_roll: wrist_roll почти не влияет на положение кончика, поэтому в
        position-only решателе он "свободный" и может уплыть за предел. Пиним его на
        заданный угол (позицию добирают остальные суставы), заодно даёт стабильный
        поворот клешни. None — не фиксировать.

        Если ошибка > tol_mm — точка вне досягаемости, углы брать нельзя.
        """
        target = np.asarray(target_xyz_mm, dtype=float)
        T = np.eye(4)
        T[:3, 3] = target / 1000.0  # мм -> метры для placo

        q = (np.zeros(len(JOINT_NAMES)) if current_joints_deg is None
             else np.asarray(current_joints_deg, dtype=float).copy())
        wr = JOINT_NAMES.index("wrist_roll")
        if fix_wrist_roll is not None:
            q[wr] = fix_wrist_roll
        orient_w = 0.0 if position_only else 0.01
        for _ in range(iters):
            q = self._kin.inverse_kinematics(q, T, position_weight=1.0,
                                             orientation_weight=orient_w)
            if fix_wrist_roll is not None:
                q[wr] = fix_wrist_roll  # держим wrist_roll на месте

        err_mm = float(np.linalg.norm(self.fk(q) - target))
        return q, err_mm, err_mm <= tol_mm


def _self_test():
    kin = ArmKinematics()
    print("Суставы:", kin.joint_names)
    print("Кончик при всех углах 0:", np.round(kin.fk(np.zeros(6)), 1), "мм\n")

    rng = np.random.default_rng(0)
    errs = []
    for _ in range(12):
        true_deg = rng.uniform(-45, 45, size=6)
        true_deg[5] = 0.0  # захват на положение кончика не влияет
        target = kin.fk(true_deg)              # заведомо достижимая точка
        q, err, ok = kin.ik(target)
        assert ok, f"не сошлось для {target}, ошибка {err:.2f} мм"
        errs.append(err)
    print(f"IK round-trip на 12 достижимых точках: "
          f"средняя ошибка {np.mean(errs):.4f} мм, макс {np.max(errs):.4f} мм")

    # заведомо недостижимая точка (2 метра вперёд) должна честно отметиться
    _, err_far, ok_far = kin.ik([2000, 0, 300])
    print(f"Недостижимая точка (2 м): ошибка {err_far:.0f} мм, достижима={ok_far}")
    assert not ok_far, "далёкая точка не должна считаться достижимой"
    print("\nСамопроверка пройдена: IK попадает в достижимые точки и честно "
          "отбраковывает недостижимые.")


if __name__ == "__main__":
    _self_test()

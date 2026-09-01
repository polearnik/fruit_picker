"""Доехать кончиком руки до точки (X, Y, Z) в системе координат РУКИ.

Читает текущие углы -> считает IK для цели -> проверяет безопасность ->
показывает план и ждёт подтверждения -> плавно ведёт руку в цель.

Система координат — основание руки (как в URDF), в миллиметрах.
ВНИМАНИЕ: это координаты руки, НЕ камеры. Связь камера->рука появится
на этапе 4 (hand-eye калибровка). Пока цель задаём в координатах руки.

Запуск:
  python arm/move_to_point.py            # цель по умолчанию: текущий кончик + 40 мм вверх
  python arm/move_to_point.py 250 0 200  # цель X=250 Y=0 Z=200 мм

Безопасность: рука в свободном пространстве, выключатель питания под рукой,
смотрите на руку. Движение плавное и медленное; при подтверждении читайте план.
"""

import sys
import time

import numpy as np

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from kinematics import ArmKinematics, JOINT_NAMES

PORT = "/dev/ttyACM0"
ROBOT_ID = "fruit_arm"

STEP_DEG = 1.0      # размер плавного шага
DT = 0.04           # пауза между шагами, сек
MAX_REL = 15.0      # аппаратный предел на команду
TOL_MM = 5.0        # цель считаем достигнутой, если IK попал ближе этого


def obs_key(obs, joint):
    """Ключ наблюдения/действия для сустава (напр. 'shoulder_pan.pos')."""
    return next(k for k in obs if joint in k and isinstance(obs[k], (int, float)))


def read_joints(arm, keys):
    obs = arm.get_observation()
    return np.array([obs[keys[j]] for j in JOINT_NAMES], dtype=float)


def main():
    kin = ArmKinematics()

    # цель из аргументов или по умолчанию
    target = None
    if len(sys.argv) == 4:
        target = np.array([float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])])

    config = SO101FollowerConfig(port=PORT, id=ROBOT_ID, use_degrees=True,
                                 max_relative_target=MAX_REL)
    arm = SO101Follower(config)
    arm.connect(calibrate=False)
    try:
        obs = arm.get_observation()
        keys = {j: obs_key(obs, j) for j in JOINT_NAMES}
        current = read_joints(arm, keys)
        current_xyz = kin.fk(current)

        if target is None:  # по умолчанию — небольшой безопасный подъём на 40 мм
            target = current_xyz + np.array([0.0, 0.0, 40.0])

        # IK от текущей позы (ближе -> меньше ход, лучше сходимость)
        goal, err, reachable = kin.ik(target, current_joints_deg=current, tol_mm=TOL_MM)

        print("\n=== ПЛАН ДВИЖЕНИЯ ===")
        print(f"кончик сейчас:  {np.round(current_xyz, 1)} мм")
        print(f"цель:           {np.round(target, 1)} мм")
        print(f"ошибка IK:      {err:.1f} мм (порог {TOL_MM})")
        print("углы сейчас ->  цель (градусы):")
        for j, c, g in zip(JOINT_NAMES, current, goal):
            print(f"  {j:15s} {c:7.1f} -> {g:7.1f}")

        # ПРОВЕРКА 1: достижимость
        if not reachable:
            print(f"\n[СТОП] Точка недостижима (ошибка {err:.0f} мм). Не двигаюсь.")
            return
        # ПРОВЕРКА 2: пределы суставов
        bad = kin.within_limits(goal)
        if bad:
            print("\n[СТОП] Суставы выходят за пределы, не двигаюсь:")
            for name, val, lo, hi in bad:
                print(f"  {name}: {val:.1f} (предел {lo:.1f}..{hi:.1f})")
            return

        # ПРОВЕРКА 3: подтверждение человеком
        ans = input("\nВыполнить движение? Смотрите на руку. [y/N] ").strip().lower()
        if ans != "y":
            print("Отменено.")
            return

        # плавное движение: интерполяция углов от current к goal
        n = max(1, int(np.max(np.abs(goal - current)) / STEP_DEG))
        print(f"Двигаюсь за {n} шагов...")
        for i in range(1, n + 1):
            q = current + (goal - current) * i / n
            arm.send_action({keys[j]: q[k] for k, j in enumerate(JOINT_NAMES)})
            time.sleep(DT)

        final_xyz = kin.fk(read_joints(arm, keys))
        print(f"\nГотово. Кончик: {np.round(final_xyz, 1)} мм, "
              f"промах от цели {np.linalg.norm(final_xyz - target):.1f} мм.")
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()

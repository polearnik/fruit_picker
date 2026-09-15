"""Показать руке точку рукой: подвести кончик куда надо и записать её XYZ.

Зачем: координаты, промеренные рулеткой, живут в вашей системе отсчёта, а
планировщик — в системе URDF (FK от основания). Любое расхождение между ними
превращается в промах. Этот скрипт снимает точку ТОЙ ЖЕ формулой FK, которой
потом пользуется планировщик, поэтому расхождению взяться неоткуда.

Как снять точку сброса в ящик:
  1. вложите яблоко в клешню (клешня остаётся под напряжением и держит его);
  2. запустите скрипт — остальные суставы обмякнут, рука станет мягкой;
  3. РУКОЙ подведите яблоко туда, где его надо выпустить: по центру ящика,
     чуть выше его края (яблоко должно просто упасть, а не лететь);
  4. нажмите Enter — скрипт напечатает строку для вставки в pick_apple.py.

ВНИМАНИЕ: при выключенном моменте рука падает под своим весом. Держите её
рукой ДО запуска и до конца работы скрипта; освободите пространство под ней.

Запуск:  python arm/teach_point.py
"""

import select
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from kinematics import ArmKinematics, JOINT_NAMES
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

PORT, ROBOT_ID = "/dev/ttyACM0", "fruit_arm"

# Клешню НЕ расслабляем: она должна продолжать держать яблоко, пока вы
# подводите руку к ящику. Обмякают только суставы, которые вы двигаете.
LIMP_JOINTS = [j for j in JOINT_NAMES if j != "gripper"]
POLL_DT = 0.15


def read_joints(arm, keys):
    obs = arm.get_observation()
    return np.array([obs[keys[j]] for j in JOINT_NAMES], dtype=float)


def main():
    kin = ArmKinematics()
    arm = SO101Follower(SO101FollowerConfig(port=PORT, id=ROBOT_ID, use_degrees=True))
    arm.connect(calibrate=False)
    try:
        obs = arm.get_observation()
        keys = {j: next(k for k in obs if j in k and isinstance(obs[k], (int, float)))
                for j in JOINT_NAMES}

        print("Держите руку! Отключаю момент на суставах:", ", ".join(LIMP_JOINTS))
        time.sleep(1.0)
        arm.bus.disable_torque(LIMP_JOINTS)

        print("Рука мягкая. Подведите кончик (или яблоко в клешне) в нужную точку")
        print("и нажмите Enter. Ctrl+C — выйти без записи.\n")

        xyz = kin.fk(read_joints(arm, keys))
        while True:
            q = read_joints(arm, keys)
            xyz = kin.fk(q)
            print(f"\r  кончик: X={xyz[0]:7.1f}  Y={xyz[1]:7.1f}  Z={xyz[2]:7.1f} мм ",
                  end="", flush=True)
            if select.select([sys.stdin], [], [], POLL_DT)[0]:
                sys.stdin.readline()
                break

        print(f"\n\nТочка снята: {np.round(xyz, 1)} мм (система руки, как в FK).")
        print("Вставьте в шапку pick_apple.py:\n")
        print(f"RELEASE_XYZ = np.array([{xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}])")
        print("\nЕсли снимали С ЯБЛОКОМ в клешне — это готовая точка сброса, "
              "поправки не нужны.")
    except KeyboardInterrupt:
        print("\nОтменено, ничего не записано.")
    finally:
        # disconnect у lerobot снимает момент со всех суставов — рука обмякнет
        # целиком, включая клешню. Держите её.
        print("Отключаюсь — рука обмякнет полностью, придержите её.")
        arm.disconnect()


if __name__ == "__main__":
    main()

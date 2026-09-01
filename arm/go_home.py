"""Развернуть руку в исходную позу (и обратно сложить).

Из сложенной позы, где суставы упёрты в механические концы, планировать захват
нельзя — решателю негде манёврировать. Этот скрипт приводит руку в удобное
исходное положение: вытянута вперёд, запас до всех пределов ~55°.

Запуск:
  python arm/go_home.py          — развернуть в исходную позу
  python arm/go_home.py park     — сложить обратно (компактно, для хранения)

ВНИМАНИЕ: из сложенной позы ход большой, рука разворачивается широко.
Освободите пространство, держите выключатель питания под рукой.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from kinematics import ArmKinematics, JOINT_NAMES
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

PORT, ROBOT_ID = "/dev/ttyACM0", "fruit_arm"

HOME_JOINTS = np.array([0., -30., 40., -20., 0., 45.])   # вытянута вперёд
PARK_JOINTS = np.array([0., -85., 85., -60., 0., 30.])   # сложена компактно

STEP_DEG, DT, MAX_REL = 1.0, 0.06, 15.0


def main():
    park = len(sys.argv) > 1 and sys.argv[1].lower().startswith("park")
    goal = PARK_JOINTS if park else HOME_JOINTS
    name = "сложенную (park)" if park else "исходную (home)"

    kin = ArmKinematics()
    bad = kin.within_limits(goal)
    if bad:
        raise SystemExit(f"Целевая поза вне пределов модели: {[b[0] for b in bad]}")

    arm = SO101Follower(SO101FollowerConfig(port=PORT, id=ROBOT_ID,
                                            use_degrees=True, max_relative_target=MAX_REL))
    arm.connect(calibrate=False)
    try:
        obs = arm.get_observation()
        keys = {j: next(k for k in obs if j in k and isinstance(obs[k], (int, float)))
                for j in JOINT_NAMES}
        current = np.array([obs[keys[j]] for j in JOINT_NAMES], dtype=float)

        print("сейчас:", np.round(current, 1))
        print(f"цель ({name}):", np.round(goal, 1))
        delta = np.max(np.abs(goal - current))
        print(f"максимальный ход сустава: {delta:.0f}°")

        ans = input("Двигаю? Смотрите на руку, освободите пространство. [y/N] ")
        if ans.strip().lower() != "y":
            print("Отменено.")
            return

        n = max(1, int(delta / STEP_DEG))
        for i in range(1, n + 1):
            q = current + (goal - current) * i / n
            arm.send_action({keys[j]: q[k] for k, j in enumerate(JOINT_NAMES)})
            time.sleep(DT)

        final = np.array([arm.get_observation()[keys[j]] for j in JOINT_NAMES], dtype=float)
        print("итог:", np.round(final, 1))
        print(f"кончик руки: {np.round(kin.fk(final), 1)} мм")
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()

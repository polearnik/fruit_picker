"""Замер нагрузки клешни — чтобы подобрать порог контакта под своё железо.

Медленно сжимает клешню от раскрытой до закрытой и печатает на каждом шаге
нагрузку сервопривода и отставание факта от команды. Ничего не решает сам —
просто показывает цифры.

Как пользоваться:
  1) python arm/gripper_probe.py            — прогон ВХОЛОСТУЮ (клешня пустая)
     запомните типичную нагрузку холостого хода (обычно небольшая).
  2) вложите яблоко между пальцами и повторите
     запомните, до какой нагрузки подскакивает при сжатии на яблоке.
  3) LOAD_THRESHOLD в arm/gripper.py поставьте ПОСЕРЕДИНЕ между этими числами.

Рука должна быть в безопасной позе; двигается только клешня.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from gripper import read_load, read_pos
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

PORT, ROBOT_ID = "/dev/ttyACM0", "fruit_arm"
OPEN_DEG, CLOSED_DEG = 45.0, 8.0
STEP_DEG, DT = 1.5, 0.12


def main():
    arm = SO101Follower(SO101FollowerConfig(port=PORT, id=ROBOT_ID,
                                            use_degrees=True, max_relative_target=15.0))
    arm.connect(calibrate=False)
    try:
        key = next(k for k in arm.get_observation() if "gripper" in k)
        print(f"Раскрываю клешню до {OPEN_DEG}°...")
        arm.send_action({key: OPEN_DEG})
        time.sleep(1.0)

        print(f"\nСжимаю до {CLOSED_DEG}°. Столбцы: команда | факт | отставание | нагрузка\n")
        angle = OPEN_DEG
        loads = []
        while angle > CLOSED_DEG:
            angle = max(CLOSED_DEG, angle - STEP_DEG)
            arm.send_action({key: angle})
            time.sleep(DT)
            load = read_load(arm.bus)
            actual = read_pos(arm.bus)
            loads.append(load)
            print(f"  {angle:6.1f}° | {actual:6.1f}° | {abs(angle-actual):5.1f}° | {load:4d}")

        print(f"\nНагрузка: минимум {min(loads)}, максимум {max(loads)}, "
              f"медиана {sorted(loads)[len(loads)//2]}")
        print("Сравните прогон вхолостую и прогон с яблоком, порог — между ними.")

        print(f"\nВозвращаю клешню в {OPEN_DEG}°")
        arm.send_action({key: OPEN_DEG})
        time.sleep(0.8)
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()

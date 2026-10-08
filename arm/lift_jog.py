"""Ручное управление подъёмником короткими толчками — проверка направления.

Команды мотору — «сырые», БЕЗ UP_SIGN: «+» значит положительный сдвиг на
моторе, «-» отрицательный. Смотрите, куда при этом едет каретка, и
впишите в arm/lift.py:
    «+» везёт вверх  ->  UP_SIGN = +1
    «-» везёт вверх  ->  UP_SIGN = -1

Каждый толчок — пол-оборота (20 мм) на медленной скорости. Остановка после
толчка — сама; между толчками момент включён.

Управление (Enter после символа):
  +   толчок положительным сдвигом
  -   толчок отрицательным сдвигом
  q   выход

Запуск:  python arm/lift_jog.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import lift as L

JOG_STEPS = L.STEPS_PER_REV // 2   # 20 мм
JOG_SPEED = 1000                   # ~10 мм/с


def main():
    lift = L.Lift.open()
    try:
        lift.setup()
        lift._w2(L.ADDR_GOAL_SPEED, JOG_SPEED, "скорость")
        print("Каретка держится. Вводите + или - (и Enter), q — выход.")
        while True:
            cmd = input("> ").strip()
            if cmd == "q":
                break
            if cmd not in ("+", "-"):
                continue
            steps = JOG_STEPS if cmd == "+" else -JOG_STEPS
            lift._w2(L.ADDR_GOAL_POS, L._sign_mag_encode(steps, 15), "толчок")
            t0 = time.time()
            peak = 0
            time.sleep(0.1)
            while lift._r1(L.ADDR_MOVING, "Moving") and time.time() - t0 < 6:
                peak = max(peak, lift.load())
                time.sleep(0.02)
            print(f"  сдвиг {steps:+d}: готово, пик нагрузки {peak}. "
                  f"Куда поехала каретка — вверх или вниз?")
    finally:
        try:
            lift._w2(L.ADDR_GOAL_SPEED, L.SPEED, "скорость")
        finally:
            lift.close()


if __name__ == "__main__":
    main()

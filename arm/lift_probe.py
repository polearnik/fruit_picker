"""Замер: что мотор подъёмника отдаёт в регистрах ВО ВРЕМЯ шагового сдвига.

Нужен, чтобы понять, как считать высоту: в шаговом режиме прошивка 3.10
отдаёт Present_Position не так, как в обычном (после сдвига на оборот там
снова ~0). Скрипт делает один оборот вверх (40 мм) и один обратно, опрашивая
регистры каждые ~20 мс, и печатает таблицу.

Перед запуском: над кареткой 100 мм свободного хода, другие скрипты руки
закрыты.

Запуск:  python arm/lift_probe.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import lift as L


def sample(lift, t0):
    return (time.time() - t0,
            lift._r2(L.ADDR_PRESENT_POS, "pos"),
            lift._r2(L.ADDR_GOAL_POS, "goal"),
            L._sign_mag_decode(lift._r2(58, "speed"), 15),
            lift._r1(L.ADDR_MOVING, "moving"),
            L._sign_mag_decode(lift._r2(L.ADDR_PRESENT_LOAD, "load"), 10))


def run(lift, steps, label):
    print(f"\n--- {label}: сдвиг {steps:+d} ---")
    print("   t, с   pos   goal   speed  moving  load")
    t0 = time.time()
    before = sample(lift, t0)
    print("%7.2f %5d %6d %7d %6d %6d  (до команды)" % before)
    lift._step(steps)
    still = 0
    while time.time() - t0 < 6:
        s = sample(lift, t0)
        print("%7.2f %5d %6d %7d %6d %6d" % s)
        still = still + 1 if s[4] == 0 and s[3] == 0 else 0
        if still >= 10:
            break
        time.sleep(0.01)


def main():
    if input("Сделать оборот ВВЕРХ (40 мм) и обратно? [y/N] ").strip().lower() != "y":
        return
    lift = L.Lift.open()
    try:
        lift.setup()
        run(lift, L.UP_SIGN * L.STEPS_PER_REV, "вверх 1 оборот")
        time.sleep(0.5)
        run(lift, -L.UP_SIGN * L.STEPS_PER_REV, "вниз 1 оборот")
    finally:
        lift.close()


if __name__ == "__main__":
    main()

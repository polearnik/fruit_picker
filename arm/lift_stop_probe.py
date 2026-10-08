"""Замер: как ОСТАНОВИТЬ мотор подъёмника посреди шагового сдвига.

Первый прогон (A, B, C — все «нет») показал: команды СКЛАДЫВАЮТСЯ — новая
прибавляется к недоеханному остатку. Поэтому теперь проверяем D: «минус
остаток» — цель становится равна текущему положению. Это то, что делает
Lift.halt(). Все движения — ВВЕРХ, от упора, по 2 оборота (80 мм) на
скорости хоминга. Остановка — через 0.8 с после старта. Два повтора.

  D: сдвиг на -(остаток)

Заодно печатает нагрузку на медленном ходу вверх — для порога хоминга.

Перед запуском: над кареткой 30 см свободного хода, рука у выключателя.

Запуск:  python arm/lift_stop_probe.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

import lift as L

ADDR_PRESENT_SPEED = 58
REVS = 2
STOP_AFTER_S = 0.8
WATCH_S = 1.5


def speed(lift):
    return L._sign_mag_decode(lift._r2(ADDR_PRESENT_SPEED, "скорость"), 15)


def remaining(lift):
    return L._sign_mag_decode(lift._r2(L.ADDR_PRESENT_POS, "остаток"), 15)


def stop_d(lift):
    lift._w2(L.ADDR_GOAL_POS, L._sign_mag_encode(-remaining(lift), 15), "минус остаток")


def trial(lift, name, stop):
    print(f"\n=== Способ {name} ===")
    stop_d(lift)  # от прошлых запусков мог остаться недоеханный хвост
    time.sleep(0.3)
    lift._w2(L.ADDR_GOAL_POS,
             L._sign_mag_encode(L.UP_SIGN * REVS * L.STEPS_PER_REV, 15), "старт")
    t0 = time.time()
    loads = []
    while time.time() - t0 < STOP_AFTER_S:
        loads.append(lift.load())
        time.sleep(0.02)
    print(f"  нагрузка на ходу вверх (скорость {L.HOME_SPEED}): "
          f"медиана {np.median(loads[len(loads)//2:]):.0f}, макс {max(loads)}")
    stop(lift)
    t1 = time.time()
    stopped_at = None
    while time.time() - t1 < WATCH_S:
        v = speed(lift)
        if v == 0 and stopped_at is None:
            stopped_at = time.time() - t1
        time.sleep(0.02)
    left = L._sign_mag_decode(lift._r2(L.ADDR_PRESENT_POS, "остаток"), 15)
    moving = lift._r1(L.ADDR_MOVING, "Moving")
    ok = stopped_at is not None and stopped_at < 0.3 and abs(left) < 200
    print(f"  после стопа: скорость 0 через "
          f"{'—' if stopped_at is None else f'{stopped_at:.2f} с'}, "
          f"остаток {left}, Moving={moving}")
    print(f"  => {'РАБОТАЕТ' if ok else 'НЕ работает (доехал старую команду)'}")
    time.sleep(0.5)
    return ok


def main():
    if input("Два коротких подъёма по 80 мм с остановкой посреди хода. "
             "Над кареткой 30 см свободно? [y/N] ").strip().lower() != "y":
        return
    lift = L.Lift.open()
    try:
        lift.setup()
        lift._w2(L.ADDR_GOAL_SPEED, L.HOME_SPEED, "скорость")
        res = {}
        for name, fn in (("D (минус остаток), раз 1", stop_d),
                         ("D (минус остаток), раз 2", stop_d)):
            res[name] = trial(lift, name, fn)
        print("\nИтог:")
        for k, v in res.items():
            print(f"  {k}: {'работает' if v else 'нет'}")
    finally:
        try:
            stop_d(lift)  # сначала погасить хвост, потом вернуть скорость
            lift._w2(L.ADDR_GOAL_SPEED, L.SPEED, "скорость")
        finally:
            lift.close()


if __name__ == "__main__":
    main()

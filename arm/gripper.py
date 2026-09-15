"""Захват «до контакта»: сжимаем клешню, пока не почувствуем яблоко.

Вместо слепого доведения до фиксированного угла закрываем клешню маленькими
шагами и после каждого проверяем два признака контакта:

  1. НАГРУЗКА (Present_Load сервопривода) выше порога — палец во что-то упёрся;
  2. ЗАСТРЕВАНИЕ — сустав не доходит до заданного угла (|цель - факт| велико),
     значит его что-то держит. Работает даже если нагрузка шумит.

Как только контакт подтверждён — останавливаемся (плюс маленький «дожим» для
надёжного удержания). Так мягкий плод не раздавливается, а мелкий не выпадает.

Порог нагрузки зависит от конкретной клешни — измерьте своим железом:
    python arm/gripper_probe.py
"""

import time

import numpy as np

# Present_Load у Feetech: младшие 10 бит — величина, бит 10 — направление.
LOAD_MAGNITUDE_MASK = 0x3FF

# Значения по умолчанию — подобраны по замеру gripper_probe.py на этой руке:
#   вхолостую: нагрузка 20..28, отставание 0.3°
#   на яблоке: нагрузка 344..500 (упирается в потолок), отставание 5.7°..16°
# Пороги ставим с большим запасом между этими режимами, ближе к холостому —
# так контакт ловится раньше и сжатие получается мягче.
DEFAULT_LOAD_THRESHOLD = 200   # 0..1000 (холостой максимум 28, контакт от ~344)
DEFAULT_STALL_DEG = 2.0        # отставание факта от команды (холостое 0.3°)
DEFAULT_STEP_DEG = 1.5         # шаг сжатия
DEFAULT_DT = 0.06              # пауза после шага, чтобы серво успел отработать
DEFAULT_CONFIRM = 4            # подряд идущих подтверждений (защита от выброса)
DEFAULT_EXTRA_SQUEEZE = 3.0    # лёгкий дожим после контакта, чтобы держало

# Насколько команда может уходить дальше фактического положения. Серво давит тем
# сильнее, чем больше это расхождение, поэтому параметр напрямую задаёт СИЛУ
# сжатия: меньше — бережнее к мягкому плоду, больше — крепче держит.
DEFAULT_MAX_OVERSHOOT_DEG = 5.0


def read_load(bus, motor="gripper"):
    """Величина нагрузки на сервоприводе, 0..1000 (без знака направления)."""
    raw = bus.read("Present_Load", motor, normalize=False)
    return int(raw) & LOAD_MAGNITUDE_MASK


def read_pos(bus, motor="gripper"):
    """Текущий угол сустава в градусах (нормализованный, как в наблюдениях)."""
    return float(bus.read("Present_Position", motor))


def close_until_contact(send_gripper, bus, start_deg, closed_deg,
                        load_threshold=DEFAULT_LOAD_THRESHOLD,
                        stall_deg=DEFAULT_STALL_DEG,
                        step_deg=DEFAULT_STEP_DEG,
                        dt=DEFAULT_DT,
                        confirm=DEFAULT_CONFIRM,
                        extra_squeeze=DEFAULT_EXTRA_SQUEEZE,
                        max_overshoot=DEFAULT_MAX_OVERSHOOT_DEG,
                        motor="gripper",
                        verbose=True):
    """Сжимает клешню от start_deg к closed_deg, пока не почувствует контакт.

    send_gripper(angle_deg) — функция, отправляющая команду на сустав клешни.
    bus — шина сервоприводов (для чтения нагрузки/позиции).

    Возвращает (итоговый угол, причина остановки):
      'contact'   — почувствовали яблоко (нагрузка/застревание);
      'closed'    — дошли до closed_deg, ничего не поймали (пусто/промах).

    closed_deg — жёсткий предел: дальше него не сжимаем никогда.
    """
    direction = np.sign(closed_deg - start_deg)
    if direction == 0:
        return start_deg, "closed"

    angle = float(start_deg)
    hits = 0
    while (closed_deg - angle) * direction > 0:
        angle = angle + direction * step_deg
        # не перескакиваем предел
        if (closed_deg - angle) * direction < 0:
            angle = float(closed_deg)

        # не уходим командой слишком далеко от факта — это ограничивает силу
        # сжатия и не даёт давить впустую, когда пальцы уже упёрлись
        actual_now = read_pos(bus, motor)
        limit = actual_now + direction * max_overshoot
        cmd = limit if (angle - limit) * direction > 0 else angle
        send_gripper(cmd)
        time.sleep(dt)

        load = read_load(bus, motor)
        actual = read_pos(bus, motor)
        lag = abs(angle - actual)

        if load >= load_threshold or lag >= stall_deg:
            hits += 1
            if verbose:
                print(f"    контакт? нагрузка={load} отставание={lag:.1f}° "
                      f"({hits}/{confirm})")
            if hits >= confirm:
                # лёгкий дожим — но он тоже обязан уважать оба ограничения:
                # жёсткий предел сжатия и максимальный перебег за факт (силу)
                squeeze = angle + direction * extra_squeeze
                squeeze_limit = actual + direction * max_overshoot
                if (squeeze - squeeze_limit) * direction > 0:
                    squeeze = squeeze_limit
                if (closed_deg - squeeze) * direction < 0:
                    squeeze = float(closed_deg)
                send_gripper(squeeze)
                time.sleep(dt)
                if verbose:
                    print(f"    яблоко схвачено на {squeeze:.1f}° "
                          f"(нагрузка {load}), сжатие остановлено")
                return squeeze, "contact"
        else:
            hits = 0

    if verbose:
        print(f"    дошёл до предела {closed_deg:.1f}° без контакта "
              "(яблоко не поймано?)")
    return float(closed_deg), "closed"

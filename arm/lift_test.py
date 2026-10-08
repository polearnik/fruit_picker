"""Первый запуск подъёмника: режим, направление, масштаб, хоминг.

Идёт по шагам и перед каждым движением спрашивает подтверждение. Ходы
маленькие (40-200 мм). Держите руку у выключателя питания.

Перед запуском:
  - мотору назначен ID 7 (arm/set_servo_id.py);
  - каретка в 100-500 мм над нижним упором (сейчас он на 660 мм),
    над ней есть 250 мм свободного хода;
  - рука в позе, где она не заденет пол, стойку и ящик;
  - pick_apple.py и прочие скрипты руки НЕ запущены (порт один).

Запуск:  python arm/lift_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import lift as L


def ask(msg):
    return input(f"{msg} [y/N] ").strip().lower() == "y"


def main():
    lift = L.Lift.open()
    try:
        mode = lift._r1(L.ADDR_MODE, "режим")
        print(f"Мотор ID {lift.id} отвечает. Режим сейчас: {mode} "
              f"({'шаговый' if mode == L.MODE_STEP else 'нужно переключить'})")

        print("\nШАГ 1. Перевод в шаговый режим и включение удержания.")
        if mode != L.MODE_STEP:
            print("  На время записи момент выключается — каретка может просесть.")
            print("  Придержите её рукой или опустите в самый низ.")
        if not ask("  Выполнить?"):
            return
        lift.setup()
        print(f"  Готово. Удержание включено, нагрузка {lift.load()}")

        print("\nШАГ 2. Направление: один оборот мотора (40 мм) со знаком «+».")
        if not ask("  Поехать?"):
            return
        s0 = lift._update()
        lift._move_steps(+L.STEPS_PER_REV)
        counted = lift._update() - s0
        print(f"  Одометр насчитал {counted} шагов из {L.STEPS_PER_REV} "
              f"({counted * L.MM_PER_STEP:.1f} мм).")
        up = ask("  Каретка поехала ВВЕРХ?")
        sign = +1 if up else -1
        print(f"  => в arm/lift.py нужно UP_SIGN = {sign:+d}"
              f"{'  (сейчас так и стоит)' if sign == L.UP_SIGN else '  <-- ИСПРАВЬТЕ'}")
        L.UP_SIGN = sign  # дальше в этом запуске — с верным знаком

        print("\nШАГ 3. Масштаб: 200 мм вверх. Отметьте каретку на стойке")
        print("  маркером ДО движения и измерьте ход рулеткой ПОСЛЕ.")
        if ask("  Поехать?"):
            s0 = lift._update()
            lift._move_steps(sign * 200 / L.MM_PER_STEP)
            mm = (lift._update() - s0) * sign * L.MM_PER_STEP
            print(f"  По одометру: {mm:.1f} мм.")
            real = input("  Сколько намерили рулеткой, мм? (Enter — пропустить) ").strip()
            if real:
                k = float(real) / mm
                print(f"  Отношение {k:.3f}. Если оно не 1.00±0.02 — в arm/lift.py "
                      f"MM_PER_REV = {L.MM_PER_REV * k:.2f}")
            print("  Возвращаю вниз на 200 мм.")
            lift._move_steps(-sign * 200 / L.MM_PER_STEP)

        print("\nШАГ 4. Хоминг: медленно вниз до нижнего упора (~6 мм/с),")
        print(f"  усилие ограничено ({L.HOME_TORQUE}/1000). Внизу ничего не должно")
        print("  попасть под каретку и руку.")
        if ask("  Выполнить?"):
            lift.home()
            print(f"  Каретка на z = {lift.z():.1f} мм.")
            if ask("  Подняться на 100 мм и вернуться?"):
                lift.move_to(L.Z_MIN_MM + 100)
                print(f"  z = {lift.z():.1f} мм")
                lift.move_to(L.Z_MIN_MM)
                print(f"  z = {lift.z():.1f} мм")
        print("\nГотово. Мотор оставлен с удержанием.")
    except L.LiftError as e:
        print(f"\n[ОШИБКА] {e}")
    finally:
        lift.close()


if __name__ == "__main__":
    main()

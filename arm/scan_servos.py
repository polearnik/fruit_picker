"""Диагностика: кто отвечает на шине сервоприводов SO-101.

Только читает (ping), ничего в сервоприводы не пишет — полностью безопасно.
Показывает, какие ID отвечают и на какой скорости. Так мы поймём, назначены ли
уникальные ID (ждём 1..6) или все сервоприводы пока с заводским ID 1.

Запуск:  python arm/scan_servos.py
Требуется: рука подключена по USB и ЗАПИТАНА от блока питания.
"""

import scservo_sdk as scs

PORT = "/dev/ttyACM0"
# STS3215 с завода на 1 Мбит/с; на всякий случай проверим и другие скорости
BAUDS = [1_000_000, 500_000, 250_000, 115_200]
PROTOCOL_END = 0  # STS/SMS Feetech
SCAN_IDS = range(1, 21)


def main():
    port = scs.PortHandler(PORT)
    if not port.openPort():
        raise SystemExit(f"Не удалось открыть {PORT}")
    packet = scs.PacketHandler(PROTOCOL_END)

    found_any = False
    for baud in BAUDS:
        port.setBaudRate(baud)
        found = []
        for sid in SCAN_IDS:
            model, comm, err = packet.ping(port, sid)
            if comm == scs.COMM_SUCCESS:
                found.append((sid, model))
        if found:
            found_any = True
            print(f"\nскорость {baud} бит/с — ответили:")
            for sid, model in found:
                print(f"  ID {sid}  (модель {model})")
        else:
            print(f"скорость {baud} бит/с — тишина")

    port.closePort()
    if not found_any:
        print("\nНи один сервопривод не ответил. Проверьте:")
        print("  1. Подключён ли блок питания к плате (сервоприводы без него молчат)")
        print("  2. Тот ли порт (ls /dev/ttyACM* /dev/ttyUSB*)")
        print("  3. Плотно ли сидят шлейфы сервоприводов")
    else:
        n = "разные ID — настройка уже сделана" if True else ""
        print("\nИтог: см. список выше. Ждём ID 1..6 (по одному на каждый сустав).")
        print("Если ответил только ID 1 — уникальные ID ещё не назначены.")


if __name__ == "__main__":
    main()

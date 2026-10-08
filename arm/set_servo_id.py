"""Назначить ID новому сервоприводу STS3215 (мотор подъёмника каретки).

С завода у каждого STS3215 ID 1 — тот же, что у shoulder_pan. Если воткнуть
новый мотор в общую шину как есть, на запрос «ID 1» ответят двое сразу, ответы
смешаются и рука перестанет читаться. Поэтому ID меняется, пока мотор на шине
ОДИН.

Порядок:
  1. Отключить от платы шлейф руки. Выключать питание не поможет: плата
     питает всё сразу. На плате должен остаться только новый мотор.
  2. Питание 7 В включено, USB подключён.
  3. python arm/set_servo_id.py            (по умолчанию новый ID 7)
  4. Вернуть шлейф руки, проверить:  python arm/scan_servos.py  -> ID 1..7

Скрипт откажется что-либо писать, если на шине отвечает не ровно один мотор.
"""

import argparse

import scservo_sdk as scs

PORT = "/dev/ttyACM0"
BAUD = 1_000_000
PROTOCOL_END = 0      # STS/SMS Feetech

# Регистры STS3215 (EEPROM)
ADDR_ID = 5
ADDR_LOCK = 55        # 0 — EEPROM открыта для записи, 1 — закрыта

ARM_IDS = range(1, 7)  # суставы руки — эти ID занимать нельзя


def scan(port, packet):
    found = []
    for sid in range(0, 254):
        _, comm, _ = packet.ping(port, sid)
        if comm == scs.COMM_SUCCESS:
            found.append(sid)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-id", type=int, default=7)
    args = ap.parse_args()
    if args.new_id in ARM_IDS or not 1 <= args.new_id <= 253:
        raise SystemExit(f"ID {args.new_id} нельзя: 1..6 заняты суставами руки")

    port = scs.PortHandler(PORT)
    if not port.openPort():
        raise SystemExit(f"Не удалось открыть {PORT}")
    port.setBaudRate(BAUD)
    packet = scs.PacketHandler(PROTOCOL_END)

    try:
        print("Опрашиваю шину (ID 0..253), может занять до минуты...")
        found = scan(port, packet)
        if len(found) != 1:
            print(f"На шине отвечают: {found or 'никто'}.")
            print("Нужен РОВНО ОДИН мотор. Отключите шлейф руки от платы,")
            print("оставьте только новый мотор и запустите снова.")
            return
        old = found[0]
        if old == args.new_id:
            print(f"У мотора уже ID {old} — ничего делать не нужно.")
            return

        ans = input(f"Мотор отвечает с ID {old}. Сменить на {args.new_id}? [y/N] ")
        if ans.strip().lower() != "y":
            print("Отменено.")
            return

        packet.write1ByteTxRx(port, old, ADDR_LOCK, 0)
        packet.write1ByteTxRx(port, old, ADDR_ID, args.new_id)
        packet.write1ByteTxRx(port, args.new_id, ADDR_LOCK, 1)

        _, comm, _ = packet.ping(port, args.new_id)
        if comm == scs.COMM_SUCCESS:
            print(f"Готово: мотор теперь ID {args.new_id} (сохранено в EEPROM).")
            print("Верните шлейф руки и проверьте: python arm/scan_servos.py")
        else:
            print(f"Мотор не отвечает на ID {args.new_id} — запустите скрипт ещё раз.")
    finally:
        port.closePort()


if __name__ == "__main__":
    main()

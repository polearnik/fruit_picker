"""Первое подключение к руке SO-101 из Python. ТОЛЬКО ЧТЕНИЕ, без движений.

Подключается к откалиброванной руке, читает и печатает угол каждого сустава.
Это проверка, что все 6 моторов на связи и калибровка (fruit_arm) подхватилась.

Запуск:  python arm/hello_arm.py

Важно: при подключении моторы, скорее всего, включат момент и «застынут»,
удерживая текущее положение (это нормально, рука не дёргается). На всякий случай
держите руку рукой и/или держите выключатель питания под рукой при первом запуске.
"""

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

PORT = "/dev/ttyACM0"
ROBOT_ID = "fruit_arm"  # имя из калибровки


def main():
    config = SO101FollowerConfig(
        port=PORT,
        id=ROBOT_ID,
        use_degrees=True,  # углы в градусах — по-человечески читаемо
    )
    arm = SO101Follower(config)

    print("Подключаюсь к руке...")
    arm.connect(calibrate=False)  # калибровку не трогаем, берём готовую
    print("Подключено. Читаю положения суставов:\n")

    obs = arm.get_observation()
    for key, value in obs.items():
        if isinstance(value, (int, float)):
            print(f"  {key:20s} = {value:8.2f}")
        else:
            print(f"  {key:20s} = {value}")

    arm.disconnect()
    print("\nГотово, отключился. Если увидели 6 значений суставов — связь и калибровка в порядке.")


if __name__ == "__main__":
    main()

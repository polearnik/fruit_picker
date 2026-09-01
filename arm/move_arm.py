"""Первое управляемое движение SO-101. Плавно и безопасно.

Двигает ОДИН сустав малой амплитудой вокруг его текущего положения и возвращает
обратно. По умолчанию — захват (gripper): самое безопасное первое движение,
рука никуда не едет. Движение идёт мелкими шагами с паузами (плавно), плюс
аппаратное ограничение max_relative_target не даёт послать резкий скачок.

Запуск:  python arm/move_arm.py

Поменять сустав/амплитуду — переменные JOINT и AMP_DEG ниже. Начинайте всегда
с малой амплитуды и безопасного сустава; рука должна быть в свободном
пространстве, выключатель питания — под рукой.
"""

import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

PORT = "/dev/ttyACM0"
ROBOT_ID = "fruit_arm"


#shoulder_pan.pos shoulder_lift.pos elbow_flex.pos wrist_flex.pos wrist_roll.pos gripper.pos
JOINT = "shoulder_lift"      # какой сустав двигаем (безопасный старт — gripper)
AMP_DEG = 20.0          # амплитуда в градусах: туда на +8, потом на -8, потом домой
STEP_DEG = 1.0         # размер одного плавного шага
DT = 0.05            # пауза между шагами, сек (меньше = быстрее)
MAX_REL = 15.0         # аппаратный предел: не больше 15° за одну команду


def smooth_move(arm, key, start, target):
    """Плавно ведёт сустав от start к target мелкими шагами."""
    n = max(1, int(abs(target - start) / STEP_DEG))
    for i in range(1, n + 1):
        val = start + (target - start) * i / n
        arm.send_action({key: val})
        time.sleep(DT)


def main():
    config = SO101FollowerConfig(
        port=PORT,
        id=ROBOT_ID,
        use_degrees=True,
        max_relative_target=MAX_REL,  # страховка от резкого скачка
    )
    arm = SO101Follower(config)
    arm.connect(calibrate=False)
    print("Подключено.")

    obs = arm.get_observation()
    # ключ действия для нужного сустава (напр. 'gripper.pos') берём из наблюдения
    print(obs)
    key = next((k for k in obs if JOINT in k and isinstance(obs[k], (int, float))), None)
    if key is None:
        arm.disconnect()
        raise SystemExit(f"Не нашёл сустав '{JOINT}' среди {list(obs)}")

    start = float(obs[key])
    print(f"Двигаю '{key}': старт {start:.1f}°, амплитуда ±{AMP_DEG}°")

    smooth_move(arm, key, start, start + AMP_DEG)   # туда
    smooth_move(arm, key, start + AMP_DEG, start - AMP_DEG)  # в другую сторону
    smooth_move(arm, key, start - AMP_DEG, start)   # домой

    joint ="gripper"
    key = next((k for k in obs if joint in k and isinstance(obs[k], (int, float))), None)

    smooth_move(arm, key, start, start + AMP_DEG)   # туда
    smooth_move(arm, key, start + AMP_DEG, start - AMP_DEG)  # в другую сторону
    smooth_move(arm, key, start - AMP_DEG, start)   # домой


    final = float(arm.get_observation()[key])
    print(f"Готово. Вернулся к {final:.1f}° (старт был {start:.1f}°).")
    arm.disconnect()


if __name__ == "__main__":
    main()

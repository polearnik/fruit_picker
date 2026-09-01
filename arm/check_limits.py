"""Диагностика: где сейчас стоят суставы относительно пределов.

Показывает три вещи по каждому суставу:
  - текущий угол (с руки);
  - физический диапазон (из калибровки lerobot — реальный ход руки);
  - модельный диапазон (из URDF — с ним работает решатель IK).

Физический ход ШИРЕ модельного, поэтому рука может стоять в позе, которую
модель считает недопустимой. Тогда планировщик ругается «сустав за пределом»,
хотя механически всё нормально. Этот скрипт показывает, так ли это.

Только читает, ничего не двигает.

Запуск:  python arm/check_limits.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from kinematics import ArmKinematics, JOINT_NAMES
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

PORT, ROBOT_ID = "/dev/ttyACM0", "fruit_arm"
CALIB = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower" / f"{ROBOT_ID}.json"


def main():
    kin = ArmKinematics()
    cal = json.loads(CALIB.read_text()) if CALIB.exists() else {}

    arm = SO101Follower(SO101FollowerConfig(port=PORT, id=ROBOT_ID, use_degrees=True))
    arm.connect(calibrate=False)
    try:
        obs = arm.get_observation()
        keys = {j: next(k for k in obs if j in k and isinstance(obs[k], (int, float)))
                for j in JOINT_NAMES}
        joints = np.array([obs[keys[j]] for j in JOINT_NAMES], dtype=float)
    finally:
        arm.disconnect()

    print(f"\n{'сустав':16s} {'сейчас':>9s}  {'физически':>18s}  {'модель URDF':>18s}  статус")
    problems = []
    for name, val in zip(JOINT_NAMES, joints):
        phys = ""
        if name in cal:
            c = cal[name]
            half = (c["range_max"] - c["range_min"]) * 360 / 4095 / 2
            phys = f"{-half:8.1f}..{half:6.1f}"
        if name in kin.limits_deg:
            lo, hi = kin.limits_deg[name]
            model = f"{lo:8.1f}..{hi:6.1f}"
            if val < lo or val > hi:
                status = "<< ВНЕ МОДЕЛИ"
                problems.append(name)
            elif val < lo + 2 or val > hi - 2:
                status = "<< у самого края"
                problems.append(name)
            else:
                status = "ок"
        else:
            model, status = "(нет в URDF)", ""
        print(f"{name:16s} {val:8.1f}°  {phys:>18s}  {model:>18s}  {status}")

    print(f"\nКончик руки сейчас: {np.round(kin.fk(joints), 1)} мм")
    if problems:
        print(f"\nСуставы вне/у края модельного диапазона: {problems}")
        print("Планировщик сам вернёт их в диапазон первым движением.")
        print("Если мешает — отведите руку рукой в более среднюю позу и повторите.")
    else:
        print("\nВсе суставы внутри модельного диапазона — планировщику ничто не мешает.")


if __name__ == "__main__":
    main()

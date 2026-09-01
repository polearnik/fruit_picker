"""Hand-eye: преобразование точек из системы камеры в систему руки.

Мост между зрением и рукой. По набору соответствий (одна и та же точка,
измеренная камерой и вычисленная по FK руки) находит жёсткое преобразование
R, t такое, что  P_рука ≈ R @ P_камера + t.  Метод — Кабш/Умейама (без масштаба,
т.к. обе системы в миллиметрах).

Самопроверка математики:  python arm/handeye.py
"""

from pathlib import Path

import numpy as np

SAVE_PATH = Path(__file__).parent / "handeye.npz"


def rigid_transform(cam_pts, arm_pts):
    """Nx3 точки в камере и в руке -> (R 3x3, t 3). P_arm ≈ R@P_cam + t."""
    cam = np.asarray(cam_pts, dtype=float)
    arm = np.asarray(arm_pts, dtype=float)
    cam_c = cam - cam.mean(0)
    arm_c = arm - arm.mean(0)
    H = cam_c.T @ arm_c
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = arm.mean(0) - R @ cam.mean(0)
    return R, t


def residuals_mm(R, t, cam_pts, arm_pts):
    """Ошибки соответствий после подгонки (мм) — для оценки качества калибровки."""
    cam = np.asarray(cam_pts, dtype=float)
    arm = np.asarray(arm_pts, dtype=float)
    pred = (R @ cam.T).T + t
    return np.linalg.norm(pred - arm, axis=1)


class HandEye:
    def __init__(self, R, t):
        self.R = np.asarray(R, dtype=float)
        self.t = np.asarray(t, dtype=float)

    def cam_to_arm(self, xyz_cam):
        """XYZ в системе камеры (мм) -> XYZ в системе руки (мм)."""
        return self.R @ np.asarray(xyz_cam, dtype=float) + self.t

    def save(self, path=SAVE_PATH):
        np.savez(path, R=self.R, t=self.t)

    @classmethod
    def load(cls, path=SAVE_PATH):
        d = np.load(path)
        return cls(d["R"], d["t"])


def _self_test():
    rng = np.random.default_rng(0)
    # задаём "истинное" преобразование камера->рука
    ang = rng.uniform(-1, 1, 3)
    from numpy import cos, sin
    Rx = np.array([[1, 0, 0], [0, cos(ang[0]), -sin(ang[0])], [0, sin(ang[0]), cos(ang[0])]])
    Ry = np.array([[cos(ang[1]), 0, sin(ang[1])], [0, 1, 0], [-sin(ang[1]), 0, cos(ang[1])]])
    Rz = np.array([[cos(ang[2]), -sin(ang[2]), 0], [sin(ang[2]), cos(ang[2]), 0], [0, 0, 1]])
    R_true = Rz @ Ry @ Rx
    t_true = rng.uniform(-300, 300, 3)

    cam = rng.uniform(-200, 200, size=(12, 3))
    arm = (R_true @ cam.T).T + t_true

    # чистые данные -> должны восстановиться точно
    R, t = rigid_transform(cam, arm)
    assert np.allclose(R, R_true, atol=1e-6) and np.allclose(t, t_true, atol=1e-6)
    print("восстановление без шума: R и t совпали точно")

    # с шумом измерения ~1 мм -> ошибка того же порядка, без развала
    arm_noisy = arm + rng.normal(0, 1.0, arm.shape)
    R2, t2 = rigid_transform(cam, arm_noisy)
    res = residuals_mm(R2, t2, cam, arm_noisy)
    print(f"с шумом 1 мм: остаточная ошибка средняя {res.mean():.2f} мм, макс {res.max():.2f} мм")
    assert res.mean() < 3.0

    # проверка применения
    he = HandEye(R2, t2)
    p_arm = he.cam_to_arm(cam[0])
    print(f"применение cam_to_arm: {np.round(p_arm,1)} vs истинное {np.round(arm[0],1)}")
    print("\nМатематика hand-eye верна.")


if __name__ == "__main__":
    _self_test()

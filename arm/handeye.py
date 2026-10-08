"""Hand-eye: преобразование точек из системы камеры в систему руки.

Мост между зрением и рукой. По набору соответствий (одна и та же точка,
измеренная камерой и вычисленная по FK руки) находит жёсткое преобразование
R, t такое, что  P_рука ≈ R @ P_камера + t.  Метод — Кабш/Умейама (без масштаба,
т.к. обе системы в миллиметрах).

Рука на подъёмнике (arm/lift.py). Основание ездит вдоль рельса, поэтому
«система руки» у каждой высоты каретки своя. Вводим неподвижную МИРОВУЮ
систему — систему основания при высоте каретки z_ref — и вектор up: куда и
на сколько смещается основание при подъёме каретки на 1 мм, в осях основания:
    P_мир   = R @ P_камера + t
    P_рука  = P_мир - (z - z_ref) * up          (z — высота каретки, мм)
up находится из той же калибровки: точки снимаются на разных высотах. Его
длина — проверка масштаба подъёмника (должна быть ≈1; 1.02 значит, что
MM_PER_REV в lift.py занижен на 2%), а направление — это «вверх» для
планировщика: рука висит на стойке боком, и её собственная ось Z вверх НЕ
смотрит.

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


def fit_with_lift(cam_pts, arm_pts, lift_z, z_ref=None, iters=200):
    """Калибровка с подъёмником: -> (R, t, up, z_ref).

    cam_pts — метка в системе камеры, arm_pts — та же метка по FK (система
    основания НА ТОЙ высоте), lift_z — высота каретки в момент снимка.
    Модель:  arm + (z - z_ref) * up = R @ cam + t.

    Чередуем два шага, каждый решается точно:
      - up известен -> R, t по Кабшу на парах (cam, arm + (z - z_ref) up);
      - R известен  -> up и t линейным МНК.
    Нужны точки минимум на двух разных высотах, иначе up не определён.
    """
    cam = np.asarray(cam_pts, float)
    arm = np.asarray(arm_pts, float)
    z = np.asarray(lift_z, float)
    if z_ref is None:
        z_ref = float(np.round(z.mean()))
    dz = z - z_ref
    if np.ptp(z) < 50:
        raise ValueError("точки сняты почти на одной высоте каретки — up не "
                         "определить; снимите ещё на высоте, отличающейся на 100+ мм")
    up = np.zeros(3)
    for _ in range(iters):
        R, t = rigid_transform(cam, arm + dz[:, None] * up)
        # dz*up - t = R@cam - arm  — линейно по (up, t)
        rhs = (R @ cam.T).T - arm                      # N x 3
        A = np.zeros((3 * len(cam), 6))
        for i in range(len(cam)):
            A[3*i:3*i+3, 0:3] = np.eye(3) * dz[i]
            A[3*i:3*i+3, 3:6] = -np.eye(3)
        sol, *_ = np.linalg.lstsq(A, rhs.reshape(-1), rcond=None)
        up_new = sol[:3]
        if np.linalg.norm(up_new - up) < 1e-9:
            up = up_new
            break
        up = up_new
    R, t = rigid_transform(cam, arm + dz[:, None] * up)
    return R, t, up, z_ref


def residuals_lift_mm(R, t, up, z_ref, cam_pts, arm_pts, lift_z):
    cam = np.asarray(cam_pts, float)
    arm = np.asarray(arm_pts, float)
    dz = np.asarray(lift_z, float) - z_ref
    pred = (R @ cam.T).T + t - dz[:, None] * up
    return np.linalg.norm(pred - arm, axis=1)


class HandEye:
    """Камера -> рука. Без подъёмника up = 0, и высота каретки не нужна."""

    def __init__(self, R, t, up=None, z_ref=0.0):
        self.R = np.asarray(R, dtype=float)
        self.t = np.asarray(t, dtype=float)
        self.up = np.zeros(3) if up is None else np.asarray(up, dtype=float)
        self.z_ref = float(z_ref)

    @property
    def has_lift(self):
        return bool(np.any(self.up))

    @property
    def up_unit(self):
        """Единичный вектор «вверх» в осях основания (вдоль рельса)."""
        if not self.has_lift:
            return np.array([0., 0., 1.])
        return self.up / np.linalg.norm(self.up)

    def cam_to_world(self, xyz_cam):
        """XYZ в системе камеры -> мировая система (основание при z = z_ref)."""
        return self.R @ np.asarray(xyz_cam, dtype=float) + self.t

    def world_to_arm(self, xyz_world, lift_z):
        """Мировая точка -> система основания при высоте каретки lift_z."""
        return np.asarray(xyz_world, float) - (lift_z - self.z_ref) * self.up

    def arm_to_world(self, xyz_arm, lift_z):
        return np.asarray(xyz_arm, float) + (lift_z - self.z_ref) * self.up

    def cam_to_arm(self, xyz_cam, lift_z=None):
        """XYZ в системе камеры (мм) -> XYZ в системе руки (мм) при высоте lift_z."""
        if self.has_lift and lift_z is None:
            raise ValueError("калибровка с подъёмником: нужна высота каретки lift_z")
        p = self.cam_to_world(xyz_cam)
        return p if lift_z is None else self.world_to_arm(p, lift_z)

    def save(self, path=SAVE_PATH):
        np.savez(path, R=self.R, t=self.t, up=self.up, z_ref=self.z_ref)

    @classmethod
    def load(cls, path=SAVE_PATH):
        d = np.load(path)
        up = d["up"] if "up" in d else None
        z_ref = float(d["z_ref"]) if "z_ref" in d else 0.0
        return cls(d["R"], d["t"], up, z_ref)


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
    # с подъёмником: up = почти единичный вектор по оси -X основания (рука
    # висит боком), масштаб чуть не 1 — как при неточном MM_PER_REV
    up_true = np.array([-0.99, 0.08, 0.03]); up_true *= 1.015 / np.linalg.norm(up_true)
    z_ref_true = 1000.0
    lz = rng.choice([700., 1000., 1300.], size=15)
    cam3 = rng.uniform(-200, 200, size=(15, 3))
    world = (R_true @ cam3.T).T + t_true
    arm3 = world - (lz - z_ref_true)[:, None] * up_true + rng.normal(0, 1.0, (15, 3))
    R3, t3, up3, zr = fit_with_lift(cam3, arm3, lz, z_ref=z_ref_true)
    res3 = residuals_lift_mm(R3, t3, up3, zr, cam3, arm3, lz)
    print(f"с подъёмником: up {np.round(up3, 3)} (истинный {np.round(up_true, 3)}), "
          f"|up|={np.linalg.norm(up3):.3f}, ошибка средняя {res3.mean():.2f} мм")
    assert np.allclose(up3, up_true, atol=0.01) and res3.mean() < 3.0
    he3 = HandEye(R3, t3, up3, zr)
    p = he3.cam_to_arm(cam3[0], lz[0])
    print(f"cam_to_arm на высоте {lz[0]:.0f}: {np.round(p,1)} vs {np.round(arm3[0],1)}")
    print("\nМатематика hand-eye верна.")


if __name__ == "__main__":
    _self_test()

"""Переиспользуемое стерео: триангуляция точки в 3D по паре кадров.

Выносит проверенную математику из stereo_detect.py, чтобы ею пользовались и
калибровка hand-eye, и захват яблока. Координаты — в системе РЕКТИФИЦИРОВАННОЙ
ЛЕВОЙ камеры, в миллиметрах.
"""

import cv2
import numpy as np


class StereoRig:
    def __init__(self, calib_path="stereo_calib.npz"):
        c = np.load(calib_path)
        self.K1, self.D1, self.R1, self.P1 = c["K1"], c["D1"], c["R1"], c["P1"]
        self.K2, self.D2, self.R2, self.P2 = c["K2"], c["D2"], c["R2"], c["P2"]
        # знак диспаритета зависит от порядка камер (см. историю проекта)
        self.disp_sign = -np.sign(self.P2[0, 3])

    def _rectify(self, pt, K, D, R, P):
        src = np.array([[pt]], dtype=np.float32)
        return cv2.undistortPoints(src, K, D, R=R, P=P)[0, 0]

    def rectified(self, uv_left, uv_right):
        """Пиксели исходных кадров -> пиксели ректифицированных (для проверки пары)."""
        rl = self._rectify(uv_left, self.K1, self.D1, self.R1, self.P1)
        rr = self._rectify(uv_right, self.K2, self.D2, self.R2, self.P2)
        return rl, rr

    def is_valid_pair(self, uv_left, uv_right, max_dy=20.0):
        """Согласованы ли детекции: близки по строке и диспаритет верного знака."""
        rl, rr = self.rectified(uv_left, uv_right)
        dy_ok = abs(rl[1] - rr[1]) < max_dy
        disp_ok = (rl[0] - rr[0]) * self.disp_sign > 0
        return dy_ok and disp_ok

    def triangulate(self, uv_left, uv_right):
        """Пара пикселей (u,v) в левом и правом кадре -> XYZ в мм (система камеры)."""
        rl, rr = self.rectified(uv_left, uv_right)
        pts4d = cv2.triangulatePoints(
            self.P1, self.P2,
            np.array([[rl[0]], [rl[1]]], dtype=np.float64),
            np.array([[rr[0]], [rr[1]]], dtype=np.float64),
        )
        return (pts4d[:3] / pts4d[3]).flatten()

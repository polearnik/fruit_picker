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
        self.calib_path = calib_path
        self.img_size = tuple(int(v) for v in c["img_size"])
        self.K1, self.D1, self.R1, self.P1 = c["K1"], c["D1"], c["R1"], c["P1"]
        self.K2, self.D2, self.R2, self.P2 = c["K2"], c["D2"], c["R2"], c["P2"]
        # знак диспаритета зависит от порядка камер (см. историю проекта)
        self.disp_sign = -np.sign(self.P2[0, 3])

    def assert_frame_size(self, width, height):
        """Проверить, что кадры того же размера, на котором снята калибровка.

        fx, fy, cx, cy — в пикселях того кадра. Другой размер (или другой
        режим камеры, который режет кадр) — и триангуляция МОЛЧА вернёт
        неверные миллиметры, а рука по ним поедет. Поэтому не предупреждение,
        а остановка.
        """
        if self.img_size != (int(width), int(height)):
            raise RuntimeError(
                f"{self.calib_path} снят на {self.img_size[0]}x{self.img_size[1]}, "
                f"а камера отдаёт {int(width)}x{int(height)}. "
                f"Переснимите калибровку: python stereo_capture.py "
                f"-> python stereo_calibrate.py, затем handeye_calibrate.py "
                f"(пересчёт старой калибровки под новый кадр — CAMERA_UPGRADE.md)")

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

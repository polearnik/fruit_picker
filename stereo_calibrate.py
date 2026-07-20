"""Шаг 2. Калибровка стереопары по снятым парам кадров.

Читает пары из calib_pairs/, вычисляет параметры каждой камеры и их
взаимное положение, сохраняет всё в stereo_calib.npz.

Запуск:  python stereo_calibrate.py

Ориентир по качеству: RMS-ошибка одиночных камер < 0.5 px, стерео < 1.0 px.
Если больше — переснимите пары (жёсткая доска, резкие кадры, больше ракурсов).
"""

from pathlib import Path

import cv2
import numpy as np

BOARD = (9, 6)          # внутренние углы, как в stereo_capture.py
SQUARE_SIZE_MM = 24.0   # ИЗМЕРЬТЕ линейкой сторону клетки на своей распечатке!

PAIRS_DIR = Path("calib_pairs")
OUT_FILE = "stereo_calib.npz"


def find_corners(img_path: Path):
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, BOARD, cv2.CALIB_CB_ADAPTIVE_THRESH)
    if not found:
        return None, gray.shape[::-1]
    # уточнение углов до субпиксельной точности — заметно улучшает калибровку
    corners = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )
    return corners, gray.shape[::-1]


def main():
    lefts = sorted(PAIRS_DIR.glob("left_*.png"))
    if len(lefts) < 10:
        raise SystemExit(f"Найдено только {len(lefts)} пар — нужно минимум 10, лучше 20-30")

    # 3D-координаты углов доски в её собственной системе (Z=0), в мм
    objp = np.zeros((BOARD[0] * BOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD[0], 0:BOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE_MM

    obj_points, pts_l, pts_r = [], [], []
    img_size = None

    for left_path in lefts:
        right_path = left_path.with_name(left_path.name.replace("left", "right"))
        corners_l, img_size = find_corners(left_path)
        corners_r, _ = find_corners(right_path)
        if corners_l is None or corners_r is None:
            print(f"Пропускаю {left_path.name}: доска не найдена")
            continue
        obj_points.append(objp)
        pts_l.append(corners_l)
        pts_r.append(corners_r)

    print(f"Использую {len(obj_points)} пар из {len(lefts)}")

    # калибровка каждой камеры по отдельности
    rms_l, K1, D1, _, _ = cv2.calibrateCamera(obj_points, pts_l, img_size, None, None)
    rms_r, K2, D2, _, _ = cv2.calibrateCamera(obj_points, pts_r, img_size, None, None)
    print(f"RMS левой: {rms_l:.3f} px, правой: {rms_r:.3f} px (хорошо < 0.5)")

    # взаимное положение камер: R — поворот, T — смещение правой относительно левой
    rms_s, K1, D1, K2, D2, R, T, _, _ = cv2.stereoCalibrate(
        obj_points, pts_l, pts_r, K1, D1, K2, D2, img_size,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    baseline = float(np.linalg.norm(T))
    print(f"RMS стерео: {rms_s:.3f} px (хорошо < 1.0)")
    print(f"База (расстояние между камерами): {baseline:.1f} мм — "
          f"сверьте с линейкой, расхождение >10% значит плохую калибровку")

    # ректификация: матрицы, приводящие обе картинки к общим строкам
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, img_size, R, T, alpha=0)

    np.savez(OUT_FILE, K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T,
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q, img_size=img_size)
    print(f"Калибровка сохранена в {OUT_FILE}. Дальше: python stereo_detect.py")


if __name__ == "__main__":
    main()

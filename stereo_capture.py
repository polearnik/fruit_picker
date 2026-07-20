"""Шаг 1. Съёмка калибровочных пар кадров с шахматной доской.

Подготовка:
  1. Распечатайте шахматную доску: https://github.com/opencv/opencv/blob/4.x/doc/pattern.png
     (9x6 внутренних углов). Наклейте лист на картон или планшет, чтобы не гнулся.
  2. Измерьте линейкой сторону клетки в мм и впишите в SQUARE_SIZE_MM
     в stereo_calibrate.py.
  3. Закрепите обе камеры жёстко на бруске (база 6-12 см), после этого
     их нельзя сдвигать до конца калибровки и работы.

Запуск:  python stereo_capture.py
  ПРОБЕЛ — сохранить пару кадров (только когда доска найдена в ОБЕИХ камерах,
           рамка углов зелёная слева и справа)
  q      — выход

Нужно 20-30 пар: доска ближе/дальше, в разных углах кадра, с наклонами
в разные стороны. Чем разнообразнее ракурсы, тем точнее калибровка.
"""

from pathlib import Path

import cv2

# Индексы камер. Дешёвые вебки обычно занимают по два устройства,
# поэтому вторая камера чаще всего /dev/video2, а не /dev/video1.
# Проверить: v4l2-ctl --list-devices
LEFT_CAM = 2
RIGHT_CAM = 0

FRAME_W, FRAME_H = 640, 480
BOARD = (9, 6)  # внутренние углы шахматной доски (столбцы, строки)

OUT_DIR = Path("calib_pairs")


def open_cam(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть камеру {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    return cap


def main():
    OUT_DIR.mkdir(exist_ok=True)
    cap_l, cap_r = open_cam(LEFT_CAM), open_cam(RIGHT_CAM)

    # быстрый флаг для поиска в реальном времени
    find_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK

    saved = len(list(OUT_DIR.glob("left_*.png")))
    print(f"Уже сохранено пар: {saved}. Цель — 20-30.")

    while True:
        ok_l, frame_l = cap_l.read()
        ok_r, frame_r = cap_r.read()
        if not (ok_l and ok_r):
            print("Потерян кадр с одной из камер")
            break

        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
        found_l, corners_l = cv2.findChessboardCorners(gray_l, BOARD, find_flags)
        found_r, corners_r = cv2.findChessboardCorners(gray_r, BOARD, find_flags)

        view_l, view_r = frame_l.copy(), frame_r.copy()
        cv2.drawChessboardCorners(view_l, BOARD, corners_l, found_l)
        cv2.drawChessboardCorners(view_r, BOARD, corners_r, found_r)

        both = cv2.hconcat([view_l, view_r])
        status = f"pairs: {saved}  board: L={'+' if found_l else '-'} R={'+' if found_r else '-'}"
        cv2.putText(both, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("stereo capture  (SPACE=save, q=quit)", both)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and found_l and found_r:
            cv2.imwrite(str(OUT_DIR / f"left_{saved:02d}.png"), frame_l)
            cv2.imwrite(str(OUT_DIR / f"right_{saved:02d}.png"), frame_r)
            saved += 1
            print(f"Сохранена пара {saved}")

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()
    print(f"Итого пар: {saved}. Дальше: python stereo_calibrate.py")


if __name__ == "__main__":
    main()

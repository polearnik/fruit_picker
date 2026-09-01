"""Шаг 3. Детекция фруктов двумя камерами + вычисление 3D-координат.

YOLO находит фрукты в левом и правом кадре, пары детекций сопоставляются,
центр каждой пары триангулируется в точку (X, Y, Z) в мм.

Система координат — левая камера (после ректификации):
  X — вправо, Y — вниз, Z — вперёд от камеры (то самое расстояние до фрукта).
Именно эти координаты на этапе 3-4 поедут манипулятору как цель захвата.

Запуск:  python stereo_detect.py   (нужен stereo_calib.npz из шага 2)
Выход:   q

Физический порядок камер (какая слева, какая справа) не важен — важно лишь,
чтобы LEFT_CAM/RIGHT_CAM совпадали с теми, что были при съёмке калибровки.
Ожидаемый знак диспаритета берётся из самой калибровки.
"""

import cv2
import numpy as np
from ultralytics import YOLO

LEFT_CAM = 0    # те же индексы, что в stereo_capture.py
RIGHT_CAM = 2
FRAME_W, FRAME_H = 1280, 960

FRUIT_CLASSES = {46: "banana", 47: "apple", 49: "orange"}
CONFIDENCE = 0.4

# После ректификации один и тот же объект лежит на одной строке в обоих
# кадрах. Пары детекций с большей разницей по вертикали отбрасываем.
MAX_Y_DIFF_PX = 20


def open_cam(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть камеру {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    return cap


def detect_fruits(model, frame):
    """[(class_id, центр рамки (x, y), рамка (x1, y1, x2, y2)), ...]"""
    results = model(frame, conf=CONFIDENCE, verbose=False)[0]
    out = []
    for box in results.boxes:
        cls_id = int(box.cls)
        if cls_id not in FRUIT_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        out.append((cls_id, ((x1 + x2) / 2, (y1 + y2) / 2), (x1, y1, x2, y2)))
    return out


def rectify_point(pt, K, D, R_rect, P_rect):
    """Пиксель исходного кадра -> пиксель ректифицированного кадра."""
    src = np.array([[pt]], dtype=np.float32)
    dst = cv2.undistortPoints(src, K, D, R=R_rect, P=P_rect)
    return dst[0, 0]  # (x, y)


def main():
    calib = np.load("stereo_calib.npz")
    K1, D1, R1, P1 = calib["K1"], calib["D1"], calib["R1"], calib["P1"]
    K2, D2, R2, P2 = calib["K2"], calib["D2"], calib["R2"], calib["P2"]

    # Знак диспаритета зависит от того, какая камера физически левее.
    # P2[0,3] = fx * Tx: при стандартном порядке (Tx < 0) диспаритет
    # rect_l - rect_r положителен, при обратном — отрицателен.
    disp_sign = -np.sign(P2[0, 3])

    model = YOLO("yolov8n.pt")
    cap_l, cap_r = open_cam(LEFT_CAM), open_cam(RIGHT_CAM)
    print("Покажите фрукт обеим камерам. Выход — q.")

    while True:
        ok_l, frame_l = cap_l.read()
        ok_r, frame_r = cap_r.read()
        if not (ok_l and ok_r):
            break

        dets_l = detect_fruits(model, frame_l)
        dets_r = detect_fruits(model, frame_r)

        # правые детекции, ещё не связанные с левыми
        free_r = list(range(len(dets_r)))

        for cls_id, center_l, (x1, y1, x2, y2) in dets_l:
            rect_l = rectify_point(center_l, K1, D1, R1, P1)

            # ищем пару: тот же класс, минимальная разница по строке
            best_j, best_rect_r, best_dy = None, None, MAX_Y_DIFF_PX
            for j in free_r:
                if dets_r[j][0] != cls_id:
                    continue
                rect_r = rectify_point(dets_r[j][1], K2, D2, R2, P2)
                dy = abs(rect_l[1] - rect_r[1])
                if dy < best_dy and (rect_l[0] - rect_r[0]) * disp_sign > 0:
                    best_j, best_rect_r, best_dy = j, rect_r, dy

            label = f"{FRUIT_CLASSES[cls_id]}: pair?"
            color = (0, 165, 255)  # оранжевый — пара не найдена

            if best_j is not None:
                free_r.remove(best_j)
                pts4d = cv2.triangulatePoints(
                    P1, P2,
                    np.array([[rect_l[0]], [rect_l[1]]], dtype=np.float64),
                    np.array([[best_rect_r[0]], [best_rect_r[1]]], dtype=np.float64),
                )
                X, Y, Z = (pts4d[:3] / pts4d[3]).flatten()  # мм
                label = f"{FRUIT_CLASSES[cls_id]} Z={Z / 10:.1f}cm ({X / 10:.0f},{Y / 10:.0f})"
                color = (0, 255, 0)
                print(f"{FRUIT_CLASSES[cls_id]}: X={X:.0f} Y={Y:.0f} Z={Z:.0f} мм")

            cv2.rectangle(frame_l, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_l, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # правый кадр показываем просто с рамками, для контроля
        for cls_id, _, (x1, y1, x2, y2) in dets_r:
            cv2.rectangle(frame_r, (x1, y1), (x2, y2), (255, 200, 0), 2)

        cv2.imshow("stereo detect  L | R  (q=quit)", cv2.hconcat([frame_l, frame_r]))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

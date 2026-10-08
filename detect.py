"""Детекция фруктов с веб-камеры в реальном времени.

Использует предобученную модель YOLO (датасет COCO), в котором есть
классы banana, apple, orange. Позже дообучим модель на свои фрукты и ягоды.

Запуск:  python detect.py
Выход:   клавиша q в окне с видео
"""

import cv2
from ultralytics import YOLO

from camera import open_cam, overlay_scale, preview

# Классы COCO, которые нас интересуют (id: имя)
FRUIT_CLASSES = {46: "banana", 47: "apple", 49: "orange"}

# Порог уверенности: детекции ниже этого значения отбрасываются
CONFIDENCE = 0.4


def main():
    # yolov8n — самая лёгкая модель, работает быстро даже без GPU.
    # При первом запуске файл модели (~6 МБ) скачается автоматически.
    model = YOLO("yolov8n.pt")

    # freeze=False: это демо, а не калибровка — пусть камера сама
    # подстраивается под свет.
    cap = open_cam(0, freeze=False)

    print("Камера запущена. Покажите ей банан, яблоко или апельсин.")
    print("Выход — клавиша q в окне с видео.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, conf=CONFIDENCE, verbose=False)[0]
        s = overlay_scale(frame)

        for box in results.boxes:
            cls_id = int(box.cls)
            if cls_id not in FRUIT_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{FRUIT_CLASSES[cls_id]} {float(box.conf):.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), int(2 * s))
            cv2.putText(frame, label, (x1, y1 - int(8 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, (0, 255, 0), int(2 * s))

            # Центр рамки — та самая точка, к которой на этапе 2
            # мы будем запрашивать расстояние у depth-камеры
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), int(4 * s), (0, 0, 255), -1)

        cv2.imshow("Fruit detection", preview(frame))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

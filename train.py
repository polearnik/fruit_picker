"""Шаг D. Дообучение YOLO на размеченных ягодах.

Берёт лёгкую модель yolov8n и дообучает её на dataset/. Результат —
runs/detect/berries/weights/best.pt, его подставляем в stereo_detect.py.

Запуск:  python train.py

На процессоре обучение идёт небыстро (десятки минут — часы в зависимости от
числа фото и EPOCHS). Это нормально. С GPU было бы в разы быстрее.
"""

from pathlib import Path

import yaml
from ultralytics import YOLO

DATASET = Path("dataset").resolve()
EPOCHS = 100
IMG_SIZE = 640
# yolov8n — самая лёгкая, быстрая на CPU. Точнее, но медленнее: yolov8s.
BASE_MODEL = "yolov8n.pt"


def main():
    data_yaml = DATASET / "data.yaml"

    # прописываем абсолютный path, чтобы train.txt/val.txt читались откуда угодно
    cfg = yaml.safe_load(data_yaml.read_text())
    cfg["path"] = str(DATASET)
    data_yaml.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))

    model = YOLO(BASE_MODEL)
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=8,
        name="berries",
        patience=20,       # ранняя остановка, если качество перестало расти
        exist_ok=True,
    )
    print("\nОбучение завершено.")
    print("Веса: runs/detect/berries/weights/best.pt")
    print("Проверка на камере: yolo predict model=runs/detect/berries/weights/best.pt source=0")
    print("Дальше — подставить best.pt в stereo_detect.py (см. README).")


if __name__ == "__main__":
    main()

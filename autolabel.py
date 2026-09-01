"""Шаг B. Черновая разметка фото ягод моделью YOLO-World (по тексту, без обучения).

Для каждого фото из dataset/images/ создаёт файл разметки в dataset/labels/
(формат YOLO) и картинку с нарисованными рамками в dataset/preview/ —
по ней удобно на глаз оценить качество. Затем делит датасет на train/val.

Это ЧЕРНОВИК: YOLO-World часть ягод пропустит, часть отметит лишнего.
После неё обязательно пройтись руками — python label_tool.py.

Запуск:  python autolabel.py
"""

import random
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLOWorld

DATASET = Path("dataset")
CONF = 0.02        # низкий порог: лучше лишнее (уберём руками), чем пропуск
VAL_FRACTION = 0.2
RANDOM_SEED = 0

# Текстовые подсказки для поиска. Ключ — подсказка, значение — id класса из
# data.yaml. Можно давать несколько формулировок на один класс для лучшего
# распознавания (напр. "ripe strawberry" и "strawberry").
PROMPTS = {
    "strawberry": 0,
    "ripe strawberry": 0,
    "raspberry": 1,
    "cherry": 2,
    "red cherry": 2,
}

COLORS = [(0, 0, 255), (255, 0, 255), (0, 165, 255), (255, 255, 0)]


def load_class_names():
    with open(DATASET / "data.yaml") as f:
        names = yaml.safe_load(f)["names"]
    return names  # {id: "name"}


def main():
    class_names = load_class_names()
    (DATASET / "labels").mkdir(exist_ok=True)
    (DATASET / "preview").mkdir(exist_ok=True)

    prompts = list(PROMPTS.keys())
    prompt_to_class = [PROMPTS[p] for p in prompts]

    model = YOLOWorld("yolov8s-world.pt")
    model.set_classes(prompts)

    images = sorted((DATASET / "images").glob("*.jpg"))
    if not images:
        raise SystemExit("В dataset/images/ нет фото — сначала python capture_dataset.py")

    total_boxes = 0
    per_class = {cid: 0 for cid in class_names}

    for img_path in images:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        results = model.predict(img, conf=CONF, verbose=False)[0]

        lines, preview = [], img.copy()
        for box in results.boxes:
            cls_id = prompt_to_class[int(box.cls)]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # YOLO-формат: класс + нормированные центр и размер
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            per_class[cls_id] += 1
            total_boxes += 1

            color = COLORS[cls_id % len(COLORS)]
            cv2.rectangle(preview, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(preview, class_names[cls_id], (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        label_path = DATASET / "labels" / f"{img_path.stem}.txt"
        label_path.write_text("\n".join(lines))
        cv2.imwrite(str(DATASET / "preview" / img_path.name), preview)

    # train/val split — списки абсолютных путей к картинкам
    rng = random.Random(RANDOM_SEED)
    shuffled = images[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * VAL_FRACTION))
    val = set(shuffled[:n_val])

    train_list = [str(p.resolve()) for p in images if p not in val]
    val_list = [str(p.resolve()) for p in images if p in val]
    (DATASET / "train.txt").write_text("\n".join(train_list))
    (DATASET / "val.txt").write_text("\n".join(val_list))

    print(f"Размечено фото: {len(images)}, черновых рамок: {total_boxes}")
    for cid, name in class_names.items():
        print(f"  {name}: {per_class[cid]}")
    print(f"Разбивка: train {len(train_list)}, val {len(val_list)}")
    print("Проверьте картинки в dataset/preview/, потом правьте: python label_tool.py")


if __name__ == "__main__":
    main()

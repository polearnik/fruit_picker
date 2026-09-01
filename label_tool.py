"""Шаг C. Ручная правка разметки (правим черновик от autolabel.py).

Простой редактор рамок на OpenCV, без сторонних программ. Читает и пишет
разметку в формате YOLO прямо в dataset/labels/.

Управление:
  n / p           — следующее / предыдущее фото (автосохранение)
  0..9            — выбрать активный класс (для новых рамок)
  ЛКМ протянуть   — нарисовать новую рамку выбранного класса
  ПКМ по рамке    — удалить рамку под курсором
  u               — отменить последнюю рамку
  s               — сохранить сейчас
  q               — выход (с сохранением)

Задача: убрать лишние рамки, дорисовать пропущенные ягоды, поправить класс.
Рамка должна плотно облегать ягоду.
"""

from pathlib import Path

import cv2
import yaml

DATASET = Path("dataset")

with open(DATASET / "data.yaml") as f:
    CLASS_NAMES = yaml.safe_load(f)["names"]  # {id: name}

COLORS = [(0, 0, 255), (255, 0, 255), (0, 165, 255), (255, 255, 0),
          (0, 255, 0), (255, 0, 0)]


def label_path(img_path):
    return DATASET / "labels" / f"{img_path.stem}.txt"


def load_boxes(img_path, w, h):
    """Читает YOLO-разметку -> список [cls, x1, y1, x2, y2] в пикселях."""
    p = label_path(img_path)
    boxes = []
    if p.exists():
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            cls, cx, cy, bw, bh = line.split()
            cls, cx, cy, bw, bh = int(cls), float(cx), float(cy), float(bw), float(bh)
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            boxes.append([cls, x1, y1, x2, y2])
    return boxes


def save_boxes(img_path, boxes, w, h):
    lines = []
    for cls, x1, y1, x2, y2 in boxes:
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = abs(x2 - x1) / w, abs(y2 - y1) / h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    label_path(img_path).write_text("\n".join(lines))


class State:
    def __init__(self):
        self.boxes = []
        self.cur_class = 0
        self.drawing = False
        self.start = (0, 0)
        self.mouse = (0, 0)


def point_in_box(px, py, box):
    _, x1, y1, x2, y2 = box
    return min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2)


def main():
    images = sorted((DATASET / "images").glob("*.jpg"))
    if not images:
        raise SystemExit("Нет фото в dataset/images/ — сначала python capture_dataset.py")

    st = State()
    win = "label tool"
    cv2.namedWindow(win)

    def on_mouse(event, x, y, flags, _):
        st.mouse = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            st.drawing = True
            st.start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and st.drawing:
            st.drawing = False
            x1, y1 = st.start
            if abs(x - x1) > 3 and abs(y - y1) > 3:  # игнор случайных кликов
                st.boxes.append([st.cur_class, float(x1), float(y1),
                                 float(x), float(y)])
        elif event == cv2.EVENT_RBUTTONDOWN:
            # удалить верхнюю рамку под курсором
            for i in range(len(st.boxes) - 1, -1, -1):
                if point_in_box(x, y, st.boxes[i]):
                    st.boxes.pop(i)
                    break

    cv2.setMouseCallback(win, on_mouse)

    idx = 0
    img = cv2.imread(str(images[idx]))
    h, w = img.shape[:2]
    st.boxes = load_boxes(images[idx], w, h)

    while True:
        view = img.copy()
        for cls, x1, y1, x2, y2 in st.boxes:
            color = COLORS[cls % len(COLORS)]
            cv2.rectangle(view, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(view, CLASS_NAMES[cls], (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if st.drawing:
            color = COLORS[st.cur_class % len(COLORS)]
            cv2.rectangle(view, st.start, st.mouse, color, 1)

        cur = CLASS_NAMES[st.cur_class]
        hud = f"[{idx + 1}/{len(images)}] class:{st.cur_class}={cur}  boxes:{len(st.boxes)}"
        cv2.rectangle(view, (0, 0), (view.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(view, hud, (8, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(win, view)

        key = cv2.waitKey(20) & 0xFF
        if key == 255:
            continue

        if ord("0") <= key <= ord("9"):
            c = key - ord("0")
            if c in CLASS_NAMES:
                st.cur_class = c
        elif key == ord("u") and st.boxes:
            st.boxes.pop()
        elif key == ord("s"):
            save_boxes(images[idx], st.boxes, w, h)
            print(f"сохранено: {images[idx].name}")
        elif key in (ord("n"), ord("p"), ord("q")):
            save_boxes(images[idx], st.boxes, w, h)  # автосохранение
            if key == ord("q"):
                break
            idx = (idx + (1 if key == ord("n") else -1)) % len(images)
            img = cv2.imread(str(images[idx]))
            h, w = img.shape[:2]
            st.boxes = load_boxes(images[idx], w, h)

    cv2.destroyAllWindows()
    print("Готово. Дальше обучение: python train.py")


if __name__ == "__main__":
    main()

"""Шаг A. Сбор фотографий ягод для датасета.

Снимает кадры с камеры и складывает их в dataset/images/.
Снимайте ягоды по-разному: разный фон, свет, расстояние, ракурс, по одной
и кучкой, частично закрытые листом. Чем разнообразнее — тем устойчивее модель.
Цель на первый рабочий датасет — 100-300 фото на каждый вид ягоды.

Запуск:  python capture_dataset.py
  ПРОБЕЛ — сохранить кадр
  a      — вкл/выкл автосъёмку (кадр каждые AUTO_EVERY сек)
  q      — выход
"""

import time
from pathlib import Path

import cv2

CAM = 0                 # обычная одна камера; для датасета стерео не нужно
FRAME_W, FRAME_H = 640, 480
AUTO_EVERY = 0.7        # секунд между кадрами в режиме автосъёмки

OUT_DIR = Path("dataset/images")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(CAM)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть камеру {CAM}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    saved = len(list(OUT_DIR.glob("*.jpg")))
    auto = False
    last = 0.0
    print(f"Уже собрано фото: {saved}. Пробел — снять, a — автосъёмка, q — выход.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        take = False
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            take = True
        if key == ord("a"):
            auto = not auto
            print("автосъёмка:", "вкл" if auto else "выкл")
        if auto and now - last >= AUTO_EVERY:
            take = True

        if take:
            fname = OUT_DIR / f"img_{int(now * 1000)}.jpg"
            cv2.imwrite(str(fname), frame)
            saved += 1
            last = now

        view = frame.copy()
        txt = f"saved: {saved}   auto: {'ON' if auto else 'off'}"
        cv2.putText(view, txt, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("capture dataset  (SPACE=save, a=auto, q=quit)", view)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Итого фото: {saved}. Дальше: python autolabel.py")


if __name__ == "__main__":
    main()

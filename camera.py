"""Открытие USB-камер: единое разрешение, формат кадра и заморозка автоматики.

Все скрипты проекта берут камеры отсюда, чтобы калибровка (стерео, hand-eye)
и работа (pick_apple) видели мир ОДИНАКОВО. Стоит развести эти настройки по
файлам — и калибровка перестанет соответствовать тому, что приходит с камеры.

Почему 1920x1080, а не 1280x960
  У Logitech C670i режим 1280x960 — это кроп 1440x1080 из кадра 1920x1080
  (проверено сравнением кадров с одной камеры: масштаб ровно 1.125 =
  1440/1280). В режиме 4:3 даром терялось 28% ширины обзора: HFOV 44.7°
  вместо 57.4° при той же вертикали 34.2°. Вертикаль у этой вебки уже на
  пределе — про выбор камер с широким углом см. CAMERA_UPGRADE.md.

Почему MJPG
  1920x1080@30 в несжатом YUYV — это ~1.5 Гбит/с, в USB 2.0 столько нет,
  драйвер молча отдаёт 5-10 fps. MJPEG сжимается на самой камере.

Почему замораживаются экспозиция и баланс белого
  Автоматика уползает — между левой и правой камерой, между съёмкой
  калибровки и работой. Для детекции по цвету (кремовая метка в hand-eye,
  красное яблоко) это смерть: порог HSV подобран под один свет, а камера
  отдаёт другой. Схема: дать автоматике сойтись на прогреве, забрать
  подобранные ею значения и зафиксировать. Стереопаре ставятся ОДНИ И ТЕ ЖЕ
  значения на обе камеры, иначе кадры разной яркости и цвета.

ВАЖНО: после смены камер, объективов или разрешения нужно заново снять
stereo_calibrate.py и handeye_calibrate.py — старые матрицы больше не годны.
"""

import cv2

# Рабочее разрешение всего проекта. Менять — только вместе с перекалибровкой.
FRAME_W, FRAME_H = 1920, 1080

# Кадров на прогрев: столько ждём, пока автоматика сойдётся, прежде чем
# забрать её значения и заморозить.
WARMUP_FRAMES = 20

# V4L2_CID_EXPOSURE_AUTO: 1 — ручной режим, 3 — автоматический.
EXPOSURE_MANUAL, EXPOSURE_AUTO = 1, 3

# Ширина окна предпросмотра. Кадр 1920, а пара рядом — 3840, в экран не влезает;
# показываем уменьшенную копию, а СОХРАНЯЕМ и считаем всегда полный кадр.
PREVIEW_W = 1280


def open_cam(index, width=FRAME_W, height=FRAME_H, freeze=True):
    """Открыть камеру: MJPG, заданное разрешение, прогрев и заморозка авто-режимов.

    freeze=False — оставить автоматику включённой (для съёмки датасета
    в меняющемся свете разнообразие как раз полезно).
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть камеру {index}")

    # Порядок важен: сначала формат, потом размер. Наоборот драйвер может
    # подобрать размер под YUYV и остаться на нём.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    for _ in range(WARMUP_FRAMES):
        cap.read()

    got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if got != (width, height):
        print(f"[camera {index}] ВНИМАНИЕ: просили {width}x{height}, "
              f"камера отдаёт {got[0]}x{got[1]} — калибровка не совпадёт")

    if freeze:
        freeze_auto(cap, index)
    return cap


def open_pair(left_index, right_index, width=FRAME_W, height=FRAME_H):
    """Открыть стереопару и заморозить ОБЕ камеры на общих значениях.

    Экспозиция и баланс белого берутся средними по паре: камеры смотрят на
    одну сцену, и одинаковые кадры лучше и для поиска пар детекций, и для
    порогов по цвету.
    """
    cap_l = open_cam(left_index, width, height, freeze=False)
    cap_r = open_cam(right_index, width, height, freeze=False)

    settings = [_auto_settings(c) for c in (cap_l, cap_r)]
    exposure = _mean([s[0] for s in settings])
    wb = _mean([s[1] for s in settings])

    freeze_auto(cap_l, left_index, exposure, wb)
    freeze_auto(cap_r, right_index, exposure, wb)
    return cap_l, cap_r


def freeze_auto(cap, index=None, exposure=None, wb=None):
    """Выключить авто-экспозицию и авто-ББ, зафиксировав текущие значения.

    exposure/wb=None — взять то, что подобрала сама камера на прогреве.
    """
    auto_exposure, auto_wb = _auto_settings(cap)
    exposure = auto_exposure if exposure is None else exposure
    wb = auto_wb if wb is None else wb

    # Ручной режим включается ПЕРВЫМ: пока активна автоматика, поля
    # exposure_time_absolute и white_balance_temperature только читаются.
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, EXPOSURE_MANUAL)
    if exposure is not None:
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    if wb is not None:
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, wb)

    for _ in range(5):  # дать новым значениям дойти до кадра
        cap.read()

    tag = "camera" if index is None else f"camera {index}"
    mode = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    if mode == EXPOSURE_AUTO:
        print(f"[{tag}] ВНИМАНИЕ: авто-экспозиция не отключилась — "
              f"яркость будет плавать между кадрами")
    print(f"[{tag}] зафиксировано: exposure={cap.get(cv2.CAP_PROP_EXPOSURE):.0f} "
          f"wb={cap.get(cv2.CAP_PROP_WB_TEMPERATURE):.0f}")


def preview_scale(*frames, width=PREVIEW_W):
    """Во сколько раз preview() уменьшит эти кадры."""
    return width / sum(f.shape[1] for f in frames)


def preview(*frames, width=PREVIEW_W):
    """Кадры рядом, уменьшенные под окно. Пропорции сохраняются.

    Подписи поверх результата рисуйте ПОСЛЕ вызова — иначе текст уменьшится
    вместе с кадром и станет нечитаемым.
    """
    scale = preview_scale(*frames, width=width)
    return cv2.hconcat([cv2.resize(f, None, fx=scale, fy=scale) for f in frames])


def overlay_scale(frame, side_by_side=1):
    """Множитель шрифта и толщины линий для рисования ПО полному кадру.

    Рамки и подписи ставятся в пиксельных координатах полного кадра (иначе
    они не совпадут с детекцией), а окно preview() уменьшено — без этого
    множителя на 1920 px подписи выходят втрое мельче, чем раньше на 1280.
    side_by_side — сколько кадров в том же окне.
    """
    return frame.shape[1] * side_by_side / PREVIEW_W


def _auto_settings(cap):
    """(экспозиция, температура ББ), подобранные автоматикой; None — не поддержано."""
    def read(prop):
        v = cap.get(prop)
        return v if v > 0 else None
    return read(cv2.CAP_PROP_EXPOSURE), read(cv2.CAP_PROP_WB_TEMPERATURE)


def _mean(values):
    known = [v for v in values if v is not None]
    return sum(known) / len(known) if known else None

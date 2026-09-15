"""Просмотр записанного эпизода: сводка и видео из кадров.

Запуск:
  python episode_review.py                  — список эпизодов
  python episode_review.py 20260901_143022  — сводка по эпизоду + собрать видео
  python episode_review.py last             — то же для последнего эпизода

Видео пишется в сам эпизод (review.mp4): левый и правый кадр рядом,
с подписью фазы и времени — удобно разобрать неудачный захват.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("episodes")
FPS = 6


def load(ep_dir):
    recs = [json.loads(l) for l in
            (ep_dir / "records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    meta_path = ep_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return recs, meta


def summarize(ep_dir, recs, meta):
    print(f"\nЭпизод: {ep_dir.name}")
    print(f"  итог:        {'УСПЕХ' if meta.get('success') else 'неудача'} — {meta.get('note','')}")
    print(f"  длительность: {meta.get('duration_s','?')} с, записей {len(recs)}")
    if "apple_arm_xyz" in meta:
        print(f"  яблоко (рука): {[round(v,1) for v in meta['apple_arm_xyz']]} мм, "
              f"дистанция {meta.get('distance_mm',0):.0f} мм")

    phases = {}
    for r in recs:
        if "phase" in r:
            phases.setdefault(r["phase"], 0)
            phases[r["phase"]] += 1
    print(f"  фазы:        {phases}")

    events = [r for r in recs if "event" in r]
    for e in events:
        rest = {k: v for k, v in e.items() if k not in ("i", "t", "event")}
        print(f"  событие t={e['t']:6.2f}s  {e['event']}: {rest}")

    # пик нагрузки на клешне — показывает, как сильно сжимали
    loads = [r["telemetry"]["gripper"]["load"] for r in recs
             if "telemetry" in r and "gripper" in r["telemetry"]
             and "load" in r["telemetry"]["gripper"]]
    if loads:
        print(f"  нагрузка клешни: макс {max(loads)}, медиана {sorted(loads)[len(loads)//2]}")
    temps = [t["temperature"] for r in recs if "telemetry" in r
             for m, t in r["telemetry"].items() if isinstance(t, dict) and "temperature" in t]
    if temps:
        print(f"  температура сервоприводов: макс {max(temps)}°C")


def make_video(ep_dir, recs):
    frames = [r for r in recs if "frames" in r]
    if not frames:
        print("  кадров нет — видео не собрать")
        return
    first = cv2.imread(str(ep_dir / frames[0]["frames"]["left"]))
    if first is None:
        print("  не читаются кадры")
        return
    h, w = first.shape[:2]
    scale = 640 / w
    tile = (int(w * scale), int(h * scale))
    out_path = ep_dir / "review.mp4"
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                         FPS, (tile[0] * 2, tile[1]))
    for r in frames:
        imgs = []
        for side in ("left", "right"):
            p = r["frames"].get(side)
            img = cv2.imread(str(ep_dir / p)) if p else None
            imgs.append(cv2.resize(img, tile) if img is not None
                        else np.zeros((tile[1], tile[0], 3), "uint8"))
        canvas = cv2.hconcat(imgs)
        cv2.putText(canvas, f"{r['t']:6.2f}s  {r.get('phase','')}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        vw.write(canvas)
    vw.release()
    print(f"  видео: {out_path} ({len(frames)} кадров)")


def main():
    if not ROOT.exists():
        raise SystemExit("Папки episodes/ ещё нет — сначала запустите pick_apple.py")
    eps = sorted(p for p in ROOT.iterdir() if p.is_dir())
    if not eps:
        raise SystemExit("Эпизодов пока нет")

    if len(sys.argv) < 2:
        print("Записанные эпизоды:")
        for p in eps:
            meta_p = p / "meta.json"
            mark = ""
            if meta_p.exists():
                m = json.loads(meta_p.read_text(encoding="utf-8"))
                mark = f"  {'успех' if m.get('success') else 'неудача'} — {m.get('note','')}"
            print(f"  {p.name}{mark}")
        print("\nПодробнее:  python episode_review.py <имя>   (или last)")
        return

    name = sys.argv[1]
    ep_dir = eps[-1] if name == "last" else ROOT / name
    if not ep_dir.exists():
        raise SystemExit(f"Нет такого эпизода: {ep_dir}")
    recs, meta = load(ep_dir)
    summarize(ep_dir, recs, meta)
    make_video(ep_dir, recs)


if __name__ == "__main__":
    main()

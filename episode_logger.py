"""Запись эпизодов сбора: кадры, углы, координаты, телеметрия, свои датчики.

Каждый запуск захвата — отдельная папка episodes/ГГГГММДД_ЧЧММСС/ со всем,
что происходило. Это материал для обучения политик и для разбора неудач
(«почему на этот раз не взял»).

Что пишется:
  meta.json     — параметры запуска, чем закончилось
  records.jsonl — по строке на такт: время, фаза, углы суставов, точка кончика,
                  цель, телеметрия сервоприводов, показания своих датчиков
  frames/       — кадры с обеих камер (jpg), имена привязаны к строкам записи

Свои датчики подключаются одной функцией: она возвращает словарь, который
попадает в каждую запись. Например:

    def my_sensors():
        return {"lux": read_light(), "distance_mm": read_range()}

    logger = EpisodeLogger(extra_sensors=my_sensors)

Телеметрия и кадры пишутся не на каждом такте, а по таймеру (FRAME_EVERY /
TELEMETRY_EVERY), чтобы не тормозить управление рукой.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# какие регистры сервоприводов писать (читаются одним запросом на всю руку)
TELEMETRY_REGISTERS = ["Present_Load", "Present_Current",
                       "Present_Temperature", "Present_Voltage"]


def _jsonable(v):
    """numpy -> обычные типы, иначе json не запишет."""
    if isinstance(v, np.ndarray):
        return [float(x) for x in v.ravel()]
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


class EpisodeLogger:
    def __init__(self, root="episodes", name=None,
                 save_frames=True, frame_every=0.25, telemetry_every=0.25,
                 extra_sensors=None, jpeg_quality=85):
        stamp = name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = Path(root) / stamp
        self.frames_dir = self.dir / "frames"
        self.dir.mkdir(parents=True, exist_ok=True)
        if save_frames:
            self.frames_dir.mkdir(exist_ok=True)

        self.save_frames = save_frames
        self.frame_every = frame_every
        self.telemetry_every = telemetry_every
        self.extra_sensors = extra_sensors
        self.jpeg_quality = jpeg_quality

        self._t0 = time.time()
        self._n = 0
        self._last_frame_t = -1e9
        self._last_telem_t = -1e9
        self._last_telemetry = {}
        self._records = open(self.dir / "records.jsonl", "w", encoding="utf-8")
        self.meta = {"started": datetime.now().isoformat()}

    # ---------- служебное ----------

    @property
    def t(self):
        return time.time() - self._t0

    def set_meta(self, **kw):
        """Параметры эпизода: калибровки, пороги, координаты цели и т.п."""
        self.meta.update(_jsonable(kw))

    def _read_telemetry(self, bus):
        """Телеметрия всех сервоприводов. Ошибки шины не должны ронять захват."""
        out = {}
        for reg in TELEMETRY_REGISTERS:
            try:
                vals = bus.sync_read(reg, normalize=False)
                for motor, v in vals.items():
                    out.setdefault(motor, {})[reg.replace("Present_", "").lower()] = int(v)
            except Exception as e:            # шина занята/сбой — пропускаем регистр
                out.setdefault("_errors", []).append(f"{reg}: {type(e).__name__}")
        return out

    # ---------- запись ----------

    def wants_frame(self):
        """Нужен ли кадр на этом такте (кадры пишутся по таймеру, а не подряд).

        Чтение камеры стоит десятки миллисекунд, а tick() зовётся на каждой точке
        траектории — поэтому спрашиваем ЗАРАНЕЕ и не читаем кадр впустую.
        """
        return self.save_frames and (self.t - self._last_frame_t >= self.frame_every)

    def tick(self, phase, joints=None, tip_xyz=None, frames=None,
             bus=None, extra=None, force=False):
        """Одна запись состояния. Кадры и телеметрия — по таймеру (или force=True)."""
        now = self.t
        rec = {"i": self._n, "t": round(now, 3), "phase": phase}

        if joints is not None:
            rec["joints"] = _jsonable(joints)
        if tip_xyz is not None:
            rec["tip_xyz"] = _jsonable(tip_xyz)

        if bus is not None and (force or now - self._last_telem_t >= self.telemetry_every):
            self._last_telemetry = self._read_telemetry(bus)
            self._last_telem_t = now
        if self._last_telemetry:
            rec["telemetry"] = self._last_telemetry

        if self.save_frames and frames and (force or now - self._last_frame_t >= self.frame_every):
            saved = {}
            for cam_name, img in frames.items():
                if img is None:
                    continue
                fname = f"{cam_name}_{self._n:06d}.jpg"
                cv2.imwrite(str(self.frames_dir / fname), img,
                            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                saved[cam_name] = f"frames/{fname}"
            if saved:
                rec["frames"] = saved
                self._last_frame_t = now

        if self.extra_sensors is not None:
            try:
                extra_vals = self.extra_sensors() or {}
            except Exception as e:
                extra_vals = {"_error": f"{type(e).__name__}: {e}"}
            rec["sensors"] = _jsonable(extra_vals)
        if extra:
            rec.setdefault("sensors", {}).update(_jsonable(extra))

        self._records.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._records.flush()
        self._n += 1
        return rec

    def event(self, name, **data):
        """Ключевой момент эпизода (контакт, промах, отмена) — отдельной строкой."""
        rec = {"i": self._n, "t": round(self.t, 3), "event": name, **_jsonable(data)}
        self._records.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._records.flush()
        self._n += 1
        return rec

    def finish(self, success, note=""):
        self.meta.update({"success": bool(success), "note": note,
                          "records": self._n,
                          "duration_s": round(self.t, 2),
                          "finished": datetime.now().isoformat()})
        (self.dir / "meta.json").write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")
        self._records.close()
        print(f"Эпизод записан: {self.dir}  "
              f"({self._n} записей, {'успех' if success else 'неудача'})")
        return self.dir

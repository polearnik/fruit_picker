"""Подъёмник: каретка с рукой на вертикальной стойке, ремень GT2, мотор STS3215.

Мотор — такой же сервопривод, как в суставах, на той же шине, ID 7
(назначается arm/set_servo_id.py). Шкив GT2 20 зубьев: один оборот мотора
(4096 позиций) — 40 мм хода каретки, на 2000 мм — около 50 оборотов.

Почему не обычный режим сервопривода
  В режиме 0 серво держит угол только в пределах ОДНОГО оборота — это 40 мм.
  Поэтому мотор переводится в шаговый режим (Operating_Mode = 3): Goal_Position
  там — не абсолютный угол, а СДВИГ от текущего положения, и после сдвига серво
  держит позицию с усилием (каретка не сползает). Абсолютную высоту считаем
  сами («одометр»).

Как мотор ведёт себя в шаговом режиме (прошивка 3.10, замеры
arm/lift_probe.py и arm/lift_stop_probe.py):
  - Present_Position — НЕ угол вала, а ОСТАТОК пути до цели, со знаком;
  - новая команда ПРИБАВЛЯЕТСЯ к остатку, а не заменяет его: был остаток 4,
    пришла команда -4096 — читается -4092;
  - поэтому «сдвиг 0», «сдвиг 1 шаг» и снятие момента мотор НЕ
    останавливают — он доезжает накопленную цель.
Отсюда одометр: держим свою цель _target (сумма всех команд), положение =
_target - остаток. А остановка — команда «минус остаток»: цель становится
равна текущему положению (halt()).

Высота z — в мм ОТ ПОЛА, «вверх» = больше. Отсчёт по нижнему упору: хоминг
(home()) медленно ведёт каретку вниз с ограниченным усилием, пока она не
встанет, и приравнивает это положение к STOP_Z_MM. Без хоминга после
включения высота неизвестна, и move_to() откажется ехать.

Почему от пола, а не от упора: упор можно переставить (сейчас временный на
660 мм), а точки, снятые в мировых координатах, — сброс в ящик, hand-eye, —
от этого не должны портиться. Переставили упор — поменяли одно STOP_Z_MM.

Проверка на железе:  python arm/lift_test.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import scservo_sdk as scs

LIFT_ID = 7
PORT = "/dev/ttyACM0"
BAUD = 1_000_000

STEPS_PER_REV = 4096
MM_PER_REV = 40.0                     # GT2 (шаг 2 мм) x 20 зубьев
MM_PER_STEP = MM_PER_REV / STEPS_PER_REV

# Какой знак сдвига поднимает каретку. Зависит от того, с какой стороны мотор
# и как лёг ремень, — определяется arm/lift_test.py и вписывается сюда.
UP_SIGN = +1   # проверено arm/lift_jog.py: «+» везёт каретку вверх

# Высота каретки над полом, когда она стоит на нижнем упоре. Мерить рулеткой
# от пола до той же точки каретки, от которой будете мерить и дальше.
# Сейчас упор временный, на 660 мм; уберёте — впишите новую высоту.
STOP_Z_MM = 660.0

# Рабочий ход, мм от пола. Внизу — чуть выше упора, вверху — с запасом от
# реального конца стойки: там упора нет, упираться нельзя.
Z_MIN_MM = STOP_Z_MM + 5.0
Z_MAX_MM = 1900.0

# Скорость и ускорение (единицы сервопривода: позиций/с и 100 позиций/с²).
# Потолок STS3215 на 7 В — около 3000 позиций/с = ~30 мм/с, то есть весь
# ход 2 м занимает больше минуты. Быстрее этим мотором не получится.
# Нагрузка здесь — это скважность ШИМ, а не усилие на ремне: у потолка
# скорости она растёт к 1000 просто потому, что мотору не хватает напряжения.
# Замер «~780/1000 на 2400» снят, когда каретка упиралась в конец стойки, и
# занижает запас. Если на этой скорости начнутся «каретка встала»/недоезды —
# вернуть 2400.
SPEED = 2800          # ~27 мм/с — вниз (помогает вес)
# Вверх медленнее: у потолка скорости нагрузка (ШИМ) уходит за Overload_Torque
# (80 %), и через Protection_Time (2 с) серво само режет момент до
# Protective_Torque (20 %) — каретка ползёт и не доезжает за отведённое время.
# Подбирать так, чтобы нагрузка на подъёме держалась ниже ~750.
SPEED_UP = 2000       # ~20 мм/с
ACCEL = 50            # ~50 мм/с²: разгон до полной скорости за ~0.6 с
# Медленно вниз к упору. На этой скорости ход берёт ~120/1000 нагрузки
# (arm/lift_stop_probe.py, после исправления направления и остановки).
# Ранние замеры 250-600 были искажены: каретка ехала вверх и упиралась в
# конец стойки, а недоеханные команды копились.
HOME_SPEED = 300      # ~3 мм/с
# Предел момента при хоминге — ВТОРАЯ защита: главная — рост нагрузки
# (см. home()). Должен быть заметно выше «обычная нагрузка + HOME_LOAD_RISE»
# (~180), но ниже того, где шкив начинает перескакивать по ремню (~250-350).
HOME_TORQUE = 320
# Упор по нагрузке: сначала на свободном спуске меряем обычную нагрузку
# (HOME_BASELINE_S после разгона), упор — когда она выросла больше чем на
# HOME_LOAD_RISE и держится так HOME_LOAD_POLLS опросов подряд (~60 мс).
HOME_SETTLE_S = 0.5           # разгон: нагрузку в это время не смотрим
HOME_BASELINE_S = 0.5         # сколько мерить обычную нагрузку после разгона
HOME_LOAD_RISE = 60
HOME_LOAD_POLLS = 3
HOME_CHUNK_STEPS = 30000      # ~290 мм за команду: меньше разгонов по пути
# Быстрый хоминг: последняя известная высота хранится в STATE_FILE, и до упора
# сначала едем на рабочей скорости, а медленно — только последние
# HOME_FAST_MARGIN_MM. Запас покрывает сползание каретки при выключенном
# питании; если её двигали руками на заметно большее — удалите STATE_FILE.
STATE_FILE = Path(__file__).with_name("lift_state.json")
HOME_FAST_MARGIN_MM = 60.0
WORK_TORQUE = 1000

# Сдвиг за одну команду. Каждый кусок — это остановка и новый разгон, поэтому
# куски крупные (~290 мм, как при хоминге): по 80 мм каретка больше
# разгонялась и тормозила, чем ехала.
CHUNK_STEPS = 30000
POLL_S = 0.02
HALT_WAIT_S = 1.0     # сколько ждать остановки после halt()
MAX_STEP_CMD = 32767  # Goal_Position: 15 бит + знак
POS_TOL_STEPS = 20    # ~0.2 мм — «доехали»
STALL_S = 0.6         # сколько стоим без движения, чтобы считать это упором

# Регистры STS3215 (см. lerobot/motors/feetech/tables.py)
ADDR_MIN_LIMIT, ADDR_MAX_LIMIT = 9, 11
ADDR_MODE = 33
ADDR_TORQUE_ENABLE = 40
ADDR_ACCEL = 41
ADDR_GOAL_POS = 42
ADDR_GOAL_SPEED = 46
ADDR_TORQUE_LIMIT = 48
ADDR_LOCK = 55
ADDR_PRESENT_POS = 56
ADDR_PRESENT_LOAD = 60
ADDR_MOVING = 66
MODE_STEP = 3


def _sign_mag_decode(v, bit):
    return -(v & ((1 << bit) - 1)) if v & (1 << bit) else v


def _sign_mag_encode(v, bit):
    return (1 << bit) | (-v) if v < 0 else v


def confirm(msg):
    """Вопрос «да/нет» в терминале; да — y/yes, а также д/да и «н» (та же
    клавиша, что y, в русской раскладке).

    Читаем байты и декодируем с errors="ignore": если набрать русскую букву и
    стереть её, терминал стирает только один из двух её байтов UTF-8, и
    обычный input() падает с UnicodeDecodeError.
    """
    print(f"{msg} [y/N] ", end="", flush=True)
    line = sys.stdin.buffer.readline().decode("utf-8", errors="ignore")
    return line.strip().lower() in ("y", "yes", "д", "да", "н")


def _load_z():
    """Последняя известная высота каретки из STATE_FILE или None."""
    try:
        return float(json.loads(STATE_FILE.read_text())["z_mm"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _save_z(z_mm):
    try:
        STATE_FILE.write_text(json.dumps({"z_mm": round(float(z_mm), 1)}))
    except OSError:
        pass


def _forget_z():
    try:
        STATE_FILE.unlink()
    except FileNotFoundError:
        pass


class LiftError(RuntimeError):
    pass


class Lift:
    """Каретка подъёмника. Использует чужой порт (шину руки) или открывает свой.

    С рукой:     lift = Lift(arm.bus.port_handler, arm.bus.packet_handler)
    Отдельно:    lift = Lift.open()     (рука при этом не должна быть подключена)
    """

    def __init__(self, port_handler, packet_handler, servo_id=LIFT_ID):
        self.port = port_handler
        self.pkt = packet_handler
        self.id = servo_id
        self._own_port = False
        self._target = 0       # сумма всех команд сдвига = цель мотора, шаги
        self._steps = 0        # одометр: _target - остаток
        self._zero = None      # _steps на нижнем упоре; None — хоминга не было
        self._check(*self.pkt.ping(self.port, self.id)[1:], "ping")
        # от прошлого запуска в регистре висит недоход прошлой команды —
        # принимаем текущее положение за 0 одометра
        self._target = self._remaining()

    @classmethod
    def open(cls, port=PORT, servo_id=LIFT_ID):
        ph = scs.PortHandler(port)
        if not ph.openPort():
            raise LiftError(f"Не удалось открыть {port}")
        ph.setBaudRate(BAUD)
        lift = cls(ph, scs.PacketHandler(0), servo_id)
        lift._own_port = True
        return lift

    def close(self):
        """Сбросить недоеханный остаток и (если порт свой) закрыть его.

        Мотор доезжает накопленную цель и без программы: без этого сброса
        Ctrl+C посреди хода не остановил бы каретку.
        """
        try:
            self._step(-self._remaining())
            if self.homed:
                _save_z(self.z())
        except LiftError:
            pass
        finally:
            if self._own_port:
                self.port.closePort()

    # --- низкий уровень ---------------------------------------------------
    def _check(self, comm, err, what):
        if comm != scs.COMM_SUCCESS:
            raise LiftError(f"мотор подъёмника (ID {self.id}) не ответил на {what}: "
                            f"{self.pkt.getTxRxResult(comm)}")
        if err:
            raise LiftError(f"мотор подъёмника: ошибка при {what}: "
                            f"{self.pkt.getRxPacketError(err)}")

    def _w1(self, addr, v, what):
        self._check(*self.pkt.write1ByteTxRx(self.port, self.id, addr, v), what)

    def _w2(self, addr, v, what):
        self._check(*self.pkt.write2ByteTxRx(self.port, self.id, addr, v), what)

    def _r1(self, addr, what):
        v, comm, err = self.pkt.read1ByteTxRx(self.port, self.id, addr)
        self._check(comm, err, what)
        return v

    def _r2(self, addr, what):
        v, comm, err = self.pkt.read2ByteTxRx(self.port, self.id, addr)
        self._check(comm, err, what)
        return v

    # --- настройка ----------------------------------------------------------
    def setup(self):
        """Шаговый режим, скорость, ускорение, включить удержание.

        Режим пишется в EEPROM — один раз навсегда, но повтор безвреден.
        Важно: в момент смены режима момент выключен, каретка держится только
        трением/противовесом. Держите её рукой или ставьте в нижнее положение.
        """
        if self._r1(ADDR_MODE, "чтение режима") != MODE_STEP:
            self._w1(ADDR_TORQUE_ENABLE, 0, "выключение момента")
            self._w1(ADDR_LOCK, 0, "разблокировка EEPROM")
            self._w2(ADDR_MIN_LIMIT, 0, "предел min")
            self._w2(ADDR_MAX_LIMIT, 0, "предел max")
            self._w1(ADDR_MODE, MODE_STEP, "режим")
            self._w1(ADDR_LOCK, 1, "блокировка EEPROM")
        self._w1(ADDR_ACCEL, ACCEL, "ускорение")
        self._w2(ADDR_GOAL_SPEED, SPEED, "скорость")
        self._w2(ADDR_TORQUE_LIMIT, WORK_TORQUE, "предел момента")
        self._w1(ADDR_TORQUE_ENABLE, 1, "включение момента")
        self._update()

    def relax(self):
        """Снять момент. ОСТОРОЖНО: без противовеса каретка упадёт."""
        _forget_z()   # каретка может сползти — сохранённой высоте не верим
        self._w1(ADDR_TORQUE_ENABLE, 0, "выключение момента")

    # --- одометр ----------------------------------------------------------
    def _remaining(self):
        """Остаток пути по текущей команде, шаги со знаком."""
        return _sign_mag_decode(self._r2(ADDR_PRESENT_POS, "чтение позиции"), 15)

    def _update(self):
        self._steps = self._target - self._remaining()
        return self._steps

    def load(self):
        """Нагрузка мотора 0..1000 (без знака)."""
        return abs(_sign_mag_decode(self._r2(ADDR_PRESENT_LOAD, "чтение нагрузки"), 10))

    @property
    def homed(self):
        return self._zero is not None

    def z(self):
        """Текущая высота каретки, мм от пола."""
        if not self.homed:
            raise LiftError("высота неизвестна — сначала home()")
        return STOP_Z_MM + (self._update() - self._zero) * UP_SIGN * MM_PER_STEP

    # --- движение ---------------------------------------------------------
    def _step(self, steps):
        # мотор прибавит сдвиг к своей цели — делаем так же у себя
        steps = int(steps)
        if abs(steps) > MAX_STEP_CMD:
            raise LiftError(f"сдвиг {steps} не влезает в регистр (макс ±{MAX_STEP_CMD})")
        self._w2(ADDR_GOAL_POS, _sign_mag_encode(steps, 15), "команда сдвига")
        self._target += steps

    def halt(self):
        """Остановить мотор посреди сдвига и держать текущее положение.

        Команды складываются, поэтому стоп — это «минус остаток»: цель
        становится равна текущему положению. Мотор тормозит с заданным
        ускорением (на скорости хоминга — доли миллиметра) и встаёт.
        Если за HALT_WAIT_S не встал — снимаем момент: каретку держат
        противовес и трение, а давить в упор и рвать ремень хуже.
        """
        self._step(-self._remaining())
        t_end = time.time() + HALT_WAIT_S
        while time.time() < t_end:
            time.sleep(POLL_S)
            if not self._r1(ADDR_MOVING, "чтение Moving"):
                self._update()
                return
        self._w1(ADDR_TORQUE_ENABLE, 0, "аварийный стоп: момент выкл")
        self._update()
        raise LiftError("мотор не остановился по команде — момент снят")

    def _move_steps(self, steps, stop_on_stall=False, timeout_s=None,
                    stop_check=None, chunk_steps=CHUNK_STEPS):
        """Сдвинуть на steps позиций, дробя на куски по chunk_steps.

        Возвращает True, если доехали; False — если встали на упор (только при
        stop_on_stall) или сработал stop_check() (вызывается на каждом опросе,
        True — немедленно остановиться). Застревание без stop_on_stall —
        ошибка: значит каретку что-то держит, ехать дальше нельзя.
        """
        target = self._update() + int(steps)
        if timeout_s is None:
            timeout_s = 5.0 + abs(steps) / max(SPEED, 1) * 2.0
        t_end = time.time() + timeout_s
        last_pos, last_move_t = self._steps, time.time()

        while True:
            left = target - self._steps
            if abs(left) <= POS_TOL_STEPS:
                return True
            chunk = int(np.clip(left, -chunk_steps, chunk_steps))
            chunk_start = self._steps
            self._step(chunk)
            # ждём, пока этот кусок доедет (или встанет). Сдвиги в шаговом
            # режиме СКЛАДЫВАЮТСЯ: если принять «Moving = 0» сразу после
            # команды, пока мотор ещё не тронулся, следующая команда удвоит ход.
            # Поэтому конец куска — только когда мотор стоит И либо прошёл
            # почти весь кусок, либо давно не двигается (упор).
            while True:
                time.sleep(POLL_S)
                pos = self._update()
                if abs(pos - last_pos) > 3:
                    last_pos, last_move_t = pos, time.time()
                if stop_check is not None and stop_check():
                    self.halt()
                    return False
                moving = self._r1(ADDR_MOVING, "чтение Moving")
                done_chunk = abs(chunk - (pos - chunk_start)) <= POS_TOL_STEPS * 5
                if not moving and (done_chunk or time.time() - last_move_t > STALL_S):
                    break
                if time.time() - last_move_t > STALL_S:
                    if stop_on_stall:
                        self.halt()
                        return False
                    self.halt()
                    raise LiftError(f"каретка встала на z≈{self._z_or_steps()} "
                                    f"(нагрузка {self.load()}) — что-то держит")
                if time.time() > t_end:
                    self.halt()
                    raise LiftError("подъёмник не доехал за отведённое время")
            if time.time() - last_move_t > STALL_S:
                if stop_on_stall:
                    return False
                raise LiftError(f"каретка не двигается на z≈{self._z_or_steps()} "
                                f"(нагрузка {self.load()})")

    def _z_or_steps(self):
        if self.homed:
            return f"{STOP_Z_MM + (self._steps - self._zero) * UP_SIGN * MM_PER_STEP:.0f} мм"
        return f"{self._steps} шагов"

    def home(self, max_travel_mm=Z_MAX_MM + 200):
        """Найти нижний упор, принять его за z = STOP_Z_MM, отойти на Z_MIN_MM.

        Рука перед этим должна быть в позе, в которой она НЕ достаёт до пола
        и до ящика, когда каретка внизу.
        """
        # Сохранённую высоту забываем сразу: если хоминг прервётся, каретка
        # окажется неизвестно где, и следующий начнёт медленно.
        saved_z = _load_z()
        _forget_z()
        fast_mm = (saved_z - STOP_Z_MM - HOME_FAST_MARGIN_MM
                   if saved_z is not None else 0.0)
        if fast_mm > 0:
            print(f"[подъёмник] хоминг: с последней высоты {saved_z:.0f} мм "
                  f"быстро вниз на {fast_mm:.0f} мм...")
            try:
                if not self._move_steps(-UP_SIGN * fast_mm / MM_PER_STEP,
                                        stop_on_stall=True):
                    print("[подъёмник] каретка встала раньше — упор ближе, "
                          "чем думали; дальше медленно")
            except BaseException:
                self.halt()
                raise

        print("[подъёмник] хоминг: медленно вниз до упора...")
        self._w2(ADDR_GOAL_SPEED, HOME_SPEED, "скорость хоминга")
        self._w2(ADDR_TORQUE_LIMIT, HOME_TORQUE, "предел момента хоминга")

        # Упор ловим по РОСТУ нагрузки: когда ремень проскальзывает по шкиву,
        # вал продолжает крутиться, и по одной остановке упор не увидеть.
        t0 = time.time()
        baseline = []
        st = {"over": 0, "limit": None, "peak": 0, "s0": None, "at_start": False}

        def hit_stop():
            ld = self.load()
            t = time.time() - t0
            if t < HOME_SETTLE_S:
                return False
            if t < HOME_SETTLE_S + HOME_BASELINE_S:
                if st["s0"] is None:
                    st["s0"] = self._steps
                baseline.append(ld)
                return False
            if st["limit"] is None:
                # Каретка уже стояла на упоре — тогда «обычная» нагрузка
                # измерена в упоре, и роста не будет. Узнаём это по тому, что
                # за время замера она почти не сдвинулась.
                moved = abs(self._steps - st["s0"])
                if moved < 0.3 * HOME_SPEED * HOME_BASELINE_S:
                    st["at_start"] = True
                    return True
                st["limit"] = float(np.median(baseline)) + HOME_LOAD_RISE
                if st["limit"] > 0.9 * HOME_TORQUE:
                    raise LiftError(
                        f"обычная нагрузка на спуске {np.median(baseline):.0f}: порог "
                        f"упора {st['limit']:.0f} упирается в предел момента "
                        f"{HOME_TORQUE} — упор не поймать. Уменьшите HOME_SPEED "
                        f"или поднимите HOME_TORQUE")
                print(f"[подъёмник] обычная нагрузка на спуске "
                      f"{np.median(baseline):.0f}, упор — выше {st['limit']:.0f}")
            st["peak"] = max(st["peak"], ld)
            if t >= st.get("next_print", 0):
                st["next_print"] = t + 1.0
                print(f"[подъёмник]   {t:5.1f} с: нагрузка {ld}, "
                      f"пройдено {abs(self._steps - st['s0']) * MM_PER_STEP:.0f} мм")
            st["over"] = st["over"] + 1 if ld > st["limit"] else 0
            return st["over"] >= HOME_LOAD_POLLS

        try:
            steps = -UP_SIGN * max_travel_mm / MM_PER_STEP
            reached = self._move_steps(
                steps, stop_on_stall=True, stop_check=hit_stop,
                chunk_steps=HOME_CHUNK_STEPS,
                timeout_s=10 + max_travel_mm / (HOME_SPEED * MM_PER_STEP) * 1.5)
            if reached:
                raise LiftError("хоминг прошёл весь ход и не нашёл упора — "
                                "проверьте UP_SIGN (возможно, ехали вверх)")
            how = ("каретка уже стояла на нём" if st["at_start"] else
                   f"пойман по росту нагрузки (пик {st['peak']})"
                   if st["over"] >= HOME_LOAD_POLLS else "пойман по остановке")
            print(f"[подъёмник] упор: {how}")
        except BaseException:
            self.halt()   # любая ошибка или Ctrl+C посреди хоминга — сначала стоп
            raise
        finally:
            self._w2(ADDR_GOAL_SPEED, SPEED, "скорость")
            self._w2(ADDR_TORQUE_LIMIT, WORK_TORQUE, "предел момента")
        self._zero = self._update()
        print(f"[подъёмник] упор найден, z = {STOP_Z_MM:.0f} мм от пола")
        self.move_to(Z_MIN_MM)

    def move_to(self, z_mm):
        """Поехать на высоту z_mm (мм от пола) и дождаться."""
        if not Z_MIN_MM - 1e-6 <= z_mm <= Z_MAX_MM + 1e-6:
            raise LiftError(f"z={z_mm:.0f} мм вне хода [{Z_MIN_MM:.0f}, {Z_MAX_MM:.0f}]")
        z0 = self.z()
        dz = z_mm - z0
        # если ход оборвётся, каретка где-то между z0 и z_mm; для быстрого
        # хоминга безопасно занизить высоту, а не завысить
        _save_z(min(z0, z_mm))
        speed = SPEED_UP if dz > 0 else SPEED
        steps = UP_SIGN * dz / MM_PER_STEP
        t0 = time.time()
        self._w2(ADDR_GOAL_SPEED, speed, "скорость")
        try:
            self._move_steps(steps, timeout_s=5.0 + abs(steps) / speed * 2.0)
        finally:
            self._w2(ADDR_GOAL_SPEED, SPEED, "скорость")
        z1, dt = self.z(), time.time() - t0
        _save_z(z1)
        if abs(z1 - z0) > 20:
            print(f"[подъёмник] {z0:.0f} -> {z1:.0f} мм за {dt:.1f} с "
                  f"({abs(z1 - z0) / dt:.0f} мм/с)")
        return z1

    def move_by(self, dz_mm):
        return self.move_to(self.z() + dz_mm)

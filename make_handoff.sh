#!/usr/bin/env bash
# Упаковать проект для переноса на другой компьютер/аккаунт.
#
#   ./make_handoff.sh              — код + калибровки (без эпизодов, лёгкий)
#   ./make_handoff.sh --episodes   — плюс записанные эпизоды (может быть тяжело)
#
# Результат: fruit-picker-handoff-ГГГГММДД.tar.gz в домашней папке.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
LEROBOT_CAL="$HOME/.cache/huggingface/lerobot/calibration/robots/so_follower/fruit_arm.json"
STAMP="$(date +%Y%m%d)"
OUT="$HOME/fruit-picker-handoff-$STAMP.tar.gz"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

WITH_EPISODES=0
[ "${1:-}" = "--episodes" ] && WITH_EPISODES=1

mkdir -p "$STAGE/fruit-picker"

# код и документы (venv, кэш и веса моделей не тащим — ставятся заново)
# venv, кэши, веса моделей и настройки IDE не переносим — ставятся/качаются заново
EXCLUDES=(--exclude=venv --exclude=__pycache__ --exclude='*.pt'
          --exclude=runs --exclude=.git --exclude=calib_pairs
          --exclude=.idea --exclude=.codegraph --exclude=weights
          --exclude=.pytest_cache --exclude='*.mp4')
[ "$WITH_EPISODES" -eq 0 ] && EXCLUDES+=(--exclude=episodes)

tar -C "$PROJECT_DIR" "${EXCLUDES[@]}" -cf - . | tar -C "$STAGE/fruit-picker" -xf -

# калибровка руки лежит вне проекта — кладём рядом
mkdir -p "$STAGE/lerobot-calibration"
if [ -f "$LEROBOT_CAL" ]; then
    cp "$LEROBOT_CAL" "$STAGE/lerobot-calibration/"
    echo "  + калибровка руки: fruit_arm.json"
else
    echo "  ! ВНИМАНИЕ: калибровка руки не найдена: $LEROBOT_CAL"
fi

cat > "$STAGE/READ_ME_FIRST.txt" <<'EOF'
Перенос проекта "робот-сборщик фруктов".

1. Распакуйте архив, папку fruit-picker положите куда удобно (напр. ~/ClaudeCode/).

2. Верните калибровку руки на место:
     mkdir -p ~/.cache/huggingface/lerobot/calibration/robots/so_follower
     cp lerobot-calibration/fruit_arm.json \
        ~/.cache/huggingface/lerobot/calibration/robots/so_follower/

3. Создайте окружение:
     cd fruit-picker
     python3 -m venv venv
     ./venv/bin/pip install ultralytics opencv-python feetech-servo-sdk placo lerobot
     ./venv/bin/pip install 'numpy<2.3'

4. Прочитайте fruit-picker/HANDOFF.md — там всё состояние проекта,
   подобранные числа и разобранные грабли. Его же дайте первым сообщением
   в новый чат с ассистентом.

Проверьте, что на месте (без них робот не заработает):
   fruit-picker/stereo_calib.npz
   fruit-picker/arm/handeye.npz
   lerobot-calibration/fruit_arm.json
EOF

# проверяем, что критичные файлы попали
MISSING=0
for f in "fruit-picker/HANDOFF.md" "fruit-picker/stereo_calib.npz" "fruit-picker/arm/handeye.npz"; do
    if [ -f "$STAGE/$f" ]; then echo "  + $f"; else echo "  ! НЕ НАЙДЕН: $f"; MISSING=1; fi
done

tar -C "$STAGE" -czf "$OUT" .
echo
echo "Архив: $OUT  ($(du -h "$OUT" | cut -f1))"
[ "$MISSING" -eq 1 ] && echo "ВНИМАНИЕ: часть критичных файлов отсутствует — см. выше."
exit 0

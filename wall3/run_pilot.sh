#!/usr/bin/env bash
# 단계 6 파일럿: 시작값 700000 ~ 700049 벽 50개 수집(10벽마다 1벽 녹화) → 시방 합격 벽만 LeRobot 변환·검사.
# 화면 없이(EGL) 돈다. nohup으로 띄우면 터미널이 닫혀도 계속된다:
#   nohup bash wall3/run_pilot.sh 20261016 > /dev/null 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
DATE="${1:?usage: run_pilot.sh YYYYMMDD}"
RAW="outputs/wall3/pilot_${DATE}"
ROOT="data/wall3_pilot_${DATE}"
for d in "$RAW" "$ROOT"; do [ -e "$d" ] && { echo "이미 있음: $d (덮어쓰지 않음)"; exit 1; }; done
mkdir -p "$(dirname "$RAW")"
LOG="${RAW}.log"
export MUJOCO_GL=egl PYTHONPATH=.
source venv/bin/activate
{
  echo "== collect $(date '+%F %T')"
  python -m wall3.collect --seed0 700000 --n 50 --out "$RAW" --workers 12 --record-every 10
  echo "== convert $(date '+%F %T')"
  python -m wall3.to_lerobot --raw "$RAW" --root "$ROOT"
  echo "== done $(date '+%F %T')"
} > "$LOG" 2>&1

#!/usr/bin/env bash
# 나7 묶음 병렬 변환: 깨끗한 합격 벽(시작값 순서)을 N묶음으로 나눠 동시에 변환한 뒤 하나로 합친다.
# 실행: bash wall3/convert_parts.sh <raw 폴더(여럿이면 쉼표로)> <벽 수> <묶음 수> <묶음 폴더> <최종 데이터셋 폴더> [to_lerobot 추가 인자...]
#   예: bash wall3/convert_parts.sh outputs/wall3/main_20261018,outputs/wall3/main_20261018_extra 1000 10 \
#         outputs/wall3/main_20261018_parts_c data/wall3_main_20261018 --exclude-seeds 710462 710548
set -euo pipefail
RAW=$1; WALLS=$2; N=$3; PARTS=$4; FINAL=$5; shift 5
IFS=',' read -r -a RAWS <<< "$RAW"
cd "$(dirname "$0")/.."
source venv/bin/activate
export PYTHONPATH=.
[ -e "$PARTS" ] && { echo "$PARTS 이미 있음"; exit 1; }
[ -e "$FINAL" ] && { echo "$FINAL 이미 있음"; exit 1; }
mkdir -p "$PARTS"
pids=()
for ((i = 0; i < N; i++)); do
  p=$(printf "%s/part_%02d" "$PARTS" "$i")
  python -m wall3.to_lerobot --raw "${RAWS[@]}" --max-walls "$WALLS" --part "$i" "$N" --root "$p" "$@" > "$p.log" 2>&1 &
  pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
[ $fail -eq 0 ] || { echo "묶음 변환 실패: $PARTS/*.log"; exit 1; }
roots=()
for ((i = 0; i < N; i++)); do roots+=("$(printf "%s/part_%02d" "$PARTS" "$i")"); done
bash wall3/merge_parts.sh "$FINAL" "${roots[@]}"          # part video check, merge, final full video check, "done:"

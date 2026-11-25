#!/usr/bin/env bash
# 묶음 데이터셋을 하나로 합치고(to_lerobot --merge: 다시 읽기 검사·원자료 대조·해시), 영상 전수 검사(check_videos)까지 통과하면
# 마지막 줄에 "done: <최종 폴더>"를 쓴다. 묶음마다 먼저 영상 전수 검사를 해서 깨진 묶음이 있으면 합치지 않는다.
# 실행: bash wall3/merge_parts.sh <최종 데이터셋 폴더> <묶음 폴더...(시작값 순서)>
set -euo pipefail
FINAL=$1; shift
cd "$(dirname "$0")/.."
source venv/bin/activate
export PYTHONPATH=.
[ -e "$FINAL" ] && { echo "$FINAL 이미 있음"; exit 1; }
for p in "$@"; do
  python -m experiments.check_videos --root "$p" --out "$p.videos.json" --workers 12 > "$p.videos.log" 2>&1 \
    || { echo "묶음 영상 깨짐: $p (see $p.videos.json)"; exit 1; }
  echo "part videos ok: $p"
done
python -m wall3.to_lerobot --merge "$@" --root "$FINAL" > "$FINAL.merge.log" 2>&1 || { echo "합치기·다시 읽기 검사 실패: $FINAL.merge.log"; exit 1; }
python -m experiments.check_videos --root "$FINAL" --out "$FINAL/meta/wall3_check_videos.json" --workers 12 > "$FINAL.videos.log" 2>&1 \
  || { echo "최종 데이터셋 영상 깨짐: $FINAL/meta/wall3_check_videos.json"; exit 1; }
echo "done: $FINAL"

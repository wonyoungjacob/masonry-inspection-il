#!/usr/bin/env bash
# main_v2 재수집 → 갈림 검사 → 변환 → 갈림 검사 → 재학습 (2026-10-22 사용자 결정: 벽돌마다 집기 · 놓기 손목 한 갈래, 1000벽 다시 채워 재학습)
#   1) 수집: 대본 main_v2(장면 main_v1 그대로), 시작값 710000부터 깨끗한 합격 벽 1002개(뒤에서 710462 · 710548을 빼도 1000개)
#   2) 원자료 갈림 검사: 모든 벽돌 · 모든 시각 · 관절 6개가 한 갈래(소수 갈래 1개도 안 됨)가 아니면 멈춤
#   3) 400벽 예비 평가(wall3-evalgates)가 끝날 때까지 기다림 (변환 메모리 압박 · 학습 GPU 경쟁 방지)
#   4) 10묶음 변환 · 합치기 · 영상 전수 검사 (convert_parts.sh)
#   5) 데이터셋 갈림 검사 (2와 같은 기준)
#   6) 재학습: train_main.sh (Diffusion s1000 → ACT s1000, 10만 스텝, 1만 스텝마다 저장, torchcodec · 작업자 4, 하나씩)
# 실행(세션과 분리된 사용자 서비스):
#   systemd-run --user --unit=wall3-v2 --property=WorkingDirectory=<repo> bash wall3/pipeline_v2.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl
REPO=$(cd .. && pwd)
TAGD=${TAGD:-20261022}
RAW=${RAW:-outputs/wall3/main_v2_$TAGD}
PARTS=${PARTS:-outputs/wall3/main_v2_${TAGD}_parts}
DS=${DS:-data/wall3_main_v2_$TAGD}
CONVLOG=${CONVLOG:-outputs/wall3/main_v2_${TAGD}_convert.log}
TRAIN_OUT=${TRAIN_OUT:-outputs/wall3/main_train_v2_$TAGD}
WORKERS=${WORKERS:-14}
# data-loader workers: 4 (2026-10-22 사용자 결정 "시간 더 걸려도 에러 안 나게"): 8개로 시작한 재학습이 메모리 압박 57%를 만들어
# systemd-oomd가 VS Code를 강제 종료함 → 절반으로 줄여 처음부터 다시. 작업자 수는 학습 결과에 영향 없음(속도 · 메모리만)
TRAIN_WORKERS=${TRAIN_WORKERS:-4}
WAIT_UNIT=${WAIT_UNIT:-wall3-evalgates.service}
LOG=${LOG:-outputs/wall3/main_v2_${TAGD}_pipeline.log}
say () { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
die () { say "STOP: $*"; exit 1; }

modes_ok () {  # modes_ok <json>: no branch anywhere (flagged moments and whole trajectory)
  python - "$1" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
bad = r["flagged"] + r["trajectory_splits"]
print(f"episodes {r['episodes']}, split moments {len(r['flagged'])}, split trajectory segments {len(r['trajectory_splits'])}")
sys.exit(1 if bad else 0)
EOF
}

# 1) collect
if [ ! -f "$RAW/collect_summary.json" ]; then
  [ -e "$RAW" ] && die "$RAW 있음(끝나지 않은 수집): 덮어쓰지 않음"
  say "collect main_v2 -> $RAW (workers $WORKERS)"
  python -m wall3.collect --seed0 710000 --n 1100 --target-pass 1002 --workers "$WORKERS" --env-version main_v1 \
    --expert-version main_v2 --record-every 100 --out "$RAW" > "$RAW.log" 2>&1 || die "collect failed ($RAW.log)"
fi
say "$(tail -1 "$RAW.log")"

# 2) branch check on the raw collection
python -m experiments.diag_demo_modes --raw "$RAW" --out "$RAW/demo_modes.json" > "$RAW/demo_modes.log" 2>&1 || die "branch scan failed"
modes_ok "$RAW/demo_modes.json" >> "$LOG" 2>&1 || die "raw collection still has wrist/joint branches ($RAW/demo_modes.log)"
say "raw branch check ok: one branch per brick, every step, joints 0-5"

# 3) wait for the gate evaluation
while systemctl --user is-active --quiet "$WAIT_UNIT"; do sleep 60; done
say "$WAIT_UNIT not running: convert"

# 4) convert (10 parts, merge, full video check)
if [ ! -d "$DS" ]; then
  bash wall3/convert_parts.sh "$RAW" 1000 10 "$PARTS" "$DS" --exclude-seeds 710462 710548 > "$CONVLOG" 2>&1 || die "convert failed ($CONVLOG)"
fi
grep -q "^done:" "$CONVLOG" || die "no 'done:' in $CONVLOG"
say "converted: $DS"

# 5) branch check on the dataset
python -m experiments.diag_demo_modes --root "$DS" --out "$RAW/demo_modes_dataset.json" > "$RAW/demo_modes_dataset.log" 2>&1 || die "dataset branch scan failed"
modes_ok "$RAW/demo_modes_dataset.json" >> "$LOG" 2>&1 || die "dataset still has branches ($RAW/demo_modes_dataset.log)"
say "dataset branch check ok"

# 6) retrain (same settings as the 2026-10-19 main training)
say "train -> $TRAIN_OUT"
DS="$DS" CONVERT_UNIT=none.service CONVERT_LOG="$CONVLOG" OUT="$TRAIN_OUT" \
  REPORT_DIR="$REPO/01_사기설/근거_우분투_wall3_재학습_v2" TAG="재학습_v2" RUNS="diffusion 1000,act 1000" CONCURRENT=1 \
  TITLE="3층 벽체 재학습 v2 (손목 한 갈래 데이터)" \
  NOTE="> 데이터 \`$DS\`: 대본 main_v2(벽돌마다 집기 · 놓기 손목 한 갈래, 역기구학 180° 뒤집힘 수정)로 다시 수집한 1000벽. 학습 설정은 본 학습(10-19)과 같음. 경위: [\`중간발표2/시연데이터_손목갈림_재수집_20261022.md\`](../중간발표2/시연데이터_손목갈림_재수집_20261022.md). 아래 '지시서와 다른 점' 표는 본 학습 때 것 그대로다." \
  FORCE_BACKEND=torchcodec FORCE_WORKERS="$TRAIN_WORKERS" SAVE_FREQ=10000 \
  bash wall3/train_main.sh > "outputs/wall3/train_main_v2_$TAGD.log" 2>&1 || die "train_main.sh exited non-zero"
say "all done"

#!/usr/bin/env bash
# 본 학습 대기·실행 (본학습 지시 2 ~ 7절). 띄워 두면 사람을 기다리지 않고:
#   1) 나7 변환(systemd 단위 $CONVERT_UNIT)이 끝날 때까지 기다림 (변환 중에는 학습·로딩 측정 안 함: 메모리 압박)
#   2) 자동 시작 조건: 합치기 완료, 다시 읽기 검사(에피소드 16,000 · 프레임 = 원자료 · 카메라 3 · 목표 정보 5 · 뺀 벽 없음), 데이터 해시
#   3) 로딩 속도 측정(검사 뒤, 학습 직전 10분 안, 본 데이터셋) → 디코더·작업자 수를 골라 6번 모두 같게
#   4) ACT s1000 → Diffusion s1000 → ACT s2000 → Diffusion s2000 → ACT s3000 → Diffusion s3000, 동시에 2개
#      감시: 로그가 10분 조용하면 끝내고 마지막 체크포인트에서 이어 함(최대 3번), 3번 다 실패하면 건너뜀
#   5) 학습 하나 끝날 때마다: 입력 특징 5개·손실 확인 → 정상 동작 확인 2벽(700300 · 700301) → 보고서 다시 씀 → 커밋·푸시
#   멈춤(7절): 시작 조건 실패, 손실 NaN·처음보다 커짐, GPU 메모리 부족이 이어 하기로도 안 풀림, 입력 특징이 5개 아님 → $OUT/STOP
# 실행(세션과 분리된 사용자 서비스):
#   systemd-run --user --unit=wall3-train --property=WorkingDirectory=<repo> \
#     bash -c 'exec bash wall3/train_main.sh > outputs/wall3/train_main.log 2>&1'
set -uo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl
DS=${DS:-data/wall3_main_20261018}
CONVERT_UNIT=${CONVERT_UNIT:-wall3-convert-d.service}
CONVERT_LOG=${CONVERT_LOG:-outputs/wall3/main_20261018_convert_d.log}
OUT=${OUT:-outputs/wall3/main_train_$(date +%Y%m%d)}
REPO=$(cd .. && pwd)
REPORT_DIR=${REPORT_DIR:-"$REPO/01_사기설/근거_우분투_wall3_본학습"}
TAG=${TAG:-본학습}             # commit message prefix: "wall3 $TAG: ..." (재학습 v2 = "재학습_v2")
TITLE=${TITLE:-}; NOTE=${NOTE:-}   # report title / note (main_train_report --title / --note; empty = 본 학습 defaults)
STEPS=${STEPS:-100000}; SAVE_FREQ=${SAVE_FREQ:-10000}; LOG_FREQ=${LOG_FREQ:-200}; STALL_S=600; CONCURRENT=${CONCURRENT:-2}
RESUME=${RESUME:-0}           # 1 = continue the runs in an existing $OUT from their checkpoints (after the 2026-10-19 reboot)
UPLOAD_ARGS=${UPLOAD_ARGS:-}  # wall3.hf_upload arguments: models are uploaded only after every run has finished
# runs to train, in order (사용자 결정 2026-10-19: 우선 seed 1000만 → RUNS="act 1000,diffusion 1000")
RUNS=${RUNS:-"act 1000,diffusion 1000,act 2000,diffusion 2000,act 3000,diffusion 3000"}
FORCE_BACKEND=${FORCE_BACKEND:-}; FORCE_WORKERS=${FORCE_WORKERS:-}   # set both to skip the loader benchmark
EXPECT_EP=${EXPECT_EP:-16000}; EXPECT_WALLS=${EXPECT_WALLS:-1000}
DRY=${DRY:-0}                 # 1 = pipeline test only: no git, no rollout, quick bench
BENCH_ARGS=${BENCH_ARGS:-}
JOURNAL=${JOURNAL:-"$REPO/실험일지.md"}
EXCLUDED="710462 710548"
stamp () { date "+%m-%d %H:%M:%S"; }
say () { echo "[$(stamp)] $*"; }
[ "$RESUME" != 1 ] && [ -e "$OUT" ] && { say "$OUT 이미 있음: 덮어쓰지 않음"; exit 1; }
mkdir -p "$OUT"
if [ "$DRY" = 1 ]; then JOURNAL="$OUT/journal_test.md"; echo "# 실험일지" > "$JOURNAL"; fi

commit () {  # commit <message> [extra paths...]
  local msg=$1; shift
  [ "$DRY" = 1 ] && { say "DRY commit: $msg"; return 0; }
  (
    flock 9
    cd "$REPO" || exit 0
    git pull -q --rebase --autostash || say "git pull failed"
    git add "$REPORT_DIR" "$@"
    git commit -q -m "$msg

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || say "nothing to commit"
    git push -q || say "git push failed"
  ) 9> "$OUT/.gitlock"
}

stop_now () {  # stop_now <reason>
  [ -f "$OUT/STOP" ] || echo "$1" > "$OUT/STOP"
  say "STOP: $1"
}

report () {  # report [--final]
  local extra=()
  [ -n "$TITLE" ] && extra+=(--title "$TITLE")
  [ -n "$NOTE" ] && extra+=(--note "$NOTE")
  python -m experiments.main_train_report --train "$OUT" --dataset "$DS" --out-dir "$REPORT_DIR" --journal "$JOURNAL" --runs "$RUNS" "${extra[@]}" "$@" >> "$OUT/report.log" 2>&1 \
    || say "report failed (see $OUT/report.log)"
}

if [ "$RESUME" = 1 ]; then
  say "resume mode: dataset start conditions passed before ($OUT/start_check.json), continuing runs from checkpoints"
else
# ---------------------------------------------------------------- 1. wait for the conversion
say "waiting for $CONVERT_UNIT (log $CONVERT_LOG)"
while systemctl --user is-active --quiet "$CONVERT_UNIT"; do sleep 60; done
say "conversion unit ended"

# ---------------------------------------------------------------- 2. start conditions
python - "$DS" "$CONVERT_LOG" "$EXCLUDED" "$EXPECT_EP" "$EXPECT_WALLS" "$OUT/check_videos.json" > "$OUT/start_check.json" 2> "$OUT/start_check.err" <<'EOF'
import json, sys
from pathlib import Path
ds, conv, excluded = Path(sys.argv[1]), Path(sys.argv[2]), [int(x) for x in sys.argv[3].split()]
n_ep, n_walls = int(sys.argv[4]), int(sys.argv[5])
chk = {}
chk["merge_done"] = conv.exists() and any(l.startswith("done:") for l in conv.read_text().splitlines())
v = json.loads((ds / "meta" / "wall3_verify.json").read_text()) if (ds / "meta" / "wall3_verify.json").exists() else {}
info = json.loads((ds / "meta" / "info.json").read_text()) if (ds / "meta" / "info.json").exists() else {}
src = json.loads((ds / "meta" / "wall3_source.json").read_text()) if (ds / "meta" / "wall3_source.json").exists() else {}
chk["verify_ok"] = bool(v.get("ok")) and not v.get("problems")
chk["episodes_16000"] = v.get("num_episodes") == n_ep and v.get("expected_episodes") == n_ep
chk["frames_equal_raw"] = v.get("num_frames") == v.get("expected_frames") and bool(v.get("raw_check", {}).get("ok"))
chk["cameras_3"] = sorted(v.get("camera_keys", [])) == ["observation.images.front", "observation.images.top", "observation.images.wrist"]
chk["target_info_5"] = info.get("features", {}).get("observation.environment_state", {}).get("shape") in ([5], (5,))
seeds = src.get("seeds_used", [])
chk["walls_1000"] = len(seeds) == n_walls
chk["excluded_absent"] = not (set(excluded) & set(seeds))
chk["hash_recorded"] = bool(v.get("dataset_sha256")) and bool(v.get("content_sha256")) and (ds / "meta" / "wall3_hashes.json").exists()
import subprocess
out = Path(sys.argv[6])
r = subprocess.run([sys.executable, "-m", "experiments.check_videos", "--root", str(ds), "--out", str(out)], capture_output=True, text=True)
chk["videos_all_decode"] = r.returncode == 0          # every mp4 decoded end to end, frames = episode lengths (2026-10-19)
chk["all_ok"] = all(chk.values())
chk["values"] = {k: v.get(k) for k in ("num_episodes", "expected_episodes", "num_frames", "expected_frames", "dataset_sha256", "content_sha256", "bytes")}
print(json.dumps(chk, indent=1))
EOF
if ! python -c "import json,sys; sys.exit(0 if json.load(open('$OUT/start_check.json'))['all_ok'] else 1)" 2>/dev/null; then
  stop_now "자동 시작 조건 실패: $(python -c "import json; d=json.load(open('$OUT/start_check.json')); print([k for k,v in d.items() if v is False])" 2>/dev/null || cat "$OUT/start_check.err")"
  mkdir -p "$REPORT_DIR"
  { echo "# 3층 벽체 본 학습 보고 (Ubuntu)"; echo; echo "> **멈춤:** $(cat "$OUT/STOP")"; echo; echo '```'; cat "$OUT/start_check.json"; echo '```'; } > "$REPORT_DIR/보고서.md"
  commit "wall3 $TAG: 자동 시작 조건 실패로 멈춤"
  exit 1
fi
say "start conditions ok"
fi

# ---------------------------------------------------------------- 3. speed setting (data loading only, on the main dataset)
if [ -n "$FORCE_BACKEND" ] && [ -n "$FORCE_WORKERS" ]; then
  BACKEND=$FORCE_BACKEND; WORKERS=$FORCE_WORKERS
  say "setting given (no benchmark)"
else
  say "bench_loader"
  python -m experiments.bench_loader --root "$DS" --out "$OUT/bench_loader.json" $BENCH_ARGS > "$OUT/bench_loader.log" 2>&1
  BACKEND=$(python -c "import json; print(json.load(open('$OUT/bench_loader.json'))['choice']['backend'])" 2>/dev/null || echo pyav)
  WORKERS=$(python -c "import json; print(json.load(open('$OUT/bench_loader.json'))['choice']['workers'])" 2>/dev/null || echo 4)
fi
echo "{\"backend\": \"$BACKEND\", \"workers\": $WORKERS, \"concurrent\": $CONCURRENT, \"save_freq\": $SAVE_FREQ, \"runs\": \"$RUNS\", \"chosen\": \"$(stamp)\"}" > "$OUT/setting.json"
say "setting: backend $BACKEND, workers $WORKERS, concurrent $CONCURRENT"
# the dataset sits on a spinning disk (sda, HDD): read it once so its pages are in the page cache before random-access
# training (13 GB, RAM 60 GB). 2026-10-19: training ran at ~1/10 of the benchmark when the pages were not cached.
say "prewarm page cache: $DS"
find "$DS" -type f \( -name "*.mp4" -o -name "*.parquet" \) -print0 | xargs -0 cat > /dev/null
say "prewarm done"

# ---------------------------------------------------------------- 4. one training under a watchdog
status () { echo "$2" > "$OUT/$1.status.json"; }

ckpt_check () {  # ckpt_check <run dir>: last checkpoint complete and loadable, else point "last" at the newest good one
  python - "$1" >> "$OUT/ckpt_check.log" 2>&1 <<'EOF'
import json, sys, time
from pathlib import Path
import torch
from safetensors.torch import load_file
from lerobot.policies.factory import get_policy_class
ck = Path(sys.argv[1]) / "checkpoints"
need = ["pretrained_model/config.json", "pretrained_model/model.safetensors", "pretrained_model/train_config.json",
        "training_state/optimizer_state.safetensors", "training_state/rng_state.safetensors", "training_state/training_step.json"]
def ok(d):
    try:
        if any(not (d / f).exists() for f in need):
            return False, "missing files"
        for f in d.rglob("*.json"):
            json.loads(f.read_text())
        for f in d.rglob("*.safetensors"):
            t = load_file(str(f))
            if any(v.is_floating_point() and not torch.isfinite(v).all() for v in t.values()):
                return False, f"non-finite {f.name}"
        cfg = json.loads((d / "pretrained_model" / "config.json").read_text())
        get_policy_class(cfg["type"]).from_pretrained(str(d / "pretrained_model"))
        return True, f"step {json.loads((d / 'training_state' / 'training_step.json').read_text())['step']}"
    except Exception as e:
        return False, repr(e)[:200]
dirs = sorted([d for d in ck.iterdir() if d.is_dir() and d.name.isdigit()], reverse=True) if ck.exists() else []
last = (ck / "last").resolve() if (ck / "last").exists() else None
print(time.strftime("[%m-%d %H:%M:%S]"), ck.parent.name, "last ->", last.name if last else None)
for d in ([last] if last else []) + [x for x in dirs if x != last]:
    good, note = ok(d)
    print("  ", d.name, "OK" if good else "BROKEN", note)
    if good:
        if last is None or d != last:
            (ck / "last").unlink(missing_ok=True); (ck / "last").symlink_to(d.name)
            print("   last ->", d.name)
        break
EOF
}

train_one () {  # train_one <policy> <seed>
  local pol=$1 seed=$2 name="${1}_s${2}"
  local out="$OUT/$name" log="$OUT/$name.log" start attempt result="failed" oom=0
  start=$(python -c "import json; print(json.load(open('$OUT/$name.status.json'))['start'])" 2>/dev/null || date +%s)
  for attempt in 1 2 3; do
    [ -f "$OUT/STOP" ] && { result="stopped"; break; }
    local -a args
    [ -d "$out/checkpoints" ] && ckpt_check "$out"
    if [ -f "$out/checkpoints/last/pretrained_model/train_config.json" ]; then
      say "TRAIN $name resume from $(readlink "$out/checkpoints/last") (attempt $attempt)"
      args=(--config_path="$out/checkpoints/last/pretrained_model/train_config.json" --resume=true
            --num_workers="$WORKERS" --dataset.video_backend="$BACKEND")
    else
      [ -e "$out" ] && mv "$out" "$out.attempt$((attempt - 1))_no_checkpoint"
      say "TRAIN $name start (attempt $attempt)"
      args=(--dataset.repo_id="local/$(basename "$DS")" --dataset.root="$DS" --dataset.video_backend="$BACKEND"
            --policy.type="$pol" --policy.device=cuda --policy.push_to_hub=false
            --steps="$STEPS" --batch_size=32 --seed="$seed" --num_workers="$WORKERS" --save_freq="$SAVE_FREQ" --log_freq="$LOG_FREQ"
            --output_dir="$out" --job_name="wall3_main_$name" --wandb.enable=false)
    fi
    status "$name" "{\"start\": $start, \"attempts\": $attempt, \"resumed\": $( [ "$RESUME" = 1 ] && echo true || echo false ), \"result\": \"running\"}"
    # each training in its own scope: if systemd-oomd kills it under memory pressure only this run dies and the watchdog
    # resumes it from its last checkpoint (before, oomd killed the whole service including the watchdog; 2026-10-19)
    # nice 10: the desktop stays responsive (2026-10-19 the machine hung under two runs); results are not affected
    systemd-run --user --scope --quiet --unit="wall3-tr-$name-$(date +%H%M%S)" -p ManagedOOMPreference=avoid \
      nice -n 10 lerobot-train "${args[@]}" >> "$log" 2>&1 &
    local pid=$!
    while kill -0 "$pid" 2>/dev/null; do
      sleep 30
      local age=$(( $(date +%s) - $(stat -c %Y "$log" 2>/dev/null || echo 0) ))
      if [ "$age" -gt "$STALL_S" ]; then say "TRAIN $name STALLED ${age}s -> kill"; kill -9 "$pid" 2>/dev/null; sleep 5; break; fi
      if grep -q "loss:nan" "$log"; then stop_now "$name 손실 NaN"; fi
      if [ -f "$OUT/STOP" ]; then kill -9 "$pid" 2>/dev/null; sleep 5; break; fi
    done
    wait "$pid" 2>/dev/null
    if grep -q "End of training" "$log"; then result="done"; break; fi
    [ -f "$OUT/STOP" ] && { result="stopped"; break; }
    grep -qE "CUDA out of memory|OutOfMemoryError" "$log" && oom=1
    sleep 10
  done
  if [ "$result" = "failed" ] && [ "$oom" = 1 ]; then result="oom"; stop_now "$name GPU 메모리 부족이 이어 하기 3번으로도 안 풀림"; fi
  [ "$result" = "failed" ] && say "TRAIN $name FAILED after 3 attempts -> skipped"
  status "$name" "{\"start\": $start, \"end\": $(date +%s), \"attempts\": $attempt, \"resumed\": $( [ "$RESUME" = 1 ] && echo true || echo false ), \"result\": \"$result\"}"
  say "TRAIN $name $result"
}

# ---------------------------------------------------------------- 5. after each training
postproc () {  # postproc <policy> <seed>
  local pol=$1 seed=$2 name="${1}_s${2}"
  local ck="$OUT/$name/checkpoints/last/pretrained_model" result
  result=$(python -c "import json; print(json.load(open('$OUT/$name.status.json'))['result'])" 2>/dev/null || echo unknown)
  if [ "$result" = "done" ]; then
    local nin
    nin=$(python -c "import json; print(len(json.load(open('$ck/config.json'))['input_features']))" 2>/dev/null || echo 0)
    [ "$nin" = "5" ] || stop_now "$name 정책 입력 특징이 5개가 아님($nin)"
    python - "$OUT/$name.log" <<'EOF' || stop_now "$name 학습 손실 발산(마지막 손실 > 처음 손실 또는 NaN)"
import sys
import numpy as np
from pathlib import Path
from experiments.mini_train_report import parse_log
rows, _ = parse_log(Path(sys.argv[1]))
a = np.array(rows)
sys.exit(1 if len(a) == 0 or not np.isfinite(a[:, 1]).all() or a[-5:, 1].mean() > a[0, 1] else 0)
EOF
    say "CHECK $name rollout 2 walls"
    [ "$DRY" = 1 ] || python -m wall3.rollout --ckpt "$ck" --env-version main_v1 --seeds 700300 700301 --out "$OUT/rollout_$name" --record-first \
      > "$OUT/rollout_$name.log" 2>&1 || say "rollout $name failed (see log)"
  fi
  report
  commit "wall3 $TAG: $name $result, 정상 동작 확인 2벽, 보고서 갱신"
  say "POST $name done ($result)"
}

# ---------------------------------------------------------------- queue: two at a time
IFS=',' read -r -a QUEUE <<< "$RUNS"
declare -A RUN=()
declare -a POST=()
next=0
report
if [ "$RESUME" = 1 ]; then
  commit "wall3 $TAG: 체크포인트에서 이어 하기 시작(순서 $RUNS, $BACKEND, 작업자 $WORKERS, 동시 $CONCURRENT)"
else
  commit "wall3 $TAG: 시작 조건 통과, 속도 설정($BACKEND, 작업자 $WORKERS, 동시 $CONCURRENT), 학습 시작"
fi
while true; do
  for key in "${!RUN[@]}"; do
    if ! kill -0 "${RUN[$key]}" 2>/dev/null; then
      unset "RUN[$key]"
      postproc $key &
      POST+=($!)
    fi
  done
  while [ "${#RUN[@]}" -lt "$CONCURRENT" ] && [ "$next" -lt "${#QUEUE[@]}" ] && [ ! -f "$OUT/STOP" ]; do
    train_one ${QUEUE[$next]} &
    RUN["${QUEUE[$next]}"]=$!
    next=$((next + 1))
    sleep 20
  done
  if [ "${#RUN[@]}" -eq 0 ] && { [ "$next" -ge "${#QUEUE[@]}" ] || [ -f "$OUT/STOP" ]; }; then break; fi
  sleep 30
done
for p in "${POST[@]}"; do wait "$p"; done
report --final
if [ -n "$UPLOAD_ARGS" ] && [ "$DRY" != 1 ]; then              # models only after every run (no big jobs during training)
  say "hf upload (models)"
  python -m wall3.hf_upload $UPLOAD_ARGS >> "$OUT/hf_upload.log" 2>&1 || say "hf upload failed (see $OUT/hf_upload.log)"
  say "hf upload finished"
fi
commit "wall3 $TAG: ${#QUEUE[@]}번 마무리$( [ -f "$OUT/STOP" ] && echo " (멈춤: $(cat "$OUT/STOP"))" ), 보고서·실험일지" "실험일지.md"
say "all done$( [ -f "$OUT/STOP" ] && echo " (STOP: $(cat "$OUT/STOP"))" )"

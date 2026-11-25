"""현실 제약 ablation 100벽 평가 (2026-11-16, 01_사기설/우분투지시_현실제약_ablation_20261116.md). 조건 하나를 묶음으로 나눠 돌리고 요약한다.

조건 이름: {act|script}_real_{none|A|B|AB} → wall3.rollout_real --controller ... [--A] [--B]
자원 관리(사용자가 컴퓨터를 볼 수 없음, 지시서 6절): 새 프로세스는 여유 메모리 ≥ 12 GB · GPU 여유 메모리 ≥ 4 GB · 1분 부하 < 24일 때만 띄운다.
프로세스마다 nice 10, OMP · MKL · OpenBLAS 스레드 1. 기다린 기록은 <out>/resource_log.txt.
출력 <out>/<조건>_part<i>/rollout.json 은 compare_ladder · primary_metrics가 그대로 읽는다. 시작값 700300 ~ 700399, 최종 평가 아님.

실행: venv/bin/python -m wall3.eval_real --out outputs/wall3/<이름> --condition script_real_none [--chunks 6 --parallel 6]
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[1]
CELLS = {"none": [], "A": ["--A"], "B": ["--B"], "AB": ["--A", "--B"]}
CONDITIONS = {f"{c}_real_{cell}": ["--controller", c] + flags for c in ("act", "script") for cell, flags in CELLS.items()}
MIN_MEM_GB, MIN_GPU_FREE_MB, MAX_LOAD = 12.0, 4096, 24.0
THREAD_ENV = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MUJOCO_GL": "egl", "PYTHONPATH": "."}


def resources():
    mem = None
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            mem = int(line.split()[1]) / 1024 / 1024
    load = os.getloadavg()[0]
    try:
        used, total = (float(x) for x in subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], text=True).split("\n")[0].split(","))
        gpu_free = total - used
    except Exception:
        gpu_free = None
    ok = mem is not None and mem >= MIN_MEM_GB and load < MAX_LOAD and (gpu_free is None or gpu_free >= MIN_GPU_FREE_MB)
    return ok, {"mem_avail_gb": round(mem or -1, 1), "load1": round(load, 1), "gpu_free_mb": gpu_free}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    ap.add_argument("--seed0", type=int, default=700300)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--chunks", type=int, default=6)
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--env-version", default="main_v1_eval")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} 이미 있음")
    out.mkdir(parents=True)
    seeds = list(range(a.seed0, a.seed0 + a.n))
    if seeds[0] < 700300 or seeds[-1] > 700399:
        raise SystemExit("시작값은 700300 ~ 700399만")
    jobs = []
    for ci, part in enumerate(np.array_split(seeds, a.chunks)):
        d = out / f"{a.condition}_part{ci}"
        cmd = ["nice", "-n", "10", sys.executable, "-m", "wall3.rollout_real", "--env-version", a.env_version,
               "--seeds", *map(str, part), "--out", str(d)] + CONDITIONS[a.condition] + (["--record-first"] if ci == 0 else [])
        jobs.append((ci, d, cmd))
    env = dict(os.environ, **THREAD_ENV)
    (out / "plan.json").write_text(json.dumps({"seeds": [seeds[0], seeds[-1]], "env_version": a.env_version, "condition": a.condition,
                                               "flags": CONDITIONS[a.condition], "jobs": [[i, str(d)] for i, d, _ in jobs],
                                               "resource_rules": {"min_mem_gb": MIN_MEM_GB, "min_gpu_free_mb": MIN_GPU_FREE_MB, "max_load1": MAX_LOAD,
                                                                  "nice": 10, "threads": 1},
                                               "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                                               "git_commit": subprocess.check_output(["git", "-C", str(SIM), "rev-parse", "HEAD"], text=True).strip()},
                                              indent=1))
    rlog = open(out / "resource_log.txt", "a")
    running, pending, done = [], list(jobs), []
    t0 = time.time()
    while pending or running:
        while pending and len(running) < a.parallel:
            ok, info = resources()
            if not ok:
                rlog.write(f"{time.strftime('%H:%M:%S')} 기다림 {info}\n"); rlog.flush()
                break
            j = pending.pop(0)
            log = open(f"{j[1]}.log", "w")
            running.append((j, subprocess.Popen(j[2], cwd=SIM, env=env, stdout=log, stderr=subprocess.STDOUT), log))
            rlog.write(f"{time.strftime('%H:%M:%S')} 시작 part{j[0]} {info}\n"); rlog.flush()
            time.sleep(20)                                   # stagger the start-up (model load, renderer init)
        time.sleep(10)
        for item in list(running):
            j, p, log = item
            if p.poll() is not None:
                running.remove(item); log.close(); done.append((j, p.returncode))
                print(time.strftime("[%H:%M:%S]"), f"part{j[0]} exit {p.returncode} ({len(done)}/{len(jobs)}, {time.time() - t0:.0f} s)", flush=True)
    from wall3.eval_gates import summarize
    walls = []
    for d in sorted(out.glob(f"{a.condition}_part*")):
        if d.is_dir() and (d / "rollout.json").exists():
            walls += json.loads((d / "rollout.json").read_text())["walls"]
    walls.sort(key=lambda w: w["seed"])
    failed = [[j[0], rc] for j, rc in done if rc != 0]
    summ = {a.condition: summarize(walls) if walls else None,
            "_meta": {"failed_jobs": failed, "walls": len(walls), "elapsed_s": time.time() - t0, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}}
    if failed or len(walls) != len(seeds):
        raise SystemExit(f"실패한 묶음 {failed}, 벽 {len(walls)}/{len(seeds)}: summary.json을 쓰지 않음(감독이 다시 시작)")
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    s = summ[a.condition]
    print(json.dumps({k: s[k] for k in ("walls", "slots_filled_mean", "walls_complete", "spec_pass")}, default=float))


if __name__ == "__main__":
    main()

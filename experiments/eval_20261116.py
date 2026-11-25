"""100벽 평가 (2026-11-16): wall3.rollout_controller을 시작값 묶음으로 나눠 동시에 돌리고 하나로 모아 요약한다 (eval_gates와 같은 요약).

조건 (결과 전에 정함, 사용자 지시 11-16)
  script_all_held+k2+k7      정책 없음: 대본 제어기(wall3/script_policy.py) + Full 검측(G0 + G1 + G2 + G2w + G3 + H)
  act_all_held+k2+k7+g2t     Full + G2t: ACT 정책 + Full 검측, G1 '쥠' 직후부터 G2가 운반(wall3/gates_carry.py)
출력 폴더 <out>/<조건>_part<i>/rollout.json 은 wall3.compare_ladder가 그대로 읽는다. 시작값 700300 ~ 700399(개발 배치), 최종 평가 아님.

실행: venv/bin/python -m experiments.eval_20261116 --out outputs/wall3/<이름> --condition script_all_held+k2+k7 [--chunks 8 --parallel 8]
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[1]
CKPT = "outputs/wall3/main_train_v2_20261022/act_s1000/checkpoints/last/pretrained_model"
GATES = ["--g1", "--g2", "--g3", "--gate-target", "held_grasp", "--gate-fix", "k2", "k7"]
CONDITIONS = {
    "script_all_held+k2+k7": ["--controller", "script"] + GATES,
    "act_all_held+k2+k7+g2t": ["--controller", "act", "--ckpt", CKPT] + GATES + ["--g2-carry"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    ap.add_argument("--seed0", type=int, default=700300)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--parallel", type=int, default=8)
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
        cmd = [sys.executable, "-m", "wall3.rollout_controller", "--env-version", a.env_version, "--seeds", *map(str, part),
               "--out", str(d)] + CONDITIONS[a.condition] + (["--record-first"] if ci == 0 else [])
        jobs.append((ci, d, cmd))
    (out / "plan.json").write_text(json.dumps({"seeds": [seeds[0], seeds[-1]], "env_version": a.env_version, "condition": a.condition,
                                               "flags": CONDITIONS[a.condition], "jobs": [[i, str(d)] for i, d, _ in jobs],
                                               "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                                               "git_commit": subprocess.check_output(["git", "-C", str(SIM), "rev-parse", "HEAD"], text=True).strip()},
                                              indent=1))
    running, pending, done = [], list(jobs), []
    t0 = time.time()
    while pending or running:
        while pending and len(running) < a.parallel:
            j = pending.pop(0)
            log = open(f"{j[1]}.log", "w")
            running.append((j, subprocess.Popen(j[2], cwd=SIM, stdout=log, stderr=subprocess.STDOUT), log))
        time.sleep(5)
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
    summ = {a.condition: summarize(walls) if walls else None,
            "_meta": {"failed_jobs": [[j[0], rc] for j, rc in done if rc != 0], "elapsed_s": time.time() - t0,
                      "finished": time.strftime("%Y-%m-%d %H:%M:%S")}}
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    s = summ[a.condition]
    print(json.dumps({k: s[k] for k in ("walls", "slots_filled_mean", "walls_complete", "spec_pass")} if s else None, default=float))


if __name__ == "__main__":
    main()

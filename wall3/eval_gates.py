"""검측 예비 평가 (2026-10-22, 결과 아님 · 문턱 임시값): 정책 × 조건(검측 없음 / G1·G2·G3 모두) × 100벽.

평가 환경 main_v1_eval(모르타르가 벽 하나 쌓는 동안 굳지 않음), 시작값은 파일럿 블록 미사용 700300 ~ 700399(검증 720000번대 ·
시험 730000번대는 쓰지 않음). 벽마다 rollout.py와 같은 규칙(도면 순서, 철수 또는 정책 시간 17 s, 검측 동작 시간은 제한에 안 넣음).
(정책, 조건)마다 시작값을 묶음으로 나눠 동시에 돌리고, 끝나면 하나로 모아 요약한다.

실행: venv/bin/python -m wall3.eval_gates --out outputs/wall3/eval_gates_<날짜> [--parallel 8] [--chunks 4]
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[1]
CKPT_DIR = "outputs/wall3/main_train_20261019"       # 본 학습 (v1 데이터); 재학습 v2 = outputs/wall3/main_train_v2_20261022


def ckpts(train_dir):
    return {p: f"{train_dir}/{p}_s1000/checkpoints/last/pretrained_model" for p in ("act", "diffusion")}
COND = {"none": [], "all": ["--g1", "--g2", "--g3"]}
# 2026-11-03 (우분투지시_검측정보진단_20261103.md 2.4): what the gates use (truth = 진단 전용) and G1 · G2 · G3 on / off
_TRUTH, _HT, _HG = ["--gate-pose-source", "truth"], ["--gate-target", "held_truth"], ["--gate-target", "held_grasp"]
COND.update({"all_truth": COND["all"] + _TRUTH, "all_truth_held": COND["all"] + _TRUTH + _HT, "all_held": COND["all"] + _HG,
             "g1": ["--g1"], "g1g2": ["--g1", "--g2"], "g1g3": ["--g1", "--g3"], "g2g3": ["--g2", "--g3"],
             "g1g2_held": ["--g1", "--g2"] + _HG, "g1g3_held": ["--g1", "--g3"] + _HG, "g2g3_held": ["--g2", "--g3"] + _HG})
# 2026-11-10 (실험파이프라인_실패원인제거_20261110.md): "<condition>+<fix>+<fix>" adds wall3/gates.py FIXES, e.g. all_held+k1+k6


def cond_flags(name):
    base, *fixes = name.split("+")
    if base not in COND:
        raise SystemExit(f"unknown condition {base!r}")
    if fixes:
        from wall3.gates import fix_fields
        fix_fields(fixes)                              # unknown fix names fail here, before any job starts
        return COND[base] + ["--gate-fix", *fixes]
    return COND[base]


PERCH_MM = 20.0          # 'filled' but the brick centre more than this above its design height (10-29: perched on a neighbour)


def perched_filled(walls):
    """(seed, slot) of 'filled' slots whose occupant (the brick matched to the slot by spec_metrics) sits more than PERCH_MM
    above the slot's design height."""
    from wall3.config import design_slots
    D = design_slots()
    out = []
    for w in walls:
        err = {b["slot"]: b["final_err_mm"] for b in w["bricks"]}
        for j, (f, i) in enumerate(zip(w["eval"]["slots_filled"], w["eval"]["match"])):
            if f and i >= 0 and (D[i]["z"] - D[j]["z"]) * 1000 + err[i][2] > PERCH_MM:
                out.append((w["seed"], j))
    return out


def summarize(walls):
    br = [b for w in walls for b in w["bricks"]]
    n = len(walls)
    filled = np.array([w["eval"]["slots_filled"] for w in walls], float)            # (walls, 16)
    lifted = np.array([[b["grasped_lifted"] for b in w["bricks"]] for w in walls], float)
    errs = [b["final_xy_err_mm"] for w in walls for b in w["bricks"] if w["eval"]["slots_filled"][b["slot"]]]
    g = [b.get("gates") for b in br if b.get("gates")]
    out = {"walls": n, "spec_pass": int(sum(w["eval"]["spec_pass"] for w in walls)),
           "slots_filled_mean": float(filled.sum(1).mean()), "iou_mean": float(np.mean([w["eval"]["iou_mean"] for w in walls])),
           "filled_rate_by_slot": filled.mean(0).round(3).tolist(), "lifted_rate_by_slot": lifted.mean(0).round(3).tolist(),
           "filled_err_mm": {"n": len(errs), "median": float(np.median(errs)) if errs else None,
                             "p90": float(np.percentile(errs, 90)) if errs else None},
           "policy_time_s_per_brick": float(np.mean([b.get("policy_time_s", b["duration_s"]) for b in br])),
           "gate_time_s_per_brick": float(np.mean([b.get("gate_time_s", 0.0) for b in br])),
           "wall_clock_s_per_wall": float(np.mean([w["wall_clock_s"] for w in walls]))}
    # 2026-11-03
    fb = filled.astype(bool)
    first = [int(np.argmin(f)) if not f.all() else len(f) for f in fb]               # first unfilled slot (0-based; 16 = complete)
    out.update(walls_complete=int(fb.all(1).sum()), slots_filled_median=float(np.median(filled.sum(1))),
               first_unfilled_hist={str(i): first.count(i) for i in range(filled.shape[1] + 1) if first.count(i)},
               filled_but_perched=len(perched_filled(walls)))
    if g:
        g1 = [e for x in g for e in x["g1"] if "grasped" in e]
        g2 = [e for x in g for e in x["g2"] if not e.get("release_here") and not e.get("skipped")]
        g3 = [e for x in g for e in x["g3"]]
        rep = [e["replace"] for e in g3 if isinstance(e.get("replace"), dict)]
        out["gates"] = {
            "g1_checks": len(g1), "g1_grasped": sum(e["grasped"] for e in g1),
            "g1_grasped_but_target_not_lifted": sum(e["grasped"] and (e.get("audit_target_rise_mm") or 0) < 10 for e in g1),
            "g2_releases": len(g2), "g2_in_hand_by_camera": sum(e.get("in_hand_src") == "camera" for e in g2),
            "g2_in_hand_err_mm_median": float(np.median([e["audit_in_hand_err_mm"] for e in g2 if e.get("audit_in_hand_err_mm") is not None]))
            if any(e.get("audit_in_hand_err_mm") is not None for e in g2) else None,
            "g3_inspections": len(g3), "g3_seen": sum(e.get("seen", False) for e in g3), "g3_bad": sum(e.get("bad", False) for e in g3),
            "g3_replaced": len(rep), "g3_meas_err_mm_median": float(np.median([e["audit_meas_err_mm"] for e in g3 if "audit_meas_err_mm" in e]))
            if any("audit_meas_err_mm" in e for e in g3) else None,
            "g3_replace_after_mm_median": float(np.median([r["audit_after_mm"] for r in rep if "audit_after_mm" in r])) if rep else None}
        # 2026-11-03: G2 / G3 interventions (the release went to the gate), releases skipped 15 cm away from the reference slot,
        # G1 'grasped' where the lifted brick (audit) was the turn's brick, in-hand pose measured (camera or truth)
        g1k = [(e, b["slot"]) for b in br if b.get("gates") for e in b["gates"]["g1"] if e.get("grasped")]
        out["gates"].update(
            g2_interventions=len([e for x in g for e in x["g2"] if not e.get("skipped")]),
            g2_skipped_far=sum(e.get("skipped") == "far" for x in g for e in x["g2"]),
            g2_in_hand_measured=sum(e.get("in_hand_src") in ("camera", "truth") for e in g2),
            g1_grasped_turn_brick=sum(e.get("audit_max_rise_brick") == k and (e.get("audit_max_rise_mm") or 0) > 10 for e, k in g1k))
        g0 = [e for x in g for e in x.get("g0", [])]                                     # 2026-11-10 K2
        if g0:
            out["gates"]["g0"] = {"grasps": len(g0), "g1_grasped": sum(e.get("g1_grasped") is True for e in g0),
                                  "tcp_off_mm_median": float(np.median([e["tcp_off_mm"] for e in g0]))}
        g1r = [e for x in g for e in x["g1"] if "retract" in e]                          # 2026-11-10 K1
        if g1r:
            out["gates"]["g1_fail_retracts"] = len(g1r)
        tg = [e for x in g for e in x.get("target", []) if e.get("used")]
        if tg:
            t1 = [e for e in tg if e.get("g1_grasped") is True]
            ok = [e for e in t1 if e.get("audit_brick") is not None]
            match = sum(not e.get("undetermined") and e["slot"] == e["audit_brick"] for e in ok)
            out["gates"]["target"] = {
                "method": sorted({e["method"] for e in tg}), "decisions": len(tg),
                "undetermined": sum(bool(e.get("undetermined")) for e in tg),
                "g1_true": len(t1), "g1_true_no_brick_lifted": len(t1) - len(ok), "g1_true_match": match,
                "match_rate": match / len(ok) if ok else None,                   # G1 'grasped' and a brick really in the hand
                "match_rate_all_g1_true": match / len(t1) if t1 else None,
                "slot_ne_turn": sum(e["slot"] != e["turn"] for e in tg)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed0", type=int, default=700300)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--policies", nargs="+", default=["act", "diffusion"])
    ap.add_argument("--conditions", nargs="+", default=["none", "all"])
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--env-version", default="main_v1_eval")
    ap.add_argument("--train-dir", default=CKPT_DIR, help="training output with {act,diffusion}_s1000")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} 이미 있음")
    out.mkdir(parents=True)
    seeds = list(range(a.seed0, a.seed0 + a.n))
    CKPT = ckpts(a.train_dir)
    jobs = []
    for pol in a.policies:
        for cond in a.conditions:
            for ci, part in enumerate(np.array_split(seeds, a.chunks)):
                d = out / f"{pol}_{cond}_part{ci}"
                cmd = [sys.executable, "-m", "wall3.rollout", "--ckpt", CKPT[pol], "--env-version", a.env_version,
                       "--seeds", *map(str, part), "--out", str(d)] + cond_flags(cond) + (["--record-first"] if ci == 0 else [])
                jobs.append((pol, cond, ci, d, cmd))
    (out / "plan.json").write_text(json.dumps({"seeds": [seeds[0], seeds[-1]], "env_version": a.env_version, "ckpt": CKPT,
                                               "conditions": {c: cond_flags(c) for c in a.conditions}, "jobs": [[p, c, i, str(d)] for p, c, i, d, _ in jobs],
                                               "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                                               "git_commit": subprocess.check_output(["git", "-C", str(SIM), "rev-parse", "HEAD"], text=True).strip()},
                                              indent=1))
    running, pending, done = [], list(jobs), []
    t0 = time.time()
    while pending or running:
        while pending and len(running) < a.parallel:
            j = pending.pop(0)
            log = open(f"{j[3]}.log", "w")
            running.append((j, subprocess.Popen(j[4], cwd=SIM, stdout=log, stderr=subprocess.STDOUT), log))
        time.sleep(5)
        for item in list(running):
            j, p, log = item
            if p.poll() is not None:
                running.remove(item); log.close(); done.append((j, p.returncode))
                print(time.strftime("[%H:%M:%S]"), f"{j[0]} {j[1]} part{j[2]} exit {p.returncode} ({len(done)}/{len(jobs)}, {time.time() - t0:.0f} s)", flush=True)
    summ = {}
    for pol in a.policies:
        for cond in a.conditions:
            walls = []
            for d in sorted(out.glob(f"{pol}_{cond}_part*")):
                if d.is_dir() and (d / "rollout.json").exists():
                    walls += json.loads((d / "rollout.json").read_text())["walls"]
            walls.sort(key=lambda w: w["seed"])
            summ[f"{pol}_{cond}"] = summarize(walls) if walls else None
    summ["_meta"] = {"failed_jobs": [[j[0], j[1], j[2], rc] for j, rc in done if rc != 0], "elapsed_s": time.time() - t0,
                     "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps({k: ({kk: v[kk] for kk in ("walls", "spec_pass", "slots_filled_mean", "iou_mean")} if isinstance(v, dict) and "walls" in v else v)
                      for k, v in summ.items()}, indent=1, default=float))


if __name__ == "__main__":
    main()

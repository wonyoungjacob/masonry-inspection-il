"""시연 데이터 진단 (2026-10-22): 벽돌(자리)마다 대본 행동이 여러 갈래로 갈리는 순간을 찾는다.

에피소드마다 순간 4개(집게 닫기 시작 · 운반 중간 · 집게 열기 시작 · 끝)의 행동(관절 6개 명령값)을 모아, 자리 × 관절마다
값을 정렬해 가장 큰 틈이 gap_rad 이상이고 작은 쪽이 min_frac 이상이면 '갈림'으로 표시한다. 시각(단계 길이)의 퍼짐도 남긴다.
정책이 여러 갈래 시연을 평균 내면 어느 갈래와도 안 맞는 동작이 나온다(ACT 2번 벽돌 집기 실패의 원인).

실행: venv/bin/python -m experiments.diag_demo_modes [--root <dataset> | --raw <collect 폴더> ...] --out <json>
"""
import argparse, glob, json
from pathlib import Path

import numpy as np
import pandas as pd

MOMENTS = ("close", "transport_mid", "open", "end")


def split_of(v, gap_rad, min_frac):
    """Cut the sorted values at every gap >= gap_rad (each cut = a separate branch). split: 2+ branches and the branches
    other than the largest hold >= min_frac of the episodes together (min_frac 0 = any single episode)."""
    v = np.sort(v)
    d = np.diff(v)
    cuts = np.flatnonzero(d >= gap_rad)
    sizes = np.diff(np.r_[0, cuts + 1, len(v)])
    centers = [float(c.mean()) for c in np.split(v, cuts + 1)]
    others = int(len(v) - sizes.max())
    return {"gap": float(d.max()) if len(d) else 0.0, "branches": sizes.tolist(), "centers": [round(c, 3) for c in centers],
            "low_n": int(sizes[0]), "high_n": int(len(v) - sizes[0]), "minority_n": others,
            "at": float((v[cuts[0]] + v[cuts[0] + 1]) / 2) if len(cuts) else None,
            "split": bool(len(sizes) >= 2 and others >= max(1, min_frac * len(v)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/wall3_main_20261019")
    ap.add_argument("--raw", nargs="+", default=None, help="collection folder(s) (wall3.collect --out) instead of a dataset")
    ap.add_argument("--gap-rad", type=float, default=0.5)
    ap.add_argument("--min-frac", type=float, default=0.0, help="0 = flag even one episode off the main branch")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    if a.raw:                                   # collection folder(s) before conversion: clean walls only, episode = steps of brick k
        eps, acts = [], {}
        for rd in a.raw:
            summ = json.loads((Path(rd) / "collect_summary.json").read_text())
            for w in summ["walls"]:
                if not w["clean_pass"]:
                    continue
                z = np.load(Path(rd) / "raw" / f"wall_{w['seed']}.npz")
                br, act = z["brick"], z["action"]
                for k in range(16):
                    i = len(eps)
                    acts[i] = act[br == k]
                    eps.append({"episode_index": i, "slot": k, "seed": w["seed"]})
        root = Path(a.raw[0])
    else:
        eps = json.loads((root / "meta" / "wall3_episodes.json").read_text())
        df = pd.concat([pd.read_parquet(f, columns=["episode_index", "action"]) for f in sorted(glob.glob(str(root / "data" / "*" / "*.parquet")))])
        acts = {i: np.stack(g.action.values) for i, g in df.groupby("episode_index")}
    vals = {k: {m: [] for m in MOMENTS} for k in range(16)}
    steps = {k: {m: [] for m in MOMENTS} for k in range(16)}
    for e in eps:
        A = acts[e["episode_index"]]
        g = A[:, 6]
        ic = int(np.argmax(g > 0.15))
        io = ic + int(np.argmax(g[ic:] < 0.15))
        idx = {"close": ic, "transport_mid": (ic + io) // 2, "open": io, "end": len(A) - 1}
        for m, t in idx.items():
            vals[e["slot"]][m].append(A[t, :6])
            steps[e["slot"]][m].append(t)
    res = {"root": str(root), "episodes": len(eps), "gap_rad": a.gap_rad, "min_frac": a.min_frac, "slots": {}}
    flagged = []
    for k in range(16):
        r = {}
        for m in MOMENTS:
            V = np.array(vals[k][m])
            T = np.array(steps[k][m])
            r[m] = {"step_mean": float(T.mean()), "step_std": float(T.std()), "step_min": int(T.min()), "step_max": int(T.max()),
                    "joints": []}
            for j in range(6):
                sp = split_of(V[:, j], a.gap_rad, a.min_frac)
                r[m]["joints"].append({"mean": float(V[:, j].mean()), "std": float(V[:, j].std()), **sp})
                if sp["split"]:
                    flagged.append({"slot": k, "moment": m, "joint": j, "branches": sp["branches"], "centers": sp["centers"],
                                    "gap_rad": round(sp["gap"], 2)})
        res["slots"][k] = r
    res["flagged"] = flagged
    # whole trajectory: every control step (episodes of a slot have the same timing within ~1 step), every arm joint
    seg = []
    for k in range(16):
        A = [acts[e["episode_index"]] for e in eps if e["slot"] == k]
        T = min(len(x) for x in A)
        X = np.stack([x[:T, :6] for x in A])                        # (episodes, T, 6)
        for j in range(6):
            on = [split_of(X[:, t, j], a.gap_rad, a.min_frac) for t in range(T)]
            t = 0
            while t < T:
                if on[t]["split"]:
                    t1 = t
                    while t1 + 1 < T and on[t1 + 1]["split"]:
                        t1 += 1
                    mn = [on[u]["minority_n"] for u in range(t, t1 + 1)]
                    seg.append({"slot": k, "joint": j, "step_from": t, "step_to": t1, "minority_n_max": int(max(mn)),
                                "max_branches": int(max(len(on[u]["branches"]) for u in range(t, t1 + 1))),
                                "branches_at_max": on[t + int(np.argmax(mn))]["branches"], "centers_at_max": on[t + int(np.argmax(mn))]["centers"],
                                "gap_rad_max": round(max(on[u]["gap"] for u in range(t, t1 + 1)), 2), "episodes": len(A)})
                    t = t1 + 1
                else:
                    t += 1
    res["trajectory_splits"] = seg
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"{len(eps)} episodes; split (gap >= {a.gap_rad} rad, minority >= {a.min_frac:.1%}):")
    for f in flagged:
        print(f"  slot {f['slot']:2d} {f['moment']:13s} joint {f['joint']}: branches {f['branches']} at {f['centers']} rad")
    print("whole trajectory (every step, joints 0-5):")
    for g in seg:
        print(f"  slot {g['slot']:2d} joint {g['joint']}: steps {g['step_from']}-{g['step_to']} ({g['step_from'] * 0.05:.2f}-{g['step_to'] * 0.05:.2f} s), "
              f"not-largest branches up to {g['minority_n_max']} / {g['episodes']} (branches {g['branches_at_max']} at {g['centers_at_max']}), up to {g['max_branches']} branches")
    print("step (20 Hz) mean ± std per slot: close / open / end")
    for k in range(16):
        r = res["slots"][k]
        print(f"  slot {k:2d}: " + " / ".join(f"{r[m]['step_mean']:.0f}±{r[m]['step_std']:.1f}" for m in ("close", "open", "end"))
              + "   max joint std at close/open: " + " / ".join(f"{max(j['std'] for j in r[m]['joints']):.2f}" for m in ("close", "open")))


if __name__ == "__main__":
    main()

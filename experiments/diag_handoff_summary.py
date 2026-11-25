"""diag_handoff v2 결과 요약 (2026-10-22): 정책 × (학습 벽 / 안 본 벽) × 넘긴 벽돌 k마다 실패 종류를 센다.

실행: python3 -m experiments.diag_handoff_summary outputs/wall3/diag_policy_20261022/handoff16
"""
import json, statistics as st, sys
from pathlib import Path


def med(v):
    return st.median(v) if v else None


def row(runs):
    b = [r["bricks"][0] for r in runs if r["expert_ok"] and r["bricks"]]
    n = len(b)
    lifted = [x for x in b if x["target_lifted"]]
    fin = [x["final"] for x in lifted]
    cl = [x["closes"][0]["target"] for x in b if x["closes"]]
    op = [x["opens"][0] for x in lifted if x["opens"]]
    return {"n": n, "expert_fail": sum(not r["expert_ok"] for r in runs),
            "lifted": len(lifted), "retract": sum(x["end_reason"] == "retract" for x in b),
            "regrasp": sum(x["n_closes"] > 1 for x in b), "other_lifted": sum(bool(x["other_lifted"]) for x in b),
            "laid_moved_2mm": sum(any(v > 2 for v in x["laid_moved_mm"].values()) for x in b),
            "laid_moved_max_mm": max([max(x["laid_moved_mm"].values(), default=0) for x in b], default=0),
            "supply_moved_2mm": sum(any(v > 2 for v in x["supply_moved_mm"].values()) for x in b),
            "close_dyaw_abs_med": med([abs(c["dyaw_deg"]) for c in cl]),
            "close_xy_med_mm": med([(c["along_mm"] ** 2 + c["across_mm"] ** 2) ** 0.5 for c in cl]),
            "open_xy_med_mm": med([o["xy_mm"] for o in op]), "open_yaw_abs_med": med([abs(o["yaw_deg"]) for o in op]),
            "final_xy_med_mm": med([f["xy_mm"] for f in fin]), "final_xy_max_mm": max([f["xy_mm"] for f in fin], default=None),
            "final_dz_med_mm": med([f["dz_mm"] for f in fin]), "final_yaw_abs_med": med([abs(f["yaw_deg"]) for f in fin]),
            "final_yaw_abs_max": max([abs(f["yaw_deg"]) for f in fin], default=None),
            "final_tilt_max": max([f["tilt_deg"] for f in fin], default=None),
            "placed_2mm": sum(f["xy_mm"] <= 2 and abs(f["yaw_deg"]) <= 2 for f in fin),
            "placed_10mm": sum(f["xy_mm"] <= 10 and abs(f["yaw_deg"]) <= 5 for f in fin)}


def main():
    d = Path(sys.argv[1])
    out = {}
    for f in sorted(d.glob("*_k*.json")):
        pol_split, k = f.stem.rsplit("_k", 1)
        out.setdefault(pol_split, {})[int(k)] = row(json.loads(f.read_text())["runs"])
    (d / "summary.json").write_text(json.dumps(out, indent=1))
    f1 = lambda v: "-" if v is None else f"{v:.0f}"
    for ps, rows in out.items():
        print(f"\n{ps}")
        print("k  | n  | 듦 | 철수 | 재집기 | 딴벽돌 | 쌓인벽돌밀림>2mm(최대) | 공급대밀림 | 닫을때 각도°/위치mm | 놓을때 xy mm/각° | 최종 xy 중앙(최대) | 최종 각° 중앙(최대) | dz | ≤2mm·2° | ≤10mm·5°")
        for k in sorted(rows):
            r = rows[k]
            print(f"{k:2d} | {r['n']:2d} | {r['lifted']:2d} | {r['retract']:2d} | {r['regrasp']:2d} | {r['other_lifted']:2d} | "
                  f"{r['laid_moved_2mm']:2d} ({r['laid_moved_max_mm']:.0f}) | {r['supply_moved_2mm']:2d} | "
                  f"{f1(r['close_dyaw_abs_med'])}/{f1(r['close_xy_med_mm'])} | {f1(r['open_xy_med_mm'])}/{f1(r['open_yaw_abs_med'])} | "
                  f"{f1(r['final_xy_med_mm'])} ({f1(r['final_xy_max_mm'])}) | {f1(r['final_yaw_abs_med'])} ({f1(r['final_yaw_abs_max'])}) | "
                  f"{f1(r['final_dz_med_mm'])} | {r['placed_2mm']:2d} | {r['placed_10mm']:2d}")


if __name__ == "__main__":
    main()

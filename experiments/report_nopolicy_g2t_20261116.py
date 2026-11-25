"""정책 없음 · G2t 결과 정리 (2026-11-16). 결과 파일만 읽는다. 시뮬레이션 없음.

주 지표는 wall3/primary_metrics.py의 compute()를 그대로 쓴다(줄 목록 ROWS만 이 비교에 맞게 바꿔 부름):
완성률(주 1), 완성 벽의 시방 항목 mm(주 2), 시방 합격 · 벽당 채운 자리(보조), Full 대비 짝 차이와 95 % 구간(벽 단위 10000번, 시드 0).
덧붙여 조건마다 누가 무엇을 했나(집기 · 놓기 · 운반 · 시간)와 벽 하나에 걸린 시뮬레이션 시간을 센다.
수치는 개발 배치 100벽(시작값 700300 ~ 700399)이며 최종 평가가 아니다.

실행: PYTHONPATH=. venv/bin/python -m experiments.report_nopolicy_g2t_20261116 --out outputs/wall3/report_nopolicy_g2t_20261116.json
"""
import argparse, json
from pathlib import Path

import numpy as np

from wall3 import primary_metrics as pm

ROWS = [
    ("Full", "eval_lf_full_20261110/act_all_held+k2+k7", "full"),
    ("Full 다시 1 (11-10)", "eval_ladder3b_20261110/act_all_held+k2+k7", "rerun"),
    ("Full 다시 2 (11-11)", "eval_lg_noG2c_20261111/act_all_held+k2+k7", "rerun"),
    ("정책 없음 (대본 + Full 검측)", "eval_nopolicy_20261116/script_all_held+k2+k7", "nopolicy"),
    ("Full + G2t (G2가 운반)", "eval_g2t_20261116/act_all_held+k2+k7+g2t", "g2t"),
    ("검측 없음 (정책만)", "eval_lf_none_20261110/act_none", "none"),
]


def roles(walls):
    """Who did what, over all walls of a condition and over its complete walls."""
    out = {}
    for name, sel in (("all", lambda w: True), ("complete", lambda w: all(w["eval"]["slots_filled"]))):
        c = {"walls": 0, "bricks": 0, "g0_grasps": 0, "g0_then_g1_ok": 0, "g1_ok": 0, "g1_not": 0, "g2_places": 0,
             "g2_far_policy_released": 0, "g3_replaced": 0, "g2t_carries": 0, "policy_s": 0.0, "gate_s": 0.0, "sim_s": []}
        for w in walls:
            if not sel(w):
                continue
            c["walls"] += 1
            c["sim_s"].append(w["sim_s"])
            for b in w["bricks"]:
                c["bricks"] += 1
                g = b.get("gates") or {}
                g0 = g.get("g0", [])
                c["g0_grasps"] += len(g0)
                c["g0_then_g1_ok"] += sum(1 for e in g0 if e.get("g1_grasped"))
                c["g1_ok"] += sum(1 for e in g.get("g1", []) if e.get("grasped"))
                c["g1_not"] += sum(1 for e in g.get("g1", []) if e.get("grasped") is False)
                c["g2_places"] += sum(1 for e in g.get("g2", []) if not e.get("skipped"))
                c["g2_far_policy_released"] += sum(1 for e in g.get("g2", []) if e.get("skipped"))
                c["g3_replaced"] += sum(1 for e in g.get("g3", []) if isinstance(e.get("replace"), dict) and e["replace"].get("result") == "replaced")
                c["g2t_carries"] += sum(1 for e in g.get("g2t", []) if e.get("result") == "carried")
                c["policy_s"] += b.get("policy_time_s") or 0.0
                c["gate_s"] += b.get("gate_time_s") or 0.0
        if c["walls"]:
            s = np.array(c.pop("sim_s"))
            c["sim_s_per_wall_median"] = float(np.median(s))
            c["policy_time_share"] = c["policy_s"] / max(c["policy_s"] + c["gate_s"], 1e-9)
            c["controller_own_grasps_g1_ok"] = c["g1_ok"] - c["g0_then_g1_ok"]
        else:
            c.pop("sim_s")
        out[name] = c
    return out


def script_turns(walls):
    t = [x for w in walls for x in w.get("script_turns", [])]
    if not t:
        return None
    return {"turns": len(t), "gave_up": sum(x["gave_up"] for x in t), "release_started": sum(x["release_started"] for x in t),
            "attempts_hist": {str(k): sum(x["attempts"] == k for x in t) for k in range(0, 6) if sum(x["attempts"] == k for x in t)}}


def first_unfilled(walls):
    f = [int(np.argmin(w["eval"]["slots_filled"])) if not all(w["eval"]["slots_filled"]) else 16 for w in walls]
    return {str(i): f.count(i) for i in range(17) if f.count(i)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} 이미 있음: 덮어쓰지 않음")
    pm.ROWS = ROWS
    res = pm.compute()
    res["note"] = res["note"] + " · 비교 줄: 정책 없음, Full + G2t (2026-11-16)"
    res["extra"] = {}
    for lab, key, kind in ROWS:
        walls, _ = pm.load(key)
        ws = [walls[s] for s in sorted(walls)]
        res["extra"][lab] = {"roles": roles(ws), "first_unfilled_hist": first_unfilled(ws), "script_turns": script_turns(ws)}
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    print(f"{'줄':32s} 채운자리  완성 시방  Δ완성 %p [95 %]          Δ채운자리 [95 %]     ②③④ 넘은 벽   벽당 sim s  정책 시간")
    for r in res["rows"]:
        v = r.get("vs_full")
        d = f"{v['complete_pp']:+6.1f} [{v['complete_ci_pp'][0]:+.0f}, {v['complete_ci_pp'][1]:+.0f}]" if v else "기준"
        df = f"{v['filled']:+6.2f} [{v['filled_ci'][0]:+.2f}, {v['filled_ci'][1]:+.2f}]" if v else ""
        it = r["items"]
        ov = " / ".join(str(it[k]["over_limit"]) if it[k] else "-" for k in ("2", "3", "4"))
        ro = res["extra"][r["label"]]["roles"]["all"]
        print(f"{r['label']:32s} {r['filled_mean']:6.2f} {r['complete']:5d} {r['spec_pass']:4d}  {d:24s} {df:24s} {ov:12s} "
              f"{ro.get('sim_s_per_wall_median', float('nan')):8.0f} {ro.get('policy_time_share', float('nan')) * 100:6.1f} %")


if __name__ == "__main__":
    main()

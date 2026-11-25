"""남은 실패의 동작 원인 분석 (2026-11-05): 이미 있는 추적 · 평가 기록만 읽는다(새 시뮬레이션 없음). 진단용이며 평가 결과가 아니다.

1. 공급대 위 실패: 처음 못 채운 벽돌이 '공급대 위 다른 곳'인 벽에서, 그 벽돌이 처음 움직인 순간에 다음 차례 벽돌이었는지(벽에 앞 자리가 다 찼는지).
2. 자리 10 반장: 정책 손이 처음 닿는 순간의 집게 명령 · 내려오는 속도 · 벽돌 중심과의 거리. 실패 벽과 자리 10을 채운 벽 비교.
3. G2 손 안 측정: 자리별로 카메라가 손 안 벽돌을 봤는지(camera / nominal = 못 봄, 기본값으로 놓음)와 놓은 뒤 오차.
   자리 5에서 못 봤을 때 G3가 다시 봤는지.
정답 좌표는 기록 · 분류에만 쓴다.

실행: PYTHONPATH=. venv/bin/python -m experiments.diag_fail_mechanism \
        --trace outputs/wall3/diag_trace_held_20261104 --evals outputs/wall3/eval_gates_info_20261104 [--out <json>]
"""
import argparse, glob, json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from wall3.config import design_slots

DT = 0.05


def yaw_of(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def load_trace(root):
    c = json.loads((root / "causes.json").read_text())
    part = {}
    for p in root.glob("part*/trace.json"):
        for w in json.loads(p.read_text())["walls"]:
            part[str(w["seed"])] = (p.parent, w)
    return c, part


def first_fail(v):
    if v["first_unfilled"] >= 16:
        return None
    f = [x for x in v["fails"] if x["slot"] == v["first_unfilled"]]
    return f[0] if f else None


def supply_table(c, part, D):
    """1. 공급대 위 실패: 다음 차례 벽돌이었나."""
    rows = []
    for s, v in c.items():
        if s.startswith("_"):
            continue
        f = first_fail(v)
        if not f or f["state"] != "공급대 위 다른 곳" or not f["moves"]:
            continue
        b = f["brick"]
        z = np.load(part[s][0] / f"trace_{s}.npz")
        P = z["pose"][f["moves"][0]["s"]]
        wall = [i for i in range(16) if np.hypot(P[i, 0] - D[i]["x"], P[i, 1] - D[i]["y"]) < 0.03 and abs(P[i, 2] - D[i]["z"]) < 0.03]
        nxt = min([i for i in range(16) if i not in wall], default=16)
        rows.append({"seed": int(s), "brick": b, "next_brick": nxt == b, "first_move_src": f["moves"][0]["src"]})
    return {"walls": len(rows), "was_next_brick": sum(r["next_brick"] for r in rows),
            "by_brick": dict(Counter(r["brick"] for r in rows)), "rows": rows}


def half10(c, part, b=10):
    """2. 자리 10 반장: 정책 손이 처음 닿는 순간."""
    out = {"fail": [], "filled": []}
    for s, v in c.items():
        if s.startswith("_"):
            continue
        f = first_fail(v)
        d, w = part[s]
        lab = "fail" if (f and f["state"] == "공급대 위 다른 곳" and f["brick"] == b) else ("filled" if w["eval"]["slots_filled"][b] else None)
        if lab is None:
            continue
        mv = [m for m in v["moves"].get(str(b), []) if m["src"] == 0]
        if not mv:
            continue
        S = mv[0]["s"]
        z = np.load(d / f"trace_{s}.npz")
        tcp, P = z["tcp"], z["pose"]
        u = np.array([np.cos(yaw_of(P[S, b, 3:7])), np.sin(yaw_of(P[S, b, 3:7]))])
        dxy = tcp[S, :2] - P[S, b, :2]
        out[lab].append({"seed": int(s), "grip": float(z["grip"][S]), "vz_m_s": float((tcp[S, 2] - tcp[S - 3, 2]) / 3 / DT),
                         "off_along_mm": float(dxy @ u) * 1000, "off_across_mm": float(dxy @ np.array([-u[1], u[0]])) * 1000,
                         "lift_mm": mv[0]["lift_mm"]})
    summ = {}
    for lab, L in out.items():
        a = lambda k: np.array([x[k] for x in L])
        summ[lab] = {"n": len(L), "grip_open": int((a("grip") < 0.1).sum()), "fast_descent(<-0.2 m/s)": int((a("vz_m_s") < -0.2).sum()),
                     "lift_ge_15mm": int((a("lift_mm") >= 15).sum()),
                     "vz_median": round(float(np.median(a("vz_m_s"))), 2),
                     "off_along_median_mm": round(float(np.median(np.abs(a("off_along_mm"))))),
                     "off_across_median_mm": round(float(np.median(np.abs(a("off_across_mm")))))}
    return {"summary": summ, "rows": out}


def g2_view(eval_root, cond):
    """3. G2 손 안 측정 출처와 놓은 뒤 오차 (평가 기록)."""
    src = defaultdict(Counter)
    after = defaultdict(list)
    g3_after_nominal5 = Counter()
    for p in sorted(glob.glob(str(eval_root / f"act_{cond}_part*/rollout.json"))):
        for w in json.loads(Path(p).read_text())["walls"]:
            for br in w["bricks"]:
                g2 = [e for e in br["gates"].get("g2", []) if not e.get("skipped")]
                for e in g2:
                    src[e["slot"]][e["in_hand_src"]] += 1
                    after[(e["slot"], e["in_hand_src"])].append(e.get("audit_after_mm") or 0.0)
                    if e["slot"] == 5 and e["in_hand_src"] == "nominal":
                        g3 = [x for x in br["gates"].get("g3", []) if x["slot"] == 5]
                        key = "G3 못 봄" if not any(x.get("seen") for x in g3) else (
                            "G3가 보고 다시 놓음" if any(isinstance(x.get("replace"), dict) for x in g3) else "G3가 봄, 다시 놓지 않음")
                        g3_after_nominal5[key] += 1

    def bins(a):
        a = np.array(a)
        return {"<=5": int((a <= 5).sum()), "5-20": int(((a > 5) & (a <= 20)).sum()), "20-200": int(((a > 20) & (a <= 200)).sum()),
                ">200": int((a > 200).sum())}
    return {"by_slot": {s: {k: {"n": n, "after_mm_median": round(float(np.median(after[(s, k)])), 1), "after_bins": bins(after[(s, k)])}
                            for k, n in src[s].items()} for s in sorted(src)},
            "slot5_nominal_then_g3": dict(g3_after_nominal5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--evals", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out and Path(a.out).exists():
        raise SystemExit(f"{a.out} 이미 있음: 덮어쓰지 않음")
    D = design_slots()
    c, part = load_trace(Path(a.trace))
    res = {"supply_table": supply_table(c, part, D), "half_brick_10": half10(c, part),
           "g2_view": {cond: g2_view(Path(a.evals), cond) for cond in ("all_held", "all_truth_held")}}
    st = res["supply_table"]
    print(f"1. 공급대 위 실패 {st['walls']}벽 중 다음 차례 벽돌: {st['was_next_brick']} | 벽돌별 {st['by_brick']}")
    print("2. 자리 10 반장, 정책 손이 처음 닿는 순간:")
    for lab, sm in res["half_brick_10"]["summary"].items():
        print(f"   {lab}: {sm}")
    print("3. G2 손 안 측정 (camera / truth = 봄, nominal = 못 봐서 기본값):")
    for cond, r in res["g2_view"].items():
        print(f"   [{cond}]")
        for s, d in r["by_slot"].items():
            print(f"   자리 {s:2d}: " + " | ".join(f"{k} {v['n']} (놓은 뒤 중앙 {v['after_mm_median']} mm, {v['after_bins']})" for k, v in d.items()))
        print(f"   자리 5 못 봄 → G3: {r['slot5_nominal_then_g3']}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()

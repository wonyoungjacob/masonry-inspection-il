"""현실 제약 ablation 결과 정리 (2026-11-16, 지시서 5절). 결과 파일만 읽는다. 시뮬레이션 없음.

8조건 = {하이브리드(act), 제어만(script)} × {none, A, B, AB}. 같은 100벽(700300 ~ 700399) 짝 비교, 95 % 부트스트랩(벽 단위 10000번, 시드 0).
  비교 1: 같은 칸에서 하이브리드 − 제어만 (완성 %p · 시방 %p · 채운 자리)
  비교 2: 제어기마다 A · B 주효과와 A × B 상호작용 (완성 %p). 주효과 A = 벽마다 ((A − none) + (AB − B)) / 2, 상호작용 = (AB − B) − (A − none)
  비교 3: 제약이 하이브리드와 제어만의 차이를 바꾸나: (하이브리드 − 제어만)_칸 − (하이브리드 − 제어만)_none
  G0v 기록: 넘겨받음 · 못 봄 · 집음(G1 쥠) · 잰 공급 벽돌 위치 오차(참값은 기록에만)
수치는 개발 배치이며 최종 평가가 아니다.

실행: PYTHONPATH=. venv/bin/python -m experiments.report_real_20261116 --out outputs/wall3/report_real_20261116.json
"""
import argparse, json
from pathlib import Path

import numpy as np

from wall3 import primary_metrics as pm

CTRL = {"act": "하이브리드 (ACT + 검측)", "script": "제어만 (대본 + 검측)"}
CELLS = ("none", "A", "B", "AB")
CELL_NAME = {"none": "제약 없음", "A": "A만", "B": "B만", "AB": "A + B"}


def key(ctrl, cell):
    return f"eval_real_{ctrl}_{cell}_20261116/{ctrl}_real_{cell}"


def per_wall(w):
    e = w["eval"]
    f = [bool(x) for x in e["slots_filled"]]
    comp = all(f)
    return {"n": sum(f), "complete": comp, "spec": bool(e["spec_pass"]), "items": pm.item_values(e) if comp else None}


def g0v_stats(walls):
    g0 = [e for w in walls for b in w["bricks"] for e in b["gates"].get("g0", [])]
    err = np.array([e["audit_meas_err_mm"] for e in g0 if "audit_meas_err_mm" in e])
    g1 = [e for w in walls for b in w["bricks"] for e in b["gates"].get("g1", [])]
    return {"g0v": len(g0), "not_seen": sum(e.get("result") == "not_seen" for e in g0),
            "grasped": sum(bool(e.get("g1_grasped")) for e in g0),
            "meas_err_mm_median": float(np.median(err)) if len(err) else None,
            "meas_err_mm_p90": float(np.percentile(err, 90)) if len(err) else None,
            "g1_ok": sum(bool(e.get("grasped")) for e in g1), "g1_not": sum(e.get("grasped") is False for e in g1),
            "g2t_carries": sum(1 for w in walls for b in w["bricks"] for e in b["gates"].get("g2t", []) if e.get("result") == "carried"),
            "sim_s_per_wall_median": float(np.median([w["sim_s"] for w in walls]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} 이미 있음: 덮어쓰지 않음")
    data, sources = {}, {}
    for c in CTRL:
        for cell in CELLS:
            walls, files = pm.load(key(c, cell))
            sources.update(files)
            data[(c, cell)] = walls
    seeds = sorted(data[("act", "none")])
    assert len(seeds) == 100 and seeds[0] == 700300 and seeds[-1] == 700399
    for k, w in data.items():
        assert sorted(w) == seeds, k
    pw = {k: [per_wall(w[s]) for s in seeds] for k, w in data.items()}
    arr = lambda k, f: np.array([float(x[f]) for x in pw[k]])
    res = {"note": "개발 배치 100벽(700300 ~ 700399) · 최종 평가 아님 · 짝 차이 95 % 부트스트랩(벽 단위 10000번, 시드 0)",
           "conditions": {}, "hybrid_minus_script": {}, "effects": {}, "did": {}, "sources": sources}
    for (c, cell), rows in pw.items():
        n, comp, spec = arr((c, cell), "n"), arr((c, cell), "complete"), arr((c, cell), "spec")
        cw = [r for r in rows if r["complete"]]
        items = {}
        for k2 in ("2", "3", "4"):
            v = np.array([r["items"][k2] for r in cw])
            items[k2] = None if not len(v) else {"median": float(np.median(v)), "max": float(v.max()), "over6": int((v > 6.0).sum())}
        res["conditions"][f"{c}_{cell}"] = {"label": f"{CTRL[c]} · {CELL_NAME[cell]}", "filled_mean": float(n.mean()),
                                            "complete": int(comp.sum()), "spec_pass": int(spec.sum()), "items": items,
                                            **g0v_stats([data[(c, cell)][s] for s in seeds])}
    for cell in CELLS:
        h, s = ("act", cell), ("script", cell)
        res["hybrid_minus_script"][cell] = {
            "complete_pp": pm.boot((arr(h, "complete") - arr(s, "complete")) * 100),
            "spec_pp": pm.boot((arr(h, "spec") - arr(s, "spec")) * 100),
            "filled": pm.boot(arr(h, "n") - arr(s, "n"))}
    for c in CTRL:
        y = {cell: arr((c, cell), "complete") * 100 for cell in CELLS}
        res["effects"][c] = {
            "A_alone": pm.boot(y["A"] - y["none"]), "B_alone": pm.boot(y["B"] - y["none"]), "AB_vs_none": pm.boot(y["AB"] - y["none"]),
            "main_A": pm.boot(((y["A"] - y["none"]) + (y["AB"] - y["B"])) / 2),
            "main_B": pm.boot(((y["B"] - y["none"]) + (y["AB"] - y["A"])) / 2),
            "interaction_AxB": pm.boot((y["AB"] - y["B"]) - (y["A"] - y["none"]))}
    d0 = (arr(("act", "none"), "complete") - arr(("script", "none"), "complete")) * 100
    for cell in ("A", "B", "AB"):
        d = (arr(("act", cell), "complete") - arr(("script", cell), "complete")) * 100
        res["did"][cell] = pm.boot(d - d0)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    f = lambda m: f"{m[0]:+.1f} [{m[1][0]:+.0f}, {m[1][1]:+.0f}]"
    print("개발 배치 100벽 · 최종 평가 아님\n")
    print(f"{'조건':28s} 채운자리 완성 시방  ②③④>6mm  G0v(못봄/집음)  잰오차중앙 mm  벽당 sim s")
    for kk, r in res["conditions"].items():
        ov = "/".join(str(r["items"][x]["over6"]) if r["items"][x] else "-" for x in ("2", "3", "4"))
        print(f"{r['label']:28s} {r['filled_mean']:6.2f} {r['complete']:4d} {r['spec_pass']:4d}  {ov:9s}  {r['g0v']:5d}({r['not_seen']}/{r['grasped']})"
              f"  {r['meas_err_mm_median'] if r['meas_err_mm_median'] is None else round(r['meas_err_mm_median'], 2)!s:>8}  {r['sim_s_per_wall_median']:7.0f}")
    print("\n같은 칸에서 하이브리드 − 제어만 (완성 %p · 시방 %p · 채운 자리)")
    for cell, r in res["hybrid_minus_script"].items():
        print(f"  {CELL_NAME[cell]:8s} 완성 {f(r['complete_pp'])}  시방 {f(r['spec_pp'])}  채운 자리 {r['filled'][0]:+.2f} [{r['filled'][1][0]:+.2f}, {r['filled'][1][1]:+.2f}]")
    print("\n제약 효과 (완성 %p, 제어기마다)")
    for c, r in res["effects"].items():
        print(f"  {CTRL[c]:20s} A만 {f(r['A_alone'])}  B만 {f(r['B_alone'])}  A+B {f(r['AB_vs_none'])}  주효과 A {f(r['main_A'])}  B {f(r['main_B'])}  A×B {f(r['interaction_AxB'])}")
    print("\n제약이 (하이브리드 − 제어만)을 바꾸나 (완성 %p, 제약 없음 대비)")
    for cell, m in res["did"].items():
        print(f"  {CELL_NAME[cell]:8s} {f(m)}")


if __name__ == "__main__":
    main()

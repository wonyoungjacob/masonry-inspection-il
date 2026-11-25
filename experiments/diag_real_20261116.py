"""현실 제약 ablation 실패 원인 집계 (2026-11-17, 진단). 결과 파일만 읽는다. 시뮬레이션 없음.

하이브리드가 못 채운 벽에서 '정책이 G0v보다 먼저 공급 벽돌을 건드렸나'를 기록(audit_*, 참값 (진단))으로 센다.
  먼저 건드림 = 공급칸 j의 첫 G0v보다 앞선 G1 순간에, 가장 많이 들린 벽돌이 j이고 5 mm 넘게 들림
  넘어뜨림 = 그 G1이 '안 쥠' (손이 빈 채로 벽돌만 옆으로 세워짐, 들림 약 16.5 mm)
  정책이 쥠 = 그 G1이 '쥠' (정책이 직접 집어 G2t가 운반)
또 공급칸 10을 G0v가 넘겨받은 순간의 손 · 칸 중심 거리(도면, G0 넘겨받기 구역 반지름 = gates.GateConfig.g0_zone_xy)를 조건마다 센다.
참고로 11-16 Full + G2t(eval_g2t_20261116)에서 'G0가 16번 모두 넘겨받음'과 완성의 관계를 센다.

실행: PYTHONPATH=. venv/bin/python -m experiments.diag_real_20261116 --out <파일.json>
"""
import argparse, glob, json
from collections import Counter
from pathlib import Path

import numpy as np

from wall3.gates import GateConfig

W = "outputs/wall3"
CELLS = ("none", "A", "B", "AB")
RISE_MM = 5.0


def load(pattern):
    walls = []
    for f in sorted(glob.glob(pattern)):
        walls += json.loads(Path(f).read_text())["walls"]
    return {w["seed"]: w for w in walls}


def events(w, gate):
    return sorted((e for b in w["bricks"] for e in b["gates"].get(gate, [])), key=lambda e: e["t"])


def touched_first(w):
    """Supply slots the policy disturbed before G0v took over that slot: [(slot, rise_mm, g1_grasped, g0v_results)]."""
    g0, g1 = events(w, "g0"), events(w, "g1")
    out = []
    for j in range(16):
        g0j = [e for e in g0 if e["supply_slot"] == j]
        t0 = g0j[0]["t"] if g0j else 10 ** 12
        pre = [e for e in g1 if e["t"] < t0 and e.get("audit_max_rise_brick") == j and e.get("audit_max_rise_mm", 0.0) > RISE_MM]
        if pre:
            out.append({"supply_slot": j, "t": pre[0]["t"], "rise_mm": round(pre[0]["audit_max_rise_mm"], 1), "g1_grasped": bool(pre[0]["grasped"]),
                        "g0v_after": [e.get("result") for e in g0j]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} 이미 있음: 덮어쓰지 않음")
    zone_mm = GateConfig().g0_zone_xy * 1000
    res = {"note": "개발 배치 100벽 · 최종 평가 아님 · 참값은 진단에만", "g0_zone_mm": zone_mm, "rise_mm": RISE_MM, "conditions": {}}
    for c in ("act", "script"):
        for cell in CELLS:
            Wd = load(f"{W}/eval_real_{c}_{cell}_20261116/{c}_real_{cell}_part*/rollout.json")
            assert len(Wd) == 100
            rows, kinds = [], Counter()
            for s, w in sorted(Wd.items()):
                comp = all(w["eval"]["slots_filled"])
                tf = touched_first(w)
                kind = ("넘어뜨림" if any(not x["g1_grasped"] for x in tf) else "정책이 쥠") if tf else "안 건드림"
                kinds[(kind, "완성" if comp else "미완성")] += 1
                if tf or not comp:
                    rows.append({"seed": s, "complete": comp, "filled": int(sum(w["eval"]["slots_filled"])),
                                 "unfilled": [i for i, v in enumerate(w["eval"]["slots_filled"]) if not v], "touched_first": tf})
            off10 = np.array([e["tcp_off_mm"] for w in Wd.values() for e in events(w, "g0") if e["supply_slot"] == 10 and e["try"] == 1
                              and not touched_first(w)])
            res["conditions"][f"{c}_{cell}"] = {
                "kinds": {f"{k[0]} · {k[1]}": v for k, v in sorted(kinds.items())}, "walls": rows,
                "g0v_slot10_takeover_off_mm": None if not len(off10) else {
                    "n": int(len(off10)), "median": float(np.median(off10)), "p90": float(np.percentile(off10, 90)), "max": float(off10.max()),
                    "over_50": int((off10 > 50).sum())}}
    old = load(f"{W}/eval_g2t_20261116/*/rollout.json")
    rel = Counter()
    for w in old.values():
        n0 = len(events(w, "g0"))
        rel[("완성" if all(w["eval"]["slots_filled"]) else "미완성", "G0 16번 모두" if n0 == 16 else f"G0 16번 미만")] += 1
    res["g2t_20261116_g0_vs_complete"] = {f"{k[0]} · {k[1]}": v for k, v in sorted(rel.items())}
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))

    print(f"개발 배치 100벽 · 최종 평가 아님 · 참값은 진단에만 (G0 넘겨받기 구역 반지름 {zone_mm:.0f} mm, 들림 > {RISE_MM:.0f} mm)\n")
    print("조건별: 정책이 G0v보다 먼저 공급 벽돌을 건드린 벽")
    for k, r in res["conditions"].items():
        o = r["g0v_slot10_takeover_off_mm"]
        os_ = "-" if o is None else f"n {o['n']}, 중앙 {o['median']:.1f}, 90 % {o['p90']:.1f}, 최대 {o['max']:.1f}, 50 mm 넘음 {o['over_50']}"
        print(f"  {k:12s} {r['kinds']}")
        print(f"  {'':12s} 공급칸 10을 G0v가 넘겨받은 순간 손 · 칸 중심 거리 mm: {os_}")
        for x in r["walls"]:
            tf = "; ".join(f"칸 {t['supply_slot']} 들림 {t['rise_mm']} mm G1 {'쥠' if t['g1_grasped'] else '안 쥠'} → G0v {t['g0v_after'] or '없음'}" for t in x["touched_first"])
            print(f"    {x['seed']} {'완성' if x['complete'] else '미완성'} {x['filled']}/16 못 채움 {x['unfilled']}  {tf or '-'}")
    print("\n(참고) 11-16 Full + G2t:", res["g2t_20261116_g0_vs_complete"])


if __name__ == "__main__":
    main()

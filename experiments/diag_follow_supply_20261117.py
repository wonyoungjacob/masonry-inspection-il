"""진단(결과 아님, 2026-11-17): 지금 ACT는 공급 벽돌이 옮겨지면 영상으로 보고 따라가는가. 결과 파일만 읽는다. 시뮬레이션 없음.

① 현실 제약 실험의 하이브리드 '없음' 대 'B'(팔레트 30 mm 이동): 같은 벽 · 같은 공급칸에서 G0v가 처음 넘겨받은 순간
   손과 도면 칸 중심의 거리(tcp_off_mm)를 짝지어 비교한다. 기록은 거리(스칼라)뿐이다.
   정책이 벽돌을 따라갔다면 손이 30 mm 옮겨지므로 거리가 크게 달라져야 한다. 참고값으로 '완전히 따라감'일 때 기대되는
   |거리 변화|를 적는다(가정: 이동 방향과 원래 치우침 방향의 각이 고르게 무작위, 시드 0).
② 예비 실행(experiments/pilot_supplyfar_20261117.py, 팔레트 90 mm 이동, 2벽씩): 채운 자리, G0v 넘겨받기 거리와 결과.

실행: PYTHONPATH=. venv/bin/python -m experiments.diag_follow_supply_20261117 --out <파일.json>
"""
import argparse, glob, json
from collections import Counter
from pathlib import Path

import numpy as np

W = "outputs/wall3"
SHIFT_B_MM = 30.0


def load(pattern):
    walls = []
    for f in sorted(glob.glob(pattern)):
        walls += json.loads(Path(f).read_text())["walls"]
    return {w["seed"]: w for w in walls}


def first_takeover(w):
    """{supply slot: tcp_off_mm of the first G0v takeover of that slot}."""
    out = {}
    for b in w["bricks"]:
        for e in b["gates"].get("g0", []):
            if e["try"] == 1 and e["supply_slot"] not in out:
                out[e["supply_slot"]] = e["tcp_off_mm"]
    return out


def expected_if_following(dn, shift_mm, seed=0):
    """Median |distance change| if the hand moved with the pallet by shift_mm (direction uniformly random w.r.t. the offset)."""
    phi = np.random.default_rng(seed).uniform(0, 2 * np.pi, len(dn))
    return float(np.median(np.abs(np.sqrt(dn ** 2 + shift_mm ** 2 + 2 * dn * shift_mm * np.cos(phi)) - dn)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} 이미 있음: 덮어쓰지 않음")
    n = load(f"{W}/eval_real_act_none_20261116/act_real_none_part*/rollout.json")
    b = load(f"{W}/eval_real_act_B_20261116/act_real_B_part*/rollout.json")
    assert len(n) == len(b) == 100
    dn, db = [], []
    for s in sorted(n):
        x, y = first_takeover(n[s]), first_takeover(b[s])
        for j in x:
            if j in y:
                dn.append(x[j]); db.append(y[j])
    dn, db = np.array(dn), np.array(db)
    ad = np.abs(db - dn)
    res = {"note": "진단 · 결과 아님 · 개발 배치(700300 ~ 700399)",
           "B_30mm": {"pairs": int(len(dn)), "median_none_mm": float(np.median(dn)), "median_B_mm": float(np.median(db)),
                      "abs_change_median_mm": float(np.median(ad)), "abs_change_p90_mm": float(np.percentile(ad, 90)),
                      "corr": float(np.corrcoef(dn, db)[0, 1]),
                      "expected_abs_change_median_if_following_mm": expected_if_following(dn, SHIFT_B_MM)},
           "pilot_90mm": {}}
    for c, name in (("act", "하이브리드"), ("script", "제어만")):
        d = json.loads(Path(f"{W}/pilot_supplyfar_{c}_90mm_20261117/rollout.json").read_text())
        g0 = [e for w in d["walls"] for x in w["bricks"] for e in x["gates"].get("g0", [])]
        off = [e["tcp_off_mm"] for e in g0 if e["try"] == 1]
        res["pilot_90mm"][name] = {
            "seeds": [w["seed"] for w in d["walls"]], "pallet_mm": [[round(v * 1000, 1) for v in w["perturb"]["pallet"]] for w in d["walls"]],
            "filled": [int(sum(w["eval"]["slots_filled"])) for w in d["walls"]],
            "g0v_first_takeovers": len(off), "g0v_takeover_off_mm": {"min": float(min(off)), "median": float(np.median(off)), "max": float(max(off))},
            "g0v_results": dict(Counter(e.get("result") for e in g0)),
            "turn_end": dict(Counter(x["end_reason"] for w in d["walls"] for x in w["bricks"])), "git_commit": d["meta"]["git_commit"]}
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))

    r = res["B_30mm"]
    print(res["note"] + "\n")
    print("① 하이브리드 '없음' 대 'B'(팔레트 30 mm): G0v가 처음 넘겨받은 순간 손 · 도면 칸 중심 거리 (같은 벽 · 같은 칸 짝)")
    print(f"   짝 {r['pairs']}개 · 중앙값 없음 {r['median_none_mm']:.1f} mm, B {r['median_B_mm']:.1f} mm")
    print(f"   |B − 없음| 중앙값 {r['abs_change_median_mm']:.1f} mm, 90 % {r['abs_change_p90_mm']:.1f} mm, 상관 {r['corr']:.2f}")
    print(f"   (참고) 손이 팔레트를 30 mm 그대로 따라갔다면 기대되는 |변화| 중앙값 {r['expected_abs_change_median_if_following_mm']:.1f} mm (방향 무작위 가정)")
    print("\n② 예비 실행: 팔레트 90 mm 이동 (2벽씩, 보고 수치 아님)")
    for name, p in res["pilot_90mm"].items():
        o = p["g0v_takeover_off_mm"]
        print(f"   {name}: 채운 자리 {p['filled']} (합 {sum(p['filled'])} / {16 * len(p['filled'])}) · G0v 첫 넘겨받기 {p['g0v_first_takeovers']}번, "
              f"도면 칸 중심과의 거리 {o['min']:.0f} ~ {o['max']:.0f} mm (중앙 {o['median']:.0f}) · G0v 결과 {p['g0v_results']} · 차례 끝 {p['turn_end']}")


if __name__ == "__main__":
    main()

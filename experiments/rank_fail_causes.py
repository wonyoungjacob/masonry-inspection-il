"""남은 실패 원인을 동작 단위로 나누고, 원인 하나를 없앨 때 기대되는 채움 수 향상으로 정렬 (2026-11-10).

이미 있는 E3(all_held) 추적 기록만 읽는다(새 시뮬레이션 없음). 진단용이며 평가 결과가 아니다. 정답 좌표는 분류에만 쓴다.
- 원인 = 처음 실패한 자리 + 동작 묶음(① 공급대 S · ② 검측 G · ③ 정책 운반 C · ④ 못 집음 N). wall3/est_fix_ceiling.py와 같은 묶음.
- 기대 향상 = 그 원인을 100 % 없앴을 때의 기대 채움 수 − 지금 (자리별 처음 실패 비율, 자리 사이 독립 가정; est_fix_ceiling.expect).
  누적 = 위 순서대로 하나씩 더 없앴을 때. 한 원인이 사라지면 뒤 자리까지 가는 벽이 늘어 뒤 원인의 향상이 커진다.
- 아래 CAUSES에 없는 (자리, 묶음)이 처음 실패로 나오면 '(새로 생김)' 원인으로 따로 세운다(제어를 넣은 뒤 분포가 바뀔 때).
- 3단(자리 12 ~ 15)은 그 자리까지 간 벽이 12벽뿐이라 최소 실패율 5 %를 둔 값도 낸다.

실행: PYTHONPATH=. venv/bin/python -m experiments.rank_fail_causes --trace outputs/wall3/diag_trace_held_20261104 --out <json>
"""
import argparse, json
from pathlib import Path

from experiments.est_fix_ceiling import load, group, hazards, expect

# 원인: 이름, 해당 (자리, 묶음) 목록
CAUSES = [
    ("R_10S", "자리 10 반장: G1이 '안 쥠'으로 손을 돌려준 직후 정책이 내려오다 침", ["10S"]),
    ("R_7G", "자리 7: G2가 손 안 벽돌을 잘못 재 놓아 이웃 끝에 걸침 (G3도 못 봄)", ["7G"]),
    ("R_6G", "자리 6: 자리 5 반장이 어긋나 있어 놓다 부딪힘 (G2 못 봄 + G3 찾는 반경 밖)", ["6G"]),
    ("R_7C", "자리 7: 정책이 옮기다 떨어뜨림", ["7C"]),
    ("R_7S", "자리 7: 공급대에서 집다 기울어 떨어짐 (G1 들기 중 · 정책)", ["7S"]),
    ("R_5", "자리 5 반장: 못 집음 · 공급대 · 검측", ["5N", "5S", "5G"]),
    ("R_11", "자리 11: 공급대 · 운반 · 못 집음", ["11S", "11C", "11N"]),
    ("R_1단", "1단 (자리 1 ~ 4): 검측 · 공급대 · 운반", ["1G", "2G", "2S", "3C", "4G", "4S"]),
    ("R_기타", "그 밖 2단 (자리 6 공급대 · 운반, 7 못 집음, 8 공급대)", ["6S", "6C", "7N", "8S"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out and Path(a.out).exists():
        raise SystemExit(f"{a.out} 이미 있음: 덮어쓰지 않음")
    c, part = load(Path(a.trace))
    rows = []
    for s, v in c.items():
        if s.startswith("_"):
            continue
        fu = v["first_unfilled"]
        f = [x for x in v["fails"] if x["slot"] == fu][0] if fu < 16 else None
        rows.append({"seed": int(s), "first_unfilled": fu, "group": group(f) if f else None})
    H = hazards(rows)
    keys = {f"{r['first_unfilled']}{r['group']}" for r in rows if r["group"]}
    covered = {k for _, _, ks in CAUSES for k in ks}
    gname = {"S": "공급대 위", "G": "검측이 다루다", "C": "정책이 옮기다", "N": "못 집음"}
    causes = list(CAUSES) + [(f"R_{k}", f"(새로 생김) 자리 {k[:-1]}: {gname[k[-1]]}", [k]) for k in sorted(keys - covered)]
    base = {fl: expect(H, {}, fl) for fl in (0.0, 0.05)}
    items = []
    for cid, name, ks in causes:
        n = sum(1 for r in rows if r["group"] and f"{r['first_unfilled']}{r['group']}" in ks)
        g = {fl: expect(H, {k: 1.0 for k in ks}, fl) for fl in (0.0, 0.05)}
        items.append({"id": cid, "name": name, "keys": ks, "walls": n,
                      "gain": round(g[0.0][0] - base[0.0][0], 2), "complete_pct": g[0.0][1],
                      "gain_floor5": round(g[0.05][0] - base[0.05][0], 2)})
    items.sort(key=lambda x: (-x["gain"], -x["walls"]))
    rm, cum = {}, []
    for it in items:
        rm.update({k: 1.0 for k in it["keys"]})
        e0, p0 = expect(H, rm, 0.0)
        e5, p5 = expect(H, rm, 0.05)
        cum.append({"id": it["id"], "filled": e0, "rate_pct": round(e0 / 16 * 100, 1), "complete_pct": p0,
                    "filled_floor5": e5, "complete_pct_floor5": p5})
    res = {"trace": a.trace, "base": {"filled": base[0.0][0], "complete_pct": base[0.0][1],
                                      "filled_floor5": base[0.05][0], "complete_pct_floor5": base[0.05][1]},
           "ranked": items, "cumulative": cum}
    print(f"지금 (모형): {base[0.0][0]}장 · 완성 {base[0.0][1]} %   [3단 최소 5 %: {base[0.05][0]}장 · {base[0.05][1]} %]")
    print("순위 · 원인 · 처음 실패 벽 · 그것만 없앨 때 +장 (완성 %) · 위에서부터 누적 장 (채움률 · 완성 %)")
    for i, (it, cu) in enumerate(zip(items, cum), 1):
        print(f"{i}. {it['id']:6s} {it['walls']:3d}벽  +{it['gain']:.2f}장 ({it['complete_pct']} %)  누적 {cu['filled']}장 "
              f"({cu['rate_pct']} % · 완성 {cu['complete_pct']} %)  [5 %: +{it['gain_floor5']:.2f} · 누적 {cu['filled_floor5']}]  {it['name']}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()

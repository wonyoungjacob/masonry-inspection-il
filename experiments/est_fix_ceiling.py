"""고칠 후보를 넣었을 때 채움 수의 상한 추정 (2026-11-10). 이미 있는 추적 기록만 읽는다(새 시뮬레이션 없음). 진단용이며 평가 결과가 아니다.

1. 자리별 처음 실패 비율(그 자리까지 온 벽 중 그 자리에서 처음 실패한 비율)을 원인 묶음별로 센다.
   ① S 공급대 위 (다음 차례 벽돌을 집다 침) · ② G 검측(G2 · G3)이 다루다 실패 · ③ C 정책이 옮기다 떨어뜨림 · ④ N 못 집음
2. 어떤 묶음을 비율 r만큼 없앴다고 가정하고, 자리마다 독립이라 보고 기대 채움 수(처음 실패 위치)와 16장 완성 비율을 계산한다.
   - 가정: 처음 실패에서 벽이 멈춤(실제로는 처음 실패 뒤 더 채운 경우가 벽당 평균 0.27장 있음). 자리 사이 독립.
   - 3단(자리 12 ~ 15)은 그 자리까지 온 벽이 12벽뿐이라 실패 비율 0은 과대평가일 수 있다 → 최소 실패율 5 %를 둔 경우도 함께 낸다.
3. 정책이 처음 그 벽돌을 움직인 순간 직전에 누가 팔을 움직였나(G1 · G2G3)와 그 뒤 시간. ① 실패와 채운 벽돌 비교.
정답 좌표는 기록 · 분류에만 쓴다.

실행: PYTHONPATH=. venv/bin/python -m experiments.est_fix_ceiling --trace outputs/wall3/diag_trace_held_20261104 --out <json>
"""
import argparse, json
from collections import Counter
from pathlib import Path

import numpy as np

DT = 0.05
SRC_NAME = ["policy", "G1", "G2G3", "other"]


def group(f):
    if f["gate_moved"]:
        return "G"
    if f["state"] == "공급대 위 다른 곳":
        return "S"
    if f["actor"] == "못 집음":
        return "N"
    return "C"


def load(root):
    c = json.loads((root / "causes.json").read_text())
    part = {}
    for p in root.glob("part*/trace.json"):
        for w in json.loads(p.read_text())["walls"]:
            part[str(w["seed"])] = (p.parent, w)
    return c, part


def hazards(rows):
    fu = np.array([r["first_unfilled"] for r in rows])
    H = []
    for s in range(16):
        reach = int((fu >= s).sum())
        H.append({"reach": reach, **{g: (sum(1 for r in rows if r["first_unfilled"] == s and r["group"] == g) / reach if reach else 0.0)
                                     for g in "SGCN"}})
    return H


def expect(H, remove, floor3=0.0):
    E, p = 0.0, 1.0
    for s in range(16):
        h = sum(H[s][g] * (1 - remove.get(f"{s}{g}", remove.get(g, 0.0))) for g in "SGCN")
        if s >= 12:
            h = max(h, floor3)
        p *= 1 - h
        E += p
    return round(E, 2), round(p * 100, 1)


SCEN = {
    "지금": {},
    "A 반 (자리 6 검측 실패 50 %)": {"6G": .5}, "A 다 (자리 6 검측 실패 100 %)": {"6G": 1},
    "B 반 (자리 10 공급대 50 %)": {"10S": .5}, "B 다 (자리 10 공급대 100 %)": {"10S": 1},
    "A+B 반": {"6G": .5, "10S": .5}, "A+B 다": {"6G": 1, "10S": 1},
    "② 검측 쪽 전부": {"G": 1}, "① 공급대 전부": {"S": 1},
    "자리 10 ① + ② 전부 반": {"10S": .5, "G": .5}, "자리 10 ① + ② 전부": {"10S": 1, "G": 1},
    "①② 반": {"S": .5, "G": .5}, "①② 전부": {"S": 1, "G": 1}, "①②④ 전부 (③만 남음)": {"S": 1, "G": 1, "N": 1},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out and Path(a.out).exists():
        raise SystemExit(f"{a.out} 이미 있음: 덮어쓰지 않음")
    c, part = load(Path(a.trace))
    rows, handoff = [], {"① 실패": Counter(), "채운 벽돌": Counter()}
    for s, v in c.items():
        if s.startswith("_"):
            continue
        d, w = part[s]
        fu = v["first_unfilled"]
        f = [x for x in v["fails"] if x["slot"] == fu][0] if fu < 16 else None
        rows.append({"seed": int(s), "first_unfilled": fu, "group": group(f) if f else None, "brick": f["brick"] if f else None,
                     "filled": int(sum(w["eval"]["slots_filled"]))})
        src = np.load(d / f"trace_{s}.npz")["src"]

        def prev(S):
            idx = np.where(src[:S] != 0)[0]
            if not len(idx):
                return "없음 (벽 시작)"
            return f"{SRC_NAME[src[idx[-1]]]} {'1.5 s 안' if (S - idx[-1]) * DT <= 1.5 else '1.5 s 뒤'}"
        if f and f["state"] == "공급대 위 다른 곳":
            m = [x for x in f["moves"] if x["src"] == 0]
            handoff["① 실패"][prev(m[0]["s"]) if m else "G1 동작 중 처음 움직임"] += 1
        for b in range(16):
            if w["eval"]["slots_filled"][b]:
                mv = [x for x in v["moves"].get(str(b), []) if x["src"] == 0]
                if mv:
                    handoff["채운 벽돌"][prev(mv[0]["s"])] += 1
    H = hazards(rows)
    res = {"trace": a.trace, "walls": len(rows), "mean_filled": float(np.mean([r["filled"] for r in rows])),
           "mean_first_unfilled": float(np.mean([r["first_unfilled"] for r in rows])),
           "hazard": H, "scenarios": {k: {"floor3_0": expect(H, v), "floor3_5pct": expect(H, v, .05)} for k, v in SCEN.items()},
           "handoff": {k: dict(v) for k, v in handoff.items()}, "rows": rows}
    print(f"벽 {res['walls']} · 평균 채움 {res['mean_filled']:.2f} · 평균 처음 실패 위치 {res['mean_first_unfilled']:.2f}")
    print("자리별 처음 실패 비율 (그 자리까지 온 벽 수):")
    for s, h in enumerate(H):
        print(f"  자리 {s:2d} ({h['reach']:3d}): " + " ".join(f"{g} {h[g]:.3f}" for g in "SGCN"))
    print("시나리오: (기대 채움 장, 16장 완성 %) — 3단 최소 실패율 0 / 5 %")
    for k, v in res["scenarios"].items():
        print(f"  {k:28s} {v['floor3_0']}  {v['floor3_5pct']}")
    print("정책이 벽돌을 처음 움직이기 직전:", res["handoff"])
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()

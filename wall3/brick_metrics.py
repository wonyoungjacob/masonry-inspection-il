"""벽돌 기준 집계 (2026-11-17, 사용자 지시 '벽돌 기준 지표를 더 메인에'). 이미 돈 결과 파일만 읽는다. 시뮬레이션 없음.

벽돌 성공 = 그 차례의 설계 자리가 채워짐 (평가지표 2.4절 조건 1의 자리별 판정: 수평 45 mm · 높이 28.5 mm 안, 기울기 45° 미만).
  정밀도 판정이 아니라 '벽돌이 그 자리에 있나'다. 정밀도는 시방 항목 ① ~ ⑤(벽 단위)가 맡는다.
시도 = 벽마다 16차례 전부(조건마다 100벽 × 16 = 1600). 포기 · 시간 초과로 끝난 차례도 시도로 센다.
구간 = 벽 단위 부트스트랩 95 % (10000번, 시드 0). 같은 벽 안의 벽돌은 서로 독립이 아니라 벽을 단위로 다시 뽑는다.
벽 완성 · 시방 합격(벽 단위)은 그대로 함께 적는다. 평가지표 문서의 주 지표(벽 완성률 · 항목별 연속 오차)를 바꾸지 않는다.

실행: PYTHONPATH=. venv/bin/python -m wall3.brick_metrics --out <파일.json>
"""
import argparse, glob, json
from pathlib import Path

import numpy as np

W = "outputs/wall3"
CONDS = [
    ("Full (ACT + 검측, 11-10)", f"{W}/eval_lf_full_20261110/*/rollout.json"),
    ("Full + G2t (하이브리드 대표, 11-16)", f"{W}/eval_g2t_20261116/*/rollout.json"),
    ("정책 없음 (대본 + 검측, 11-16)", f"{W}/eval_nopolicy_20261116/*/rollout.json"),
] + [(f"현실 제약 {'하이브리드' if c == 'act' else '제어만'} · {n}", f"{W}/eval_real_{c}_{cell}_20261116/{c}_real_{cell}_part*/rollout.json")
     for c in ("act", "script") for cell, n in (("none", "없음"), ("A", "A"), ("B", "B"), ("AB", "A + B"))]
PAIRS = [("Full + G2t (하이브리드 대표, 11-16)", "정책 없음 (대본 + 검측, 11-16)")] + [
    (f"현실 제약 하이브리드 · {n}", f"현실 제약 제어만 · {n}") for n in ("없음", "A", "B", "A + B")]


def load(pattern):
    walls = []
    for f in sorted(glob.glob(pattern)):
        walls += json.loads(Path(f).read_text())["walls"]
    return {w["seed"]: w for w in walls}


def boot(x, n=10000, seed=0):
    """95 % interval of the mean of per-wall values x (wall-level resampling)."""
    r = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = x[r.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} 이미 있음: 덮어쓰지 않음")
    res = {"note": "개발 배치 100벽(700300 ~ 700399) · 최종 평가 아님 · 벽돌 성공 = 설계 자리가 채워짐(평가지표 2.4절 조건 1의 자리별 판정) · "
                   "구간 = 벽 단위 부트스트랩 95 % (10000번, 시드 0)", "conditions": {}, "pairs": {}}
    per = {}
    for name, pat in CONDS:
        d = load(pat)
        if len(d) != 100:
            raise SystemExit(f"{name}: 벽 {len(d)}개 (100이어야 함)")
        filled = {s: int(sum(w["eval"]["slots_filled"])) for s, w in d.items()}
        per[name] = filled
        f = np.array([filled[s] for s in sorted(filled)])
        lo, hi = boot(f / 16 * 100)
        res["conditions"][name] = {"bricks_ok": int(f.sum()), "bricks_tried": 16 * len(f), "brick_pct": float(f.sum() / (16 * len(f)) * 100),
                                   "brick_pct_ci": [lo, hi], "walls_complete": int((f == 16).sum()),
                                   "walls_spec_pass": int(sum(bool(w["eval"].get("spec_pass")) for w in d.values())),
                                   "min_bricks_in_a_wall": int(f.min())}
    for h, c in PAIRS:
        seeds = sorted(per[h])
        assert seeds == sorted(per[c])
        dlt = np.array([per[h][s] - per[c][s] for s in seeds]) / 16 * 100
        lo, hi = boot(dlt)
        res["pairs"][f"{h} − {c}"] = {"brick_pct_diff": float(dlt.mean()), "ci": [lo, hi], "bricks_diff": int(sum(per[h][s] - per[c][s] for s in seeds))}
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))

    print(res["note"] + "\n")
    print(f"{'조건':34s} {'벽돌 성공 / 시도':>16s} {'벽돌 %':>8s} {'[95 %]':>16s} {'완성 벽':>6s} {'시방 합격 벽':>8s}")
    for name, r in res["conditions"].items():
        print(f"{name:34s} {r['bricks_ok']:>9d} / {r['bricks_tried']:<5d} {r['brick_pct']:>7.1f} [{r['brick_pct_ci'][0]:5.1f}, {r['brick_pct_ci'][1]:5.1f}] "
              f"{r['walls_complete']:>6d} {r['walls_spec_pass']:>8d}")
    print("\n짝 차이 (하이브리드 − 제어만, 벽돌 %p)")
    for k, r in res["pairs"].items():
        print(f"  {k}: {r['brick_pct_diff']:+.1f} %p [{r['ci'][0]:+.1f}, {r['ci'][1]:+.1f}] (벽돌 {r['bricks_diff']:+d}개)")


if __name__ == "__main__":
    main()

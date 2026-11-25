"""제어 사다리 비교 (2026-11-10, 01_사기설/실험파이프라인_실패원인제거_20261110.md 4절 판정 기준).

eval_gates 출력 폴더(여럿 가능)의 조건들을 같은 시작값끼리 짝지어 기준 조건과 비교한다.
- 벽당 채운 자리 평균 · 16장 완성 벽 · 시방 합격 · 처음 못 채운 자리 분포
- 기준과의 짝 차이 평균과 95 % 부트스트랩 구간(벽 단위로 다시 뽑기 10000번, 난수 시드 0), 늘어난 벽 · 줄어든 벽 수
결과 수치는 개발 시작값(700300 ~ 700399)이며 최종 평가가 아니다.

실행: PYTHONPATH=. venv/bin/python -m wall3.compare_ladder --evals <dir> [<dir> ...] --base act_all_held [--out <json>]
"""
import argparse, json
from pathlib import Path

import numpy as np


def load(dirs):
    conds = {}
    for d in dirs:
        for p in sorted(Path(d).glob("*_part*/rollout.json")):
            name = p.parent.name.rsplit("_part", 1)[0]
            key = f"{Path(d).name}/{name}"
            for w in json.loads(p.read_text())["walls"]:
                conds.setdefault(key, {})[w["seed"]] = w
    return conds


def stats(walls):
    f = np.array([sum(w["eval"]["slots_filled"]) for w in walls], float)
    first = [int(np.argmin(w["eval"]["slots_filled"])) if not all(w["eval"]["slots_filled"]) else 16 for w in walls]
    return {"walls": len(walls), "filled_mean": round(float(f.mean()), 2), "fill_rate_pct": round(float(f.mean()) / 16 * 100, 1),
            "complete": int((f == 16).sum()), "spec_pass": int(sum(w["eval"]["spec_pass"] for w in walls)),
            "first_unfilled_hist": {str(i): first.count(i) for i in range(17) if first.count(i)}}


def paired(a, b, n_boot=10000):
    d = b - a
    rng = np.random.default_rng(0)
    bs = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(1)
    return {"diff_mean": round(float(d.mean()), 2), "ci95": [round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)],
            "walls_up": int((d > 0).sum()), "walls_down": int((d < 0).sum()), "walls_same": int((d == 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", nargs="+", required=True)
    ap.add_argument("--base", required=True, help="condition name (folder prefix without _part), e.g. act_all_held; "
                                                    "or <eval dir name>/<condition>")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out and Path(a.out).exists():
        raise SystemExit(f"{a.out} 이미 있음: 덮어쓰지 않음")
    conds = load(a.evals)
    base = [k for k in conds if k == a.base or k.endswith("/" + a.base)]
    if len(base) != 1:
        raise SystemExit(f"기준 조건이 하나가 아님: {base} (있는 조건: {sorted(conds)})")
    base = base[0]
    seeds = sorted(conds[base])
    res = {"base": base, "seeds": [seeds[0], seeds[-1], len(seeds)], "conditions": {}}
    fb = np.array([sum(conds[base][s]["eval"]["slots_filled"]) for s in seeds], float)
    for k in sorted(conds):
        miss = [s for s in seeds if s not in conds[k]]
        if miss:
            res["conditions"][k] = {"error": f"시작값 {len(miss)}개 없음"}
            continue
        walls = [conds[k][s] for s in seeds]
        r = stats(walls)
        if k != base:
            r["vs_base"] = paired(fb, np.array([sum(w["eval"]["slots_filled"]) for w in walls], float))
        res["conditions"][k] = r
    print(f"기준 {base}, 시작값 {seeds[0]} ~ {seeds[-1]} ({len(seeds)}벽). 개발 시작값이며 최종 평가 아님.")
    for k, r in res["conditions"].items():
        if "error" in r:
            print(f"  {k}: {r['error']}")
            continue
        v = r.get("vs_base")
        print(f"  {k:45s} {r['filled_mean']:5.2f}장 ({r['fill_rate_pct']:4.1f} %) 완성 {r['complete']:3d} 시방 {r['spec_pass']:3d}"
              + (f" | 기준 대비 {v['diff_mean']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}] 늘어남 {v['walls_up']} 줄어듦 {v['walls_down']}" if v else " | 기준"))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

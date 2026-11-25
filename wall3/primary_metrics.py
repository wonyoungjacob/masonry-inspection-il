"""평가지표 md 2.5절의 주 지표를 역방향 제거 결과에서 다시 계산한다 (2026-11-11). 결과 파일만 읽는다. 시뮬레이션 없음.

- 주 지표 1: 벽 완성(16자리 모두 채움) → 조건별 완성률, Full 대비 짝 차이(%p)와 95 % 구간
- 주 지표 2: 완성된 벽의 시방 항목 ①~⑤ 측정값(mm) → 조건별 중앙값 · 95백분위, Full 대비 짝 차이(두 조건 모두 완성된 벽만)
- 보조: 시방 합격, 벽당 채운 자리(채움 판정은 시방보다 훨씬 느슨: 수평 45 mm · 높이 28.5 mm · 기울기 45°)
- 벽마다 16자리 채움 여부 · 완성 · 시방 합격 · 불합격 항목을 함께 저장한다(자리별 모자이크 그림용).
짝 차이 구간은 compare_ladder.py와 같은 방식(벽 단위로 다시 뽑기 10000번, 난수 시드 0).
수치는 개발 배치 100벽(시작값 700300 ~ 700399)이며 최종 평가가 아니다. 참값 줄은 '참값 (진단)'.

실행: PYTHONPATH=. venv/bin/python -m wall3.primary_metrics [--out outputs/wall3/primary_metrics.json]
"""
import argparse, hashlib, json, os
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[1]
# 결과 폴더. 다른 PC에서는 결과 폴더 복사본을 WALL3_OUT으로 준다.
W = Path(os.environ["WALL3_OUT"]).resolve() if os.environ.get("WALL3_OUT") else SIM / "outputs" / "wall3"


def relkey(p):
    """출처 기록용 경로: 원래 위치면 sim/ 기준(outputs/wall3/...), 복사본이면 outputs/wall3/... 형태로 맞춘다."""
    p = Path(p).resolve()
    try:
        return str(p.relative_to(SIM))
    except ValueError:
        return str(Path("outputs") / "wall3" / p.relative_to(W))

# (줄 이름, 평가 폴더/조건, 종류) — 역방향 제거 표와 같은 줄 + Full 다시 잰 두 번(실행 간 흔들림)
ROWS = [
    ("Full", "eval_lf_full_20261110/act_all_held+k2+k7", "full"),
    ("Full − G2w", "eval_lf_noG2w_20261110/act_all_held+k2", "loo"),
    ("Full − G3", "eval_lf_noG3_20261110/act_g1g2_held+k2+k7", "loo"),
    ("Full − G1", "eval_lf_noG1fix_20261111/act_g2g3_held+k2+k7", "loo"),
    ("Full − G0", "eval_lf_noG0_20261110/act_all_held+k7", "loo"),
    ("Full − G2", "eval_lf_noG2_20261110/act_g1g3_held+k2", "loo"),
    ("Full − H", "eval_lf_noH_20261110/act_all+k2+k7", "loo"),
    ("검측 없음", "eval_lf_none_20261110/act_none", "none"),
    ("참값 (진단)", "eval_lf_truth_20261110/act_all_truth_held+k2+k7", "diag"),
    ("Full 다시 1 (11-10)", "eval_ladder3b_20261110/act_all_held+k2+k7", "rerun"),
    ("Full 다시 2 (11-11)", "eval_lg_noG2c_20261111/act_all_held+k2+k7", "rerun"),
]
# 항목별 스칼라 측정값(mm)과 허용 한계(mm). ①은 2·3단 줄눈의 설계 10 mm 대비 최대 편차와 1단 아래 틈을 따로 둔다.
ITEMS = [
    ("1u", "① 가로줄눈 (2 · 3단, |t − 10|)", 3.0),
    ("1b", "① 1단 아래 틈", 3.0),
    ("2", "② 선 및 면", 6.0),
    ("3", "③ 수직도", 6.0),
    ("4", "④ 평면상 위치", 6.0),
    ("5", "⑤ 벽 두께 (+ 쪽)", 13.0),
]


def item_values(e):
    b = e["1_bed_joint"]
    return {"1u": max(abs(b["upper_min_mm"] - 10.0), abs(b["upper_max_mm"] - 10.0)), "1b": b["base_max_mm"],
            "2": e["2_line_plane"]["value_mm"], "3": e["3_plumb"]["value_mm"], "4": e["4_position"]["value_mm"],
            "5": e["5_thickness"]["max_mm"]}


def item_pass(e):
    return {"1": e["1_bed_joint"]["pass"], "2": e["2_line_plane"]["pass"], "3": e["3_plumb"]["pass"],
            "4": e["4_position"]["pass"], "5": e["5_thickness"]["pass"]}


def boot(d, n_boot=10000):
    rng = np.random.default_rng(0)
    bs = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(1)
    return float(d.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def load(key):
    d, cond = key.split("/", 1)
    walls, files = {}, {}
    for p in sorted((W / d).glob(f"{cond}_part*/rollout.json")):
        files[relkey(p)] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        for w in json.loads(p.read_bytes())["walls"]:
            walls[w["seed"]] = w
    return walls, files


def compute():
    """-> 결과 dict (그림 스크립트도 이 함수를 그대로 부른다)."""
    data = {}
    sources = {}
    for lab, key, kind in ROWS:
        walls, files = load(key)
        sources.update(files)
        data[lab] = (key, kind, walls)
    seeds = sorted(data["Full"][2])
    assert len(seeds) == 100 and seeds[0] == 700300 and seeds[-1] == 700399, seeds[:3]
    for lab, (key, kind, walls) in data.items():
        assert sorted(walls) == seeds, f"{lab}: 시작값이 Full과 다름"

    def per_wall(w):
        e = w["eval"]
        f = [bool(x) for x in e["slots_filled"]]
        comp = all(f)
        ip = item_pass(e)
        return {"seed": w["seed"], "filled": f, "n": sum(f), "complete": comp, "spec": bool(e["spec_pass"]),
                "items": item_values(e) if comp else None, "fail_items": [k for k, v in ip.items() if not v] if comp else None}

    full = {w["seed"]: per_wall(w) for w in data["Full"][2].values()}
    res = {"note": "개발 배치 100벽(시작값 700300 ~ 700399) · 최종 평가 아님. 짝 차이 구간: 벽 단위 부트스트랩 10000번, 시드 0",
           "items": [{"key": k, "name": n, "limit_mm": t} for k, n, t in ITEMS], "rows": [], "sources": sources}
    for lab, (key, kind, walls) in data.items():
        pw = [per_wall(walls[s]) for s in seeds]
        n = np.array([p["n"] for p in pw], float)
        comp = np.array([p["complete"] for p in pw], float)
        spec = np.array([p["spec"] for p in pw], float)
        row = {"label": lab, "key": key, "kind": kind, "walls": len(pw), "filled_mean": float(n.mean()),
               "complete": int(comp.sum()), "spec_pass": int(spec.sum()),
               "complete_not_spec": int(((comp == 1) & (spec == 0)).sum()),
               "fill_hist": {str(k): int((n == k).sum()) for k in range(17) if (n == k).sum()}}
        cw = [p for p in pw if p["complete"]]
        row["items"] = {}
        for k, nm, t in ITEMS:
            v = np.array([p["items"][k] for p in cw], float)
            row["items"][k] = None if len(v) == 0 else {"median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
                                                         "max": float(v.max()), "n": len(v), "over_limit": int((v > t).sum())}
        row["fail_item_counts"] = {}
        for p in cw:
            for k in p["fail_items"]:
                row["fail_item_counts"][k] = row["fail_item_counts"].get(k, 0) + 1
        if lab != "Full":
            fc = np.array([full[s]["complete"] for s in seeds], float)
            fs = np.array([full[s]["spec"] for s in seeds], float)
            fn = np.array([full[s]["n"] for s in seeds], float)
            m, ci = boot((comp - fc) * 100)
            row["vs_full"] = {"complete_pp": m, "complete_ci_pp": ci,
                              "spec_pp": boot((spec - fs) * 100)[0], "spec_ci_pp": boot((spec - fs) * 100)[1],
                              "filled": boot(n - fn)[0], "filled_ci": boot(n - fn)[1]}
            both = [s for s in seeds if full[s]["complete"] and pw[seeds.index(s)]["complete"]]
            row["vs_full"]["items_pairs"] = len(both)
            row["vs_full"]["items"] = {}
            if len(both) >= 5:
                for k, nm, t in ITEMS:
                    d = np.array([pw[seeds.index(s)]["items"][k] - full[s]["items"][k] for s in both], float)
                    m2, ci2 = boot(d)
                    row["vs_full"]["items"][k] = {"diff_mean": m2, "ci95": ci2}
        row["walls_detail"] = [{"seed": p["seed"], "filled": p["filled"], "complete": p["complete"], "spec": p["spec"],
                                "items": p["items"], "fail_items": p["fail_items"]} for p in pw]
        res["rows"].append(row)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(W / "primary_metrics.json"))
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} 이미 있음: 덮어쓰지 않음")
    res = compute()
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(res["note"])
    print(f"{'조건':<18}{'채운자리':>8}{'완성':>6}{'시방':>6}{'완성−불합격':>8}  완성률 차이 %p [95 %]       ②p95  ③p95  ④p95  불합격 항목")
    for r in res["rows"]:
        v = r.get("vs_full")
        cc = "기준" if v is None else f"{v['complete_pp']:+6.1f} [{v['complete_ci_pp'][0]:+6.1f}, {v['complete_ci_pp'][1]:+6.1f}]"
        it = r["items"]
        p95 = "  ".join(f"{it[k]['p95']:5.2f}" if it[k] else "    –" for k in ("2", "3", "4"))
        print(f"{r['label']:<18}{r['filled_mean']:8.2f}{r['complete']:6d}{r['spec_pass']:6d}{r['complete_not_spec']:8d}  {cc:<28} {p95}  {r['fail_item_counts']}")
    print("만듦:", out)


if __name__ == "__main__":
    main()

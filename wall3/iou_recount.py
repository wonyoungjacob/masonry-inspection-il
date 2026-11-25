"""벽돌별 3D IoU 재집계 (2026-11-25, 최종보고서 v1에 쓸 값 확인). 이미 돈 결과 파일(rollout.json)만 읽는다. 시뮬레이션 없음.

자리별 IoU는 롤아웃 때 채점 코드(wall3/spec_metrics.py 3절 부분, 10-15 이후 그대로)가 시뮬레이터 참값 자세로 계산해
rollout.json의 eval.iou에 남긴 값을 그대로 쓴다. 채운 자리(eval.slots_filled)만 계산돼 있고 빈 자리는 0이다.
  - IoU 평균: 1600자리(빈 자리 0) · IoU 중앙값: 채운 자리만
  - 통과 벽돌: 자리 IoU ≥ 기준. 온장 · 반장 기준이 다르다(사용자 지시 11-25).
      기준 A (하한) 온장 0.749 · 반장 0.702 / 기준 B (상한) 온장 0.959 · 반장 0.953 / 참고 모두 0.875
  - 대응 차이: 같은 시작값(벽)끼리 뺀 값. 구간 = 벽 단위 부트스트랩 95 % (10000번, 시드 0).
    통과 수 차이는 100벽 합으로 적는다(벽당 평균의 구간 × 100).
벽돌의 최종 회전은 rollout.json에 없다(중심 오차 final_err_mm만 있음). 그래서 모서리 8개 판정은 하지 않는다.

실행: PYTHONPATH=. venv/bin/python -m wall3.iou_recount --out outputs/wall3/iou_recount.json
"""
import argparse, glob, hashlib, json
from pathlib import Path

import numpy as np

W = "outputs/wall3"
SEEDS = list(range(700300, 700400))
THR = {"A": (0.749, 0.702), "B": (0.959, 0.953), "ref": (0.875, 0.875)}     # (온장, 반장)

# (이름, 묶음, rollout.json 패턴, 기존 보고 값 (채운 벽돌, 완성 벽, 시방 합격 벽))
CONDS = [
    ("① 처음 정책 + G1 · G2 · G3 (10-22)", "설계", f"{W}/eval_gates_20261022/act_all_part*/rollout.json", (208, 0, 0)),
    ("② 시연 통일 재학습 (10-26)", "설계", f"{W}/eval_gates_v2_20261026/act_all_part*/rollout.json", (538, 0, 0)),
    ("③ + H (11-04)", "설계", f"{W}/eval_gates_info_20261104/act_all_held_part*/rollout.json", (855, 12, 11)),
    ("④ + G0 (11-10)", "설계", f"{W}/eval_ladder1_20261110/act_all_held+k2_part*/rollout.json", (1330, 59, 59)),
    ("⑤ Full (11-10)", "설계", f"{W}/eval_lf_full_20261110/act_all_held+k2+k7_part*/rollout.json", (1365, 60, 60)),
    ("⑥ Full + G2t (11-16)", "설계", f"{W}/eval_g2t_20261116/act_all_held+k2+k7+g2t_part*/rollout.json", (1462, 83, 79)),
    ("⑦ 끝 구성 (11-17)", "설계", f"{W}/eval_real_act_none_20261116/act_real_none_part*/rollout.json", (1600, 100, 96)),
    ("검측 없음 (정책만, 11-10)", "설계", f"{W}/eval_lf_none_20261110/act_none_part*/rollout.json", (200, 0, 0)),
    ("Full − H", "역방향", f"{W}/eval_lf_noH_20261110/act_all+k2+k7_part*/rollout.json", (182, 0, 0)),
    ("Full − G2", "역방향", f"{W}/eval_lf_noG2_20261110/act_g1g3_held+k2_part*/rollout.json", (768, 17, 14)),
    ("Full − G0", "역방향", f"{W}/eval_lf_noG0_20261110/act_all_held+k7_part*/rollout.json", (843, 9, 9)),
    ("Full − G1 (11-11 재측정)", "역방향", f"{W}/eval_lf_noG1fix_20261111/act_g2g3_held+k2+k7_part*/rollout.json", (900, 0, 0)),
    ("Full − G3", "역방향", f"{W}/eval_lf_noG3_20261110/act_g1g2_held+k2+k7_part*/rollout.json", (1067, 23, 2)),
    ("Full − G2w", "역방향", f"{W}/eval_lf_noG2w_20261110/act_all_held+k2_part*/rollout.json", (1340, 60, 60)),
    ("끝 구성 × 오차 없음", "오차", f"{W}/eval_real_act_none_20261116/act_real_none_part*/rollout.json", (1600, 100, 96)),
    ("끝 구성 × A", "오차", f"{W}/eval_real_act_A_20261116/act_real_A_part*/rollout.json", (1564, 94, 93)),
    ("끝 구성 × B", "오차", f"{W}/eval_real_act_B_20261116/act_real_B_part*/rollout.json", (1600, 100, 100)),
    ("끝 구성 × A + B", "오차", f"{W}/eval_real_act_AB_20261116/act_real_AB_part*/rollout.json", (1552, 92, 92)),
]
_N = [c[0] for c in CONDS]
PAIRS = ([(_N[i + 1], _N[i], "설계: 직전 단계 대비") for i in range(6)]
         + [(n, "⑤ Full (11-10)", "역방향: Full 대비") for n in _N[8:14]]
         + [(n, "끝 구성 × 오차 없음", "오차: 오차 없음 대비 (참고)") for n in _N[15:18]])


def sha16(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def load(pattern):
    """{seed: wall} from every file matching pattern, and [(path, sha16)]. Exactly the 100 seeds 700300 ~ 700399, each once."""
    files = sorted(glob.glob(pattern))
    walls, seen = {}, []
    for f in files:
        for w in json.loads(Path(f).read_text())["walls"]:
            if w["seed"] in walls:
                raise ValueError(f"{pattern}: 시작값 {w['seed']} 중복")
            walls[w["seed"]] = w
        seen.append((f, sha16(f)))
    if sorted(walls) != SEEDS:
        raise ValueError(f"{pattern}: 시작값이 700300 ~ 700399 100개가 아님 ({len(walls)}개)")
    return walls, seen


def thresholds(half, which):
    """Per-slot IoU threshold array (16,) for criterion which, given half-brick flags (16,)."""
    full_t, half_t = THR[which]
    return np.where(np.asarray(half, bool), half_t, full_t)


def per_wall(w, half):
    """Per-wall arrays from one wall record: filled (16,) bool, iou (16,) float (0 where empty)."""
    ev = w["eval"]
    filled = np.asarray(ev["slots_filled"], bool)
    iou = np.asarray(ev["iou"], float)
    if filled.shape != (16,) or iou.shape != (16,):
        raise ValueError(f"벽 {w['seed']}: 자리 수가 16이 아님")
    if np.any(iou[~filled] != 0.0):
        raise ValueError(f"벽 {w['seed']}: 빈 자리 IoU가 0이 아님")
    return filled, iou


def boot(x, n=10000, seed=0):
    """95 % interval of the mean of per-wall values x (wall-level resampling)."""
    r = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = x[r.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarize(walls, half):
    """Condition summary and per-wall vectors (seed order)."""
    F, I = [], []
    for s in SEEDS:
        f, i = per_wall(walls[s], half)
        F.append(f); I.append(i)
    F, I = np.array(F), np.array(I)
    vec = {"filled": F.sum(1)}
    out = {"bricks_filled": int(F.sum()), "walls_complete": int((F.sum(1) == 16).sum()),
           "walls_spec_pass": int(sum(bool(walls[s]["eval"].get("spec_pass")) for s in SEEDS)),
           "iou_mean_1600": float(I.mean()),
           "iou_median_filled": float(np.median(I[F])) if F.any() else None}
    for k in THR:
        p = (I >= thresholds(half, k)[None]) & F
        vec[f"pass_{k}"] = p.sum(1)
        out[f"pass_{k}"] = int(p.sum())
        out[f"pass_{k}_full"] = int(p[:, ~np.asarray(half, bool)].sum())
        out[f"pass_{k}_half"] = int(p[:, np.asarray(half, bool)].sum())
    return out, vec


def pair(vh, vc, key, scale):
    d = np.asarray(vh[key], float) - np.asarray(vc[key], float)
    lo, hi = boot(d)
    return {"diff": float(d.mean() * scale), "ci": [lo * scale, hi * scale]}


def threshold_check():
    """IoU of a design box against itself moved by (dx, dy, dz) mm, for the full and the half brick (참고: 기준 값의 근거 확인)."""
    from wall3.volume import box, intersection
    from wall3.config import L_FULL, L_HALF, W as BW, H as BH
    res = {}
    for name, L in (("온장", L_FULL), ("반장", L_HALF)):
        hs = np.array([L, BW, BH]) / 2
        a = box(np.zeros(3), np.eye(3), hs)
        for d in ((6, 6, 3), (0, 6, 0), (6, 0, 0), (0, 0, 3)):
            b = box(np.array(d) / 1000.0, np.eye(3), hs)
            v = intersection([a, b])
            res[f"{name} 이동 {d} mm"] = float(v / (a[4] + b[4] - v))
    return res


def main():
    from wall3.config import design_slots
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} 이미 있음: 덮어쓰지 않음")
    half = [bool(s["half"]) for s in design_slots()]
    res = {"note": "개발 배치 100벽(700300 ~ 700399) · 최종 평가 아님 · 저장된 rollout.json만 읽음(새 롤아웃 없음) · "
                   "IoU = rollout 때 채점 코드가 참값 자세로 계산한 자리별 값(빈 자리 0) · 구간 = 벽 단위 부트스트랩 95 % (10000번, 시드 0)",
           "thresholds": {k: {"온장": v[0], "반장": v[1]} for k, v in THR.items()},
           "half_slots": [k for k, h in enumerate(half) if h], "threshold_check": threshold_check(),
           "conditions": {}, "pairs": {}, "files": {}}
    vecs = {}
    for name, group, pat, exp in CONDS:
        walls, files = load(pat)
        s, v = summarize(walls, half)
        s["group"] = group
        s["expected"] = {"bricks_filled": exp[0], "walls_complete": exp[1], "walls_spec_pass": exp[2]}
        s["matches_reported"] = (s["bricks_filled"], s["walls_complete"], s["walls_spec_pass"]) == exp
        res["conditions"][name] = s
        res["files"][name] = [{"path": f, "sha16": h} for f, h in files]
        vecs[name] = v
    for h, c, kind in PAIRS:
        res["pairs"][f"{h} − {c}"] = {"kind": kind,
                                      "filled_per_wall": pair(vecs[h], vecs[c], "filled", 1.0),
                                      **{f"pass_{k}_sum": pair(vecs[h], vecs[c], f"pass_{k}", 100.0) for k in THR}}
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))

    print(res["note"] + "\n")
    print(f"{'조건':30s} {'채운':>5s} {'보고값':>4s} {'완성':>4s} {'시방':>4s} {'IoU평균':>8s} {'IoU중앙(채움)':>12s} {'A':>5s} {'B':>5s} {'0.875':>5s}")
    for name, r in res["conditions"].items():
        med = f"{r['iou_median_filled']:.4f}" if r["iou_median_filled"] is not None else "-"
        print(f"{name:30s} {r['bricks_filled']:>5d} {'같음' if r['matches_reported'] else '다름':>4s} {r['walls_complete']:>4d} {r['walls_spec_pass']:>4d} "
              f"{r['iou_mean_1600']:>8.4f} {med:>12s} {r['pass_A']:>5d} {r['pass_B']:>5d} {r['pass_ref']:>5d}")
    print("\n대응 차이 (벽당 채운 자리 · 통과 수는 100벽 합)")
    for k, r in res["pairs"].items():
        f_, pa, pb = r["filled_per_wall"], r["pass_A_sum"], r["pass_B_sum"]
        print(f"  [{r['kind']}] {k}: 채운 {f_['diff']:+.2f} [{f_['ci'][0]:+.2f}, {f_['ci'][1]:+.2f}] · "
              f"A {pa['diff']:+.0f} [{pa['ci'][0]:+.0f}, {pa['ci'][1]:+.0f}] · B {pb['diff']:+.0f} [{pb['ci'][0]:+.0f}, {pb['ci'][1]:+.0f}]")
    print("\n기준 값 확인 (설계 상자를 옮긴 IoU):", {k: round(v, 4) for k, v in res["threshold_check"].items()})


if __name__ == "__main__":
    main()

"""진단(결과 아님, 2026-11-25): 끝 구성 × A에서 기준 B(IoU ≥ 0.959 · 0.953) 통과가 줄어든 것이 벽 등록 오차(A1) 때문인가.
결과 파일만 읽는다. 시뮬레이션 없음.

A1은 벽마다 '믿는 도면'을 (wall_dx, wall_dy, wall_dth)만큼 옮긴다(rollout.json의 perturb). 검측은 믿는 도면에 놓으므로
벽돌이 참 설계 자리에서 그만큼 밀리면, 참 설계 자리와 잰 IoU가 내려간다. 확인하는 것:
  ① 벽마다 등록 오차 크기 |(wall_dx, wall_dy)|와 그 벽의 기준 B 통과 수 · 채운 자리 IoU 중앙값의 순위 상관
  ② 채운 벽돌의 중심 오차(final_err_mm, 참 설계 자리 기준)와 등록 오차 방향의 관계: 벽마다 중심 오차 x · y 중앙값 대 wall_dx · wall_dy
실행: PYTHONPATH=. venv/bin/python -m experiments.diag_iou_A_20261125 --out outputs/wall3/diag_iou_A_20261125.json
"""
import argparse, json
from pathlib import Path

import numpy as np

from wall3.config import design_slots
from wall3.iou_recount import CONDS, SEEDS, load, per_wall, thresholds


def rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def spearman(x, y):
    return float(np.corrcoef(rank(np.asarray(x)), rank(np.asarray(y)))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} 이미 있음: 덮어쓰지 않음")
    half = [bool(s["half"]) for s in design_slots()]
    tB = thresholds(half, "B")
    pats = {c[0]: c[2] for c in CONDS}
    res = {"note": "진단 · 결과 아님 · 개발 배치(700300 ~ 700399)"}
    for name in ("끝 구성 × A", "끝 구성 × A + B"):
        walls, _ = load(pats[name])
        off, rot, passB, med, ex, ey, dx, dy = [], [], [], [], [], [], [], []
        for s in SEEDS:
            w = walls[s]
            p = w["perturb"]
            f, iou = per_wall(w, half)
            off.append(np.hypot(p["wall_dx"], p["wall_dy"]) * 1000)
            rot.append(abs(np.degrees(p["wall_dth"])))
            passB.append(int(((iou >= tB) & f).sum()))
            med.append(float(np.median(iou[f])) if f.any() else 0.0)
            e = np.array([b["final_err_mm"] for b in w["bricks"] if f[b["slot"]]])
            ex.append(float(np.median(e[:, 0]))); ey.append(float(np.median(e[:, 1])))
            dx.append(p["wall_dx"] * 1000); dy.append(p["wall_dy"] * 1000)
        off, passB = np.array(off), np.array(passB)
        small, large = off < 1.0, off >= 2.0
        res[name] = {
            "spearman_offset_vs_passB": spearman(off, passB), "spearman_offset_vs_median_iou": spearman(off, med),
            "spearman_rot_vs_passB": spearman(rot, passB),
            "passB_per_wall_mean_offset_lt_1mm": {"walls": int(small.sum()), "mean": float(passB[small].mean()) if small.any() else None},
            "passB_per_wall_mean_offset_ge_2mm": {"walls": int(large.sum()), "mean": float(passB[large].mean()) if large.any() else None},
            "corr_center_err_x_vs_wall_dx": float(np.corrcoef(ex, dx)[0, 1]), "corr_center_err_y_vs_wall_dy": float(np.corrcoef(ey, dy)[0, 1]),
            "slope_center_err_x_vs_wall_dx": float(np.polyfit(dx, ex, 1)[0]), "slope_center_err_y_vs_wall_dy": float(np.polyfit(dy, ey, 1)[0]),
            "offset_mm": {"median": float(np.median(off)), "max": float(off.max())}}
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

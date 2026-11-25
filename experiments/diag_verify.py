"""원인 추적 보고서(실패원인_추적_20261028.md) 검증 (2026-10-29). 진단용이며 평가 결과가 아니다.

1. 처음 실패 뒤 차례들: 정책이 '그 차례 벽돌'에 손을 가져가나, 빈 공급칸 근처에 머무나 (diag_trace 100벽).
   diag_trace_causes의 '빈 공급 위치 집기'는 집게를 오므린 곳에서 가장 가까운 처음 공급칸을 거리 제한 없이 골랐다. 여기서는 5 cm 안만 센다.
2. 평가가 '채움'으로 셌지만 높이가 +20 ~ 28.5 mm 떠 있는 자리(이웃 벽돌에 걸친 모습. 평가 기준 |dz| ≤ 28.5 mm 바로 안).
3. 벽마다 벽돌 1~6이 처음 들린(3 cm) 차례 (diag_video 실행).

실행: venv/bin/python -m experiments.diag_verify outputs/wall3/diag_trace_20261028 \
        "outputs/wall3/eval_gates_v2_20261026/act_all_part*/rollout.json" outputs/wall3/diag_video_20261029
"""
import collections, glob, json, sys
from pathlib import Path

import numpy as np

SRC_NAME = {0: "정책", 1: "G1", 2: "G2·G3", 3: "마무리"}


def after_failure(trace_root):
    n = collections.Counter()
    for p in sorted(Path(trace_root).glob("part*/trace.json")):
        for w in json.loads(p.read_text())["walls"]:
            z = np.load(p.parent / f"trace_{w['seed']}.npz")
            P, tcp, k, src = z["pose"], z["tcp"], z["k"], z["src"]
            first = next((j for j in range(16) if not w["eval"]["slots_filled"][j]), 16)
            sup = P[0, :, :3]
            for kk in range(16):
                idx = np.where(k == kk)[0]
                if len(idx) == 0:
                    continue
                g = "뒤" if kk > first else "앞"
                n[g, "차례 수"] += 1
                n[g, "어떤 벽돌이든 2 cm 넘게 들림"] += any((P[idx, b, 2] - P[idx[0], b, 2]).max() > 0.02 for b in range(16))
                low = idx[(src[idx] == 0) & (tcp[idx, 2] < 0.43)]          # policy steps with the hand down at supply height
                own = np.linalg.norm(P[idx[0], kk, :3] - sup[kk]) < 0.01
                n[g, "차례 시작 때 제 벽돌이 공급칸에 그대로"] += own
                if own:
                    n[g, "  그중 손이 제 벽돌 5 cm 안까지 내려감"] += bool(len(low)) and bool(
                        (np.linalg.norm(tcp[low, :2] - P[idx[0], kk, :2], axis=1) < 0.05).any())
                empty = sup[np.linalg.norm(P[idx[0], :, :3] - sup, axis=1) >= 0.01]
                if len(low) and len(empty):
                    de = np.linalg.norm(tcp[low, None, :2] - empty[None, :, :2], axis=2).min(1)
                    n[g, "손이 빈 공급칸 5 cm 안에 낮게 1 s 이상"] += (de < 0.05).sum() >= 20
    return n


def perched_filled(eval_glob):
    rows, shifted = [], 0
    for p in sorted(glob.glob(eval_glob)):
        for w in json.loads(Path(p).read_text())["walls"]:
            f = w["eval"]["slots_filled"]
            first = next((j for j in range(16) if not f[j]), 16)
            hits = [b["slot"] for b in w["bricks"] if f[b["slot"]] and 20 < b["final_err_mm"][2] < 28.5 and b["final_xy_err_mm"] < 45]
            rows += [(w["seed"], j + 1) for j in hits]
            shifted += bool(hits) and min(hits) < first
    return rows, shifted


def first_lift(video_root):
    out = {}
    p = Path(video_root) / "part0/trace.json"
    for w in json.loads(p.read_text())["walls"]:
        z = np.load(p.parent / f"trace_{w['seed']}.npz")
        P, k, src, t = z["pose"], z["k"], z["src"], z["t"] * 0.05
        row = []
        for b in range(6):
            i = np.where(P[:, b, 2] - P[0, b, 2] > 0.03)[0]
            row.append(f"{b + 1}: 차례 {k[i[0]] + 1}({SRC_NAME[int(src[i[0]])]}, t {t[i[0]]:.0f} s)" if len(i) else f"{b + 1}: 안 들림")
        out[w.get("name", w["seed"])] = (w["seed"], row, w.get("same_as_eval"))
    return out


def main():
    trace_root, eval_glob, video_root = sys.argv[1:4]
    n = after_failure(trace_root)
    print("1. 차례별 행동 (앞 = 처음 못 채운 자리의 차례까지, 뒤 = 그 뒤 차례)")
    for key in sorted({x[1] for x in n} - {"차례 수"}):
        print(f"   {key}: 앞 {n['앞', key]}/{n['앞', '차례 수']}, 뒤 {n['뒤', key]}/{n['뒤', '차례 수']}")
    rows, shifted = perched_filled(eval_glob)
    print(f"2. 평가가 채움으로 셌지만 +20 ~ 28.5 mm 떠 있는 자리: {len(rows)}곳, {len({s for s, _ in rows})}벽,"
          f" 자리 {sorted(collections.Counter(j for _, j in rows).items())}; 그 때문에 '처음 못 채운 자리'가 뒤로 밀린 벽 {shifted}")
    print("3. 벽돌이 처음 들린 차례 (diag_video)")
    for name, (seed, row, same) in first_lift(video_root).items():
        print(f"   {name} {seed} 평가와 같음 {same and same['slots_filled']}: " + " | ".join(row))


if __name__ == "__main__":
    main()

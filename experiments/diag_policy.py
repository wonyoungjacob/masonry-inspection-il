"""정책 진단 (2026-10-22): 셋째 벽돌부터 못 집는 이유 가리기. 시뮬레이션 없이 학습 데이터셋으로만 본다.

1) 학습 장면 그대로(열린 고리): 데이터셋의 벽돌(에피소드)마다 첫 장면(철수 자세)을 정책에 넣어 예측한 행동 묶음을 실제 대본 행동과
   비교한다. 특히 어깨 회전(관절 0: 공급대 왼쪽/오른쪽 결정)의 0.5 · 1.0 · 1.5 s 뒤 값. 여기서 이미 틀리면 학습 · 조건 문제,
   맞으면 실행 중 누적 오차(분포 이탈) 문제.
2) 목표 정보 바꿔 넣기: 같은 장면에 목표 정보(observation.environment_state)만 다른 자리 것으로 바꿔 넣어 예측이 따라 바뀌는지.
   안 바뀌면 정책이 목표 정보를 거의 쓰지 않는다.

실행: source venv/bin/activate && python -m experiments.diag_policy --ckpt <pretrained_model> --out <json> [--walls 20]
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
HS = (10, 20, 30)          # control steps ahead (0.5, 1.0, 1.5 s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="data/wall3_main_20261019")
    ap.add_argument("--walls", type=int, default=20)
    ap.add_argument("--swap-walls", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from eval_policy import load_policy
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    policy, cfg, pre, post = load_policy(a.ckpt, "cuda", None, None)
    root = Path(a.root)
    eps = json.loads((root / "meta" / "wall3_episodes.json").read_text())
    walls = sorted({e["wall_index"] for e in eps})[: a.walls]
    use = [e for e in eps if e["wall_index"] in walls]
    ds = LeRobotDataset(f"local/{root.name}", root=str(root), episodes=[e["episode_index"] for e in use], video_backend="torchcodec")
    hf = ds._ensure_reader().hf_dataset                    # actions only, no video decoding
    # LeRobotDataset(episodes=...) indexes rows within the subset; map episode_index -> first row in this subset
    first_row, row = {}, 0
    for e in use:
        first_row[e["episode_index"]] = row
        row += e["frames"]
    is_diff = cfg.type == "diffusion"
    n_obs = getattr(cfg, "n_obs_steps", 1)
    targets = {e["slot"]: np.asarray(e["target"], np.float32) for e in use}

    def chunk(item, target=None, seed=0):
        obs = {}
        for k in cfg.input_features:
            v = item[k]
            if k == "observation.environment_state" and target is not None:
                v = torch.from_numpy(target)
            v = v.unsqueeze(0)
            obs[k] = v
        obs["task"] = [item["task"]]
        b = pre(obs)
        policy.reset()
        if is_diff:
            for k in list(b.keys()):
                if isinstance(b[k], torch.Tensor) and k.startswith("observation"):
                    b[k] = b[k].unsqueeze(1).repeat_interleave(n_obs, dim=1)
            g = torch.Generator(device="cuda").manual_seed(seed)
            noise = torch.randn((1, cfg.horizon, cfg.output_features["action"].shape[0]), generator=g, device="cuda")
            act = policy.predict_action_chunk(b, noise=noise)
        else:
            act = policy.predict_action_chunk(b)
        act = post(act)
        return act.squeeze(0).detach().cpu().numpy()

    rows = []
    with torch.inference_mode():
        for e in use:
            r0 = first_row[e["episode_index"]]
            item = ds[r0]
            assert int(item["episode_index"]) == e["episode_index"] and int(item["frame_index"]) == 0, (e, item["episode_index"])
            pred = chunk(item)
            true = np.stack([np.asarray(hf[r0 + t]["action"], np.float32) for t in range(max(HS) + 1) if t < e["frames"]])
            rec = {"wall": e["wall_index"], "seed": e["seed"], "slot": e["slot"]}
            for h in HS:
                if h < len(pred) and h < len(true):
                    rec[f"pan_true_{h}"] = float(true[h, 0]); rec[f"pan_pred_{h}"] = float(pred[h, 0])
                    rec[f"joint_err_{h}"] = float(np.abs(pred[h, :6] - true[h, :6]).max())
            rows.append(rec)
        # target swap on a few walls: the scene of slot k, the target of slot j
        swap = []
        for e in use:
            if e["wall_index"] not in walls[: a.swap_walls]:
                continue
            item = ds[first_row[e["episode_index"]]]
            for j, tj in targets.items():
                pred = chunk(item, tj)
                swap.append({"wall": e["wall_index"], "scene_slot": e["slot"], "target_slot": j, "pan_pred_30": float(pred[min(30, len(pred) - 1), 0])})
    # summaries
    by_slot = {}
    for k in range(16):
        r = [x for x in rows if x["slot"] == k]
        if not r:
            continue
        by_slot[k] = {h: {"pan_true_mean": float(np.mean([x[f"pan_true_{h}"] for x in r])), "pan_pred_mean": float(np.mean([x[f"pan_pred_{h}"] for x in r])),
                          "joint_err_max_median_rad": float(np.median([x[f"joint_err_{h}"] for x in r]))} for h in HS if f"pan_true_{h}" in r[0]}
    M = np.full((16, 16), np.nan)
    for k in range(16):
        for j in range(16):
            v = [s["pan_pred_30"] for s in swap if s["scene_slot"] == k and s["target_slot"] == j]
            if v:
                M[k, j] = float(np.mean(v))
    true30 = np.array([by_slot[j][30]["pan_true_mean"] if j in by_slot and 30 in by_slot[j] else np.nan for j in range(16)])
    follow = {}
    for k in range(16):
        row_ = M[k]
        if np.isfinite(row_).sum() >= 8:
            follow[k] = float(np.corrcoef(row_[np.isfinite(row_)], true30[np.isfinite(row_)])[0, 1])
    res = {"ckpt": a.ckpt, "policy": cfg.type, "walls": len(walls), "by_slot": by_slot, "rows": rows,
           "swap_pan30_matrix": M.round(3).tolist(), "true_pan30_by_slot": true30.round(3).tolist(),
           "swap_follow_corr_by_scene_slot": follow,
           "swap_spread_by_scene_slot": {k: float(np.nanmax(M[k]) - np.nanmin(M[k])) for k in range(16) if np.isfinite(M[k]).any()},
           "true_spread": float(np.nanmax(true30) - np.nanmin(true30))}
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(cfg.type)
    for k, v in by_slot.items():
        h = v.get(30, {})
        print(f"slot {k:2d}: pan@1.5s true {h.get('pan_true_mean', float('nan')):+.2f} pred {h.get('pan_pred_mean', float('nan')):+.2f}  "
              f"joint err median {h.get('joint_err_max_median_rad', float('nan')):.3f} rad   swap: follow corr {follow.get(k, float('nan')):+.2f} "
              f"spread {res['swap_spread_by_scene_slot'].get(k, float('nan')):.2f} (true spread {res['true_spread']:.2f})")


if __name__ == "__main__":
    main()

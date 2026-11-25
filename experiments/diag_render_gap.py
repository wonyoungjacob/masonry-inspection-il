"""정책 진단 3 (2026-10-22): 같은 장면, 입력 영상만 다르게. 학습 데이터 영상(압축 후 복원) vs 시뮬레이터 직접 렌더.

학습 시작값의 벽을 대본으로 벽돌 k 직전까지 다시 쌓아(학습 때와 같은 장면) 벽돌 k 첫 장면에서
  (a) 데이터셋 첫 프레임(영상 압축 · 복원본) 입력,
  (b) 시뮬레이터가 지금 렌더한 입력,
  (c) (b)의 영상 + (a)의 상태 · 목표
으로 정책 예측을 각각 구하고, 대본이 집게를 닫는 순간(데이터셋 행동)의 손끝 위치(정기구학, 명령값 기준)를 비교한다.
영상 픽셀 차이(평균 절대값, 0 ~ 255)도 남긴다. 진단용.

실행: PYTHONPATH=. MUJOCO_GL=egl venv/bin/python -m experiments.diag_render_gap --ckpt <pretrained_model> --walls 0 1 2 --slots 0 1 2 3 --out <json>
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import mujoco

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
CAMS = ("front", "top", "wrist")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="data/wall3_main_20261019")
    ap.add_argument("--walls", type=int, nargs="+", required=True, help="wall_index in the dataset")
    ap.add_argument("--slots", type=int, nargs="+", default=list(range(16)))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from eval_policy import load_policy
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from wall3.config import scene_config
    from wall3.scene import WallEnv
    from wall3.expert import WallExpert
    from wall3.rollout import obs_of
    policy, cfg, pre, post = load_policy(a.ckpt, "cuda", None, None)
    cfg_in = set(cfg.input_features)
    is_diff = cfg.type == "diffusion"
    root = Path(a.root)
    eps = json.loads((root / "meta" / "wall3_episodes.json").read_text())
    use = [e for e in eps if e["wall_index"] in a.walls and e["slot"] in a.slots]
    ds = LeRobotDataset(f"local/{root.name}", root=str(root), episodes=[e["episode_index"] for e in use], video_backend="torchcodec")
    hf = ds._ensure_reader().hf_dataset
    first_row, row = {}, 0
    for e in use:
        first_row[e["episode_index"]] = row
        row += e["frames"]
    env = WallEnv(scene_config("main_v1"))
    ex = WallExpert(env)
    m = env.model
    fk = mujoco.MjData(m)

    def tcp_of(ctrl):
        fk.qpos[:] = env.data.qpos
        fk.qpos[env.arm_qadr] = ctrl[:6]
        mujoco.mj_kinematics(m, fk)
        return fk.site_xpos[env.tcp_sid].copy()

    def predict(obs):
        b = pre(dict(obs))
        policy.reset()
        if is_diff:
            for k in list(b.keys()):
                if isinstance(b[k], torch.Tensor) and k.startswith("observation"):
                    b[k] = b[k].unsqueeze(1).repeat_interleave(cfg.n_obs_steps, dim=1)
            g = torch.Generator(device="cuda").manual_seed(0)
            noise = torch.randn((1, cfg.horizon, cfg.output_features["action"].shape[0]), generator=g, device="cuda")
            act = policy.predict_action_chunk(b, noise=noise)
        else:
            act = policy.predict_action_chunk(b)
        return post(act).squeeze(0).detach().cpu().numpy()

    res = []
    with torch.inference_mode():
        for e in sorted(use, key=lambda x: (x["wall_index"], x["slot"])):
            k = e["slot"]
            r0 = first_row[e["episode_index"]]
            item = ds[r0]
            assert int(item["episode_index"]) == e["episode_index"] and int(item["frame_index"]) == 0
            act_true = np.stack([np.asarray(hf[r0 + t]["action"], np.float32) for t in range(e["frames"])])
            ic = int(np.argmax(act_true[:, 6] > 0.15))                     # expert starts closing the fingers
            r = ex.run(e["seed"], settle_s=0.0, max_bricks=k)
            assert not r.get("fail")
            o_sim = obs_of(env, k, cfg_in, torch)
            o_ds = {key: item[key].unsqueeze(0) for key in cfg_in}
            o_ds["task"] = [item["task"]]
            o_mix = dict(o_sim)
            for key in ("observation.state", "observation.environment_state"):
                o_mix[key] = o_ds[key]
            rec = {"wall": e["wall_index"], "seed": e["seed"], "slot": k, "close_step": ic,
                   "state_diff_max": float((o_sim["observation.state"] - o_ds["observation.state"]).abs().max()),
                   "img_mad": {c: float((o_sim[f"observation.images.{c}"] - o_ds[f"observation.images.{c}"]).abs().mean() * 255) for c in CAMS}}
            p_true = tcp_of(act_true[ic])
            for name, o in (("dataset", o_ds), ("sim", o_sim), ("sim_img_ds_state", o_mix)):
                pred = predict(o)
                h = min(ic, len(pred) - 1)
                d = tcp_of(pred[h]) - p_true
                rec[name] = {"tcp_err_at_close_mm": [round(float(v) * 1000, 1) for v in d], "tcp_err_xy_mm": round(float(np.hypot(*d[:2])) * 1000, 1),
                             "grip_at_close": round(float(pred[h, 6]), 3), "h": h}
            res.append(rec)
            print(f"wall {e['wall_index']} slot {k:2d}: img diff (0-255) " + " ".join(f"{c} {rec['img_mad'][c]:.1f}" for c in CAMS)
                  + f" | TCP xy err at close: dataset {rec['dataset']['tcp_err_xy_mm']:.0f} mm, sim {rec['sim']['tcp_err_xy_mm']:.0f} mm", flush=True)
    Path(a.out).write_text(json.dumps({"ckpt": a.ckpt, "policy": cfg.type, "rows": res}, indent=1))
    env.close()


if __name__ == "__main__":
    main()

"""학습 정책을 MuJoCo 창에서 실시간으로 돌려 보기 (정책 단독, G1·G2·G3 끔, 결과로 쓰지 않음).

rollout.py와 같은 규칙: 벽돌마다 정책 reset, 도면 순서표로 넘어감(철수 완료 또는 벽돌당 17 s). 오른쪽 위에 정책이 보는 카메라 3대를 겹쳐 띄운다.
실행: DISPLAY=:1 venv/bin/python -m wall3.view_policy --ckpt <pretrained_model> --seed 700300 [--speed 1.0]
"""
import argparse, sys, time
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=700300)
    ap.add_argument("--env-version", default="main_v1")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    import mujoco, mujoco.viewer, torch
    from eval_policy import load_policy
    from wall3.config import scene_config, RETRACT_Q
    from wall3.scene import WallEnv
    from wall3.rollout import obs_of, ALLOWED, LIMIT_S
    if not any(lo <= a.seed <= hi for lo, hi in ALLOWED):
        raise SystemExit(f"시작값은 {ALLOWED}만")
    policy, pcfg, pre, post = load_policy(a.ckpt, "cuda", None, None)
    cfg_in = set(pcfg.input_features)
    env = WallEnv(scene_config(a.env_version))
    q_ret = np.array(RETRACT_Q)
    env.reset(a.seed, arm_q=q_ret)
    v = mujoco.viewer.launch_passive(env.model, env.data, show_left_ui=False, show_right_ui=False)
    v.cam.lookat[:] = [0.0, 0.0, 0.3]; v.cam.distance = 2.8; v.cam.azimuth = 135; v.cam.elevation = -25
    dt = env.cfg.ctrl_dt / a.speed
    name = a.label or f"{pcfg.type} policy"
    for k in range(env.n):
        policy.reset()
        start, left, still = env.t, False, 0
        while env.t - start < int(round(LIMIT_S / env.cfg.ctrl_dt)) and v.is_running():
            t0 = time.time()
            obs = obs_of(env, k, cfg_in, torch)
            with torch.inference_mode():
                act = post(policy.select_action(pre(obs))).squeeze(0).cpu().numpy().astype(np.float64)
            env.step(act[:6], float(act[6]))
            vp = v.viewport
            s = max(128, min(256, vp.height // 3)) if vp is not None else 256
            v.set_images([(mujoco.MjrRect(vp.width - s, vp.height - (j + 1) * s, s, s),
                           np.ascontiguousarray(np.kron(env.render(c), np.ones((s // 128, s // 128, 1), np.uint8))))
                          for j, c in enumerate(("front", "top", "wrist"))])
            v.set_texts((None, None, f"{name}  seed {a.seed}  brick {k + 1}/16  t {(env.t - start) * env.cfg.ctrl_dt:4.1f} s",
                         "right: policy cameras front / top / wrist"))
            v.sync()
            q = env.arm_q(); dev = np.abs(q - q_ret).max()
            left = left or dev > 0.3
            still = still + 1 if (left and dev < 0.05 and np.abs(env.data.qvel[env.arm_dofadr]).max() < 0.05) else 0
            if still >= 3:
                break
            time.sleep(max(0.0, dt - (time.time() - t0)))
        if not v.is_running():
            return
    v.set_texts((None, None, f"{name}: wall done - close the window (Esc)", ""))
    while v.is_running():
        v.sync(); time.sleep(0.05)


if __name__ == "__main__":
    main()

"""정책 진단 2 (2026-10-22): 앞 벽돌을 대본이 쌓고 벽돌 k0부터 정책에 넘긴다(검측 없음).

정책이 셋째 벽돌부터 못 집는 이유가
  (가) 자기가 앞서 쌓은 벽돌 · 자세가 학습 장면과 달라서(누적 오차 · 분포 이탈)인지,
  (나) 학습 장면과 똑같아도 못 하는지(장면 렌더 차이 · 벽돌 안 실행 중 오차)인지 가린다.
학습 시작값(7100xx, 학습 장면 그대로 재현, main_v1)과 안 본 시작값(7001xx, main_v1_eval)을 함께 본다. 진단용이며 평가 결과가 아니다.
벽돌마다 시작 후 0.5 · 1.0 · 1.5 s의 어깨 회전(관절 0), 들어 올린 벽돌(목표 · 다른 벽돌), 끝난 이유, 최종 위치 오차를 남긴다.

실행: PYTHONPATH=. MUJOCO_GL=egl venv/bin/python -m experiments.diag_handoff --ckpt <pretrained_model> --env-version main_v1 \
        --seeds 710000 710001 --k0 2 3 --out-dir <dir> --tag train [--record <mp4>]
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

import mujoco
import numpy as np

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
LIMIT_S = 17.0                     # same as rollout.py
HS = (10, 20, 30)


def close_event(env, k):
    from wall3.poses import yaw_of_quat
    tp, tR = env.tcp()
    out = {"t": int(env.t)}
    d = [np.linalg.norm(env.brick_pose(i)[0][:2] - tp[:2]) for i in range(env.n)]
    for name, i in (("target", k), ("nearest", int(np.argmin(d)))):
        bp, bq = env.brick_pose(i)
        yb = yaw_of_quat(bq)
        ax, ay = np.array([np.cos(yb), np.sin(yb)]), np.array([-np.sin(yb), np.cos(yb)])
        r = tp[:2] - bp[:2]
        yt = np.arctan2(tR[1, 0], tR[0, 0])
        dyaw = (yt - yb + np.pi / 2) % np.pi - np.pi / 2
        out[name] = {"brick": i, "along_mm": round(float(r @ ax) * 1000, 1), "across_mm": round(float(r @ ay) * 1000, 1),
                     "dz_mm": round(float(tp[2] - bp[2]) * 1000, 1), "dyaw_deg": round(float(np.degrees(dyaw)), 1)}
    return out


def pose_err(env, k):
    """Brick k vs its design pose: xy / z (mm), yaw (deg, folded: the brick is symmetric under 180 deg), tilt (deg)."""
    from wall3.poses import yaw_of_quat
    s = env.design[k]
    p, q = env.brick_pose(k)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    tilt = np.degrees(np.arccos(np.clip(R[8], -1.0, 1.0)))
    dyaw = (yaw_of_quat(q) - s["yaw"] + np.pi / 2) % np.pi - np.pi / 2
    return {"xy_mm": round(float(np.hypot(p[0] - s["x"], p[1] - s["y"])) * 1000, 2), "dx_mm": round(float(p[0] - s["x"]) * 1000, 2),
            "dy_mm": round(float(p[1] - s["y"]) * 1000, 2), "dz_mm": round(float(p[2] - s["z"]) * 1000, 2),
            "yaw_deg": round(float(np.degrees(dyaw)), 2), "tilt_deg": round(float(tilt), 2)}


def policy_brick(env, k, policy, pre, post, cfg_in, torch, q_ret, rec=None):
    """One brick with the policy, same loop and end rule as rollout.run_wall (no gates)."""
    from wall3.rollout import obs_of
    policy.reset()
    dt = env.cfg.ctrl_dt
    start = env.t
    z0 = {i: env.brick_pose(i)[0][2] for i in range(env.n)}
    zmax = dict(z0)
    left, still, reason = False, 0, "timeout"
    pan = []
    closes = []                    # each time the finger command starts closing: TCP vs target brick (and the nearest brick)
    opens = []                     # each time it opens again: target brick vs its design pose at that moment
    g_prev = 1.0
    p_start = {i: env.brick_pose(i)[0] for i in range(env.n)}
    while env.t - start < int(round(LIMIT_S / dt)):
        obs = obs_of(env, k, cfg_in, torch)
        with torch.inference_mode():
            act = post(policy.select_action(pre(obs))).squeeze(0).detach().cpu().numpy().astype(np.float64)
        if rec is not None:
            rec.frame(f"policy brick {k + 1}", k)
        if act[6] > 0.15 and g_prev <= 0.15:
            closes.append(close_event(env, k))
        if act[6] <= 0.15 and g_prev > 0.15 and closes:
            opens.append({"t": int(env.t), **pose_err(env, k)})
        g_prev = act[6]
        env.step(act[:6], float(act[6]))
        q = env.arm_q()
        pan.append(float(q[0]))
        for i in range(env.n):
            zmax[i] = max(zmax[i], env.brick_pose(i)[0][2])
        dev = np.abs(q - q_ret).max()
        if dev > 0.3:
            left = True
        qd = np.abs(env.data.qvel[env.arm_dofadr]).max()
        still = still + 1 if (left and dev < 0.05 and qd < 0.05) else 0
        if still >= 3:
            reason = "retract"
            break
    fin = pose_err(env, k)
    moved = {i: float(np.linalg.norm(env.brick_pose(i)[0] - p_start[i]) * 1000) for i in range(env.n) if i != k}
    return {"slot": k, "end_reason": reason, "duration_s": (env.t - start) * dt,
            "pan_at": {str(h): (pan[h - 1] if len(pan) >= h else None) for h in HS},
            "target_lifted": bool(zmax[k] > z0[k] + 0.02), "target_max_lift_mm": float((zmax[k] - z0[k]) * 1000),
            "n_closes": len(closes), "closes": closes, "opens": opens,
            "other_lifted": [i for i in range(env.n) if i != k and zmax[i] > z0[i] + 0.02],
            "laid_moved_mm": {str(i): round(v, 1) for i, v in moved.items() if i < k and v > 1.0},      # bricks already in the wall
            "supply_moved_mm": {str(i): round(v, 1) for i, v in moved.items() if i > k and v > 1.0},    # bricks still on the table
            "final": fin, "final_xy_err_mm": fin["xy_mm"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--env-version", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--k0", type=int, nargs="+", required=True, help="first brick given to the policy (bricks 0..k0-1 by the expert)")
    ap.add_argument("--n-bricks", type=int, default=1)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--expert-version", default=None, help="expert for bricks 0..k0-1 (default: same as --env-version; v2 data = main_v2)")
    ap.add_argument("--tag", required=True, help="file name prefix: <out-dir>/<tag>_k<k0>.json")
    ap.add_argument("--record", default=None, help="mp4 of the first seed (first k0 only)")
    a = ap.parse_args()
    if any(720000 <= s < 740000 for s in a.seeds):
        raise SystemExit("검증 · 시험 시작값은 쓰지 않음")
    outs = {k0: Path(a.out_dir) / f"{a.tag}_k{k0}.json" for k0 in a.k0}
    for o in outs.values():
        if o.exists():
            raise SystemExit(f"{o} 이미 있음")
    import torch
    from eval_policy import load_policy
    from wall3.config import scene_config, RETRACT_Q
    from wall3.scene import WallEnv
    from wall3.expert import WallExpert, expert_config
    policy, pcfg, pre, post = load_policy(a.ckpt, "cuda", None, None)
    cfg_in = set(pcfg.input_features)
    env = WallEnv(scene_config(a.env_version))
    ex = WallExpert(env, expert_config(a.expert_version or a.env_version))
    q_ret = np.array(RETRACT_Q)
    commit = subprocess.check_output(["git", "-C", str(SIM), "rev-parse", "HEAD"], text=True).strip()
    for j, k0 in enumerate(a.k0):
        res = []
        for i, seed in enumerate(a.seeds):
            t0 = time.time()
            rec = None
            if a.record and i == 0 and j == 0:
                from wall3.view_collect import Recorder
                rec = Recorder(env, Path(a.record), label=f"{pcfg.type} handoff k0={k0} seed {seed}")
            r = ex.run(seed, settle_s=0.0, max_bricks=k0, on_step=(lambda e, act, ph, kk: rec.frame(f"expert {ph}", kk)) if rec else None)
            ok = not r.get("fail")
            bricks = []
            if ok:
                for k in range(k0, min(env.n, k0 + a.n_bricks)):
                    bricks.append(policy_brick(env, k, policy, pre, post, cfg_in, torch, q_ret, rec))
            if rec is not None:
                rec.close()
            res.append({"seed": seed, "expert_ok": ok, "bricks": bricks, "wall_clock_s": time.time() - t0})
            print(f"[{pcfg.type}] seed {seed} k0 {k0}: " + "; ".join(
                f"brick {b['slot']} lifted {b['target_lifted']} closes {b['n_closes']} other {b['other_lifted']} {b['end_reason']} "
                f"final xy {b['final']['xy_mm']:.1f} mm yaw {b['final']['yaw_deg']:.1f} laid moved {b['laid_moved_mm']}" for b in bricks), flush=True)
            outs[k0].write_text(json.dumps({"meta": {"ckpt": a.ckpt, "policy": pcfg.type, "env_version": a.env_version, "expert_version": a.expert_version or a.env_version, "k0": k0,
                                                     "seeds": a.seeds, "git_commit": commit, "script": "diag_handoff v2"},
                                            "runs": res}, indent=1, default=float))
    env.close()


if __name__ == "__main__":
    main()

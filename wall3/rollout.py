"""학습 정책 롤아웃 (2차 지시 가2: 미니 학습 점검용, 결과로 쓰지 않음).

정책 단독. 다음 벽돌은 도면 순서표(자리 0 → 15)가 정하고, 성공을 확인하지 않고 정해진 규칙으로 넘어간다:
  - 정책의 철수 완료: 팔이 철수 자세에서 한 번 벗어났다가(어느 관절이든 0.3 rad 이상) 다시 모든 관절이 철수 자세 0.05 rad 안,
    관절 속도 0.05 rad/s 미만으로 3스텝 머묾 (로봇 자신의 관절 상태만 봄)
  - 또는 벽돌당 시간 제한 17 s (340스텝)
정책 관측: 카메라 3대(128²), 관절 6 + 집게(왼쪽 구동 관절), 목표 정보(도면 목표 x·y·z·yaw + 반장). 정답 좌표는 채점에만 쓴다.
벽돌마다 정책을 reset 한다(벽돌 한 장 = 에피소드 1개로 학습했으므로).
G1(--g1, 나4): 집게를 닫고 1.5 cm 들어 올리면 정책을 멈추고 하중·각도로 쥠을 확인(wall3/g1.py). 못 쥐었으면 집게를 열고 정책을
reset 해 다시 집게 함(벽돌당 시간 제한 안에서 횟수 제한 없음). 쥐었으면 정책을 reset 하고 이어 감. 확인 동작 시간도 벽돌 시간에 넣는다.

실행: source venv/bin/activate && MUJOCO_GL=egl PYTHONPATH=. python -m wall3.rollout --ckpt <pretrained_model> \\
        --env-version pilot_20261016 --seeds 700100 ... --out outputs/wall3/<이름> [--record-first] [--n-action-steps N]
"""
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
CAMS = ("front", "top", "wrist")
LIMIT_S = 17.0
ALLOWED = [(700053, 700199),          # pilot block, unused seeds (2차 지시: 700100 ~ 700119 for the mini rollouts)
           (700300, 700399)]          # 본학습 확인 2벽(700300 · 700301), 검측 예비 확인(2026-10-20, 결과 아님)


def obs_of(env, k, cfg_in, torch):
    from wall3.to_lerobot import target_of, TASK
    obs = {}
    for c in CAMS:
        key = f"observation.images.{c}"
        if key in cfg_in:
            obs[key] = torch.from_numpy(env.render(c)).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    st = np.r_[env.arm_q(), float(env.data.joint("g_finger_joint").qpos[0])].astype(np.float32)
    obs["observation.state"] = torch.from_numpy(st).unsqueeze(0)
    obs["observation.environment_state"] = torch.from_numpy(target_of(env.design[k])).unsqueeze(0)
    obs["task"] = [TASK]
    return obs


def run_wall(env, policy, pre, post, cfg_in, seed, torch, rec=None, q_ret=None, gates=None):
    from wall3.spec_metrics import evaluate_env
    env.reset(seed, arm_q=q_ret)
    dt = env.cfg.ctrl_dt
    t0 = time.time()
    bricks = []
    last = np.r_[q_ret, env.cfg.gripper.open_ctrl]
    n_limit = int(round(LIMIT_S / dt))
    if gates is not None:
        gates.replaced = set()
    for k in range(env.n):
        policy.reset()
        start = env.t
        if gates is not None:
            gates.start_brick(k, start + n_limit)
            gates.on_frame = (lambda ph, kk: rec.frame(f"gate {ph}", kk)) if rec is not None else None
            gates.ex.on_step = (lambda e, a, ph, kk: rec.frame(f"gate {ph}", kk)) if rec is not None else None
        z0 = env.brick_pose(k)[0][2]
        zmax = z0
        left, still, reason = False, 0, "timeout"
        qhist = []
        gate_steps = 0                                           # gate actions do not count in the brick time limit (2026-10-22)
        while env.t - start - gate_steps < n_limit:
            obs = obs_of(env, k, cfg_in, torch)
            with torch.inference_mode():
                act = post(policy.select_action(pre(obs))).squeeze(0).detach().cpu().numpy().astype(np.float64)
            if rec is not None:
                rec.frame(f"policy brick {k + 1}", k)
            t_b = env.t
            if gates is not None and gates.before(act):          # G2 / G3 took over this release
                last = np.asarray(env.last_action, float).copy()
                gate_steps += env.t - t_b
            else:
                env.step(act[:6], float(act[6]))
                last = act
                if gates is not None:
                    t_m = env.t
                    gates.after(act)                              # G1 (or carrying bookkeeping)
                    gate_steps += env.t - t_m
            if gates is not None and gates.pop_gate_retract():   # 2026-11-10 K1: a gate retract is not the policy's retract
                left, still = False, 0                           # (after G1 in after(), or after G0 + G1 inside before())
            q = env.arm_q()
            qhist.append(q)
            zmax = max(zmax, env.brick_pose(k)[0][2])
            dev = np.abs(q - q_ret).max()
            if dev > 0.3:
                left = True
            qd = np.abs(env.data.qvel[env.arm_dofadr]).max()
            still = still + 1 if (left and dev < 0.05 and qd < 0.05) else 0
            if still >= 3:
                reason = "retract"
                break
        moved_last3s = float(np.abs(np.array(qhist[-60:]).max(0) - np.array(qhist[-60:]).min(0)).max()) if len(qhist) >= 60 else None
        bricks.append({"slot": k, "start": int(start), "end": int(env.t), "duration_s": (env.t - start) * dt, "end_reason": reason,
                       "policy_time_s": (env.t - start - gate_steps) * dt, "gate_time_s": gate_steps * dt,
                       "grasped_lifted": bool(zmax > z0 + 0.02), "max_lift_mm": float((zmax - z0) * 1000),
                       "stalled": bool(reason == "timeout" and moved_last3s is not None and moved_last3s < 0.02),
                       "joint_range_last3s_rad": moved_last3s,
                       **({"gates": gates.end_brick()} if gates is not None else {})})
    for _ in range(int(round(2.0 / dt))):                 # 2 s settle holding the last command
        env.step(last[:6], float(last[6]))
        if rec is not None:
            rec.frame("settle", -1)
    ev = evaluate_env(env)
    for b in bricks:
        s = env.design[b["slot"]]
        p, q = env.brick_pose(b["slot"])
        from wall3.poses import yaw_of_quat
        b["final_err_mm"] = ((p - [s["x"], s["y"], s["z"]]) * 1000).round(2).tolist()
        b["final_xy_err_mm"] = float(np.hypot(*b["final_err_mm"][:2]))
        b["iou"] = ev["iou"][b["slot"]]
    return {"seed": seed, "bricks": bricks, "eval": ev, "wall_clock_s": time.time() - t0, "sim_s": env.t * dt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--env-version", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--record-first", action="store_true")
    ap.add_argument("--n-action-steps", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--g1", action="store_true", help="G1 grasp check on (condition 2/5/6/8)")
    ap.add_argument("--g2", action="store_true", help="G2 pre-release correction (top camera)")
    ap.add_argument("--g3", action="store_true", help="G3 post-release inspection + one re-placement (top camera)")
    ap.add_argument("--g1-thr-load", type=float, default=4.5, help="N (임시값)")
    ap.add_argument("--g1-thr-angle", type=float, default=0.47, help="rad (임시값)")
    ap.add_argument("--gate-pose-source", choices=("camera", "truth"), default="camera",
                    help="G2 · G3 brick poses from the camera, or the simulator truth (진단 전용, 2026-11-03)")
    ap.add_argument("--gate-target", choices=("turn", "held_truth", "held_grasp"), default="turn",
                    help="G2 · G3 reference slot: turn k / brick lifted (truth, 진단) / nearest supply slot at the G1 moment")
    ap.add_argument("--gate-fix", nargs="*", default=[],
                    help="added controls (wall3/gates.py FIXES, 2026-11-10 실험파이프라인_실패원인제거): k1 (needs --g1), k2, k6 (needs --g3), k7 (needs --g2), k8 (needs --g2, 2026-11-11 G2c SDCG)")
    a = ap.parse_args()
    need = {"k1": a.g1, "k6": a.g3, "k2": True, "k7": a.g2, "k8": a.g2}
    if any(not need.get(f, True) for f in a.gate_fix):
        raise SystemExit("--gate-fix k1 은 --g1, k6 은 --g3, k7 · k8 은 --g2 와 함께만")
    if (a.gate_pose_source != "camera" or a.gate_target != "turn") and not (a.g2 or a.g3):
        raise SystemExit("--gate-pose-source / --gate-target 는 --g2 또는 --g3 와 함께만")
    if any(not any(lo <= s <= hi for lo, hi in ALLOWED) for s in a.seeds):
        raise SystemExit(f"롤아웃 시작값은 파일럿 블록 미사용 값 {ALLOWED}만")
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} 이미 있음: 덮어쓰지 않음")
    out.mkdir(parents=True)
    import torch
    from eval_policy import load_policy
    from wall3.config import scene_config, RETRACT_Q
    from wall3.scene import WallEnv
    policy, pcfg, pre, post = load_policy(a.ckpt, a.device, None, a.n_action_steps or None)
    cfg_in = set(pcfg.input_features)
    env = WallEnv(scene_config(a.env_version))
    q_ret = np.array(RETRACT_Q)
    gates = None
    if a.g1 or a.g2 or a.g3:
        from wall3.g1 import G1Config
        from wall3.gates import WallGates, GateConfig, fix_fields
        gates = WallGates(env, policy, g1=a.g1, g2=a.g2, g3=a.g3,
                          g1_cfg=G1Config(thr_load=a.g1_thr_load, thr_angle=a.g1_thr_angle),
                          cfg=GateConfig(pose_source=a.gate_pose_source, target=a.gate_target, **fix_fields(a.gate_fix)))
    meta = {"ckpt": a.ckpt, "env_version": a.env_version, "policy_type": pcfg.type, "n_action_steps": pcfg.n_action_steps,
            "input_features": {k: list(v.shape) for k, v in pcfg.input_features.items()},
            "git_commit": subprocess.check_output(["git", "-C", str(SIM), "rev-parse", "HEAD"], text=True).strip(), "limit_s": LIMIT_S,
            "gates": (gates.summary_config() if gates is not None else None),
            "gate_pose_source": a.gate_pose_source, "gate_target": a.gate_target, "gate_fix": a.gate_fix,
            "scene_sha256": hashlib.sha256(json.dumps(env.cfg.to_dict(), sort_keys=True, default=float).encode()).hexdigest()}
    res = []
    for i, seed in enumerate(a.seeds):
        rec = None
        if a.record_first and i == 0:
            from wall3.view_collect import Recorder
            rec = Recorder(env, out / f"wall_{seed}.mp4", label=f"{pcfg.type} {a.env_version} seed {seed}")
        r = run_wall(env, policy, pre, post, cfg_in, seed, torch, rec, q_ret, gates)
        if rec is not None:
            rec.close()
        res.append(r)
        ev = r["eval"]
        b = r["bricks"]
        print(f"[{pcfg.type}] seed {seed}: lifted {sum(x['grasped_lifted'] for x in b)}/16, retract {sum(x['end_reason'] == 'retract' for x in b)}/16, "
              f"stalled {sum(x['stalled'] for x in b)}, filled {sum(ev['slots_filled'])}/16, spec {ev['spec_pass']}, IoU mean {ev['iou_mean']:.3f}, "
              f"median xy err {np.median([x['final_xy_err_mm'] for x in b]):.1f} mm, wall {r['wall_clock_s']:.0f} s (sim {r['sim_s']:.0f} s)"
              + (f", G1 checks {sum('grasped' in c for x in b for c in x['gates']['g1'])} (not grasped {sum(c.get('grasped') is False for x in b for c in x['gates']['g1'])}),"
                 f" G2 {sum(not e.get('skipped') for x in b for e in x['gates']['g2'])} (far {sum(e.get('skipped') == 'far' for x in b for e in x['gates']['g2'])}), G3 {sum(len(x['gates']['g3']) for x in b)}"
                 f" (re-placed {sum(isinstance(e.get('replace'), dict) for x in b for e in x['gates']['g3'])})" if gates is not None else ""), flush=True)
        (out / "rollout.json").write_text(json.dumps({"meta": meta, "walls": res}, indent=1, default=float))
    env.close()


if __name__ == "__main__":
    main()

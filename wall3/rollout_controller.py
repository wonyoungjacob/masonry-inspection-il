"""롤아웃 (2026-11-16): 정책 자리에 학습 정책(act) 또는 대본 제어기(script), 검측은 Full과 같은 코드. G2t(운반까지 G2)를 고를 수 있다.

벽마다 wall3/rollout.py의 run_wall을 그대로 쓴다(도면 순서, 철수 또는 정책 시간 17 s, 검측 동작 시간은 제한에 안 넣음, 채점 같음).
  --controller act    : 학습 정책 (wall3.rollout과 같음)
  --controller script : wall3/script_policy.py (정책 없음: 도면 순서 + 대본 동작, 참값 안 씀). 카메라 영상은 렌더하지 않는다.
  --g2-carry          : wall3/gates_carry.py (G2t)
참값 금지(사용자 조건 11-16): --gate-pose-source truth · --gate-target held_truth는 받지 않는다.

실행: MUJOCO_GL=egl PYTHONPATH=. venv/bin/python -m wall3.rollout_controller --controller script --env-version main_v1_eval \\
        --seeds 700300 ... --out outputs/wall3/<이름> --g1 --g2 --g3 --gate-target held_grasp --gate-fix k2 k7 [--g2-carry]
"""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
ALLOWED = (700300, 700399)


def sha16(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=("act", "script"), required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--env-version", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--record-first", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--g1", action="store_true")
    ap.add_argument("--g2", action="store_true")
    ap.add_argument("--g3", action="store_true")
    ap.add_argument("--g1-thr-load", type=float, default=4.5)
    ap.add_argument("--g1-thr-angle", type=float, default=0.47)
    ap.add_argument("--gate-target", choices=("turn", "held_grasp"), default="turn")
    ap.add_argument("--gate-fix", nargs="*", default=[])
    ap.add_argument("--g2-carry", action="store_true", help="G2t: G2 carries from the G1 'grasped' moment (needs --g1 --g2)")
    a = ap.parse_args()
    if a.controller == "act" and not a.ckpt:
        raise SystemExit("--controller act 는 --ckpt 필요")
    if a.g2_carry and not (a.g1 and a.g2):
        raise SystemExit("--g2-carry 는 --g1 --g2 와 함께만")
    if any(not ALLOWED[0] <= s <= ALLOWED[1] for s in a.seeds):
        raise SystemExit(f"시작값은 {ALLOWED}만")
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} 이미 있음: 덮어쓰지 않음")
    out.mkdir(parents=True)
    import torch
    from wall3.config import scene_config, RETRACT_Q
    from wall3.scene import WallEnv
    from wall3.rollout import run_wall, LIMIT_S
    env = WallEnv(scene_config(a.env_version))
    q_ret = np.array(RETRACT_Q)
    if a.controller == "act":
        from eval_policy import load_policy
        policy, pcfg, pre, post = load_policy(a.ckpt, a.device, None, None)
        cfg_in = set(pcfg.input_features)
        ctrl_meta = {"policy_type": pcfg.type, "ckpt": a.ckpt, "n_action_steps": pcfg.n_action_steps}
    else:
        from wall3.script_policy import ScriptPolicy
        policy = ScriptPolicy(env)
        pre = post = (lambda x: x)
        cfg_in = {"observation.state", "observation.environment_state"}
        ctrl_meta = {"policy_type": "script", "script": policy.config(),
                     "script_sha16": sha16(SIM / "wall3/script_policy.py")}
    gates = None
    if a.g1 or a.g2 or a.g3:
        from wall3.g1 import G1Config
        from wall3.gates import WallGates, GateConfig, fix_fields
        cls = WallGates
        if a.g2_carry:
            from wall3.gates_carry import WallGatesCarry as cls
        gates = cls(env, policy, g1=a.g1, g2=a.g2, g3=a.g3, g1_cfg=G1Config(thr_load=a.g1_thr_load, thr_angle=a.g1_thr_angle),
                    cfg=GateConfig(pose_source="camera", target=a.gate_target, **fix_fields(a.gate_fix)))
    meta = {"controller": a.controller, **ctrl_meta, "env_version": a.env_version, "limit_s": LIMIT_S, "g2_carry": a.g2_carry,
            "git_commit": subprocess.check_output(["git", "-C", str(SIM), "rev-parse", "HEAD"], text=True).strip(),
            "code_sha16": {f: sha16(SIM / f) for f in ("wall3/rollout_controller.py", "wall3/rollout.py", "wall3/gates.py",
                                                       "wall3/gates_carry.py", "wall3/script_policy.py")},
            "gates": (gates.summary_config() if gates is not None else None),
            "gate_pose_source": "camera", "gate_target": a.gate_target, "gate_fix": a.gate_fix,
            "scene_sha256": hashlib.sha256(json.dumps(env.cfg.to_dict(), sort_keys=True, default=float).encode()).hexdigest()}
    res = []
    for i, seed in enumerate(a.seeds):
        rec = None
        if a.record_first and i == 0:
            from wall3.view_collect import Recorder
            rec = Recorder(env, out / f"wall_{seed}.mp4", label=f"{a.controller} {a.env_version} seed {seed}")
        n_log = len(getattr(policy, "log", []))
        r = run_wall(env, policy, pre, post, cfg_in, seed, torch, rec, q_ret, gates)
        if rec is not None:
            rec.close()
        if a.controller == "script":
            r["script_turns"] = policy.log[n_log:]
        res.append(r)
        ev, b = r["eval"], r["bricks"]
        print(f"[{a.controller}{' +G2t' if a.g2_carry else ''}] seed {seed}: filled {sum(ev['slots_filled'])}/16, spec {ev['spec_pass']}, "
              f"retract {sum(x['end_reason'] == 'retract' for x in b)}/16, wall {r['wall_clock_s']:.0f} s (sim {r['sim_s']:.0f} s)", flush=True)
        (out / "rollout.json").write_text(json.dumps({"meta": meta, "walls": res}, indent=1, default=float))
    env.close()


if __name__ == "__main__":
    main()

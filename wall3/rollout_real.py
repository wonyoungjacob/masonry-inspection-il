"""현실 제약 ablation 롤아웃 (2026-11-16, 01_사기설/우분투지시_현실제약_ablation_20261116.md).

정책 자리: --controller act (하이브리드) | script (제어만, 대본 제어기). 검측(8조건 공통): G0v + G1 + G2 + G2w + G2t + G3 + H
(wall3/gates_real.py). 제약: --A (계측 · 센서 오차 묶음), --B (공급 팔레트 30 mm 이동). 벽마다 오차는
wall3/realism.make_perturb(시작값)로 정해지고 rollout.json의 벽 기록에 남는다. 벽 실행 · 채점은 wall3.rollout.run_wall 그대로.

실행: MUJOCO_GL=egl PYTHONPATH=. venv/bin/python -m wall3.rollout_real --controller script --env-version main_v1_eval \\
        --seeds 700300 ... --out outputs/wall3/<이름> [--A] [--B]
"""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
ALLOWED = (700300, 700399)
CKPT = "outputs/wall3/main_train_v2_20261022/act_s1000/checkpoints/last/pretrained_model"
WALL_SIDE_Y = 0.20          # m: the supply table is at y < -0.17, the wall at y = 0.6


def sha16(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def script_real_class():
    from wall3.script_policy import ScriptPolicy
    from wall3.ik import down_R

    class ScriptPolicyReal(ScriptPolicy):
        """대본 제어기 + 믿는 도면(A1). G2t가 운반 · 놓기를 끝내고 손을 돌려주면(빈손, 손이 벽 쪽) 그 차례는 끝난 것으로 본다."""

        def __init__(self, env, cfg=None):
            super().__init__(env, cfg)
            self.design_belief = [dict(d) for d in env.design]

        def _plan(self, q, finger):
            t = self.turn
            if not t["release_started"] and t["attempts"] >= 1 and not self.holding(finger):
                p, _ = self.ex.ik.fk(q)
                if p[1] > WALL_SIDE_Y:
                    t["release_started"] = True
                    t["delivered_by_gate"] = True
            super()._plan(q, finger)

        def _transport(self, q):
            c, ex, k = self.cfg, self.ex, self.k
            s = self.design_belief[k]
            segs = []
            p, R = ex.ik.fk(q)
            q_now = q
            pos, _ = self.env.supply_pose(k)
            z_app_supply = pos[2] + self.dzp + c.approach
            if p[2] < z_app_supply - 0.005:
                path = self._cart(q_now, np.r_[p[:2], z_app_supply], R, c.t_up)
                if path is not None:
                    segs.append((path, None, self.cg, k, "up"))
                    q_now = path[-1]
            tgt = np.array([s["x"], s["y"], s["z"] + self.dzp + c.approach])
            cands = []
            for flip in (0.0, np.pi):
                qw = ex._solve(tgt, down_R(s["yaw"] + flip), self.q_ret, self.q_ret[0])
                if qw is not None:
                    cands.append((abs(qw[5] - q_now[5]) + 0.3 * abs(qw[5] - self.q_ret[5]), qw))
            if not cands:
                self.turn["gave_up"] = True
                return segs + self._retract(q_now)
            q_w = min(cands, key=lambda x: x[0])[1]
            segs.append((self._joint(q_now, q_w, c.lag_wall), None, self.cg, k, "to_wall"))
            n = max(1, int(round(c.t_open / self.dt)))
            start = self.cg - c.release_drop
            opening = [max(self.gc.open_ctrl, start - (start - self.gc.open_ctrl) * (j + 1) / n) for j in range(n)]
            segs.append((np.repeat(q_w[None], n + 1, axis=0), None, [start] + opening, None, "release"))
            return segs

    return ScriptPolicyReal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=("act", "script"), required=True)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--env-version", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--record-first", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--A", action="store_true", help="계측 · 센서 오차 묶음")
    ap.add_argument("--B", action="store_true", help="공급 팔레트 30 mm 이동")
    a = ap.parse_args()
    if any(not ALLOWED[0] <= s <= ALLOWED[1] for s in a.seeds):
        raise SystemExit(f"시작값은 {ALLOWED}만")
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} 이미 있음: 덮어쓰지 않음")
    out.mkdir(parents=True)
    import torch
    if os.environ.get("OMP_NUM_THREADS"):
        torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    from wall3.config import scene_config, RETRACT_Q
    from wall3.scene import WallEnv
    from wall3.rollout import run_wall, LIMIT_S
    from wall3.g1 import G1Config
    from wall3.gates import GateConfig, fix_fields
    from wall3.gates_real import WallGatesReal
    from wall3.realism import make_perturb, shift_pallet, noisy_ft, belief_design, SALT
    env = WallEnv(scene_config(a.env_version))
    q_ret = np.array(RETRACT_Q)
    holder = {"pert": None}
    orig_reset = env.reset

    def reset(seed, arm_q=None):
        orig_reset(seed, arm_q=arm_q)
        if holder["pert"] is not None:
            shift_pallet(env, holder["pert"])
        return env
    env.reset = reset
    if a.controller == "act":
        from eval_policy import load_policy
        policy, pcfg, pre, post = load_policy(a.ckpt, a.device, None, None)
        cfg_in = set(pcfg.input_features)
        ctrl_meta = {"policy_type": pcfg.type, "ckpt": a.ckpt, "n_action_steps": pcfg.n_action_steps}
    else:
        policy = script_real_class()(env)
        pre = post = (lambda x: x)
        cfg_in = {"observation.state", "observation.environment_state"}
        ctrl_meta = {"policy_type": "script", "script": policy.config()}
    gates = WallGatesReal(env, policy, g1=True, g2=True, g3=True, g1_cfg=G1Config(thr_load=4.5, thr_angle=0.47),
                          cfg=GateConfig(pose_source="camera", target="held_grasp", **fix_fields(["k2", "k7"])))
    files = ["wall3/rollout_real.py", "wall3/gates_real.py", "wall3/realism.py", "wall3/gates_carry.py",
             "wall3/gates.py", "wall3/rollout.py", "wall3/script_policy.py", "wall3/perception.py"]
    meta = {"controller": a.controller, **ctrl_meta, "A": a.A, "B": a.B, "env_version": a.env_version, "limit_s": LIMIT_S,
            "gates": {**gates.summary_config(), "g0": "G0v (wrist camera)", "g2_carry": True, "ff": "nominal"},
            "git_commit": subprocess.check_output(["git", "-C", str(SIM), "rev-parse", "HEAD"], text=True).strip(),
            "code_sha16": {f: sha16(SIM / f) for f in files},
            "threads": os.environ.get("OMP_NUM_THREADS"),
            "scene_sha256": hashlib.sha256(json.dumps(env.cfg.to_dict(), sort_keys=True, default=float).encode()).hexdigest()}
    res = []
    for i, seed in enumerate(a.seeds):
        p = make_perturb(seed, a.A, a.B)
        holder["pert"] = p
        gates.set_perturb(p, np.random.default_rng([seed, SALT, 1]), np.random.default_rng([seed, SALT, 2]))
        noisy_ft(env, p, np.random.default_rng([seed, SALT, 3]))
        if a.controller == "script":
            policy.design_belief = belief_design(env.design, p)
        rec = None
        if a.record_first and i == 0:
            from wall3.view_collect import Recorder
            rec = Recorder(env, out / f"wall_{seed}.mp4", label=f"{a.controller} A={int(a.A)} B={int(a.B)} seed {seed}")
        n_log = len(getattr(policy, "log", []))
        r = run_wall(env, policy, pre, post, cfg_in, seed, torch, rec, q_ret, gates)
        if rec is not None:
            rec.close()
        r["perturb"] = p.to_dict()
        if a.controller == "script":
            r["script_turns"] = policy.log[n_log:]
        res.append(r)
        ev, b = r["eval"], r["bricks"]
        g0 = [e for x in b for e in x["gates"].get("g0", [])]
        print(f"[{a.controller} A={int(a.A)} B={int(a.B)}] seed {seed}: filled {sum(ev['slots_filled'])}/16, spec {ev['spec_pass']}, "
              f"G0v {len(g0)} (not seen {sum(e.get('result') == 'not_seen' for e in g0)}, grasped {sum(bool(e.get('g1_grasped')) for e in g0)}), "
              f"wall {r['wall_clock_s']:.0f} s (sim {r['sim_s']:.0f} s)", flush=True)
        (out / "rollout.json").write_text(json.dumps({"meta": meta, "walls": res}, indent=1, default=float))
    env.close()


if __name__ == "__main__":
    main()

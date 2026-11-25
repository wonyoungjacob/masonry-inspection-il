"""wall3.script_policy (정책 없음: 대본 제어기) · wall3.gates_carry (G2t) 점검. 2026-11-16.

1. 대본 제어기 소스에 참값 · 접촉 · 모르타르 상태 · 검측 내부를 읽는 이름이 없다.
2. 실제로 벽돌 두 장을 쌓는 동안, 대본 제어기가 호출된 때에 환경의 참값 함수가 한 번도 불리지 않는다(불리면 예외).
   검측(gates)은 기록용으로 참값을 읽으므로 그 호출은 허용한다(대본 제어기 호출 안에서만 막음).
3. G2t: G1 '쥠' 직후 운반 기록이 남고, 그 벽돌을 G2가 놓는다.
MuJoCo EGL 렌더가 필요하다(검측 카메라). 시작값 700300(개발 배치, 동작 확인만).
"""
import importlib.util
import os
import re
import unittest
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[1]
SRC = SIM / "wall3" / "script_policy.py"
FORBIDDEN = ["brick_pose", "brick_vel", "lock_b", "xpos", "xmat", "contacts", "bed_laid", "bed_count", "in_slot", "on_table",
             "cured", "render", "wrist_ft", "audit", "gates", "lock_events", "closure()"]


def code_only(text):
    """Source without docstrings and comments."""
    text = re.sub(r'"""[\s\S]*?"""', "", text)
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


class TestSource(unittest.TestCase):
    def test_no_truth_names(self):
        code = code_only(SRC.read_text(encoding="utf-8"))
        for name in FORBIDDEN:
            self.assertNotIn(name, code, name)


def run_turns(carry, n_turns=2, seed=700300):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import torch
    from wall3.config import scene_config, RETRACT_Q
    from wall3.scene import WallEnv
    from wall3.g1 import G1Config
    from wall3.gates import WallGates, GateConfig, fix_fields
    from wall3.gates_carry import WallGatesCarry
    from wall3.rollout import obs_of, LIMIT_S
    from wall3.script_policy import ScriptPolicy
    env = WallEnv(scene_config("main_v1_eval"))
    pol = ScriptPolicy(env)
    cls = WallGatesCarry if carry else WallGates
    gates = cls(env, pol, g1=True, g2=True, g3=True, g1_cfg=G1Config(thr_load=4.5, thr_angle=0.47),
                cfg=GateConfig(pose_source="camera", target="held_grasp", **fix_fields(["k2", "k7"])))
    guard = {"on": False, "hits": []}
    for name in ("brick_pose", "brick_poses", "brick_vel", "contacts", "tcp", "closure", "in_slot", "on_table", "wrist_ft"):
        orig = getattr(env, name)

        def wrapped(*a, _o=orig, _n=name, **kw):
            if guard["on"]:
                guard["hits"].append(_n)
                raise AssertionError(f"script controller called env.{_n}")
            return _o(*a, **kw)
        setattr(env, name, wrapped)
    q_ret = np.array(RETRACT_Q)
    env.reset(seed, arm_q=q_ret)
    cfg_in = {"observation.state", "observation.environment_state"}
    n_limit = int(round(LIMIT_S / env.cfg.ctrl_dt))
    evs = []
    for k in range(n_turns):
        pol.reset()
        gates.start_brick(k, env.t + n_limit)
        start, gate_steps, left, still = env.t, 0, False, 0
        while env.t - start - gate_steps < n_limit:
            obs = obs_of(env, k, cfg_in, torch)
            guard["on"] = True
            try:
                act = pol.select_action(obs).squeeze(0).numpy().astype(np.float64)
            finally:
                guard["on"] = False
            t_b = env.t
            if gates.before(act):
                gate_steps += env.t - t_b
            else:
                env.step(act[:6], float(act[6]))
                t_m = env.t
                gates.after(act)
                gate_steps += env.t - t_m
            dev = np.abs(env.arm_q() - q_ret).max()
            left = left or dev > 0.3
            qd = np.abs(env.data.qvel[env.arm_dofadr]).max()
            still = still + 1 if (left and dev < 0.05 and qd < 0.05) else 0
            if still >= 3:
                break
        evs.append(gates.end_brick())
    placed = [float(np.hypot(*(env.brick_pose(k)[0][:2] - [env.design[k]["x"], env.design[k]["y"]]))) for k in range(n_turns)]
    env.close()
    return guard, evs, placed, pol.log


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch 필요 (requirements-train.txt)")
class TestRuntime(unittest.TestCase):
    def test_script_no_truth_two_bricks(self):
        guard, evs, placed, log = run_turns(carry=False)
        self.assertEqual(guard["hits"], [])
        for k, e in enumerate(evs):
            self.assertTrue(any(not g.get("skipped") for g in e["g2"]), f"turn {k}: G2 did not place")
            self.assertTrue(log[k]["release_started"], f"turn {k}")
        self.assertTrue(all(d < 0.005 for d in placed), placed)        # both bricks within 5 mm of their slots (audit)

    def test_g2t_carry(self):
        guard, evs, placed, log = run_turns(carry=True)
        self.assertEqual(guard["hits"], [])
        for k, e in enumerate(evs):
            self.assertTrue(any(c.get("result") == "carried" for c in e.get("g2t", [])), f"turn {k}: no G2t carry")
            self.assertTrue(any(not g.get("skipped") for g in e["g2"]), f"turn {k}: G2 did not place")
        self.assertTrue(all(d < 0.005 for d in placed), placed)


if __name__ == "__main__":
    unittest.main()

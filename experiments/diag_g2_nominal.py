"""검측 조건 하나로 벽을 다시 돌리며 매 스텝 기록 · 진단 화면을 남긴다 (2026-11-11, 진단 · 평가 아님).

두 가지를 보려고 만들었다.
1) Full 추적(diag_trace_lf_full_20261110)에서 자리 6 처음 실패 9벽 중 8벽은 첫 G2가 손 안 측정에 실패한 채 놓아 7 ~ 11 mm 어긋났다.
   gates.debug_dir를 켜서 시점마다 끝을 못 본 작업대 카메라 화면을 저장한다(gates.py의 기존 진단 출력).
2) Full − G1(G1 꺼짐, 11-11 리셋 고친 뒤)에서 42벽이 자리 4에서 처음 실패했다(놓인 벽돌 4가 나중에 19 cm 밀림).
   --no-g1로 같은 조건을 돌려 매 스텝 벽돌 자세 · 손 · 누가 움직였나(정책 / G0 / G1 / G2·G3)를 남긴다(diag_video와 같은 기록 형식).

실행: MUJOCO_GL=egl PYTHONPATH=. venv/bin/python -m experiments.diag_g2_nominal --seeds 700315 --out outputs/wall3/diag_g2_nominal_20261111
      (G1 꺼짐: --no-g1 --seeds 700302 --out ...)
"""
import argparse, json
from pathlib import Path

import numpy as np

SRC = {"policy": 0, "g1": 1, "g2g3": 2, "other": 3, "g0": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/wall3/main_train_v2_20261022/act_s1000/checkpoints/last/pretrained_model")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate-fix", nargs="*", default=["k2", "k7"])
    ap.add_argument("--no-g1", action="store_true", help="G1 off (Full − G1)")
    a = ap.parse_args()
    if any(not (700300 <= s <= 700399) for s in a.seeds):
        raise SystemExit("시작값은 700300 ~ 700399만")
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} 이미 있음: 덮어쓰지 않음")
    out.mkdir(parents=True)
    import torch
    from eval_policy import load_policy
    from wall3.config import scene_config, RETRACT_Q
    from wall3.scene import WallEnv
    from wall3.g1 import G1Config
    from wall3.gates import WallGates, GateConfig, fix_fields
    from wall3.rollout import run_wall
    policy, pcfg, pre, post = load_policy(a.ckpt, "cuda", None, None)
    env = WallEnv(scene_config("main_v1_eval"))
    gates = WallGates(env, policy, g1=not a.no_g1, g2=True, g3=True, g1_cfg=G1Config(thr_load=4.5, thr_angle=0.47),
                      cfg=GateConfig(pose_source="camera", target="held_grasp", **fix_fields(a.gate_fix)))
    state, log = {"src": SRC["policy"]}, {}
    step0 = env.step

    def step(arm_ctrl, grip_cmd):
        r = step0(arm_ctrl, grip_cmd)
        tp, R = env.tcp()
        log["t"].append(env.t)
        log["k"].append(getattr(gates, "k", -1))
        log["tgt"].append(getattr(gates, "tgt", -1))
        log["src"].append(state["src"])
        log["pose"].append(env.brick_poses().astype(np.float32))
        log["tcp"].append(np.r_[tp, np.arctan2(R[1, 0], R[0, 0])].astype(np.float32))
        log["lock"].append(env.lock_b)
        log["grip"].append(float(grip_cmd))
        return r
    env.step = step

    def mark(fn, code):
        def g(*x, **kw):
            prev = state["src"]
            state["src"] = code
            try:
                return fn(*x, **kw)
            finally:
                state["src"] = prev
        return g
    gates.before = mark(gates.before, SRC["g2g3"])
    gates.after = mark(gates.after, SRC["g1"])
    gates._g0_grasp = mark(gates._g0_grasp, SRC["g0"])

    res = []
    for seed in a.seeds:
        for key in ("t", "k", "tgt", "src", "pose", "tcp", "lock", "grip"):
            log[key] = []
        gates.debug_dir = out / str(seed)
        r = run_wall(env, policy, pre, post, set(pcfg.input_features), seed, torch, None, np.array(RETRACT_Q), gates)
        np.savez_compressed(out / f"trace_{seed}.npz", t=np.array(log["t"], np.int32), k=np.array(log["k"], np.int8),
                            tgt=np.array(log["tgt"], np.int8), src=np.array(log["src"], np.int8), pose=np.stack(log["pose"]),
                            tcp=np.stack(log["tcp"]), lock=np.array(log["lock"], np.int8), grip=np.array(log["grip"], np.float32))
        r["seed"] = seed
        res.append(r)
        g2 = [e for b in r["bricks"] for e in b["gates"]["g2"] if not e.get("skipped")]
        print(seed, "filled", int(sum(r["eval"]["slots_filled"])), "nominal G2:",
              [(e["slot"], e["t"]) for e in g2 if e.get("in_hand_src") == "nominal"], flush=True)
    (out / "trace.json").write_text(json.dumps({"meta": {"no_g1": a.no_g1, "gate_fix": a.gate_fix, "src_codes": SRC},
                                                "walls": res}, indent=1, default=float))
    env.close()


if __name__ == "__main__":
    main()

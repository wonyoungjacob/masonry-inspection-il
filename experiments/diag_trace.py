"""정책 + 검측 벽 쌓기 추적 (2026-10-28): 100벽 평가(eval_gates)와 같은 벽을 같은 코드(rollout.run_wall)로 다시 돌리며
매 제어 스텝 벽돌 16장의 자세 · 손(TCP) · 쥔 벽돌 · 누가 움직였는지(정책 / G1 / G2·G3 / 마무리)를 남긴다. 진단용이며 평가 결과가 아니다.

평가 기록에는 차례(k)마다 G1 · G2 · G3 사건만 있고, 차례 사이에 벽돌이 어디로 갔는지(누가 집어 어디에 놓았는지, 다른 벽돌에
밀렸는지)는 없다. 이 추적으로 '못 쌓은 벽돌'마다 원인을 정답 좌표로 가린다(정답 좌표는 기록 · 분석에만, 정책 · 검측 판단에는 안 씀).
같은 시작값이면 평가와 같은 결과가 나오는지(재현)도 벽마다 비교해 남긴다.

실행: PYTHONPATH=. MUJOCO_GL=egl venv/bin/python -m experiments.diag_trace --ckpt <pretrained_model> \
        --seeds 700300 ... --out <dir> [--compare outputs/wall3/eval_gates_v2_20261026/act_all_part*/rollout.json]
"""
import argparse, glob, hashlib, json, subprocess, sys, time
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
SRC = {"policy": 0, "g1": 1, "g2g3": 2, "other": 3, "g0": 4}     # who issued the step: policy loop / G1 check / G2·G3 placing / settle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--env-version", default="main_v1_eval")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate-pose-source", choices=("camera", "truth"), default="camera", help="= rollout (2026-11-03)")
    ap.add_argument("--gate-target", choices=("turn", "held_truth", "held_grasp"), default="turn", help="= rollout (2026-11-03)")
    ap.add_argument("--gate-fix", nargs="*", default=[], help="= rollout (2026-11-10)")
    ap.add_argument("--compare", nargs="*", default=[], help="rollout.json files of the evaluation run to compare with")
    a = ap.parse_args()
    if any(not (700300 <= s <= 700399) for s in a.seeds):
        raise SystemExit("시작값은 평가와 같은 700300 ~ 700399만")
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} 이미 있음: 덮어쓰지 않음")
    out.mkdir(parents=True)
    ref = {}
    for f in a.compare:
        for p in glob.glob(f):
            for w in json.loads(Path(p).read_text())["walls"]:
                ref[w["seed"]] = w
    import torch
    from eval_policy import load_policy
    from wall3.config import scene_config, RETRACT_Q
    from wall3.scene import WallEnv
    from wall3.g1 import G1Config
    from wall3.gates import WallGates, GateConfig, fix_fields
    from wall3.rollout import run_wall
    policy, pcfg, pre, post = load_policy(a.ckpt, "cuda", None, None)
    cfg_in = set(pcfg.input_features)
    env = WallEnv(scene_config(a.env_version))
    q_ret = np.array(RETRACT_Q)
    gates = WallGates(env, policy, g1=True, g2=True, g3=True, g1_cfg=G1Config(thr_load=4.5, thr_angle=0.47),
                      cfg=GateConfig(pose_source=a.gate_pose_source, target=a.gate_target, **fix_fields(a.gate_fix)))   # = rollout --g1 --g2 --g3
    meta = {"ckpt": a.ckpt, "env_version": a.env_version, "policy_type": pcfg.type, "gates": gates.summary_config(),
            "gate_pose_source": a.gate_pose_source, "gate_target": a.gate_target, "gate_fix": a.gate_fix,
            "git_commit": subprocess.check_output(["git", "-C", str(SIM), "rev-parse", "HEAD"], text=True).strip(),
            "scene_sha256": hashlib.sha256(json.dumps(env.cfg.to_dict(), sort_keys=True, default=float).encode()).hexdigest(),
            "script": "diag_trace", "src_codes": SRC}

    # step log: wrap env.step (same object every caller uses) and mark the gate calls; the rollout itself is unchanged
    state = {"src": SRC["policy"]}
    log = {}
    step0 = env.step

    def step(arm_ctrl, grip_cmd):
        r = step0(arm_ctrl, grip_cmd)
        P = env.brick_poses()
        tp, R = env.tcp()
        log["t"].append(env.t)
        log["k"].append(getattr(gates, "k", -1))
        log["tgt"].append(getattr(gates, "tgt", -1))                 # gate reference slot (= k for target turn)
        log["src"].append(state["src"])
        log["pose"].append(P.astype(np.float32))
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
    gates._g0_grasp = mark(gates._g0_grasp, SRC["g0"])         # 2026-11-10 K2 (inside before)

    res = []
    for seed in a.seeds:
        for key in ("t", "k", "tgt", "src", "pose", "tcp", "lock", "grip"):
            log[key] = []
        r = run_wall(env, policy, pre, post, cfg_in, seed, torch, None, q_ret, gates)
        np.savez_compressed(out / f"trace_{seed}.npz", t=np.array(log["t"], np.int32), k=np.array(log["k"], np.int8),
                            tgt=np.array(log["tgt"], np.int8),
                            src=np.array(log["src"], np.int8), pose=np.stack(log["pose"]), tcp=np.stack(log["tcp"]),
                            lock=np.array(log["lock"], np.int8), grip=np.array(log["grip"], np.float32))
        r["lock_events"] = env.lock_events
        if seed in ref:
            o = ref[seed]
            r["same_as_eval"] = {"slots_filled": o["eval"]["slots_filled"] == r["eval"]["slots_filled"],
                                 "max_final_err_diff_mm": float(max(abs(x - y) for b0, b1 in zip(o["bricks"], r["bricks"])
                                                                    for x, y in zip(b0["final_err_mm"], b1["final_err_mm"]))),
                                 "sim_s": [o["sim_s"], r["sim_s"]]}
        res.append(r)
        print(f"seed {seed}: filled {sum(r['eval']['slots_filled'])}/16, {r['wall_clock_s']:.0f} s"
              + (f", same as eval {r['same_as_eval']}" if "same_as_eval" in r else ""), flush=True)
        (out / "trace.json").write_text(json.dumps({"meta": meta, "walls": res}, indent=1, default=float))
    env.close()


if __name__ == "__main__":
    main()

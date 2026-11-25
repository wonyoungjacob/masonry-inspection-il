"""나4 G1 판정 시험: 대본으로 정상 집기 16장 + 일부러 빈손 집기(벽돌 옆) 16번 → 하중·각도 분포, 각도 확인 순간의 벽돌 밀림.

정상: 시작값 700000의 공급 칸 벽돌 0 ~ 15를 대본 그대로 집고 1.5 cm 들어 올린 뒤 G1 확인.
빈손: 같은 벽돌의 긴 축 방향 바깥(같은 줄 이웃과 먼 쪽), 벽돌 끝에서 63 mm(패드 반폭 33 + 여유 30) 떨어진 곳을 같은 높이에서 집음.
문턱은 정하지 않는다(분포만). 정답 좌표는 시험 배치와 밀림 확인에만 쓴다.

실행: MUJOCO_GL=egl venv/bin/python -m wall3.test_g1 --out outputs/wall3/<이름> [--env-version main_v1]
"""
import argparse, json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from wall3 import gripper as G
from wall3.config import scene_config
from wall3.ik import down_R
from wall3.poses import yaw_of_quat
from wall3.scene import WallEnv, _quat_rotmat
from wall3.expert import WallExpert
from wall3.g1 import G1, G1Config

SEED = 700000


def empty_side(env, k):
    pb, qb = env.brick_pose(k)
    ax = np.array([np.cos(yaw_of_quat(qb)), np.sin(yaw_of_quat(qb)), 0.0])
    best = None
    for sgn in (1.0, -1.0):
        gap = np.inf
        for j in range(env.n):
            pj = env.brick_pose(j)[0]
            if j == k or abs(pj[1] - pb[1]) > 0.10:
                continue
            dx = (pj[0] - pb[0]) * sgn
            if dx > 0:
                gap = min(gap, dx - env.design[j]["L"] / 2)
        if best is None or gap > best[1]:
            best = (sgn, gap)
    return ax * best[0], float(best[1])


def trial(args):
    env_version, mode, k = args
    env = WallEnv(scene_config(env_version))
    ex = WallExpert(env)
    env.reset(SEED, arm_q=ex.q_ret)
    g1 = G1(env, G1Config())
    ex.on_step, ex.q_cmd, ex.grip_cmd, ex.fail, ex.touch, ex.brick_k = None, ex.q_ret.copy(), 0.0, None, [], k
    ex.in_hand, ex.max_drift, ex.steps = None, 0.0, 0
    c = ex.cfg
    pb, qb = env.brick_pose(k)
    yb = yaw_of_quat(qb)
    L = env.design[k]["L"]
    p, clear = pb.copy(), None
    if mode == "empty":
        u, clear = empty_side(env, k)
        p = pb + u * (L / 2 + 0.033 + 0.030)
    p = p + [0, 0, ex.dzp]
    R = down_R(yb)
    q_app = ex._solve(p + [0, 0, c.approach], R, ex.q_ret, ex.q_ret[0])
    ex._joint_move(q_app, 0.0, None, "to_supply")
    ex._cart_move(p, R, c.t_down, 0.0, None, "grasp_descend")
    gc = env.cfg.gripper
    n = int(round(c.t_close / ex.dt))
    cg = ex.grip_close_cmd()
    ex._hold(n, [min(cg, G.ramp(gc, True, (i + 1) * ex.dt, gc.open_ctrl)) for i in range(n)], None, "grasp_close")
    z0 = env.brick_pose(k)[0][2]
    ex._cart_move(p + [0, 0, g1.cfg.lift], R, 0.3, cg, None, "grasp_lift")
    q_hold, grip_hold = env.last_action[:6].copy(), float(env.last_action[6])

    def rel():
        pt, Rt = env.tcp()
        pb_, qb_ = env.brick_pose(k)
        return Rt.T @ (pb_ - pt), _quat_rotmat(qb_)

    # the angle check's effect on the brick: in-hand pose just before the squeeze vs after the restore
    marks = {}

    def on_step(label):
        if label == "g1_load":
            marks["before"] = rel()
        if label == "g1_restore":
            marks["after"] = rel()

    r = g1.measure(q_hold, grip_hold, on_step)
    (p0, R0), (p1, R1) = marks["before"], marks["after"]
    tilt = float(np.degrees(np.arccos(np.clip((np.trace(R0.T @ R1) - 1) / 2, -1, 1))))
    out = {"mode": mode, "seed": SEED, "brick": k, "half": bool(env.design[k]["half"]), "empty_side_gap_m": clear,
           "load_N": r["load_N"], "angle_rad": r["angle_rad"], "tare_N": r["tare_N"], "fz_samples": r["fz_samples"],
           "brick_lifted_mm": float((env.brick_pose(k)[0][2] - z0) * 1000),
           "slip_during_angle_check_mm": float(np.linalg.norm(p1 - p0) * 1000), "rot_during_angle_check_deg": tilt,
           "expert_fail": ex.fail, "touch": ex.touch[:5],
           "lock_events": [{kk: v for kk, v in e.items() if kk in ("event", "t", "reason", "brick")} for e in env.lock_events]}
    env.close()
    return out


def stats(v):
    v = np.asarray(v, float)
    return {"n": int(len(v)), "min": float(v.min()), "median": float(np.median(v)), "max": float(v.max())} if len(v) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--env-version", default="main_v1")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} 이미 있음")
    out.mkdir(parents=True)
    with Pool(16) as p:
        rows = p.map(trial, [(a.env_version, m, k) for m in ("normal", "empty") for k in range(16)])
    summ = {}
    for m in ("normal", "empty"):
        r = [x for x in rows if x["mode"] == m]
        summ[m] = {"load_N": stats([x["load_N"] for x in r]), "angle_rad": stats([x["angle_rad"] for x in r]),
                   "load_N_full": stats([x["load_N"] for x in r if not x["half"]]), "load_N_half": stats([x["load_N"] for x in r if x["half"]]),
                   "angle_rad_full": stats([x["angle_rad"] for x in r if not x["half"]]), "angle_rad_half": stats([x["angle_rad"] for x in r if x["half"]]),
                   "brick_lifted_ge_10mm": sum(x["brick_lifted_mm"] > 10 for x in r),
                   "slip_during_angle_check_mm": stats([x["slip_during_angle_check_mm"] for x in r]),
                   "rot_during_angle_check_deg": stats([x["rot_during_angle_check_deg"] for x in r]),
                   "locked": sum(any(e["event"] == "lock" for e in x["lock_events"]) for x in r),
                   "releases": sorted({e["reason"] for x in r for e in x["lock_events"] if e["event"] == "release"}),
                   "expert_fail": [x["expert_fail"] for x in r if x["expert_fail"]]}
    n, e = summ["normal"], summ["empty"]
    summ["separation"] = {"load_gap_N (normal min - empty max)": n["load_N"]["min"] - e["load_N"]["max"],
                          "angle_gap_rad (empty min - normal max)": e["angle_rad"]["min"] - n["angle_rad"]["max"]}
    res = {"env_version": a.env_version, "g1_config": G1Config().to_dict(), "summary": summ, "trials": rows}
    (out / "test_g1.json").write_text(json.dumps(res, indent=1, default=float, ensure_ascii=False))
    print(json.dumps(summ, indent=1, default=float, ensure_ascii=False))


if __name__ == "__main__":
    main()

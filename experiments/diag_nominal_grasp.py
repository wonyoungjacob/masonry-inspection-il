"""도면 공급칸 위치로 집으면 충분한가 (2026-11-10, 실험파이프라인_실패원인제거_20261110.md K2 사전 확인). 대본 동작만, 정책 없음.

공급대 벽돌은 도면 공급칸 중심에서 ±10 mm · ±10° 흩어져 놓인다(P6). 대본의 집기 동작(접근 → 내려감 → 닫기 → 1.5 cm 들기 → G1 판정)을
  truth  : 벽돌의 참 위치 · 방향으로 (= 시연, 비교 기준)
  nominal: 도면 공급칸 중심 · 방향 0°로 (검측이 실제로 쓸 수 있는 정보만)
두 가지로 하고, 들렸는지 · G1 판정 · 손 안에서 벽돌 중심이 얼마나 벗어났나(길이 · 폭 방향) · 기울기 · 다른 벽돌이 움직였나를 잰다.
참값은 기록에만 쓴다.

실행: MUJOCO_GL=egl PYTHONPATH=. venv/bin/python -m experiments.diag_nominal_grasp --seeds 700300 ... --out <json>
"""
import argparse, json
from pathlib import Path

import numpy as np

from wall3 import gripper as G
from wall3.config import scene_config
from wall3.expert import WallExpert
from wall3.g1 import G1, G1Config
from wall3.ik import down_R
from wall3.poses import yaw_of_quat
from wall3.scene import WallEnv, _quat_rotmat


def grasp(env, ex, g1, k, mode):
    c = ex.cfg
    pb, qb = env.brick_pose(k)
    if mode == "truth":
        p, yaw0 = pb.copy(), yaw_of_quat(qb)
    else:
        p, yaw0 = env.supply_pose(k)                           # drawing: slot centre, table height + H / 2, yaw 0 (no scatter)
    best = None
    for yaw in (yaw0, yaw0 + np.pi):
        R = down_R(yaw)
        q_app = ex._solve(p + [0, 0, ex.dzp + c.approach], R, ex.q_ret, ex.q_ret[0])
        if q_app is not None and (best is None or abs(q_app[5] - ex.q_ret[5]) < abs(best[1][5] - ex.q_ret[5])):
            best = (R, q_app)
    R, q_app = best
    before = env.brick_poses().copy()
    ex._joint_move(q_app, 0.0, None, "to_supply")
    ex._cart_move(p + [0, 0, ex.dzp], R, c.t_down, 0.0, None, "grasp_descend")
    gc = env.cfg.gripper
    n = int(round(c.t_close / ex.dt))
    cg = ex.grip_close_cmd()
    ex._hold(n, [min(cg, G.ramp(gc, True, (i + 1) * ex.dt, gc.open_ctrl)) for i in range(n)], None, "grasp_close")
    ex._cart_move(p + [0, 0, ex.dzp + g1.cfg.lift], R, 0.3, cg, None, "grasp_lift")
    m = g1.measure(env.last_action[:6].copy(), float(env.last_action[6]))
    ok = g1.decide(m)
    pt, Rt = env.tcp()
    pk, qk = env.brick_pose(k)
    rel = Rt.T @ (pk - pt)
    Rb = _quat_rotmat(qk)
    others = [j for j in range(env.n) if j != k]
    moved = np.linalg.norm(env.brick_poses()[others, :3] - before[others, :3], axis=1).max()
    return {"k": k, "mode": mode, "lifted": bool(pk[2] - pb[2] > 0.010), "g1_grasped": ok, "load_N": round(m["load_N"], 2),
            "along_mm": round(float(rel[0]) * 1000, 1), "across_mm": round(float(rel[1]) * 1000, 1),
            "tilt_deg": round(float(np.degrees(np.arccos(np.clip(abs(Rb[2, 2]), -1, 1)))), 1),
            "aim_err_mm": round(float(np.hypot(*(p[:2] - pb[:2]))) * 1000, 1),
            "yaw_err_deg": round(float(np.degrees((yaw0 - yaw_of_quat(qb) + np.pi / 2) % np.pi - np.pi / 2)), 1),
            "others_moved_mm": round(float(moved) * 1000, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--env-version", default="main_v1_eval")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"{a.out} 이미 있음: 덮어쓰지 않음")
    if any(not 700300 <= s <= 700399 for s in a.seeds):
        raise SystemExit("시작값은 700300 ~ 700399만")
    env = WallEnv(scene_config(a.env_version))
    ex = WallExpert(env)
    rows = []
    for seed in a.seeds:
        for mode in ("truth", "nominal"):
            for k in range(env.n):
                env.reset(seed, arm_q=ex.q_ret)
                ex.on_step, ex.q_cmd, ex.grip_cmd, ex.fail, ex.touch, ex.brick_k = None, ex.q_ret.copy(), 0.0, None, [], k
                ex.in_hand, ex.max_drift, ex.steps = None, 0.0, 0
                ex._watch = lambda held, phase: None
                g1 = G1(env, G1Config(thr_load=4.5, thr_angle=0.47))
                rows.append({"seed": seed, **grasp(env, ex, g1, k, mode)})
    summ = {}
    for mode in ("truth", "nominal"):
        R = [r for r in rows if r["mode"] == mode]
        a_ = lambda key: np.array([r[key] for r in R], float)
        summ[mode] = {"n": len(R), "lifted": int(sum(r["lifted"] for r in R)), "g1_grasped": int(sum(r["g1_grasped"] for r in R)),
                      "along_abs_mm_median": float(np.median(np.abs(a_("along_mm")))), "along_abs_mm_max": float(np.abs(a_("along_mm")).max()),
                      "across_abs_mm_max": float(np.abs(a_("across_mm")).max()), "tilt_deg_max": float(a_("tilt_deg").max()),
                      "others_moved_mm_max": float(a_("others_moved_mm").max()), "aim_err_mm_max": float(a_("aim_err_mm").max()),
                      "fail_k": sorted({r["k"] for r in R if not (r["lifted"] and r["g1_grasped"])})}
    print(json.dumps(summ, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps({"summary": summ, "rows": rows}, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()

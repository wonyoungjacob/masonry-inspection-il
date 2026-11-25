"""나2 결정 시험: 쥔 동안 고정 + 현실성 조건 (2차 지시 나2 결정 "시험" 표).

1. 스모크 3벽 (대본, main_v1): 손안 밀림 최대, 잡기 48번 고정 여부, 과부하 해제 수, 벽돌 한 장 시간
2. 고정 순간 튐 (고정 직후 첫 물리 스텝의 손안 위치 변화)
3. 나쁜 잡기: ① 빈손(벽돌 끝 밖 125 mm) ② 한쪽 패드만(벽돌 끝 +5 mm, 집게 yaw +20°) ③ 끝 가까이(끝에서 30 mm)
4. 과부하: 정상으로 잡아 든 뒤 받침(탁자)에 20 mm 눌러 박기 → 해제되고 손안에서 밀리는지
5. 파일럿 재현: pilot_20261016으로 700000 재현이 나1과 같은지

실행: MUJOCO_GL=egl venv/bin/python -m wall3.test_lock --out outputs/wall3/<이름>
"""
import argparse, json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from wall3.config import scene_config, L_FULL, L_HALF, H
from wall3.ik import down_R
from wall3.poses import yaw_of_quat
from wall3.scene import WallEnv, _quat_rotmat
from wall3.expert import WallExpert
from wall3.run_smoke import one as smoke_one

SIM = Path(__file__).resolve().parents[1]


def trial(args):
    mode, seed, k = args
    env = WallEnv(scene_config("main_v1"))
    ex = WallExpert(env)
    env.reset(seed, arm_q=ex.q_ret)
    ex.on_step, ex.q_cmd, ex.grip_cmd, ex.fail, ex.touch, ex.brick_k = None, ex.q_ret.copy(), 0.0, None, [], k
    ex.in_hand, ex.max_drift, ex.steps = None, 0.0, 0
    c = ex.cfg
    pb, qb = env.brick_pose(k)
    yb = yaw_of_quat(qb)
    L = env.design[k]["L"]
    ax = np.array([np.cos(yb), np.sin(yb), 0.0])
    yaw, p = yb, pb.copy()
    if mode == "empty":
        p = pb + ax * (L / 2 + 0.030 + 0.014 + 0.050)
    elif mode == "one_pad":
        p, yaw = pb + ax * (L / 2 + 0.005), yb + np.deg2rad(20)
    elif mode == "near_end":
        p = pb + ax * (L / 2 - 0.030)
    p = p + [0, 0, ex.dzp]
    R = down_R(yaw)
    q_app = ex._solve(p + [0, 0, c.approach], R, ex.q_ret, ex.q_ret[0])
    if q_app is None:
        return {"mode": mode, "seed": seed, "brick": k, "error": "ik"}
    ex._joint_move(q_app, 0.0, None, "to_supply")
    ex._cart_move(p, R, c.t_down, 0.0, None, "grasp_descend")
    from wall3 import gripper as G
    gc = env.cfg.gripper
    n = int(round(c.t_close / ex.dt))
    cg = ex.grip_close_cmd()
    ex._hold(n, [min(cg, G.ramp(gc, True, (i + 1) * ex.dt, gc.open_ctrl)) for i in range(n)], None, "grasp_close")
    z0 = env.brick_pose(k)[0][2]
    ex._cart_move(p + [0, 0, c.approach], R, c.t_up, cg, None, "grasp_lift")
    ex._hold(20, cg, k, "hold")
    pb2, qb2 = env.brick_pose(k)
    tilt = float(np.degrees(np.arccos(np.clip(abs(_quat_rotmat(qb2)[2, 2]), -1, 1))))
    out = {"mode": mode, "seed": seed, "brick": k, "half": bool(env.design[k]["half"]),
           "lifted_mm": float((pb2[2] - z0) * 1000), "tilt_after_lift_deg": tilt, "collision_or_fail": ex.fail}
    if mode == "overload":
        # press the held brick 20 mm into the supply table (collision watch off: the press is the point of the test)
        ex.fail = None
        ex._watch = lambda held, phase: None
        rel0 = ex._brick_in_tcp(k)[0]
        ex._cart_move(p - [0, 0, 0.020], R, 0.6, cg, k, "press")
        ex._hold(10, cg, k, "press_hold")
        out["press_fail"] = ex.fail
        out["after_press_brick_z_mm"] = float(env.brick_pose(k)[0][2] * 1000)
        out["supply_brick_z_mm"] = float(z0 * 1000)
        out["in_hand_shift_after_press_mm"] = ((ex._brick_in_tcp(k)[0] - rel0) * 1000).round(2).tolist()
    out["events"] = [e for e in env.lock_events]
    out["locked"] = any(e["event"] == "lock" for e in env.lock_events)
    out["releases"] = [e["reason"] for e in env.lock_events if e["event"] == "release"]
    out["locked_at_end"] = env.lock_b >= 0
    env.close()
    return out


def pilot_repro(_):
    from wall3.spec_metrics import evaluate_env
    env = WallEnv(scene_config("pilot_20261016"))
    ex = WallExpert(env)
    r = ex.run(700000)
    ev = evaluate_env(env)
    old = json.loads((SIM / "outputs/wall3/pilot_20261016/raw/wall_700000.json").read_text())
    same = all(abs(ev[k]["value_mm"] - old["eval"][k]["value_mm"]) < 1e-9 for k in ("2_line_plane", "3_plumb", "4_position"))
    same &= abs(ev["iou_mean"] - old["eval"]["iou_mean"]) < 1e-12
    same &= [[b["start"], b["end"]] for b in r["bricks"]] == [[b["start"], b["end"]] for b in old["bricks"]]
    return {"identical_to_pilot": bool(same), "locks": len(env.lock_events)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} 이미 있음")
    out.mkdir(parents=True)
    res = {}
    with Pool(16) as p:
        smoke = p.starmap(smoke_one, [(s, "main_v1") for s in (700000, 700001, 700002)])
        trials = p.map(trial, [(m, s, k) for m in ("normal", "empty", "one_pad", "near_end", "overload")
                               for s in (700000, 700001, 700002, 700003) for k in (0, 5)])
        repro = p.map(pilot_repro, [0])[0]
    # 1, 2. smoke
    br = [b for w in smoke for b in w["bricks"]]
    locks = [e for b in br for e in b.get("lock_events", []) if e["event"] == "lock"]
    rel = [e for b in br for e in b.get("lock_events", []) if e["event"] == "release"]
    res["smoke"] = {"walls": [w["seed"] for w in smoke], "fail": [w.get("fail") for w in smoke], "spec_pass": [w["eval"]["spec_pass"] for w in smoke],
                    "grasps": len(br), "grasps_locked": sum(b.get("locks", 0) >= 1 for b in br),
                    "locks_total": len(locks), "releases_by_reason": {r: sum(e["reason"] == r for e in rel) for r in sorted({e["reason"] for e in rel})},
                    "overload_releases": sum(e["reason"] in ("force", "torque", "no_contact") for e in rel),
                    "in_hand_drift_transport_mm_max": max(b["drift_transport_mm"] for b in br),
                    "in_hand_drift_max_mm": max(b["max_drift_mm"] for b in br),
                    "lock_max_dev_mm": max(b.get("lock_max_dev_mm", 0) for b in br),
                    "lock_jump_mm_max": max(e.get("jump_mm", 0) for e in locks),
                    "brick_time_s_max": max(b["duration_s"] for b in br),
                    "pad_normal_N_at_lock": {"min": min(min(e["N_left"], e["N_right"]) for e in locks), "median": float(np.median([e["N_left"] + e["N_right"] for e in locks]) / 2)},
                    "grasp_offset_mm": {"median": float(np.median([e["grasp_offset_mm"] for e in locks])), "max": max(e["grasp_offset_mm"] for e in locks)},
                    "grasp_yaw_offset_deg": {"median": float(np.median([e["grasp_yaw_offset_deg"] for e in locks])), "max": max(e["grasp_yaw_offset_deg"] for e in locks)},
                    "items_mm": [{k: w["eval"][k].get("value_mm") for k in ("2_line_plane", "3_plumb", "4_position")} for w in smoke]}
    res["trials"] = trials
    summ = {}
    for m in ("normal", "empty", "one_pad", "near_end", "overload"):
        t = [x for x in trials if x["mode"] == m]
        summ[m] = {"n": len(t), "locked": sum(x.get("locked", False) for x in t), "releases": sum(len(x.get("releases", [])) for x in t),
                   "release_reasons": sorted({r for x in t for r in x.get("releases", [])}),
                   "lifted_ge_50mm": sum(x.get("lifted_mm", 0) > 50 for x in t), "tilt_max_deg": max(x.get("tilt_after_lift_deg", 0) for x in t)}
    res["trial_summary"] = summ
    res["pilot_repro"] = repro
    (out / "test_lock.json").write_text(json.dumps(res, indent=1, default=float, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k != "trials"}, indent=1, default=float, ensure_ascii=False))


if __name__ == "__main__":
    main()

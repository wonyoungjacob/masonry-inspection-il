"""나2 손안 밀림 진단과 해결책 비교 (2차 지시 나2).

1) 진단 (--trace): 한 벽돌 운반 구간(to_wall)에서 물리 스텝(1 ms)마다 손안 상대 위치, 패드 접촉력(법선·접선 합과 마찰 한계 비율),
   TCP 가속도를 기록하고, 밀림 증분이 제어 주기(50 ms) 안의 어느 물리 스텝에서 생기는지 본다.
2) 비교: 같은 시작값·같은 벽돌들의 운반 구간 손안 밀림(벽 위에서 재조준하기 전 최대값)을 방법별로 잰다.

실행: MUJOCO_GL=egl venv/bin/python -m experiments.diag_slip --out outputs/wall3/<이름> [--trace]
"""
import argparse, json, time
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

from wall3.config import scene_config
from wall3.scene import WallEnv
from wall3.expert import WallExpert

VARIANTS = {
    "A_implicitfast_linear (pilot)": dict(integrator="implicitfast", arm_damping="actuator", ctrl_interp="linear"),
    "B_implicitfast_hermite": dict(integrator="implicitfast", arm_damping="actuator", ctrl_interp="hermite"),
    "C_implicitfast_lowpass": dict(integrator="implicitfast", arm_damping="actuator", ctrl_interp="lowpass"),
    "D_euler_jointdamping_linear": dict(integrator="euler", arm_damping="joint", ctrl_interp="linear"),
    "E_euler_jointdamping_hermite": dict(integrator="euler", arm_damping="joint", ctrl_interp="hermite"),
    "F_euler_wristkd150_linear": dict(integrator="euler", arm_damping="actuator", arm_kd=(500, 500, 500, 150, 150, 150), ctrl_interp="linear"),
    "G_implicit_linear": dict(integrator="implicit", arm_damping="actuator", ctrl_interp="linear"),
    # H: Euler; passive joint damping (10 10 5 2 2 2) moved into the servo damping, wrists' servo damping 150 (+2)
    "H_euler_nopassive_wristkd152": dict(integrator="euler", arm_damping="actuator", arm_kd=(510, 510, 505, 152, 152, 152),
                                         arm_joint_damping=(0, 0, 0, 0, 0, 0), ctrl_interp="linear"),
    "J_euler_wristkd150_noslip20": dict(integrator="euler", arm_damping="actuator", arm_kd=(500, 500, 500, 150, 150, 150),
                                        ctrl_interp="linear", noslip_iterations=20),
    "K_euler_wristkd150_noslip100": dict(integrator="euler", arm_damping="actuator", arm_kd=(500, 500, 500, 150, 150, 150),
                                         ctrl_interp="linear", noslip_iterations=100),
    "L_rk4_wristkd150": dict(integrator="rk4", arm_damping="actuator", arm_kd=(500, 500, 500, 150, 150, 150), ctrl_interp="linear"),
    "M_euler_wristkd150_dt0.5ms (diagnostic only)": dict(integrator="euler", arm_damping="actuator", arm_kd=(500, 500, 500, 150, 150, 150),
                                                         ctrl_interp="linear", timestep=0.0005),
    "N_rk4_kd500": dict(integrator="rk4", arm_damping="actuator", ctrl_interp="linear"),
    "I_implicitfast_nopassive": dict(integrator="implicitfast", arm_damping="actuator", arm_kd=(510, 510, 505, 502, 502, 502),
                                     arm_joint_damping=(0, 0, 0, 0, 0, 0), ctrl_interp="linear"),
}


def compare(args):
    name, kw, seed, n_bricks = args
    env = WallEnv(scene_config("main_v1", mortar_stiffness="brick_sim", **kw))
    ex = WallExpert(env)
    t0 = time.time()
    r = ex.run(seed, max_bricks=n_bricks, settle_s=0.0)
    out = {"variant": name, "config": kw, "seed": seed, "fail": r.get("fail"), "wall_s": time.time() - t0,
           "transport_drift_mm": [round(b.get("drift_transport_mm", float("nan")), 2) for b in r["bricks"]],
           "brick_time_s": [b.get("duration_s") for b in r["bricks"]],
           "end_err_mm": [b.get("end_err_mm") for b in r["bricks"]]}
    env.close()
    return out


def trace(seed=700000, brick=2):
    """Per physics step log of one transport (pilot physics)."""
    env = WallEnv(scene_config("pilot_20261016"))
    ex = WallExpert(env)
    m, d = env.model, env.data
    rows = []
    f6 = np.zeros(6)
    orig = env.step

    def step(arm, grip):
        if ex.brick_k != brick or ex.in_hand is None or getattr(ex, "_phase", "") != "to_wall":
            return orig(arm, grip)
        new = np.r_[np.asarray(arm, float), float(grip)]
        tp_prev = None
        for j, c in enumerate(env._interp(env.last_action, new)):
            d.ctrl[env.arm_act] = c[:6]; d.ctrl[env.grip_act] = [c[6], -c[6]]
            mujoco.mj_step(m, d)
            rel = ex._brick_in_tcp(brick)[0]
            fn = ft = 0.0; ratio = 0.0
            for i in range(d.ncon):
                cc = d.contact[i]
                if env.brick_gid[brick] in (cc.geom1, cc.geom2):
                    o = m.geom(cc.geom2 if cc.geom1 == env.brick_gid[brick] else cc.geom1).name
                    if o.startswith("g_"):
                        mujoco.mj_contactForce(m, d, i, f6)
                        fn += f6[0]; t_ = np.hypot(f6[1], f6[2]); ft += t_
                        ratio = max(ratio, t_ / (cc.friction[0] * f6[0] + 1e-9))
            rows.append({"t": env.t, "j": j, "rel": rel.tolist(), "Fn": fn, "Ft": ft, "max_ratio": ratio,
                         "tcp": env.tcp()[0].tolist()})
        env.prev_action = env.last_action; env.last_action = new; env.t += 1
        env._update_cure()

    env.step = step
    orig_send = ex._send

    def send(q, qd, qdd, grip, held, phase):
        ex._phase = phase
        return orig_send(q, qd, qdd, grip, held, phase)

    ex._send = send
    ex.run(seed, max_bricks=brick + 1, settle_s=0.0)
    P = np.array([r["rel"] for r in rows]); T = np.array([r["tcp"] for r in rows])
    acc = np.zeros(len(rows))
    acc[1:-1] = np.linalg.norm(T[2:] - 2 * T[1:-1] + T[:-2], axis=1) / 1e-6
    dslip = np.r_[0, np.linalg.norm(np.diff(P, axis=0), axis=1)]
    J = np.array([r["j"] for r in rows])
    by_j = {int(j): {"slip_um_mean": float(dslip[J == j].mean() * 1e6),
                     "tcp_acc_mean": float(acc[J == j].mean()), "tcp_acc_max": float(acc[J == j].max()),
                     "Fn_mean": float(np.mean([r["Fn"] for r, jj in zip(rows, J) if jj == j])),
                     "max_ratio_mean": float(np.mean([r["max_ratio"] for r, jj in zip(rows, J) if jj == j]))} for j in range(50)}
    return {"seed": seed, "brick": brick, "n_physics_steps": len(rows), "total_slip_mm": float(np.linalg.norm(P[-1] - P[0]) * 1000),
            "path_slip_mm": float(dslip.sum() * 1000), "tcp_acc_max": float(acc.max()), "by_substep": by_j,
            "Fn_mean": float(np.mean([r["Fn"] for r in rows])), "Ft_mean": float(np.mean([r["Ft"] for r in rows])),
            "frac_steps_at_cone": float(np.mean([r["max_ratio"] > 0.99 for r in rows]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[700000])
    ap.add_argument("--bricks", type=int, default=6)
    ap.add_argument("--only", nargs="*", default=[])
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} 이미 있음")
    out.mkdir(parents=True)
    res = {}
    if a.trace:
        res["trace"] = trace()
        tr = res["trace"]
        print(f"trace: total slip {tr['total_slip_mm']:.2f} mm over {tr['n_physics_steps']} steps, path {tr['path_slip_mm']:.2f} mm, "
              f"TCP acc max {tr['tcp_acc_max']:.1f} m/s2, Fn mean {tr['Fn_mean']:.1f} N, Ft mean {tr['Ft_mean']:.1f} N, "
              f"steps with a contact at the cone {tr['frac_steps_at_cone']:.2f}")
        for j in (0, 1, 2, 5, 10, 25, 48, 49):
            b = tr["by_substep"][j]
            print(f"   substep {j:2d}: slip {b['slip_um_mean']:.1f} um/step, TCP acc mean {b['tcp_acc_mean']:.2f} max {b['tcp_acc_max']:.2f} m/s2, Fn {b['Fn_mean']:.1f} N, max ratio {b['max_ratio_mean']:.2f}")
    jobs = [(n, kw, s, a.bricks) for n, kw in VARIANTS.items() for s in a.seeds if not a.only or n[0] in a.only]
    with Pool(min(len(jobs), 10)) as p:
        res["compare"] = p.map(compare, jobs)
    for r in res["compare"]:
        dr = [x for x in r["transport_drift_mm"] if x == x]
        print(f"{r['variant']:32s} seed {r['seed']} fail {r['fail']} drift max {max(dr) if dr else None} mm {r['transport_drift_mm']} "
              f"time max {max([t for t in r['brick_time_s'] if t] or [0]):.2f} s")
    (out / "diag_slip.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()

"""단계 1 재개 2차(A1+): ①②③ + ④ 링크 구속 강성(2F-85 값) + ⑤ 패드 평행 결합. 쥐는 힘(토크 한계)을 훑는다.

실행: MUJOCO_GL=egl venv/bin/python -m experiments.sweep_gripper140_a1plus --out outputs/wall3/<이름>
"""
import argparse, itertools, json, subprocess
from multiprocessing import Pool
from pathlib import Path

from wall3 import gripper as G
from experiments.diag_gripper140_a import run, SIM, W

TORQUES = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]
DZ = [0.0, 0.004]
SOLVER = dict(impratio=20.0, noslip=0)       # set by --solver
REST = 0.5


def _init(solver, rest):
    global REST
    SOLVER.update(solver); REST = rest


def job(args):
    tq, dz, kind = args
    cfg = G.a1plus(tq)
    r = run(cfg, kind, dz=dz, rest_after_move=REST, **SOLVER)
    r.update(torque_limit=tq, dz=dz)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--solver", choices=["robosuite", "brick_sim"], default="robosuite",
                    help="robosuite: impratio 20, noslip 0 / brick_sim(지난 실험 scene.py): impratio 10, noslip 3")
    ap.add_argument("--rest", type=float, default=0.5, help="hold after the move, counted in the slip (s)")
    a = ap.parse_args()
    global REST
    REST = a.rest
    if a.solver == "brick_sim":
        SOLVER.update(impratio=10.0, noslip=3)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} 이미 있음: 덮어쓰지 않음")
    out.mkdir(parents=True)
    grid = [(tq, dz, k) for tq, dz, k in itertools.product(TORQUES, DZ, ("full", "half"))] + [(tq, 0.0, None) for tq in TORQUES]
    with Pool(24, initializer=_init, initargs=(dict(SOLVER), REST)) as p:
        res = p.map(job, grid)
    meta = {"git_commit": subprocess.check_output(["git", "-C", str(SIM), "rev-parse", "HEAD"], text=True).strip(),
            "calibration_90mm": G.calibrate(G.a1plus(4.0), W), "config_example": G.a1plus(4.0).to_dict(), "solver": SOLVER, "rest_after_move_s": REST}
    (out / "result.json").write_text(json.dumps({"meta": meta, "runs": res}, indent=1, default=list))
    empty = {r["torque_limit"]: r["closure_rad"] for r in res if r["brick"] == "none"}
    for r in sorted([r for r in res if r["brick"] != "none"], key=lambda r: (r["torque_limit"], r["dz"], r["brick"])):
        print(f"tau {r['torque_limit']:4.1f} dz {r['dz']*1000:+.0f} {r['brick']:4s} pass {r.get('pass')!s:5s} "
              f"slip {r.get('lift_slip_max_mm', 0):.2f}/{r.get('move_slip_max_mm', 0):.2f} tilt {r.get('lift_tilt_max_deg', 0):.2f}/{r.get('move_tilt_max_deg', 0):.2f} "
              f"N {r['hold_normal_N']['left']:.1f}/{r['hold_normal_N']['right']:.1f} oth {r['hold_normal_N']['other_gripper']:.1f} "
              f"closure {r['closure_rad']:.3f}/empty {empty[r['torque_limit']]:.3f} placed {r.get('placed_xy_err_mm', 0):.2f} mm {r.get('placed_tilt_deg', 0):.2f} deg")


if __name__ == "__main__":
    main()

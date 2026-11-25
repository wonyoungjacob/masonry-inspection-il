"""단계 1 재개: 허용 범위(② 쥐는 힘, ③ 집는 높이) 격자 탐색. 결과는 --out 폴더의 result.json.

실행: MUJOCO_GL=egl venv/bin/python -m experiments.sweep_gripper140_a --out outputs/wall3/<이름>
"""
import argparse, itertools, json, subprocess
from multiprocessing import Pool
from pathlib import Path

from wall3 import gripper as G
from experiments.diag_gripper140_a import run, SIM

TORQUES = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]       # N·m (pad normal force about 10 ~ 30 N)
DZ = [-0.015, -0.008, 0.0, 0.008, 0.015]            # m, pad tip edge above brick mid-height
GAINS = [(100.0, 1.0)]


def job(args):
    kp, kv, tq, dz, kind = args
    cfg = G.GripperConfig(kp=kp, kv=kv, torque_limit=tq)
    r = run(cfg, kind, dz=dz)
    r.update(kp=kp, kv=kv, torque_limit=tq, dz=dz)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} 이미 있음: 덮어쓰지 않음")
    out.mkdir(parents=True)
    grid = [(kp, kv, tq, dz, kind) for (kp, kv), tq, dz, kind in itertools.product(GAINS, TORQUES, DZ, ("full", "half"))]
    grid += [(20.0, 0.0, 0.0, dz, kind) for dz in DZ for kind in ("full", "half")]   # robosuite gain, no limit
    with Pool(a.workers) as p:
        res = p.map(job, grid)
    meta = {"git_commit": subprocess.check_output(["git", "-C", str(SIM), "rev-parse", "HEAD"], text=True).strip()}
    (out / "result.json").write_text(json.dumps({"meta": meta, "runs": res}, indent=1))
    best = sorted(res, key=lambda r: max(r.get("lift_slip_max_mm", 1e9), r.get("move_slip_max_mm", 1e9)))
    print("n", len(res), "pass", sum(r.get("pass", False) for r in res))
    for r in best[:12]:
        print(f"kp {r['kp']:5.0f} tau {r['torque_limit']:3.1f} dz {r['dz']*1000:+5.1f} {r['brick']:4s} slip {r['lift_slip_max_mm']:.2f}/{r['move_slip_max_mm']:.2f} "
              f"tilt {r['lift_tilt_max_deg']:.2f}/{r['move_tilt_max_deg']:.2f} N {r['hold_normal_N']['left']:.1f}/{r['hold_normal_N']['right']:.1f} closure {r['closure_rad']:.3f}")


if __name__ == "__main__":
    main()

"""나3 모르타르 강성 보정 시험 (2차 지시 나3).

같은 시작값 3벽(700000 ~ 700002)을 대본으로 쌓아 보정 전(brick_sim 공식)과 보정 후(impedance 공식, main_v1)를 비교한다.
침하 = 정착(고정) 순간 벽돌이 모르타르 판에 파고든 깊이(접촉 거리 평균). 온장·반장 따로.
완료 기준: 온장 침하 0.8 ~ 1.2 mm, 스모크 3벽 완주, ① 가로줄눈 측정값 7 ~ 13 mm 안. 파일럿 재현(pilot_20261016)도 다시 확인.

실행: MUJOCO_GL=egl venv/bin/python -m wall3.test_mortar --out outputs/wall3/<이름>
"""
import argparse, json, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from wall3.config import scene_config
from wall3.scene import WallEnv
from wall3.expert import WallExpert, expert_config
from wall3.spec_metrics import evaluate_env
from wall3.run_smoke import check_marks
from wall3.test_lock import pilot_repro

SEEDS = (700000, 700001, 700002)


def one(args):
    name, kw, seed = args[:3]
    ekw = args[3] if len(args) > 3 else {}
    env = WallEnv(scene_config("main_v1", **kw))
    ex = WallExpert(env, expert_config("main_v1", **ekw))
    peak = np.zeros(env.n)                                   # deepest bed penetration before the brick cured (transient)
    step0 = env.step

    def step(arm, grip):
        step0(arm, grip)
        for i in range(env.n):
            if env.design[i]["course"] > 1 and not env.cured[i]:
                v = env.bed_sag(i)
                if v == v:
                    peak[i] = max(peak[i], v)

    env.step = step
    t0 = time.time()
    r = ex.run(seed)
    ev = evaluate_env(env)
    br = r["bricks"]
    rel = [e for b in br for e in b.get("lock_events", []) if e["event"] == "release"]
    out = {"variant": name, "seed": seed, "fail": r.get("fail"), "wall_s": time.time() - t0, "spec_pass": ev["spec_pass"],
           "items_mm": {k: ev[k].get("value_mm") for k in ("2_line_plane", "3_plumb", "4_position")},
           "bed_joint_mm": [ev["1_bed_joint"]["upper_min_mm"], ev["1_bed_joint"]["upper_max_mm"]], "bed_joint_pass": ev["1_bed_joint"]["pass"],
           "thickness": {k: ev["5_thickness"].get(k) for k in ("max_mm", "pass")},
           "iou_mean": ev["iou_mean"], "mark_problems": check_marks(r),
           "brick_time_s_max": max(b["duration_s"] for b in br),
           "grasps_locked": sum(b.get("locks", 0) >= 1 for b in br), "grasps": len(br),
           "overload_releases": sum(e["reason"] != "open" for e in rel),
           "in_hand_drift_max_mm": max(b["max_drift_mm"] for b in br),
           "sag_at_cure_mm": r["sag_at_cure_mm"], "sag_end_mm": r["sag_end_mm"], "sag_peak_mm": (peak * 1000).tolist(),
           "expert": ekw,
           "half": [bool(s["half"]) for s in env.design], "course": [int(s["course"]) for s in env.design]}
    env.close()
    return out


def sag_stats(rows, key, half):
    v = [x for r in rows for x, h, c in zip(r[key], r["half"], r["course"]) if c > 1 and h == half and x is not None]
    return {"n": len(v), "mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))} if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--place-gaps", type=float, nargs="*", default=[], help="mm; compare the expert's release height instead")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} 이미 있음")
    out.mkdir(parents=True)
    variants = {"before_brick_sim": (dict(mortar_stiffness="brick_sim"), {}), "after_impedance (main_v1)": (dict(), {})}
    if a.place_gaps:
        variants = {f"after_impedance place_gap {g:g} mm": (dict(), dict(place_gap=g / 1000)) for g in a.place_gaps}
    with Pool(9) as p:
        rows = p.map(one, [(n, kw, s, ekw) for n, (kw, ekw) in variants.items() for s in SEEDS])
        repro = p.map(pilot_repro, [0])[0]
    res = {"walls": rows, "pilot_repro": repro, "summary": {}}
    for n in variants:
        r = [x for x in rows if x["variant"] == n]
        res["summary"][n] = {"fail": [x["fail"] for x in r], "spec_pass": sum(x["spec_pass"] for x in r), "walls": len(r),
                             "sag_full_at_cure_mm": sag_stats(r, "sag_at_cure_mm", False), "sag_half_at_cure_mm": sag_stats(r, "sag_at_cure_mm", True),
                             "sag_full_end_mm": sag_stats(r, "sag_end_mm", False), "sag_half_end_mm": sag_stats(r, "sag_end_mm", True),
                             "sag_full_peak_mm": sag_stats(r, "sag_peak_mm", False), "sag_half_peak_mm": sag_stats(r, "sag_peak_mm", True),
                             "bed_joint_mm": [min(x["bed_joint_mm"][0] for x in r), max(x["bed_joint_mm"][1] for x in r)],
                             "items_mm_max": {k: max(x["items_mm"][k] for x in r) for k in r[0]["items_mm"]},
                             "brick_time_s_max": max(x["brick_time_s_max"] for x in r),
                             "grasps_locked": [sum(x["grasps_locked"] for x in r), sum(x["grasps"] for x in r)],
                             "overload_releases": sum(x["overload_releases"] for x in r),
                             "in_hand_drift_max_mm": max(x["in_hand_drift_max_mm"] for x in r),
                             "mark_problems": sum(len(x["mark_problems"]) for x in r)}
    (out / "test_mortar.json").write_text(json.dumps(res, indent=1, default=float, ensure_ascii=False))
    print(json.dumps(res["summary"], indent=1, default=float, ensure_ascii=False))
    print("pilot repro", repro)


if __name__ == "__main__":
    main()

"""단계 7 보고용 파일럿 집계 (시뮬레이션 없음).

실행: venv/bin/python -m experiments.summarize_pilot --raw outputs/wall3/pilot_<날짜> --root data/wall3_pilot_<날짜> --out <요약 json>
"""
import argparse, json
from collections import Counter
from pathlib import Path

import numpy as np

from wall3.run_smoke import check_marks

ITEMS = ("1_bed_joint", "2_line_plane", "3_plumb", "4_position", "5_thickness")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, nargs="+")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raws = [Path(r) for r in a.raw]
    walls = sorted([json.loads(p.read_text()) for r in raws for p in r.glob("raw/wall_*.json")], key=lambda w: w["seed"])
    cfgs = [json.loads((r / "collect_config.json").read_text()) for r in raws]
    summs = [json.loads((r / "collect_summary.json").read_text()) for r in raws]
    cfg = cfgs[0]
    summ = {"elapsed_s": sum(s["elapsed_s"] for s in summs), "workers": summs[0]["workers"]}
    n = len(walls)
    passed = [w for w in walls if w["eval"]["spec_pass"] and not w.get("fail")]
    fails = [w for w in walls if not (w["eval"]["spec_pass"] and not w.get("fail"))]

    item_fail = Counter()
    item_fail_slot = {k: Counter() for k in ITEMS}
    reasons = []
    for w in fails:
        ev = w["eval"]
        r = {"seed": w["seed"], "aborted": w.get("fail"), "exists": ev["exists"], "items": {}}
        for k in ITEMS:
            if not ev[k]["pass"]:
                item_fail[k] += 1
                slot = ev[k].get("worst_slot")
                if k == "1_bed_joint":
                    slot = ev[k].get("slots")
                if k == "2_line_plane":
                    slot = ev[k]["worst"].get("plane_slot")
                if k == "5_thickness":
                    slot = ev[k].get("worst_x_m")
                item_fail_slot[k][str(slot)] += 1
                v = ev[k].get("value_mm", ev[k].get("max_mm"))
                r["items"][k] = {"value_mm": v, "tol_mm": ev[k]["tol_mm"], "slot": slot}
        reasons.append(r)

    times = np.array([b["duration_s"] for w in walls for b in w["bricks"] if "duration_s" in b])
    slots = np.array([b["slot"] for w in walls for b in w["bricks"] if "duration_s" in b])
    over = Counter(int(s) for s, t in zip(slots, times) if t > 10.0)
    per_slot_t = {int(k): {"median": float(np.median(times[slots == k])), "max": float(times[slots == k].max())} for k in range(16) if (slots == k).any()}
    marks = {w["seed"]: check_marks(w) for w in walls}
    marks_pass_ok = all(not marks[w["seed"]] for w in passed)

    drift = np.array([b["drift_transport_mm"] for w in walls for b in w["bricks"] if "drift_transport_mm" in b])
    err = np.array([np.hypot(*b["end_err_mm"][:2]) for w in walls for b in w["bricks"] if "end_err_mm" in b])
    dist = np.array([w["base_dist_m"][b["slot"]] for w in walls for b in w["bricks"] if "end_err_mm" in b])
    bins = [(0.60, 0.65), (0.65, 0.70), (0.70, 0.76)]
    by_dist = {f"{lo:.2f}-{hi:.2f} m": {"n": int(((dist >= lo) & (dist < hi)).sum()),
                                        "median_mm": float(np.median(err[(dist >= lo) & (dist < hi)])),
                                        "p95_mm": float(np.percentile(err[(dist >= lo) & (dist < hi)], 95))} for lo, hi in bins}
    iou_mean = [w["eval"]["iou_mean"] for w in walls]
    vals = {k: [w["eval"][k].get("value_mm", w["eval"][k].get("max_mm")) for w in walls] for k in ITEMS[1:4]}
    touch = sum(w.get("n_touch_events", 0) for w in walls)

    lock = None
    br_all = [b for w in walls for b in w["bricks"] if "duration_s" in b]
    if br_all and "locks" in br_all[0]:                  # main_v1 grip lock (2차 나2 결정)
        rel = [e for b in br_all for e in b.get("lock_events", []) if e["event"] == "release"]
        lk = [e for b in br_all for e in b.get("lock_events", []) if e["event"] == "lock"]
        lock = {"grasps": len(br_all), "grasps_locked": sum(b["locks"] >= 1 for b in br_all), "locks": len(lk),
                "releases_by_reason": dict(Counter(e["reason"] for e in rel)),
                "overload_releases": sum(e["reason"] != "open" for e in rel),
                "lock_jump_mm_max": max([e.get("jump_mm", 0.0) for e in lk] or [0.0]),
                "lock_max_dev_mm": max(b.get("lock_max_dev_mm", 0.0) for b in br_all),
                "pad_normal_N_min": min([min(e["N_left"], e["N_right"]) for e in lk] or [None]),
                "grasp_offset_mm": {"median": float(np.median([e["grasp_offset_mm"] for e in lk])), "max": float(np.max([e["grasp_offset_mm"] for e in lk]))} if lk else None,
                "in_hand_drift_max_mm": float(max(b["max_drift_mm"] for b in br_all))}
        sag = [(v, w["design"][i]["half"]) for w in walls for i, v in enumerate(w.get("sag_at_cure_mm", [])) if v is not None]
        if sag:
            lock["sag_at_cure_mm"] = {h: {"n": sum(x[1] == hv for x in sag), "min": float(min(x[0] for x in sag if x[1] == hv)),
                                          "median": float(np.median([x[0] for x in sag if x[1] == hv])), "max": float(max(x[0] for x in sag if x[1] == hv))}
                                      for h, hv in (("full", False), ("half", True)) if any(x[1] == hv for x in sag)}
    raw_bytes = [w["raw_bytes"] for w in walls]
    wall_time = [w["time_s"]["total"] for w in walls]
    ds_meta = Path(a.root) / "meta"
    verify = json.loads((ds_meta / "wall3_verify.json").read_text()) if (ds_meta / "wall3_verify.json").exists() else {}
    ds_bytes = verify.get("bytes")
    sim_s = float(np.mean([w["steps"] for w in walls])) * 0.05
    workers = summ["workers"]
    out = {
        "n_walls_attempted": n, "seeds_attempted": [walls[0]["seed"], walls[-1]["seed"]], "n_spec_pass": len(passed), "pass_rate": len(passed) / n, "n_aborted": sum(bool(w.get("fail")) for w in walls),
        "abort_reasons": [w["fail"] for w in walls if w.get("fail")],
        "item_fail_counts": dict(item_fail), "item_fail_slots": {k: dict(v) for k, v in item_fail_slot.items() if v},
        "fail_details": reasons,
        "item_values_mm": {k: {"median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "max": float(np.max(v))} for k, v in vals.items()},
        "bed_joint_mm": [min(w["eval"]["1_bed_joint"]["upper_min_mm"] for w in walls), max(w["eval"]["1_bed_joint"]["upper_max_mm"] for w in walls)],
        "iou_mean": {"median": float(np.median(iou_mean)), "min": float(np.min(iou_mean))},
        "sensitivity_pass": {f: sum(w["sensitivity"][f] for w in walls) for f in ("1mm", "2mm", "3mm")},
        "brick_time_s": {"n": int(len(times)), "median": float(np.median(times)), "p95": float(np.percentile(times, 95)), "max": float(times.max()),
                         "over_10s": int((times > 10).sum()), "over_10s_slots": dict(over), "per_slot": per_slot_t},
        "marks": {"all_pass_walls_16_ok": marks_pass_ok, "problems": {str(k): v for k, v in marks.items() if v}},
        "in_hand_drift_transport_mm": {"median": float(np.median(drift)), "p95": float(np.percentile(drift, 95)), "max": float(drift.max())},
        "placement_xy_err_mm_by_base_dist": by_dist, "robot_contact_events": int(touch),
        "beds_laid": {"all": all(w["bed_laid_t"]["1"] >= 0 and w["bed_laid_t"]["2"] >= 0 for w in walls)},
        "grip_lock": lock,
        "cost": {"wall_sim_seconds": sim_s, "wall_compute_s_per_worker": {"median": float(np.median(wall_time)), "max": float(np.max(wall_time))},
                 "workers": workers, "collect_elapsed_s": summ["elapsed_s"],
                 "raw_mb_per_wall": float(np.mean(raw_bytes) / 1e6),
                 "dataset_mb_per_pass_wall": (ds_bytes / 1e6 / len(passed)) if ds_bytes and passed else None,
                 "estimate_1000_pass_walls": {
                     "walls_to_attempt": 1000 / (len(passed) / n) if passed else None,
                     "collect_hours_at_this_parallelism": (1000 / (len(passed) / n)) * summ["elapsed_s"] / n / 3600 if passed else None,
                     "convert_hours": (verify.get("build_s", 0) / max(len(passed), 1)) * 1000 / 3600 if verify else None,
                     "raw_gb": (1000 / (len(passed) / n)) * np.mean(raw_bytes) / 1e9 if passed else None,
                     "dataset_gb": (ds_bytes / len(passed)) * 1000 / 1e9 if ds_bytes and passed else None}},
        "dataset": {"root": a.root, "verify": verify},
        "reproducibility": {"collect_runs": [{"raw_dir": str(r), "git_commit": c["git_commit"], "git_dirty_wall3": c["git_dirty"],
                                              "seeds_attempted": [w["seed"] for w in s["walls"]]} for r, c, s in zip(raws, cfgs, summs)],
                            "assets": cfg["assets"], "raw_sha256": {str(w["seed"]): w["raw_sha256"] for w in walls}},
    }
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    print(json.dumps({k: v for k, v in out.items() if k not in ("fail_details", "reproducibility", "dataset")}, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()

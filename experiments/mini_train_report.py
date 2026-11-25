"""미니 학습 집계: lerobot-train 로그에서 학습 곡선(손실)과 속도를 뽑고, 롤아웃 결과를 요약한다 (2차 지시 가1·가2).

실행: venv/bin/python -m experiments.mini_train_report --train outputs/wall3/mini_train_<날짜> --rollouts <폴더> ... --out <요약.json> --png <곡선.png>
"""
import argparse, json, re
from pathlib import Path

import numpy as np

STEP = re.compile(r"step:(\d+(?:\.\d+)?)([KM]?) .*?loss:([0-9.]+).*?updt_s:([0-9.]+) data_s:([0-9.]+)")


def parse_log(p: Path):
    rows = []
    for m in STEP.finditer(p.read_text(errors="ignore")):
        n, u, loss, upd, dat = m.groups()
        step = float(n) * {"": 1, "K": 1e3, "M": 1e6}[u]
        rows.append((step, float(loss), float(upd), float(dat)))
    speed = [float(x) for x in re.findall(r"([0-9.]+)step/s", p.read_text(errors="ignore"))]
    return rows, speed


def summarize_rollout(p: Path):
    d = json.loads((p / "rollout.json").read_text())
    walls = d["walls"]
    br = [b for w in walls for b in w["bricks"]]
    return {"policy": d["meta"]["policy_type"], "ckpt": d["meta"]["ckpt"], "n_action_steps": d["meta"]["n_action_steps"], "walls": len(walls),
            "seeds": [w["seed"] for w in walls],
            "lifted": sum(b["grasped_lifted"] for b in br), "bricks": len(br),
            "retract_end": sum(b["end_reason"] == "retract" for b in br), "stalled": sum(b["stalled"] for b in br),
            "slots_filled_mean": float(np.mean([sum(w["eval"]["slots_filled"]) for w in walls])),
            "spec_pass": sum(w["eval"]["spec_pass"] for w in walls),
            "iou_mean": float(np.mean([w["eval"]["iou_mean"] for w in walls])),
            "final_xy_err_mm_median": float(np.median([b["final_xy_err_mm"] for b in br])),
            "final_xy_err_mm_of_lifted_median": float(np.median([b["final_xy_err_mm"] for b in br if b["grasped_lifted"]])) if any(b["grasped_lifted"] for b in br) else None,
            "wall_clock_s": {"median": float(np.median([w["wall_clock_s"] for w in walls])), "max": float(np.max([w["wall_clock_s"] for w in walls]))},
            "sim_s_per_wall_median": float(np.median([w["sim_s"] for w in walls]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--rollouts", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--png", default="")
    a = ap.parse_args()
    tr = Path(a.train)
    out = {"training": {}, "rollouts": [summarize_rollout(Path(r)) for r in a.rollouts]}
    curves = {}
    for log in sorted(tr.glob("*.log")):
        rows, speed = parse_log(log)
        if not rows:
            continue
        s = np.array(rows)
        curves[log.stem] = s
        out["training"][log.stem] = {"steps_logged": int(s[-1, 0]), "loss_first": float(s[0, 1]), "loss_last": float(s[-1, 1]),
                                    "loss_min": float(s[:, 1].min()),
                                    "loss_at": {str(int(k)): float(s[np.argmin(np.abs(s[:, 0] - k)), 1]) for k in (1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000) if s[-1, 0] >= k},
                                    "update_s_median": float(np.median(s[:, 2])), "data_s_median": float(np.median(s[:, 3])),
                                    "steps_per_s_median": float(np.median(speed[len(speed) // 10:])) if speed else None}
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    if a.png and curves:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, len(curves), figsize=(5 * len(curves), 3.2))
        for ax, (k, s) in zip(np.atleast_1d(axs), curves.items()):
            ax.semilogy(s[:, 0], s[:, 1], lw=1)
            ax.set_title(f"{k} (mini, pilot 50 walls)")
            ax.set_xlabel("step"); ax.set_ylabel("train loss"); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(a.png, dpi=110)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()

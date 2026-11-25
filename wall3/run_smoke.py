"""단계 4 스모크: 대본으로 벽 N개를 쌓고 벽돌별 시작·끝 표시와 시방 판정을 검사한다.

검사: 벽돌 16장 모두 시작·끝 기록, 끝 > 시작, 다음 벽돌 시작 = 이전 벽돌 끝(철수 자세에서 출발), 끝에서 벽돌 정지,
벽돌 한 장 10초 이내. 결과는 --out/smoke.json.
실행: MUJOCO_GL=egl venv/bin/python -m wall3.run_smoke --env-version main_v1 --seeds 700000 700001 700002 --out outputs/wall3/<이름>
"""
import argparse, json, subprocess, time
from multiprocessing import Pool
from pathlib import Path

from functools import partial

from wall3.config import scene_config
from wall3.scene import WallEnv
from wall3.expert import WallExpert, expert_config
from wall3.spec_metrics import evaluate_env, sensitivity

SIM = Path(__file__).resolve().parents[1]


def check_marks(r, n=16, limit_s=10.0):
    b = r["bricks"]
    probs = []
    if len(b) != n or any("start" not in x for x in b):
        probs.append(f"bricks with marks {sum('start' in x for x in b)}/{n}")
    prev_end = 0
    for x in b:
        if "start" not in x:
            continue
        if x["end"] <= x["start"]:
            probs.append(f"slot {x['slot']}: end <= start")
        if x["start"] != prev_end:
            probs.append(f"slot {x['slot']}: start {x['start']} != previous end {prev_end}")
        if not x["static_at_end"]:
            probs.append(f"slot {x['slot']}: brick moving at end")
        if x["duration_s"] > limit_s:
            probs.append(f"slot {x['slot']}: {x['duration_s']:.2f} s > {limit_s} s")
        prev_end = x["end"]
    return probs


def one(seed, env_version):
    env = WallEnv(scene_config(env_version))
    ex = WallExpert(env)
    t0 = time.time()
    r = ex.run(seed)
    r["wall_clock_s"] = time.time() - t0
    ev = evaluate_env(env)
    r["eval"] = ev
    r["sensitivity"] = sensitivity(env)
    r["mark_problems"] = check_marks(r)
    env.close()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--env-version", required=True)
    a = ap.parse_args()
    if any(not 700000 <= s <= 700049 for s in a.seeds):
        raise SystemExit("파일럿 시작값 범위(700000 ~ 700049)만 쓴다")
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} 이미 있음: 덮어쓰지 않음")
    out.mkdir(parents=True)
    with Pool(min(len(a.seeds), 16)) as p:
        res = p.map(partial(one, env_version=a.env_version), a.seeds)
    meta = {"git_commit": subprocess.check_output(["git", "-C", str(SIM), "rev-parse", "HEAD"], text=True).strip(),
            "env_version": a.env_version, "scene": scene_config(a.env_version).to_dict(), "expert": expert_config(a.env_version).to_dict()}
    (out / "smoke.json").write_text(json.dumps({"meta": meta, "walls": res}, indent=1, default=float, ensure_ascii=False))
    for r in res:
        times = [b["duration_s"] for b in r["bricks"] if "duration_s" in b]
        ev = r["eval"]
        print(f"seed {r['seed']}: fail {r.get('fail')} marks {16 - len([b for b in r['bricks'] if 'start' not in b])}/16 problems {r['mark_problems']} "
              f"| brick time max {max(times):.2f} median {sorted(times)[len(times) // 2]:.2f} s | spec {ev['spec_pass']} "
              f"② {ev['2_line_plane']['value_mm']:.2f} ③ {ev['3_plumb']['value_mm']:.2f} ④ {ev['4_position']['value_mm']:.2f} "
              f"① {ev['1_bed_joint']['upper_min_mm']:.2f}~{ev['1_bed_joint']['upper_max_mm']:.2f} ⑤ {ev['5_thickness']['max_mm']:.2f} "
              f"IoU {ev['iou_mean']:.4f}/{ev['iou_min']:.4f} | sens {r['sensitivity']} | wall clock {r['wall_clock_s']:.1f} s")


if __name__ == "__main__":
    main()

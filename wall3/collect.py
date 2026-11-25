"""벽 단위 원자료 수집 (지시서 단계 5·6). 시작값마다 대본으로 벽을 쌓고, 매 제어 스텝을 기록해 벽 하나 = 파일 하나로 저장한다.

기록(매 스텝, 행동을 보내기 직전의 관측): 카메라 3대(front·top·wrist, 128²), 관절 6 + 집게(왼쪽 구동 관절 각도), 행동(관절 목표 6 +
집게 명령), 집게 닫힘 각도(G1용), 손목 힘·토크(집게 ft 사이트, 세계 좌표), 벽돌 16장 참값 자세, TCP 자세, 현재 벽돌 번호, 대본 단계.
벽 단위: 벽돌별 시작·끝 스텝, 시방 5항목·벽돌별 IoU 판정, 실패 사유, 설정·커밋·자산 해시.
중단된 벽은 저장하지 않는다(벽마다 끝까지 돈 뒤 임시 파일 → 이름 바꾸기). 같은 출력 폴더가 있으면 멈춘다.

실행: MUJOCO_GL=egl venv/bin/python -m wall3.collect --env-version main_v1 --seed0 700000 --n 50 --out outputs/wall3/<이름> [--record-every 10]
      깨끗한 합격 벽 개수를 채울 때까지: --target-pass K (시작값을 seed0부터 차례로, 깨끗한 합격 + 도는 중 < K인 동안만 새 벽 시작, 최대 --n번)
      멈춤 규칙(추가 지시 2): --stop-overload-frac 0.01 --stop-pass-rate-below <재확인 − 0.10> (--stop-min-n 벽을 마친 뒤부터)
깨끗한 합격 = 시방 합격 + 완주 + 대본 과부하 해제 0번(추가 지시 2).
"""
import argparse, hashlib, json, os, subprocess, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[1]
PHASES = ["to_supply", "grasp_descend", "grasp_close", "grasp_lift", "to_wall", "wait_bed", "place_align", "place_descend",
          "place_servo", "place_lower", "place_unload", "place_open", "retreat", "to_retract", "settle"]
CAMS = ("front", "top", "wrist")
PILOT = (700000, 709999)        # pilot block: 700000 ~ 700049 first, then continued from 700050 (2026-10-16:
                                 # 50 *passing* walls); 700200 ~ 700209 = 2차 나5 recheck
MAIN = (710000, 719999)         # main collection block (2차 나6): frozen main_v1 only. 720000 ~ (val), 730000 ~ (test) stay untouched


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def asset_info():
    out = {}
    for name, p in (("ur10e_xml", SIM / "menagerie/universal_robots_ur10e/ur10e.xml"),
                    ("gripper_xml", SIM / "third_party/robosuite_gripper140/robotiq_gripper_140.xml")):
        out[name] = sha256(p)
    for name, p in (("menagerie_source", SIM / "menagerie/universal_robots_ur10e/SOURCE.txt"),
                    ("robosuite_source", SIM / "third_party/robosuite_gripper140/SOURCE.txt")):
        out[name] = p.read_text().strip() if p.exists() else None
    return out


class StepLog:
    def __init__(self, env):
        self.env = env
        self.rows = {k: [] for k in ("t", "phase", "brick", "state", "action", "closure", "fingers", "ft", "bricks", "tcp")}
        if env.cfg.grip_lock:
            self.rows["lock"] = []          # info only (현실성 조건 6): locked, locked brick, pad normal force left / right
        self.img = {c: [] for c in CAMS}

    def __call__(self, env, action, phase, k):
        d = env.data
        self.rows["t"].append(env.t)
        self.rows["phase"].append(PHASES.index(phase))
        self.rows["brick"].append(k)
        q = env.arm_q()
        fl = float(d.joint("g_finger_joint").qpos[0]); fr = float(-d.joint("g_right_outer_knuckle_joint").qpos[0])
        self.rows["state"].append(np.r_[q, fl].astype(np.float32))
        self.rows["action"].append(np.asarray(action, np.float32))
        self.rows["closure"].append(np.float32(0.5 * (fl + fr)))
        self.rows["fingers"].append(np.array([fl, fr], np.float32))
        f, tq = env.wrist_ft()
        self.rows["ft"].append(np.r_[f, tq].astype(np.float32))
        self.rows["bricks"].append(env.brick_poses().astype(np.float32))
        p, R = env.tcp()
        import mujoco
        qt = np.zeros(4); mujoco.mju_mat2Quat(qt, R.ravel())
        self.rows["tcp"].append(np.r_[p, qt].astype(np.float32))
        if "lock" in self.rows:
            self.rows["lock"].append(env.lock_state())
        for c in CAMS:
            self.img[c].append(env.render(c))

    def arrays(self):
        out = {k: np.asarray(v) for k, v in self.rows.items()}
        for c in CAMS:
            out[f"img_{c}"] = np.asarray(self.img[c], np.uint8)
        return out


def one_wall(args):
    seed, out_dir, record, env_version, expert_version = args
    from wall3.config import scene_config
    from wall3.scene import WallEnv
    from wall3.expert import WallExpert, expert_config
    from wall3.spec_metrics import evaluate_env, sensitivity
    t0 = time.time()
    env = WallEnv(scene_config(env_version))
    ex = WallExpert(env, expert_config(expert_version))
    log = StepLog(env)
    rec = None
    if record:
        from wall3.view_collect import Recorder
        rec = Recorder(env, Path(out_dir) / "video" / f"wall_{seed}.mp4", label=f"seed {seed}")

    def on_step(env_, action, phase, k):
        log(env_, action, phase, k)
        if rec is not None:
            rec.frame(phase, k)

    r = ex.run(seed, on_step=on_step)
    if rec is not None:
        rec.frame("end"); rec.close()
    t_run = time.time() - t0
    ev = evaluate_env(env)
    r["eval"] = ev
    r["sensitivity"] = sensitivity(env)
    r["final_brick_poses"] = env.brick_poses().tolist()
    r["design"] = env.design
    r["env_version"] = env_version
    r["expert_version"] = expert_version
    r["supply_initial"] = [[p.tolist(), float(y)] for p, y in env.supply]
    r["base_dist_m"] = [float(np.hypot(s["x"], s["y"])) for s in env.design]
    # 추가 지시 2: a wall whose expert grasp was released by condition ③ (force / twist / contact loss) even once is not
    # clean (dropped from the main dataset even if it passes the spec)
    over = [dict(e, slot=b["slot"]) for b in r["bricks"] for e in b.get("lock_events", []) if e["event"] == "release" and e["reason"] != "open"]
    r["overload_releases"] = len(over)
    r["overload_events"] = over
    r["clean_pass"] = bool(ev["spec_pass"] and not r.get("fail") and not over)
    arr = log.arrays()
    raw = Path(out_dir) / "raw"
    tmp = raw / f".wall_{seed}.tmp.npz"
    np.savez_compressed(tmp, **arr)
    final = raw / f"wall_{seed}.npz"
    os.replace(tmp, final)
    r["raw_file"] = final.name
    r["raw_sha256"] = sha256(final)
    r["raw_bytes"] = final.stat().st_size
    r["frames"] = int(len(arr["t"]))
    r["time_s"] = {"simulate_render": t_run, "total": time.time() - t0}
    (raw / f"wall_{seed}.json").write_text(json.dumps(r, indent=1, default=float, ensure_ascii=False))
    env.close()
    return {"seed": seed, "fail": r.get("fail"), "spec_pass": ev["spec_pass"], "clean_pass": r["clean_pass"],
            "overload_releases": r["overload_releases"], "frames": r["frames"], "raw_bytes": r["raw_bytes"], "time_s": r["time_s"]["total"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed0", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--record-every", type=int, default=0, help="record every N-th wall to video/ (0 = none)")
    ap.add_argument("--target-pass", type=int, default=0, help="stop once this many walls pass the spec (0 = run all --n)")
    ap.add_argument("--env-version", required=True, help="pilot_20261016 | main_v1 (wall3.config.ENV_VERSIONS)")
    ap.add_argument("--expert-version", default=None, help="wall3.expert.EXPERT_VERSIONS key (default: same as --env-version)")
    ap.add_argument("--stop-overload-frac", type=float, default=0.0, help="stop when walls with an overload release exceed this fraction")
    ap.add_argument("--stop-pass-rate-below", type=float, default=0.0, help="stop when the spec pass rate falls below this")
    ap.add_argument("--stop-min-n", type=int, default=100, help="walls finished before the stop rules apply")
    a = ap.parse_args()
    a.expert_version = a.expert_version or a.env_version
    seeds = list(range(a.seed0, a.seed0 + a.n))
    in_pilot = PILOT[0] <= seeds[0] and seeds[-1] <= PILOT[1]
    in_main = MAIN[0] <= seeds[0] and seeds[-1] <= MAIN[1] and a.env_version == "main_v1"
    if not (in_pilot or in_main):
        raise SystemExit("시작값은 파일럿 블록(700000 ~ 709999) 또는 main_v1 본 수집 블록(710000 ~ 719999)만 (720000 ~ 검증·시험 예약)")
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} 이미 있음: 덮어쓰지 않음")
    (out / "raw").mkdir(parents=True)
    from wall3.config import scene_config
    from wall3.expert import expert_config
    meta = {"git_commit": subprocess.check_output(["git", "-C", str(SIM), "rev-parse", "HEAD"], text=True).strip(),
            "git_dirty": bool(subprocess.check_output(["git", "-C", str(SIM), "status", "--porcelain", "--", "wall3"], text=True).strip()),
            "assets": asset_info(), "env_version": a.env_version, "scene": scene_config(a.env_version).to_dict(), "expert_version": a.expert_version, "expert": expert_config(a.expert_version).to_dict(),
            "seeds": seeds, "phases": PHASES, "cams": CAMS, "args": vars(a), "nproc": os.cpu_count(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "collect_config.json").write_text(json.dumps(meta, indent=1, default=float, ensure_ascii=False))
    jobs = [(s, str(out), bool(a.record_every) and (s - a.seed0) % a.record_every == 0, a.env_version, a.expert_version) for s in seeds]
    t0 = time.time()
    res, stop = [], None

    def check_stop():
        n = len(res)
        if n < a.stop_min_n:
            return None
        n_over = sum(r["overload_releases"] > 0 for r in res)
        if a.stop_overload_frac and n_over > a.stop_overload_frac * n:
            return f"overload walls {n_over}/{n} > {a.stop_overload_frac:.0%}"
        rate = sum(r["spec_pass"] for r in res) / n
        if a.stop_pass_rate_below and rate < a.stop_pass_rate_below:
            return f"spec pass rate {rate:.3f} < {a.stop_pass_rate_below:.3f} after {n} walls"
        return None

    with Pool(a.workers) as p:
        # seeds go in order; with --target-pass a new seed is started only while (clean passes + walls running) < target,
        # so the attempted seeds are one contiguous block and the run ends with exactly the target (if --n allows)
        pending, nxt = [], 0
        while True:
            while nxt < len(jobs) and len(pending) < a.workers and stop is None and (
                    not a.target_pass or sum(r["clean_pass"] for r in res) + len(pending) < a.target_pass):
                pending.append(p.apply_async(one_wall, (jobs[nxt],)))
                nxt += 1
            if not pending:
                break
            done = [x for x in pending if x.ready()]
            if not done:
                time.sleep(0.5)
                continue
            for x in done:
                pending.remove(x)
                r = x.get()
                res.append(r)
                print(f"[{len(res):4d}] seed {r['seed']} fail {r['fail']} spec {r['spec_pass']} clean {r['clean_pass']} "
                      f"overload {r['overload_releases']} frames {r['frames']} raw {r['raw_bytes'] / 1e6:.0f} MB {r['time_s']:.0f} s  "
                      f"(clean {sum(q['clean_pass'] for q in res)}, elapsed {time.time() - t0:.0f} s)", flush=True)
            if stop is None:
                stop = check_stop()
                if stop:
                    print(f"STOP: {stop} (finishing the {len(pending)} walls already running)", flush=True)
    res.sort(key=lambda r: r["seed"])
    summ = {"n": len(res), "n_pass": sum(r["spec_pass"] for r in res), "n_clean_pass": sum(r["clean_pass"] for r in res),
            "n_overload_walls": sum(r["overload_releases"] > 0 for r in res),
            "n_overload_walls_spec_pass": sum(r["overload_releases"] > 0 and r["spec_pass"] for r in res),
            "n_fail_run": sum(bool(r["fail"]) for r in res), "stopped": stop,
            "seeds_attempted": [res[0]["seed"], res[-1]["seed"]] if res else None,
            "elapsed_s": time.time() - t0, "workers": a.workers, "target_pass": a.target_pass,
            "stop_rules": {"overload_frac": a.stop_overload_frac, "pass_rate_below": a.stop_pass_rate_below, "min_n": a.stop_min_n},
            "walls": res, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "collect_summary.json").write_text(json.dumps(summ, indent=1, default=float, ensure_ascii=False))
    print(f"done: {summ['n_pass']}/{summ['n']} spec pass, {summ['n_clean_pass']} clean, {summ['n_overload_walls']} overload walls, "
          f"{summ['n_fail_run']} aborted, {summ['elapsed_s']:.0f} s" + (f", STOPPED: {stop}" if stop else ""))
    if stop:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

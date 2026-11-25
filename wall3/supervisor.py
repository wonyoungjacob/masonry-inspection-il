"""남은 계산 작업 감독 (2026-11-04, 우분투지시_검측정보진단_20261103.md B · C · D).

사람을 기다리지 않고 단계를 차례로 돌린다. 끝남은 결과 파일(summary.json / 종료 코드 파일)로만 판정한다.
돌고 있는지는 python 프로세스의 실행 인자를 원소 단위로 정확히 맞춰 본다(셸 명령줄 문자열에 속지 않게).
작업이 죽으면: 남은 하위 프로세스를 정리하고, 끊긴 출력은 <out>_STOPPED_<n>으로 이름만 바꿔 두고(덮어쓰기 · 합치기 없음)
처음부터 다시 돌린다. 작업마다 최대 MAX_RETRY번까지. 넘으면 멈추고 이유를 상태 파일에 쓴다.
감독 자신은 systemd 사용자 서비스(Restart=always)로 돌고, 상태를 파일에 남기므로 다시 떠도 이어서 한다.
계산만 한다: 보고서 · 커밋은 하지 않는다.

실행: venv/bin/python -m wall3.supervisor   (상태: outputs/wall3/supervisor_20261104/STATUS.txt)
     다른 계획: venv/bin/python -m wall3.supervisor --plan experiments.plan_ladder_20261110
     (계획 모듈은 SUP_NAME · STAGES와, 있으면 post_stage(sname, st)를 둔다. 2026-11-10)
"""
import collections, json, os, shlex, shutil, signal, subprocess, sys, time, traceback
from pathlib import Path

SIM = Path(__file__).resolve().parents[1]
W = "outputs/wall3"
SUP = SIM / W / "supervisor_20261104"
STATE = SUP / "state.json"
CKPT = f"{W}/main_train_v2_20261022/act_s1000/checkpoints/last/pretrained_model"
A = f"{W}/eval_gates_info_20261104"
REPORT = SIM.parent / "01_사기설" / "근거_우분투_wall3_검측정보진단_20261103"
POLL = 60
MAX_RETRY = 2
PY = "venv/bin/python"


def chunk_seeds(i, n=100, chunks=5, s0=700300):
    k = n // chunks
    return list(range(s0 + i * k, s0 + (i + 1) * k))


def trace_jobs(root, src, tgt, cond):
    return [dict(name=f"{Path(root).name}_part{i}", kind="rc", module="experiments.diag_trace", out=f"{root}/part{i}",
                 cmd=[PY, "-m", "experiments.diag_trace", "--ckpt", CKPT, "--env-version", "main_v1_eval",
                      "--seeds", *map(str, chunk_seeds(i)), "--out", f"{root}/part{i}",
                      "--gate-pose-source", src, "--gate-target", tgt,
                      "--compare", f"{A}/act_{cond}_part{i}/rollout.json"]) for i in range(5)]


def causes_job(root):
    return [dict(name=f"{Path(root).name}_causes", kind="rc", module="experiments.diag_trace_causes", out=f"{root}/causes.json",
                 cmd=[PY, "-m", "experiments.diag_trace_causes", root, "--aim-max-mm", "50", "--perch-mm", "20",
                      "--out", f"{root}/causes.json"], log=f"{root}/causes.txt")]


def pick_video_walls():
    """가장 많이 채운 벽 1개 + 가장 흔한 '처음 못 채운 자리' 원인의 대표 벽 1개 (작은 시작값 우선, 둘이 같으면 다음 벽)."""
    walls = []
    for i in range(5):
        walls += json.loads((SIM / A / f"act_all_held_part{i}/rollout.json").read_text())["walls"]
    best = max(walls, key=lambda w: (sum(w["eval"]["slots_filled"]), -w["seed"]))["seed"]
    c = json.loads((SIM / W / "diag_trace_held_20261104/causes.json").read_text())
    first = {}
    for s, r in c.items():
        if s.startswith("_") or r["first_unfilled"] >= 16:
            continue
        st = next((f["state"] for f in r["fails"] if f["slot"] == r["first_unfilled"]), None)
        if st:
            first[int(s)] = st
    top = collections.Counter(first.values()).most_common(1)[0][0]
    cand = [s for s in sorted(first) if first[s] == top]
    rep = next((s for s in cand if s != best), cand[0])
    return best, rep, top


def video_cmd(st):
    best, rep, top = pick_video_walls()
    st["video_pick"] = {"best": best, "representative": rep, "top_first_cause": top}
    out = f"{W}/diag_video_held_20261104"
    return [PY, "-m", "experiments.diag_video", "--ckpt", CKPT, "--env-version", "main_v1_eval", "--seeds", str(best), str(rep),
            "--names", f"가장많이_{best}", f"흔한원인_{rep}", "--out", out, "--gate-target", "held_grasp",
            "--compare", *[f"{A}/act_all_held_part{i}/rollout.json" for i in range(5)]]


FIGS = f"{W}/diag_camera_figs_20261104"
STAGES = [
    ("B", [dict(name="B_ablation", kind="summary", module="wall3.eval_gates", out=f"{W}/eval_gates_ablation_20261104",
                cmd=[PY, "-m", "wall3.eval_gates", "--out", f"{W}/eval_gates_ablation_20261104",
                     "--train-dir", f"{W}/main_train_v2_20261022", "--policies", "act",
                     "--conditions", "g1", "g1g2", "g1g3", "g2g3", "g1g2_held", "g1g3_held", "g2g3_held",
                     "--chunks", "5", "--parallel", "8", "--env-version", "main_v1_eval"])]),
    ("C_trace", trace_jobs(f"{W}/diag_trace_held_20261104", "camera", "held_grasp", "all_held")),
    ("C_causes", causes_job(f"{W}/diag_trace_held_20261104")),
    ("C_video", [dict(name="C_video", kind="rc", module="experiments.diag_video", out=f"{W}/diag_video_held_20261104", cmd=video_cmd)]),
    # 시간이 남으면 (지시 C): 참값 (진단) all_truth_held 추적
    ("C_truth_trace", trace_jobs(f"{W}/diag_trace_truth_held_20261104", "truth", "held_truth", "all_truth_held")),
    ("C_truth_causes", causes_job(f"{W}/diag_trace_truth_held_20261104")),
    ("D", [dict(name="D_figs", kind="rc", module="experiments.diag_camera_figs", out=FIGS,
                cmd=[PY, "-m", "experiments.diag_camera_figs", "--ckpt", CKPT, "--env-version", "main_v1_eval", "--seed", "700300",
                     "--brick", "2", "--gate-pose-source", "camera", "--gate-target", "held_grasp",
                     "--out", FIGS, "--fig-dir", f"{FIGS}_그림"], extra=[f"{FIGS}_그림"])]),
]


def log(msg):
    SUP.mkdir(parents=True, exist_ok=True)
    with open(SUP / "log.txt", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def save(st):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1, ensure_ascii=False))
    tmp.replace(STATE)
    lines = [f"갱신 {time.strftime('%Y-%m-%d %H:%M:%S')}  상태: {st['status']}  {st.get('reason', '')}"]
    for sname, jobs in STAGES:
        for j in jobs:
            js = st["jobs"].get(j["name"], {})
            lines.append(f"{sname:15s} {j['name']:40s} {js.get('state', '대기'):8s} 재시작 {js.get('retries', 0)}")
    if "video_pick" in st:
        lines.append(f"영상 벽: {st['video_pick']}")
    (SUP / "STATUS.txt").write_text("\n".join(lines) + "\n")


def procs():
    """(pid, argv) for python processes only."""
    out = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            argv = (p / "cmdline").read_bytes().split(b"\0")
            argv = [a.decode(errors="replace") for a in argv if a]
        except OSError:
            continue
        if argv and "python" in Path(argv[0]).name:
            out.append((int(p.name), argv))
    return out


def running(j):
    for pid, argv in procs():
        if j["module"] in argv and j["out"] in argv:
            return pid
    return None


def kill_leftovers(j):
    """the job's own process and its children (argv has an element that is or starts with out)."""
    pids = [pid for pid, argv in procs() if any(a == j["out"] or a.startswith(j["out"] + "/") or a.startswith(j["out"] + "_part")
                                               for a in argv)]
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        time.sleep(5)
    return pids


def paths(j):
    out = SIM / j["out"]
    logf = SIM / j.get("log", j["out"] + ".log")
    return out, logf, Path(str(logf) + ".rc")


def done(j):
    out, logf, rc = paths(j)
    if j["kind"] == "summary":
        return (out / "summary.json").exists()
    return rc.exists() and rc.read_text().strip() == "0"


def set_aside(j, n):
    for p in [*paths(j), *(SIM / e for e in j.get("extra", []))]:
        if p.exists():
            dst = p.with_name(f"{p.name}_STOPPED_{n}")
            while dst.exists():
                dst = dst.with_name(dst.name + "x")
            p.rename(dst)
            log(f"보관: {p} -> {dst.name}")


def launch(j, st):
    out, logf, rc = paths(j)
    if out.exists():
        raise RuntimeError(f"{out} 이미 있음: 덮어쓰지 않음")
    cmd = j["cmd"](st) if callable(j["cmd"]) else j["cmd"]
    out.parent.mkdir(parents=True, exist_ok=True) if j["kind"] == "rc" else None
    env = dict(os.environ, MUJOCO_GL="egl", PYTHONPATH=".")
    sh = f"{shlex.join(cmd)} > {shlex.quote(str(logf))} 2>&1; echo $? > {shlex.quote(str(rc))}"
    p = subprocess.Popen(["bash", "-c", sh], cwd=SIM, env=env, start_new_session=True,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log(f"시작 {j['name']} (wrapper pid {p.pid}): {shlex.join(cmd)}")
    return cmd


def post_stage(sname, st):
    """끝난 단계의 산출물을 보고서 폴더로 복사 (없는 곳에만, 덮어쓰지 않음)."""
    if sname == "D":
        dst = REPORT / "그림"
        src = SIM / f"{FIGS}_그림"
        big = [f.name for f in src.glob("*.png") if f.stat().st_size > 1_000_000]
        if big:
            st["notes"].append(f"D 그림 1 MB 넘음: {big}")
        if not dst.exists():
            shutil.copytree(src, dst)
            log(f"복사 {src} -> {dst}")
    if sname == "C_video":
        dst = REPORT / "확인영상"
        if not dst.exists():
            dst.mkdir(parents=True)
            for f in (SIM / W / "diag_video_held_20261104").rglob("*"):
                if f.suffix in (".mp4", ".png"):
                    shutil.copy2(f, dst / f.name)
            log(f"복사 영상 -> {dst}")
    if sname == "B":
        s = json.loads((SIM / W / "eval_gates_ablation_20261104/summary.json").read_text())
        if s["_meta"]["failed_jobs"]:
            st["notes"].append(f"B 실패한 작업: {s['_meta']['failed_jobs']}")


def tick(st):
    for sname, jobs in STAGES:
        if st["stages"].get(sname) == "done":
            continue
        all_done = True
        for j in jobs:
            js = st["jobs"].setdefault(j["name"], {"state": "대기", "retries": 0, "missing": 0})
            if done(j):
                if js["state"] != "끝":
                    js["state"] = "끝"
                    log(f"끝 {j['name']}")
                continue
            all_done = False
            if running(j):
                js["state"], js["missing"] = "실행중", 0
                continue
            out, logf, rc = paths(j)
            started = out.exists() or logf.exists() or rc.exists()
            if not started:
                js["cmd"] = launch(j, st)
                js["state"], js["missing"] = "실행중", 0
                continue
            js["missing"] += 1                       # 두 번 연속 안 보여야 죽은 것으로 본다
            if js["missing"] < 2:
                continue
            why = f"rc={rc.read_text().strip()}" if rc.exists() else "프로세스 없음, 결과 없음"
            if js["retries"] >= MAX_RETRY:
                st["status"], st["reason"] = "멈춤", f"{j['name']} {MAX_RETRY}번 다시 시작해도 실패 ({why}). 로그: {logf}"
                js["state"] = "실패"
                log(st["reason"])
                return False
            killed = kill_leftovers(j)
            js["retries"] += 1
            log(f"죽음 감지 {j['name']} ({why}), 남은 프로세스 {killed} 정리, {js['retries']}번째 다시 시작")
            set_aside(j, js["retries"])
            js["cmd"] = launch(j, st)
            js["state"], js["missing"] = "실행중", 0
        if not all_done:
            return True                              # 이 단계가 끝나야 다음 단계로
        post_stage(sname, st)
        st["stages"][sname] = "done"
        log(f"단계 끝 {sname}")
        save(st)
    st["status"], st["reason"] = "모두 끝", ""
    return False


def use_plan(name):
    """2026-11-10: run another plan module's stages with its own state folder (the jobs logic above is unchanged)."""
    global SUP, STATE, STAGES, post_stage
    import importlib
    m = importlib.import_module(name)
    SUP = SIM / W / m.SUP_NAME
    STATE = SUP / "state.json"
    STAGES = m.STAGES
    post_stage = getattr(m, "post_stage", lambda sname, st: None)


def main():
    if "--plan" in sys.argv:
        use_plan(sys.argv[sys.argv.index("--plan") + 1])
    SUP.mkdir(parents=True, exist_ok=True)
    st = json.loads(STATE.read_text()) if STATE.exists() else {"status": "실행중", "stages": {}, "jobs": {}, "notes": []}
    log(f"감독 시작 pid {os.getpid()}")
    while True:
        if st["status"] in ("멈춤", "모두 끝"):
            save(st)
            time.sleep(POLL * 5)                     # 서비스는 살아 있되 아무것도 안 띄움
            st = json.loads(STATE.read_text())       # 사람이 상태를 고치면 다시 이어서
            continue
        st["status"] = "실행중"
        try:
            tick(st)
        except Exception:
            st["status"], st["reason"] = "멈춤", "감독 오류: " + traceback.format_exc(limit=3).replace("\n", " | ")
            log(st["reason"])
        st["heartbeat"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save(st)
        time.sleep(POLL)


if __name__ == "__main__":
    main()

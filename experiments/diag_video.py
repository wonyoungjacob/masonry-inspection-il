"""팀원 눈 확인용 영상 + 추적 (2026-10-29): diag_trace와 같은 방식(같은 rollout.run_wall, 같은 검측 설정)으로 벽을 한 프로세스에서
하나씩 다시 돌리며, 같은 실행에서 (1) 매 스텝 기록(trace_<seed>.npz, diag_trace_causes로 분류 가능) (2) 영상 (3) 마지막 장면 그림을 남긴다.
영상과 자동 분류가 같은 실행에서 나오므로 서로 맞춰 볼 수 있다. 진단용이며 평가 결과가 아니다.

화면 표시: 차례 번호, 지금 움직이는 주체(정책 / 검측). 벽돌 번호 · 분류 결과는 표시하지 않는다(눈 확인을 독립으로 하려고).

실행: PYTHONPATH=. MUJOCO_GL=egl venv/bin/python -m experiments.diag_video --ckpt <pretrained_model> \
        --seeds 700301 ... --out <dir> [--compare outputs/wall3/eval_gates_v2_20261026/act_all_part*/rollout.json]
"""
import argparse, glob, hashlib, json, subprocess, sys
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
SRC = {"policy": 0, "g1": 1, "g2g3": 2, "other": 3, "g0": 4}


def wall_camera(azimuth, elevation=-25.0, distance=1.45, lookat=(0.0, 0.6, 0.09)):
    """Free camera on the wall. azimuth -90: from the side opposite the robot (robot and supply table behind the wall)."""
    import mujoco
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def make_recorder(env, path, label):
    import av
    from PIL import Image, ImageDraw, ImageFont
    from wall3.view_collect import Recorder, FONT

    class Rec(Recorder):
        """Recorder with a banner: turn number and who is moving (policy / gate). Every 2nd step kept (2x speed)."""

        def __init__(self, env, path, width=640, height=480, label=""):          # = Recorder.__init__ with a 40 px taller frame
            import mujoco
            self.env, self.w, self.h, self.side = env, width, height, height // 3
            self.cam = wall_camera(-90.0, -32.0, 2.1, (0.0, 0.45, 0.15))   # wall in front, robot and supply table behind it
            self.r_free = mujoco.Renderer(env.model, height, width)
            self.font = ImageFont.truetype(str(FONT), 15)
            self.big = ImageFont.truetype(str(FONT), 22)
            self.label, self.path = label, Path(path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.c = av.open(str(self.path), "w")
            self.s = self.c.add_stream("libx264", rate=20)
            self.s.width, self.s.height, self.s.pix_fmt = width + self.side, height + 40, "yuv420p"
            self.s.options = {"crf": "28", "preset": "medium"}
            self.frames, self.n = 0, 0

        def frame(self, phase="", k=-1):
            self.n += 1
            if self.n % 2:
                return
            env = self.env
            self.r_free.update_scene(env.data, camera=self.cam, scene_option=env.vopt)
            canvas = Image.new("RGB", (self.w + self.side, self.h + 40), (20, 20, 20))
            canvas.paste(Image.fromarray(self.r_free.render()), (0, 40))
            d = ImageDraw.Draw(canvas)
            for i, name in enumerate(("front", "top", "wrist")):
                im = Image.fromarray(env.render(name)).resize((self.side, self.side), Image.NEAREST)
                canvas.paste(im, (self.w, 40 + i * self.side))
                d.text((self.w + 4, 40 + i * self.side + 2), name, fill=(255, 255, 0), font=self.font)
            if phase.startswith("policy"):
                who, col = "정책(학습한 로봇)이 움직임", (40, 140, 60)
            elif phase == "settle":
                who, col = "끝(멈춰 둠)", (90, 90, 90)
            else:
                who, col = f"검측이 움직임 ({phase.replace('gate ', '')})", (180, 50, 50)
            d.rectangle([0, 0, self.w + self.side, 40], fill=col)
            turn = f"차례 {k + 1}/16" if k >= 0 else "차례 -"
            d.text((10, 7), f"{self.label}   {turn}   {who}   t {env.t * env.cfg.ctrl_dt:6.1f} s", fill=(255, 255, 255), font=self.big)
            for p in self.s.encode(av.VideoFrame.from_image(canvas)):
                self.c.mux(p)
            self.frames += 1

    return Rec(env, path, label=label)


def stills(env, out, name):
    """Final scene, 960x640: front (side opposite the robot), from above, at an angle."""
    import mujoco
    from PIL import Image
    r = mujoco.Renderer(env.model, 640, 960)
    for tag, cam in (("정면", wall_camera(-90.0)), ("위에서", wall_camera(-90.0, -89.0, 1.6)),
                     ("비스듬히", wall_camera(-50.0, -35.0, 1.6))):
        r.update_scene(env.data, camera=cam, scene_option=env.vopt)
        Image.fromarray(r.render()).save(out / f"{name}_마지막_{tag}.png")
    r.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--env-version", default="main_v1_eval")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True, help="file name per seed (e.g. 벽A)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate-pose-source", choices=("camera", "truth"), default="camera", help="= rollout (2026-11-03)")
    ap.add_argument("--gate-target", choices=("turn", "held_truth", "held_grasp"), default="turn", help="= rollout (2026-11-03)")
    ap.add_argument("--gate-fix", nargs="*", default=[], help="= rollout (2026-11-10)")
    ap.add_argument("--compare", nargs="*", default=[])
    a = ap.parse_args()
    if any(not (700300 <= s <= 700399) for s in a.seeds):
        raise SystemExit("시작값은 평가와 같은 700300 ~ 700399만")
    if len(a.names) != len(a.seeds):
        raise SystemExit("--names 수가 --seeds와 다름")
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} 이미 있음: 덮어쓰지 않음")
    part = out / "part0"
    part.mkdir(parents=True)
    ref = {}
    for f in a.compare:
        for p in glob.glob(f):
            for w in json.loads(Path(p).read_text())["walls"]:
                ref[w["seed"]] = w
    import torch
    from eval_policy import load_policy
    from wall3.config import scene_config, RETRACT_Q
    from wall3.scene import WallEnv
    from wall3.g1 import G1Config
    from wall3.gates import WallGates, GateConfig, fix_fields
    from wall3.rollout import run_wall
    policy, pcfg, pre, post = load_policy(a.ckpt, "cuda", None, None)
    cfg_in = set(pcfg.input_features)
    env = WallEnv(scene_config(a.env_version))
    q_ret = np.array(RETRACT_Q)
    gates = WallGates(env, policy, g1=True, g2=True, g3=True, g1_cfg=G1Config(thr_load=4.5, thr_angle=0.47),
                      cfg=GateConfig(pose_source=a.gate_pose_source, target=a.gate_target, **fix_fields(a.gate_fix)))
    meta = {"ckpt": a.ckpt, "env_version": a.env_version, "policy_type": pcfg.type, "gates": gates.summary_config(),
            "gate_pose_source": a.gate_pose_source, "gate_target": a.gate_target, "gate_fix": a.gate_fix,
            "git_commit": subprocess.check_output(["git", "-C", str(SIM), "rev-parse", "HEAD"], text=True).strip(),
            "scene_sha256": hashlib.sha256(json.dumps(env.cfg.to_dict(), sort_keys=True, default=float).encode()).hexdigest(),
            "script": "diag_video", "src_codes": SRC, "names": dict(zip(map(str, a.seeds), a.names))}

    state = {"src": SRC["policy"]}
    log = {}
    step0 = env.step

    def step(arm_ctrl, grip_cmd):
        r = step0(arm_ctrl, grip_cmd)
        tp, R = env.tcp()
        log["t"].append(env.t)
        log["k"].append(getattr(gates, "k", -1))
        log["tgt"].append(getattr(gates, "tgt", -1))                 # gate reference slot (= k for target turn)
        log["src"].append(state["src"])
        log["pose"].append(env.brick_poses().astype(np.float32))
        log["tcp"].append(np.r_[tp, np.arctan2(R[1, 0], R[0, 0])].astype(np.float32))
        log["lock"].append(env.lock_b)
        log["grip"].append(float(grip_cmd))
        return r
    env.step = step

    def mark(fn, code):
        def g(*x, **kw):
            prev = state["src"]
            state["src"] = code
            try:
                return fn(*x, **kw)
            finally:
                state["src"] = prev
        return g
    gates.before = mark(gates.before, SRC["g2g3"])
    gates.after = mark(gates.after, SRC["g1"])
    gates._g0_grasp = mark(gates._g0_grasp, SRC["g0"])         # 2026-11-10 K2 (inside before)

    res = []
    for seed, name in zip(a.seeds, a.names):
        for key in ("t", "k", "tgt", "src", "pose", "tcp", "lock", "grip"):
            log[key] = []
        rec = make_recorder(env, out / "video" / f"{name}.mp4", label=name)
        r = run_wall(env, policy, pre, post, cfg_in, seed, torch, rec, q_ret, gates)
        rec.close()
        stills(env, out / "video", name)
        np.savez_compressed(part / f"trace_{seed}.npz", t=np.array(log["t"], np.int32), k=np.array(log["k"], np.int8),
                            tgt=np.array(log["tgt"], np.int8),
                            src=np.array(log["src"], np.int8), pose=np.stack(log["pose"]), tcp=np.stack(log["tcp"]),
                            lock=np.array(log["lock"], np.int8), grip=np.array(log["grip"], np.float32))
        r["lock_events"] = env.lock_events
        r["name"] = name
        if seed in ref:
            o = ref[seed]
            r["same_as_eval"] = {"slots_filled": o["eval"]["slots_filled"] == r["eval"]["slots_filled"],
                                 "max_final_err_diff_mm": float(max(abs(x - y) for b0, b1 in zip(o["bricks"], r["bricks"])
                                                                    for x, y in zip(b0["final_err_mm"], b1["final_err_mm"]))),
                                 "sim_s": [o["sim_s"], r["sim_s"]]}
        res.append(r)
        print(f"{name} seed {seed}: filled {sum(r['eval']['slots_filled'])}/16, {r['wall_clock_s']:.0f} s"
              + (f", same as eval {r['same_as_eval']}" if "same_as_eval" in r else ""), flush=True)
        (part / "trace.json").write_text(json.dumps({"meta": meta, "walls": res}, indent=1, default=float))
    env.close()


if __name__ == "__main__":
    main()

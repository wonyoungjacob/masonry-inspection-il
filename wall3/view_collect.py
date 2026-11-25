"""수집 화면 보기·녹화 (지시서 단계 4·6 '화면 보기').

보기:  bash wall3/view_collect.sh --seed 700000 --walls 1 [--speed 1.0]
       MuJoCo 뷰어 창(mujoco.viewer.launch_passive)에서 역기구학 대본이 벽을 쌓는 모습을 실시간 속도로 보여 준다.
       창 렌더링은 화면용이므로 MUJOCO_GL=egl을 쓰지 않는다(sh 스크립트가 glfw로 둔다). 데스크톱 화면에 연결된 터미널에서 실행.
       마우스: 끌기 = 회전, 휠 = 확대, Esc = 닫기. 벽이 끝나면 시방 판정을 출력하고 3초 뒤 다음 벽으로 간다.
녹화:  MUJOCO_GL=egl venv/bin/python -m wall3.view_collect --seed 700000 --record outputs/wall3/<이름>/wall.mp4
       화면 없이 자유 시점(벽 전체와 공급대) + 정책 카메라 3대(front·top·wrist, 128 × 128 확대)를 한 화면에 붙여 mp4로 저장한다.
       --short N 이면 N배속으로 줄인 사본도 만든다(보고서용).
"""
import argparse, json, sys, time
from pathlib import Path

import mujoco
import numpy as np

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from wall3.config import scene_config                      # noqa: E402
from wall3.scene import WallEnv                            # noqa: E402
from wall3.expert import WallExpert                        # noqa: E402
from wall3.spec_metrics import evaluate_env                # noqa: E402

FONT = SIM / "tools/fonts/NotoSansKR-Regular.ttf"


def free_camera():
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.05, 0.25]
    cam.distance = 2.9
    cam.azimuth = 300.0             # wall in front, robot and supply table behind it
    cam.elevation = -32.0
    return cam


def summary(ev):
    pick = {"1_bed_joint": "upper_min_mm", "2_line_plane": "value_mm", "3_plumb": "value_mm", "4_position": "value_mm",
            "5_thickness": "max_mm"}
    items = {k: (ev[k]["pass"], round(float(ev[k].get(f) or 0), 2)) for k, f in pick.items()}
    return f"시방 합격 {ev['spec_pass']} | 존재 {ev['exists']} | " + ", ".join(f"{k[:1]} {'O' if p else 'X'} {v} mm" for k, (p, v) in items.items()) \
        + f" | IoU 평균 {ev['iou_mean']:.3f} 최소 {ev['iou_min']:.3f}"


# ---------------------------------------------------------------- viewer
def view(a):
    import mujoco.viewer
    env = WallEnv(scene_config(a.env_version))
    ex = WallExpert(env)
    for w in range(a.walls):
        seed = a.seed + w
        env.reset(seed, arm_q=ex.q_ret)
        with mujoco.viewer.launch_passive(env.model, env.data) as v:
            fc = free_camera()
            v.cam.lookat[:] = fc.lookat; v.cam.distance = fc.distance
            v.cam.azimuth = fc.azimuth; v.cam.elevation = fc.elevation
            v.opt.sitegroup[:] = 0                          # hide robosuite's helper sites
            last = [time.time()]

            def pace(env_, action, phase, k):
                if not v.is_running():
                    raise KeyboardInterrupt
                v.sync()
                dt = env.cfg.ctrl_dt / a.speed
                now = time.time()
                if now - last[0] < dt:
                    time.sleep(dt - (now - last[0]))
                last[0] = time.time()

            try:
                r = ex.run(seed, on_step=pace)
            except KeyboardInterrupt:
                return
            v.sync()
            ev = evaluate_env(env)
            times = [b.get("duration_s") for b in r["bricks"] if b.get("duration_s")]
            print(f"[seed {seed}] 실패 {r.get('fail')} | 벽돌 {len(times)}장, 한 장 최대 {max(times) if times else 0:.2f} s | {summary(ev)}", flush=True)
            t_end = time.time()
            while v.is_running() and time.time() - t_end < (3.0 if w < a.walls - 1 else 1e9):
                v.sync(); time.sleep(0.05)


# ---------------------------------------------------------------- recording
class Recorder:
    """Composite frames: free view (W×H) | policy cameras stacked (H/3 each)."""

    def __init__(self, env, path, width=640, height=480, fps=20, label=""):
        import av
        from PIL import ImageFont
        self.env, self.w, self.h = env, width, height
        self.side = height // 3
        self.cam = free_camera()
        self.r_free = mujoco.Renderer(env.model, height, width)
        self.font = ImageFont.truetype(str(FONT), 15) if FONT.exists() else None
        self.label = label
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.c = av.open(str(self.path), "w")
        self.s = self.c.add_stream("libx264", rate=fps)
        self.s.width, self.s.height = width + self.side, height
        self.s.pix_fmt = "yuv420p"
        self.s.options = {"crf": "26", "preset": "medium"}
        self.frames = 0

    def frame(self, phase="", k=-1):
        from PIL import Image, ImageDraw
        env = self.env
        self.r_free.update_scene(env.data, camera=self.cam, scene_option=env.vopt)
        big = Image.fromarray(self.r_free.render())
        canvas = Image.new("RGB", (self.w + self.side, self.h), (20, 20, 20))
        canvas.paste(big, (0, 0))
        d = ImageDraw.Draw(canvas)
        for i, name in enumerate(("front", "top", "wrist")):
            im = Image.fromarray(env.render(name)).resize((self.side, self.side), Image.NEAREST)
            canvas.paste(im, (self.w, i * self.side))
            d.text((self.w + 4, i * self.side + 2), name, fill=(255, 255, 0), font=self.font)
        txt = f"{self.label}  t {env.t * env.cfg.ctrl_dt:6.2f} s  벽돌 {k + 1 if k >= 0 else '-'}/16  {phase}"
        d.text((8, 6), txt, fill=(255, 255, 255), font=self.font)
        for p in self.s.encode(__import__("av").VideoFrame.from_image(canvas)):
            self.c.mux(p)
        self.frames += 1

    def close(self):
        for p in self.s.encode():
            self.c.mux(p)
        self.c.close()
        self.r_free.close()


def speedup(src, dst, factor, width=None):
    """Keep every factor-th frame (same fps) and optionally downscale; returns the size in bytes."""
    import av
    inp = av.open(str(src))
    vs = inp.streams.video[0]
    out = av.open(str(dst), "w")
    w = width or vs.codec_context.width
    h = int(round(vs.codec_context.height * w / vs.codec_context.width / 2) * 2)
    st = out.add_stream("libx264", rate=int(vs.average_rate or 20))
    st.width, st.height, st.pix_fmt = w, h, "yuv420p"
    st.options = {"crf": "28", "preset": "medium"}
    for i, fr in enumerate(inp.decode(video=0)):
        if i % factor:
            continue
        img = fr.to_image().resize((w, h))
        for p in st.encode(av.VideoFrame.from_image(img)):
            out.mux(p)
    for p in st.encode():
        out.mux(p)
    out.close(); inp.close()
    return Path(dst).stat().st_size


def record(a):
    env = WallEnv(scene_config(a.env_version))
    ex = WallExpert(env)
    rec = Recorder(env, a.record, label=f"{a.env_version} seed {a.seed}")
    env.reset(a.seed, arm_q=ex.q_ret)
    rec.frame("start")

    def on_step(env_, action, phase, k):
        rec.frame(phase, k)

    r = ex.run(a.seed, on_step=on_step)
    rec.frame("end")
    rec.close()
    ev = evaluate_env(env)
    info = {"seed": a.seed, "fail": r.get("fail"), "frames": rec.frames, "file": str(a.record),
            "brick_times_s": [b.get("duration_s") for b in r["bricks"]], "spec_pass": ev["spec_pass"], "summary": summary(ev)}
    if a.short:
        dst = Path(a.record).with_name(Path(a.record).stem + f"_x{a.short}.mp4")
        info["short"] = {"file": str(dst), "factor": a.short, "bytes": speedup(a.record, dst, a.short, width=a.short_width)}
    Path(a.record).with_suffix(".json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
    print(json.dumps(info, ensure_ascii=False, indent=1))
    env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=700000, help="초기 배치 시작값 (파일럿 700000 ~ 700049)")
    ap.add_argument("--walls", type=int, default=1, help="보기: 연달아 볼 벽 개수")
    ap.add_argument("--speed", type=float, default=1.0, help="보기: 재생 속도 배수 (1 = 실시간)")
    ap.add_argument("--record", default="", help="녹화 mp4 경로 (화면 없이, EGL)")
    ap.add_argument("--short", type=int, default=0, help="녹화: N배속 축약본도 저장")
    ap.add_argument("--short-width", type=int, default=640)
    ap.add_argument("--env-version", default="main_v1", help="pilot_20261016 | main_v1")
    a = ap.parse_args()
    if not 700000 <= a.seed <= 700049 or a.seed + a.walls - 1 > 700049:
        raise SystemExit("파일럿 시작값 범위(700000 ~ 700049)만 쓴다")
    if a.record:
        if Path(a.record).exists():
            raise SystemExit(f"{a.record} 이미 있음: 덮어쓰지 않음")
        record(a)
    else:
        view(a)


if __name__ == "__main__":
    main()

"""카메라 입력 그림 (2026-11-03, 피드백 ⑥ "로봇이 카메라에서 받는 정보가 뭔지", 우분투지시_검측정보진단_20261103.md 3절 D).
진단용이며 평가 결과가 아니다.

벽 하나를 rollout.run_wall로 (지정한 검측 설정, 기본 all_held = 카메라 · 손에 든 벽돌 기준) 돌리며 같은 실행에서 세 가지를 남긴다.
  1. 정책 입력: 지정한 벽돌(기본 3번째, 0부터 2)을 집기 직전 · 놓기 직전에 정책이 받은 front · top · wrist 128² 영상 그대로와
     8배 확대본(nearest). 집기 직전 = 그 벽돌을 G1이 '쥠'으로 판정하기 전 마지막으로 정책이 집게를 닫기 시작한 스텝,
     놓기 직전 = 그 벽돌을 든 채 정책이 집게를 풀기 시작해 G2가 가로챈(또는 15 cm 밖이라 정책이 혼자 푼) 스텝.
  2. G2 측정: 손 안 그 벽돌을 처음 잰 순간의 top 1280 × 960 컬러에, 벽돌로 잡힌 화소 윤곽 · 찾은 중심과 방향 · 숫자를 덧그림.
  3. G3 측정: 놓인 그 벽돌을 판정한 순간의 같은 덧그림.
어느 순간인지 고르는 데와 '참값과의 차'에만 시뮬레이터 참값을 쓴다(그림 고르기 · 표시 전용). 검측 · 정책 동작은 바꾸지 않는다:
측정 함수는 원래 함수를 그대로 부르고 결과도 그대로 돌려주며, 그 순간의 화소만 옆에 복사해 둔다.

실행: PYTHONPATH=. MUJOCO_GL=egl venv/bin/python -m experiments.diag_camera_figs --ckpt <pretrained_model> \
        --seed 700300 --out outputs/wall3/<이름> --fig-dir ../01_사기설/<보고서 폴더>/그림
"""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
CAMS = ("front", "top", "wrist")
CLOSE = 0.15            # grip command crossing this upward = the policy starts closing (= diag_trace_causes)
MAX_PNG = 1_000_000


def held_pixels(cam, tcp_p, tcp_R, L, dz_pad, z_tol=0.006):
    """The pixels BrickPoseEstimator.held_brick keeps as the held brick's top face (same steps, read only)."""
    from wall3.config import L_FULL, W, H
    from wall3.perception import CHROMA_MIN, MIN_PX
    ax = np.array([tcp_R[0, 0], tcp_R[1, 0]]); ax /= np.linalg.norm(ax)
    ay = np.array([-ax[1], ax[0]])
    z_nom = tcp_p[2] - dz_pad + H / 2
    rgb, hgt = cam.last_rgb, cam._hgt
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    red = (r - b >= CHROMA_MIN) & (r >= g) & (g >= b - 10)
    ys0, xs0 = np.nonzero(red & (np.abs(hgt - z_nom) < 0.03))
    if len(ys0) < MIN_PX:
        return None
    p0 = cam.xyz(ys0, xs0)
    r0 = p0[:, :2] - tcp_p[:2]
    near = (np.abs(r0 @ ay) < W / 2 + 0.01) & (np.abs(r0 @ ax) < L_FULL)
    if near.sum() < MIN_PX:
        return None
    z_top = float(np.percentile(p0[near, 2], 80))
    ys, xs = np.nonzero(red & (np.abs(hgt - z_top) < z_tol))
    rel = cam.xyz(ys, xs)[:, :2] - tcp_p[:2]
    sel = (np.abs(rel @ ay) < W / 2 + 0.01) & (np.abs(rel @ ax) < L_FULL)
    return ys[sel], xs[sel]


def save_png(im, path):
    """PNG at most 1 MB: as is, else 256-colour palette."""
    im.save(path, optimize=True)
    if path.stat().st_size > MAX_PNG:
        im.quantize(256).save(path, optimize=True)
    return path.stat().st_size


def overlay(cap, path, title, font_path):
    """Top 1280 x 960 colour + the pixels taken as the brick (outline), found centre / direction, numbers."""
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    rgb = cap["rgb"].copy()
    m = np.zeros(rgb.shape[:2], np.uint8)
    m[cap["ys"], cap["xs"]] = 1
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(rgb, cnts, -1, (255, 255, 0), 2)
    c, a = np.array(cap["uv_c"]), np.array(cap["uv_a"])
    cv2.arrowedLine(rgb, tuple(int(v) for v in c), tuple(int(v) for v in a), (255, 40, 40), 3, tipLength=0.25)
    cv2.circle(rgb, tuple(int(v) for v in c), 6, (255, 40, 40), -1)
    t = np.array(cap["uv_t"])
    cv2.drawMarker(rgb, tuple(int(v) for v in t), (40, 255, 40), cv2.MARKER_CROSS, 22, 2)
    im = Image.fromarray(rgb)
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(str(font_path), 22)
    lines = [title] + cap["text"]
    w = max(d.textlength(s, font=f) for s in lines) + 20
    d.rectangle([0, 0, w, 10 + 30 * len(lines)], fill=(0, 0, 0))
    for i, s in enumerate(lines):
        d.text((10, 6 + 30 * i), s, fill=(255, 255, 255), font=f)
    leg = "노란 선: 벽돌로 잡힌 화소 윤곽 · 빨간 점과 화살표: 찾은 중심과 방향 · 초록 +: 참값 중심"
    lw = d.textlength(leg, font=f) + 20
    d.rectangle([0, im.height - 40, lw, im.height], fill=(0, 0, 0))
    d.text((10, im.height - 34), leg, fill=(255, 255, 255), font=f)
    return save_png(im, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--env-version", default="main_v1_eval")
    ap.add_argument("--seed", type=int, default=700300)
    ap.add_argument("--brick", type=int, default=2, help="0-based brick / slot (default 2 = the 3rd)")
    ap.add_argument("--gate-pose-source", choices=("camera", "truth"), default="camera")
    ap.add_argument("--gate-target", choices=("turn", "held_truth", "held_grasp"), default="held_grasp")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig-dir", required=True)
    a = ap.parse_args()
    if not (700300 <= a.seed <= 700399):
        raise SystemExit("시작값은 평가와 같은 700300 ~ 700399만")
    out, figd = Path(a.out), Path(a.fig_dir)
    for d in (out, figd):
        if d.exists():
            raise SystemExit(f"{d} 이미 있음: 덮어쓰지 않음")
    out.mkdir(parents=True)
    figd.mkdir(parents=True)
    import torch
    from PIL import Image
    from eval_policy import load_policy
    from wall3 import rollout as RO
    from wall3.config import scene_config, RETRACT_Q, H
    from wall3.scene import WallEnv
    from wall3.g1 import G1Config
    from wall3.gates import WallGates, GateConfig
    from wall3.perception import BrickPoseEstimator
    from wall3.poses import yaw_of_quat
    from wall3.view_collect import FONT
    B = a.brick
    policy, pcfg, pre, post = load_policy(a.ckpt, "cuda", None, None)
    cfg_in = set(pcfg.input_features)
    env = WallEnv(scene_config(a.env_version))
    gates = WallGates(env, policy, g1=True, g2=True, g3=True, g1_cfg=G1Config(thr_load=4.5, thr_angle=0.47),
                      cfg=GateConfig(pose_source=a.gate_pose_source, target=a.gate_target))
    cam = gates.cam
    st = {"src": "policy", "obs": {}, "closings": [], "pick": None, "place": None, "g2": None, "g3": None, "g3_last": None}

    # 1) what the policy got: the images obs_of rendered for it (uint8 back from the float tensor, exact)
    obs0 = RO.obs_of

    def obs_hook(env_, k, cfg_in_, torch_):
        o = obs0(env_, k, cfg_in_, torch_)
        if st["place"] is None:
            st["obs"][int(env_.t)] = (int(k), {c: (o[f"observation.images.{c}"][0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                                               for c in CAMS})
            for t in [t for t in st["obs"] if t < env_.t - 600 and t not in (st["closings"][-1:] or [None])]:
                del st["obs"][t]                              # keep 30 s and the last closing
        return o
    RO.obs_of = obs_hook

    step0 = env.step

    def step(arm_ctrl, grip_cmd):
        t0 = int(env.t)
        prev = float(env.last_action[6]) if hasattr(env, "last_action") else 0.0
        r = step0(arm_ctrl, grip_cmd)
        if st["src"] == "policy" and st["pick"] is None and prev <= CLOSE < float(grip_cmd):
            tp, _ = env.tcp()
            if float(np.hypot(*(tp[:2] - env.brick_pose(B)[0][:2]))) < 0.05:
                st["closings"].append(t0)                    # obs at t0 produced this action
        return r
    env.step = step

    after0, before0 = gates.after, gates.before

    def after(act):
        st["src"] = "g1"
        try:
            r = after0(act)
            if st["pick"] is None and gates.carrying and gates.held_b == B and st["closings"]:
                st["pick"] = st["closings"][-1]
            return r
        finally:
            st["src"] = "policy"

    def before(act):
        st["src"] = "gate"
        try:
            t0, held, carrying = int(env.t), gates.held_b, gates.carrying
            n2 = len(gates.ev["g2"])
            st["placing"] = st["place"] is None and st["pick"] is not None and carrying and held == B   # G3 runs inside before0
            r = before0(act)
            if st["place"] is None and st["pick"] is not None and carrying and held == B and (r or len(gates.ev["g2"]) > n2):
                st["place"] = t0
                st["place_by"] = "G2" if r else "정책 (15 cm 밖)"
            return r
        finally:
            st["src"], st["placing"] = "policy", False
    gates.after, gates.before = after, before

    # 2) G2: the held brick measurement (original function, result unchanged)
    hb0 = cam.held_brick

    def held_hook(tcp_p, tcp_R, L, dz_pad, *x, **kw):
        r = hb0(tcp_p, tcp_R, L, dz_pad, *x, **kw)
        if r is not None and st["g2"] is None and st["pick"] is not None and gates.held_b == B and st["src"] == "gate":
            px = held_pixels(cam, tcp_p, tcp_R, L, dz_pad)
            bp, bq = env.brick_pose(B)
            pos = np.asarray(r["pos"], float)
            top = np.r_[pos[:2], pos[2] + H / 2]
            tip = top + 0.06 * np.r_[np.cos(r["yaw"]), np.sin(r["yaw"]), 0.0]
            e = float(np.hypot(*(pos[:2] - bp[:2]))) * 1000
            ey = float(np.degrees((r["yaw"] - yaw_of_quat(bq) + np.pi / 2) % np.pi - np.pi / 2))
            st["g2"] = {"t": int(env.t), "rgb": cam.last_rgb.copy(), "ys": px[0], "xs": px[1],
                        "uv_c": cam.project(top)[:2], "uv_a": cam.project(tip)[:2], "uv_t": cam.project(np.r_[bp[:2], bp[2] + H / 2])[:2],
                        "meas": {"x_mm": pos[0] * 1000, "y_mm": pos[1] * 1000, "yaw_deg": float(np.degrees(r["yaw"])),
                                 "along_mm": r["along_m"] * 1000, "ends": r["ends"]},
                        "err_mm": e, "err_yaw_deg": ey,
                        "text": [f"측정(이 1장): 중심 x {pos[0] * 1000:.1f} mm · y {pos[1] * 1000:.1f} mm · 방향 {np.degrees(r['yaw']):.1f}°"
                                 f" (방향은 집게 축 = 관절값)",
                                 f"집은 위치: 벽돌 중심에서 길이 방향 {r['along_m'] * 1000:+.1f} mm (보인 끝: {'양쪽' if all(r['ends'].values()) else '한쪽'})",
                                 f"참값과의 차: 중심 {e:.2f} mm · 방향 {ey:+.2f}°"]}
        return r
    cam.held_brick = held_hook

    # 3) G3: the laid brick (same search as BrickPoseEstimator.find, keeping the detection's pixels)
    fit0 = BrickPoseEstimator._fit

    def fit_hook(self, pc, hgt):
        d = fit0(self, pc, hgt)
        if d is not None:
            d["_ys"], d["_xs"] = pc["ys"], pc["xs"]
        return d
    BrickPoseEstimator._fit = fit_hook
    xyB = np.array([env.design[B]["x"], env.design[B]["y"]])

    def find_hook(xy, r=0.05, z_range=None, L=None, yaw_prior=None):
        best, bd, bdet = None, r, None                                  # = BrickPoseEstimator.find
        for det in cam.detect():
            if z_range is not None and not (z_range[0] <= det["z_top"] <= z_range[1]):
                continue
            p = cam.pose_of(det, L, yaw_prior)
            dd = float(np.hypot(*(p["pos"][:2] - np.asarray(xy)[:2])))
            if dd < bd:
                best, bd, bdet = p, dd, det
        if best is not None and st["g3"] is None and (st["place"] is not None or st.get("placing")) and gates.held_b == B and \
                float(np.hypot(*(np.asarray(xy)[:2] - xyB))) < 1e-6:
            bp, bq = env.brick_pose(B)
            pos = np.asarray(best["pos"], float)
            top = np.r_[pos[:2], pos[2] + H / 2]
            tip = top + 0.06 * np.r_[np.cos(best["yaw"]), np.sin(best["yaw"]), 0.0]
            e = float(np.hypot(*(pos[:2] - bp[:2]))) * 1000
            ey = float(np.degrees((best["yaw"] - yaw_of_quat(bq) + np.pi / 2) % np.pi - np.pi / 2))
            cap = {"t": int(env.t), "rgb": cam.last_rgb.copy(), "ys": bdet["_ys"], "xs": bdet["_xs"],
                   "uv_c": cam.project(top)[:2], "uv_a": cam.project(tip)[:2], "uv_t": cam.project(np.r_[bp[:2], bp[2] + H / 2])[:2],
                   "full": bool(best["full"]), "meas": {"x_mm": pos[0] * 1000, "y_mm": pos[1] * 1000, "z_mm": pos[2] * 1000,
                                                        "yaw_deg": float(np.degrees(best["yaw"])), "tilt_deg": best["tilt_deg"]},
                   "err_mm": e, "err_yaw_deg": ey,
                   "text": [f"측정: 중심 x {pos[0] * 1000:.1f} mm · y {pos[1] * 1000:.1f} mm · z {pos[2] * 1000:.1f} mm · 방향 {np.degrees(best['yaw']):.1f}°"
                            f" · 기울기 {best['tilt_deg']:.1f}°",
                            f"설계 자리와의 차: {float(np.hypot(*(pos[:2] - xyB))) * 1000:.2f} mm ({'전체가 보임' if best['full'] else '일부만 보임'})",
                            f"참값과의 차: 중심 {e:.2f} mm · 방향 {ey:+.2f}°"]}
            if best["full"]:
                st["g3"] = cap
            else:
                st["g3_last"] = cap
        return best
    cam.find = find_hook

    meta = {"ckpt": a.ckpt, "env_version": a.env_version, "seed": a.seed, "brick": B, "gates": gates.summary_config(),
            "git_commit": subprocess.check_output(["git", "-C", str(SIM), "rev-parse", "HEAD"], text=True).strip(),
            "scene_sha256": hashlib.sha256(json.dumps(env.cfg.to_dict(), sort_keys=True, default=float).encode()).hexdigest(),
            "script": "diag_camera_figs"}
    r = RO.run_wall(env, policy, pre, post, cfg_in, a.seed, torch, None, np.array(RETRACT_Q), gates)
    env.close()
    if st["g3"] is None:
        st["g3"] = st["g3_last"]
    files, info = {}, {}
    for tag, t in (("집기직전", st["pick"]), ("놓기직전", st["place"])):
        if t is None or t not in st["obs"]:
            info[tag] = None
            continue
        k, ims = st["obs"][t]
        info[tag] = {"obs_t": t, "turn_k": k, "time_s": t * env.cfg.ctrl_dt}
        for c in CAMS:
            p1 = figd / f"정책입력_{tag}_{c}_128.png"
            p8 = figd / f"정책입력_{tag}_{c}_8배.png"
            files[p1.name] = save_png(Image.fromarray(ims[c]), p1)
            files[p8.name] = save_png(Image.fromarray(ims[c]).resize((1024, 1024), Image.NEAREST), p8)
    if st.get("place_by") and info.get("놓기직전"):
        info["놓기직전"]["by"] = st["place_by"]
    for tag, title in (("g2", f"G2: 손 안 벽돌 측정 (벽돌 {B + 1}, top 1280 × 960)"), ("g3", f"G3: 놓인 벽돌 측정 (벽돌 {B + 1}, top 1280 × 960)")):
        cap = st[tag]
        if cap is None:
            info[tag] = None
            continue
        p = figd / f"{'G2_손안측정' if tag == 'g2' else 'G3_놓인벽돌측정'}.png"
        files[p.name] = overlay(cap, p, title, FONT)
        info[tag] = {k: v for k, v in cap.items() if k in ("t", "meas", "err_mm", "err_yaw_deg", "full", "text")}
    res = {"meta": meta, "frames": info, "files_bytes": files,
           "wall": {"slots_filled": r["eval"]["slots_filled"], "filled": int(sum(r["eval"]["slots_filled"])),
                    "bricks": [{k: b[k] for k in ("slot", "end_reason", "final_err_mm")} for b in r["bricks"]]}}
    (out / "figs.json").write_text(json.dumps(res, indent=1, default=float, ensure_ascii=False))
    (out / "rollout_wall.json").write_text(json.dumps(r, indent=1, default=float))
    print(json.dumps({"frames": info, "files_bytes": files, "filled": res["wall"]["filled"]}, indent=1, default=float, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""현실 제약 A · B 주입 (2026-11-16, 01_사기설/우분투지시_현실제약_ablation_20261116.md 4절).

벽마다 시작값에서 정해지는 난수(default_rng([시작값, 20261116]))로 오차를 뽑는다. 항목 순서가 고정이라 같은 벽은 8조건 모두 같은 값이고,
꺼진 묶음(A 또는 B)의 항목은 0이다.
  A1 벽 등록 오차      검측 · 대본이 믿는 설계 자리(도면) = 실제 설계 자리를 벽 가운데 기준으로 회전 + 평행 이동. 채점은 실제 설계 자리
  A2 작업대 카메라 보정 측정 모듈이 믿는 카메라 설치 자세 = 실제 자세 × 작은 회전, + 작은 이동
  A3 손목 카메라 보정   같은 방식 (G0v)
  A4 깊이 잡음          영상마다 매끄러운 오차 면(8 × 6 격자 쌍선형 보간) + 화소 잡음
  A5 손목 힘 센서       z 치우침(벽마다) + 읽을 때마다 잡음
  B  공급 팔레트       reset 직후 공급대 벽돌 16장 전체를 무작위 방향으로 30 mm 평행 이동(도면 · 검측 · 대본 · 정책은 모름)
참값은 오차를 '만드는' 데만 쓴다(시뮬레이터가 현실 역할). 검측 · 대본은 오차가 들어간 값만 본다.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import mujoco
import numpy as np

from wall3.config import WALL_Y
from wall3.perception import BrickPoseEstimator, CHROMA_MIN, FLAT_MAX, MIN_PX, _rect

SALT = 20261116
A_WALL_MM, A_WALL_DEG = 3.0, 0.3
A_TOP_MM, A_TOP_DEG = 1.0, 0.05
A_WRIST_MM, A_WRIST_DEG = 1.0, 0.3
A_DEPTH_TOP_MM, A_DEPTH_WRIST_MM, A_DEPTH_PIX_MM = 1.0, 0.5, 0.2
A_FT_BIAS_N, A_FT_SIGMA_N = 1.0, 0.5
B_PALLET_MM = 30.0


def small_rot(r):
    """Rotation matrix for small rotation angles r = (rx, ry, rz) rad (Rodrigues)."""
    R = np.zeros((3, 3))
    cv2.Rodrigues(np.asarray(r, float).reshape(3, 1), R)
    return R


@dataclass
class Perturb:
    seed: int
    A: bool
    B: bool
    wall_dx: float = 0.0          # m
    wall_dy: float = 0.0
    wall_dth: float = 0.0         # rad, about the wall centre (0, WALL_Y)
    top_t: tuple = (0.0, 0.0, 0.0)        # m (world)
    top_r: tuple = (0.0, 0.0, 0.0)        # rad (camera frame)
    wrist_t: tuple = (0.0, 0.0, 0.0)
    wrist_r: tuple = (0.0, 0.0, 0.0)
    depth_top: float = 0.0        # m, smooth field sigma
    depth_wrist: float = 0.0
    depth_pix: float = 0.0
    ft_bias: float = 0.0          # N (z)
    ft_sigma: float = 0.0         # N per axis per reading
    pallet: tuple = (0.0, 0.0)    # m

    def to_dict(self):
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}


def make_perturb(seed: int, A: bool, B: bool) -> Perturb:
    r = np.random.default_rng([int(seed), SALT])
    u = lambda lim, n=None: r.uniform(-lim, lim, n)
    wall = (u(A_WALL_MM / 1000), u(A_WALL_MM / 1000), np.deg2rad(u(A_WALL_DEG)))
    top = (tuple(u(A_TOP_MM / 1000, 3)), tuple(np.deg2rad(u(A_TOP_DEG, 3))))
    wrist = (tuple(u(A_WRIST_MM / 1000, 3)), tuple(np.deg2rad(u(A_WRIST_DEG, 3))))
    ft_bias = u(A_FT_BIAS_N)
    ang = r.uniform(0.0, 2 * np.pi)
    pallet = (B_PALLET_MM / 1000 * np.cos(ang), B_PALLET_MM / 1000 * np.sin(ang))
    p = Perturb(seed=int(seed), A=bool(A), B=bool(B))
    if A:
        p.wall_dx, p.wall_dy, p.wall_dth = (float(x) for x in wall)
        p.top_t, p.top_r = tuple(float(x) for x in top[0]), tuple(float(x) for x in top[1])
        p.wrist_t, p.wrist_r = tuple(float(x) for x in wrist[0]), tuple(float(x) for x in wrist[1])
        p.depth_top, p.depth_wrist, p.depth_pix = A_DEPTH_TOP_MM / 1000, A_DEPTH_WRIST_MM / 1000, A_DEPTH_PIX_MM / 1000
        p.ft_bias, p.ft_sigma = float(ft_bias), A_FT_SIGMA_N
    if B:
        p.pallet = tuple(float(x) for x in pallet)
    return p


def belief_design(design, p: Perturb):
    """The drawing as the robot believes it (A1): every slot turned by wall_dth about (0, WALL_Y), then shifted."""
    c, s = np.cos(p.wall_dth), np.sin(p.wall_dth)
    out = []
    for d in design:
        x, y = d["x"], d["y"] - WALL_Y
        e = dict(d)
        e["x"] = float(c * x - s * y + p.wall_dx)
        e["y"] = float(s * x + c * y + WALL_Y + p.wall_dy)
        e["yaw"] = float(d["yaw"] + p.wall_dth)
        out.append(e)
    return out


def shift_pallet(env, p: Perturb):
    """B: move every supply brick by p.pallet right after env.reset (the drawing is not changed)."""
    if not any(p.pallet):
        return
    d = env.data
    for i, a in enumerate(env.brick_qadr):
        d.qpos[a] += p.pallet[0]
        d.qpos[a + 1] += p.pallet[1]
        pos, yaw = env.supply[i]
        env.supply[i] = (pos + [p.pallet[0], p.pallet[1], 0.0], yaw)
    mujoco.mj_forward(env.model, d)


def noisy_ft(env, p: Perturb, rng):
    """A5: the wrist force/torque reading with a z bias and per-reading noise (installed on the env instance)."""
    orig = type(env).wrist_ft.__get__(env)

    def wrist_ft():
        F, T = orig()
        if p.ft_sigma > 0 or p.ft_bias:
            F = F + np.array([0.0, 0.0, p.ft_bias]) + rng.normal(0.0, p.ft_sigma, 3)
        return F, T
    env.wrist_ft = wrist_ft


class NoisyEstimator(BrickPoseEstimator):
    """BrickPoseEstimator whose camera pose belief (A2 / A3) and depth image (A4) carry the wall's errors.
    z_min: pixels at or below this height are not brick candidates. The wrist camera sees the wooden supply table, whose colour
    passes the brick colour test (the top camera never sees it), so G0v sets z_min just above the table top (2026-11-16 test)."""

    def __init__(self, env, cam="top", height=960, width=1280, z_min=None):
        super().__init__(env, cam, height, width)
        self.z_min = z_min
        self.set_noise((0, 0, 0), (0, 0, 0), 0.0, 0.0, None)

    def detect(self, force: bool = False):
        if self.z_min is None:
            return super().detect(force)
        key = int(self.env.t)
        if not force and self._key == key:
            return self._dets
        rgb, hgt = self.render()
        r, g, b = rgb[..., 0].astype(np.int16), rgb[..., 1].astype(np.int16), rgb[..., 2].astype(np.int16)
        colour = (r - b >= CHROMA_MIN) & (r >= g) & (g >= b - 10) & (hgt < 0.6) & (hgt > self.z_min)
        gy, gx = np.gradient(hgt * 1000.0)
        flat = np.hypot(gx, gy) < FLAT_MAX
        mask = cv2.morphologyEx((colour & flat).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, lab = cv2.connectedComponents(mask, connectivity=8)
        pieces = []
        for k in range(1, n):
            sel = lab == k
            if sel.sum() < MIN_PX:
                continue
            ys, xs = np.nonzero(sel)
            pts = self.xyz(ys, xs)
            c2, ex, lo, sh = _rect(pts[:, :2])
            pieces.append({"ys": ys, "xs": xs, "pts": pts, "c": c2, "ex": ex, "long": lo, "short": sh})
        pieces = self._merge(pieces)
        dets = [d for pc in pieces for d in [self._fit(pc, hgt)] if d is not None]
        self._key, self._dets = key, dets
        return dets

    def set_noise(self, t_err, r_err, sigma_smooth, sigma_pix, rng):
        self.t_err = np.asarray(t_err, float)
        self.R_err = small_rot(r_err)
        self.sig_s, self.sig_p, self.rng = float(sigma_smooth), float(sigma_pix), rng
        self._key = None

    def render(self):
        d = self.env.data
        self.r_rgb.update_scene(d, camera=self.cam_name, scene_option=self.vopt); rgb = self.r_rgb.render().copy()
        self.r_dep.update_scene(d, camera=self.cam_name, scene_option=self.vopt); dep = self.r_dep.render().copy()
        if self.rng is not None and (self.sig_s > 0 or self.sig_p > 0):
            field = cv2.resize(self.rng.normal(0.0, self.sig_s, (6, 8)).astype(np.float32), (self.w, self.h),
                               interpolation=cv2.INTER_LINEAR)
            dep = dep + field + self.rng.normal(0.0, self.sig_p, dep.shape).astype(np.float32)
        self._R = d.cam_xmat[self.cam_id].reshape(3, 3).copy() @ self.R_err      # the pose the module believes (calibration)
        self._t = d.cam_xpos[self.cam_id].copy() + self.t_err
        self._dep = dep
        R, t = self._R, self._t
        hgt = t[2] + dep * (R[2, 0] * self._dirs[..., 0] + R[2, 1] * self._dirs[..., 1] - R[2, 2])
        self.n_renders += 1
        self.last_rgb, self._hgt = rgb, hgt
        return rgb, hgt

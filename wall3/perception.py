"""검측용 벽돌 위치 추정: 작업대 위 카메라(top) 고해상도 RGB-D (실험설계 md 10.3절, G2 · G3용).

brick_sim/perception.py(소형 실험, 2026-10-08)를 3단 벽 장면에 옮긴 것. 정책 입력(128²)과는 별개로 같은 top 카메라를
1280 × 960 RGB + 깊이로 렌더해, 카메라 내부 값(fovy)·설치 자세(캘리브레이션)와 도면의 벽돌 크기만으로 벽돌 자세를 잰다.
**시뮬레이터의 참값 자세는 전혀 쓰지 않는다**(검측 판단에도, 짝짓기에도). 어떤 검출이 어떤 벽돌인지는 작업 순서로 정한다:
손에 든 벽돌 = 집게(관절값 순기구학) 바로 아래, 놓은 벽돌 = 그 설계 자리 근처. 참값은 오차 기록(감사)에만 쓴다(gates.py).

소형 실험과 다른 점: 적벽돌 색, 줄눈 10 mm(같은 단 벽돌이 붙지 않아 한 덩어리를 나누는 처리 없음), 온장 190 · 반장 90 mm
(반장 윗면은 정사각형이라 방향이 모호 → 사전 정보로 결정), 모르타르 판(회색)과 공급대는 색으로 걸러짐(공급대는 화면 밖).

파이프라인: 렌더 → 화소마다 세계 좌표 → 벽돌 색 · 평평한 윗면 화소 → 연결 덩어리 → (집게에 잘린 조각 합치기) → 평면 맞춤(기울기),
최소 사각형(방향), 가려지지 않은 모서리 + 알려진 L × W로 중심, 윗면 높이 − H/2로 중심 높이.
"""
from __future__ import annotations

import numpy as np
import mujoco
import cv2

from wall3.config import L_FULL, L_HALF, W, H

CHROMA_MIN = 40          # R − B: 적벽돌(밝은 윗면 · 어두운 옆면 모두 ≥ 40), 모르타르 판 · 바닥 · 로봇은 회색/검정/파랑
MIN_PX = 30
OCCLUDER_DZ = 0.003      # m: 덩어리 윗면보다 이만큼 높은, 덩어리 아닌 화소 = 팔/집게 (그 옆 모서리는 믿지 않음)
FLAT_MAX = 0.45          # mm/px: 윗면으로 볼 기울기 상한 (옆면 띠 제외)
MERGE_L, MERGE_W = 0.015, 0.012
EDGE_BAND_PX = 3
EDGE_OCC_FRAC = 0.25
PLANE_INLIER = 0.003
SIDE_TOL = 0.008
HALF_FULL_SPLIT = 0.14   # m: 보이는 길이가 이보다 길면 온장


def _rect(points2d):
    (cx, cy), (w, h), ang = cv2.minAreaRect(np.asarray(points2d, dtype=np.float32))
    a = np.deg2rad(ang)
    ex = np.array([np.cos(a), np.sin(a)])
    if h > w:
        w, h = h, w
        ex = np.array([-ex[1], ex[0]])
    return np.array([cx, cy]), ex, float(w), float(h)


def _fold(a):
    return (a + np.pi / 2) % np.pi - np.pi / 2


class BrickPoseEstimator:
    def __init__(self, env, cam: str = "top", height: int = 960, width: int = 1280):
        self.env, self.cam_name, self.h, self.w = env, cam, int(height), int(width)
        m = env.model
        self.cam_id = m.camera(cam).id
        self.fovy = float(m.cam_fovy[self.cam_id])
        self.f = (self.h / 2) / np.tan(np.deg2rad(self.fovy) / 2)
        self.r_rgb = mujoco.Renderer(m, self.h, self.w)
        self.r_dep = mujoco.Renderer(m, self.h, self.w)
        self.r_dep.enable_depth_rendering()
        u = (np.arange(self.w) + 0.5 - self.w / 2) / self.f
        v = -(np.arange(self.h) + 0.5 - self.h / 2) / self.f
        self._dirs = np.stack(np.meshgrid(u, v), axis=-1)
        self._key, self._dets = None, None
        self.n_renders = 0
        # a real camera does not see the simulator's visual-only markers (e.g. the gripper's green axis cylinder site,
        # which covered brick ends in the 2026-10-20 tests): render sites off for this inspection camera only
        self.vopt = mujoco.MjvOption()
        self.vopt.sitegroup[:] = 0

    def close(self):
        for r in (self.r_rgb, self.r_dep):
            r.close()

    # ------------------------------------------------------------------ camera
    def render(self):
        d = self.env.data
        self.r_rgb.update_scene(d, camera=self.cam_name, scene_option=self.vopt); rgb = self.r_rgb.render().copy()
        self.r_dep.update_scene(d, camera=self.cam_name, scene_option=self.vopt); dep = self.r_dep.render().copy()
        self._R = d.cam_xmat[self.cam_id].reshape(3, 3).copy(); self._t = d.cam_xpos[self.cam_id].copy(); self._dep = dep
        R, t = self._R, self._t
        hgt = t[2] + dep * (R[2, 0] * self._dirs[..., 0] + R[2, 1] * self._dirs[..., 1] - R[2, 2])
        self.n_renders += 1
        self.last_rgb, self._hgt = rgb, hgt
        return rgb, hgt

    def xyz(self, ys, xs):
        dep = self._dep[ys, xs]
        p = np.stack([self._dirs[ys, xs, 0] * dep, self._dirs[ys, xs, 1] * dep, -dep], axis=-1)
        return p @ self._R.T + self._t

    def project(self, p):
        pc = self._R.T @ (np.asarray(p, dtype=np.float64) - self._t)
        z = -pc[2]
        return self.f * pc[0] / z + self.w / 2, -self.f * pc[1] / z + self.h / 2, z

    # ------------------------------------------------------------------ detection
    def detect(self, force: bool = False):
        """All bricks visible now (memoised per control step)."""
        key = int(self.env.t)
        if not force and self._key == key:
            return self._dets
        rgb, hgt = self.render()
        r, g, b = rgb[..., 0].astype(np.int16), rgb[..., 1].astype(np.int16), rgb[..., 2].astype(np.int16)
        colour = (r - b >= CHROMA_MIN) & (r >= g) & (g >= b - 10) & (hgt < 0.6)
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

    def _merge(self, pieces):
        """Pieces of one brick cut apart by the gripper fingers / wrist: merged when the union still fits one full brick."""
        changed = True
        while changed and len(pieces) > 1:
            changed = False
            for i in range(len(pieces)):
                for j in range(i + 1, len(pieces)):
                    a, b = pieces[i], pieces[j]
                    if float(np.hypot(*(a["c"] - b["c"]))) > L_FULL or abs(np.median(a["pts"][:, 2]) - np.median(b["pts"][:, 2])) > 0.01:
                        continue
                    pts = np.concatenate([a["pts"], b["pts"]])
                    c2, ex, lo, sh = _rect(pts[:, :2])
                    if lo <= L_FULL + MERGE_L and sh <= W + MERGE_W:
                        pieces[i] = {"ys": np.concatenate([a["ys"], b["ys"]]), "xs": np.concatenate([a["xs"], b["xs"]]),
                                     "pts": pts, "c": c2, "ex": ex, "long": lo, "short": sh, "merged": True}
                        del pieces[j]
                        changed = True
                        break
                if changed:
                    break
        return pieces

    @staticmethod
    def _edge_occluded(ys, xs, u, v, u_ext, v_ext, occ_d):
        out = {}
        for name, coord, ext in (("u-", u, u_ext[0]), ("u+", u, u_ext[1]), ("v-", v, v_ext[0]), ("v+", v, v_ext[1])):
            band = np.abs(coord - ext) < 0.002
            out[name] = True if band.sum() < 3 else bool(np.mean(occ_d[ys[band], xs[band]]) > EDGE_OCC_FRAC)
        return out

    def _fit(self, pc, hgt):
        ys, xs, pts = pc["ys"], pc["xs"], pc["pts"]
        n_px = int(len(ys))
        cen = pts.mean(0); nrm = np.linalg.svd(pts - cen, full_matrices=False)[2][2]
        nrm = -nrm if nrm[2] < 0 else nrm
        inl = np.abs((pts - cen) @ nrm) < PLANE_INLIER
        if inl.sum() >= 20:
            cen = pts[inl].mean(0); nrm = np.linalg.svd(pts[inl] - cen, full_matrices=False)[2][2]
            nrm = -nrm if nrm[2] < 0 else nrm
            inl = np.abs((pts - cen) @ nrm) < PLANE_INLIER
            pts, ys, xs = pts[inl], ys[inl], xs[inl]
        if len(pts) < MIN_PX:
            return None
        tilt = float(np.degrees(np.arccos(np.clip(nrm[2], -1, 1))))
        z_top = float(np.percentile(pts[:, 2], 90))
        y0, y1, x0, x1 = max(ys.min() - 8, 0), min(ys.max() + 9, hgt.shape[0]), max(xs.min() - 8, 0), min(xs.max() + 9, hgt.shape[1])   # clip at the image border
        z_ref = float(np.median(pts[:, 2]))
        own = np.zeros((y1 - y0, x1 - x0), np.uint8); own[ys - y0, xs - x0] = 1
        own = cv2.dilate(own, np.ones((5, 5), np.uint8)).astype(bool)
        occ = ((hgt[y0:y1, x0:x1] > z_ref + OCCLUDER_DZ) & ~own).astype(np.uint8)
        occ_d = cv2.dilate(occ, np.ones((2 * EDGE_BAND_PX + 1,) * 2, np.uint8)).astype(bool)
        c2, ex, _, _ = _rect(pts[:, :2])
        ex3 = np.array([ex[0], ex[1], 0.0]); ex3 -= (ex3 @ nrm) * nrm; ex3 /= np.linalg.norm(ex3)
        ey3 = np.cross(nrm, ex3)
        rel = pts - cen
        u, v = rel @ ex3, rel @ ey3
        u_ext, v_ext = (float(u.min()), float(u.max())), (float(v.min()), float(v.max()))
        edges = self._edge_occluded(ys - y0, xs - x0, u, v, u_ext, v_ext, occ_d)
        return {"cen": cen, "nrm": nrm, "ex3": ex3, "ey3": ey3, "u_ext": u_ext, "v_ext": v_ext, "edges": edges,
                "long": u_ext[1] - u_ext[0], "short": v_ext[1] - v_ext[0], "tilt_deg": tilt, "z_top": z_top, "n_px": n_px,
                "merged": bool(pc.get("merged", False))}

    # ------------------------------------------------------------------ one brick from a detection
    def pose_of(self, det, L=None, yaw_prior=None):
        """Brick centre / yaw from a detection and the drawing's size L (None: guessed from the visible length).
        yaw_prior resolves the direction when the visible part cannot tell it (half brick, short fragment).
        Returns dict(pos, yaw, tilt_deg, full, ambiguous, L)."""
        if L is None:
            L = L_FULL if det["long"] > HALF_FULL_SPLIT else L_HALF
        ex3, ey3 = det["ex3"], det["ey3"]
        u_ext, v_ext, edges = det["u_ext"], det["v_ext"], det["edges"]
        ambiguous = det["long"] < W + SIDE_TOL                     # visible part no longer than W: direction unknown
        yaw = float(np.arctan2(ex3[1], ex3[0]))
        if ambiguous and yaw_prior is not None and abs(_fold(yaw + np.pi / 2 - yaw_prior)) < abs(_fold(yaw - yaw_prior)):
            # swap the in-plane axes: u' = v, v' = -u
            ex3, ey3 = ey3, -ex3
            yaw = float(np.arctan2(ex3[1], ex3[0]))
            u_ext, v_ext = (v_ext[0], v_ext[1]), (-u_ext[1], -u_ext[0])
            e = edges
            edges = {"u-": e["v-"], "u+": e["v+"], "v-": e["u+"], "v+": e["u-"]}
        # along: from the trusted end(s) and L
        if edges["u-"] and not edges["u+"]:
            uc = u_ext[1] - L / 2
        elif edges["u+"] and not edges["u-"]:
            uc = u_ext[0] + L / 2
        else:
            uc = (u_ext[0] + u_ext[1]) / 2
        if edges["v-"] and not edges["v+"]:
            vc = v_ext[1] - W / 2
        elif edges["v+"] and not edges["v-"]:
            vc = v_ext[0] + W / 2
        else:
            vc = (v_ext[0] + v_ext[1]) / 2
        top = det["cen"] + uc * ex3 + vc * ey3
        if det["tilt_deg"] < 3.0:
            top[2] = det["z_top"]
        pos = top - det["nrm"] * (H / 2)
        full = (abs((u_ext[1] - u_ext[0]) - L) < SIDE_TOL and abs((v_ext[1] - v_ext[0]) - W) < SIDE_TOL
                and not any(edges.values()))
        return {"pos": pos, "yaw": yaw, "tilt_deg": det["tilt_deg"], "full": bool(full), "ambiguous": bool(ambiguous),
                "L": L, "n_px": det["n_px"], "edges": edges}

    def find(self, xy, r=0.05, z_range=None, L=None, yaw_prior=None):
        """The brick whose estimated centre is nearest to xy (within r): the role is decided by the work order
        (in the hand: below the TCP; laid: at its design slot). None if nothing there."""
        best, bd = None, r
        for det in self.detect():
            if z_range is not None and not (z_range[0] <= det["z_top"] <= z_range[1]):
                continue
            p = self.pose_of(det, L, yaw_prior)
            dd = float(np.hypot(*(p["pos"][:2] - np.asarray(xy)[:2])))
            if dd < bd:
                best, bd = p, dd
        return best

    def surface_z(self, x, y, yaw, offsets=(-0.07, 0.07), across=(-0.03, 0.03), r=0.006, z_lo=0.0, z_hi=0.35):
        """Height of the mortar bed / support beside (x, y): median height of grey (non-brick) flat pixels around points
        along the wall axis and across it. None if not visible."""
        self.detect()
        zs = []
        ex, ey = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
        rgb = self.last_rgb
        for o in offsets:
            for a in across:
                c = np.array([x, y]) + o * ex + a * ey
                uu, vv, zc = self.project((c[0], c[1], 0.2))
                rp = int(np.ceil(r * self.f / zc))
                v0, v1, u0, u1 = int(vv) - rp, int(vv) + rp + 1, int(uu) - rp, int(uu) + rp + 1
                if v0 < 0 or u0 < 0 or v1 > self.h or u1 > self.w:
                    continue
                ys, xs = np.mgrid[v0:v1, u0:u1]
                ys, xs = ys.ravel(), xs.ravel()
                p = self.xyz(ys, xs)
                col = rgb[ys, xs].astype(np.int16)
                grey = (col[:, 0] - col[:, 2]) < CHROMA_MIN / 2
                ok = (np.abs(p[:, 0] - c[0]) < r) & (np.abs(p[:, 1] - c[1]) < r) & (p[:, 2] > z_lo) & (p[:, 2] < z_hi) & grey
                if ok.sum() >= 5:
                    zs.append(float(np.median(p[ok, 2])))
        return float(np.median(zs)) if zs else None


    def held_brick(self, tcp_p, tcp_R, L, dz_pad, z_tol=0.006, probe=0.004):
        """The brick in the hand, with the gripper's own geometry as prior (what the robot knows from its joints):
        the brick hangs dz_pad below the TCP, its long axis along the TCP x axis (the jaw closes across W) and its
        width centred between the fingers. Only the grasp position ALONG the brick is unknown: it is read from the
        visible end(s). Brick-coloured pixels at the held brick's top height around the TCP are projected on the TCP
        axis; an extreme is a real end when just beyond it the camera sees something lower (floor / bed / gap), not the
        hand above it and not more brick. Measure with the brick lifted clear of its neighbours (they would be at the
        same height). Returns dict(pos, yaw, along_m, ends, n_px) or None."""
        ax = np.array([tcp_R[0, 0], tcp_R[1, 0]]); ax /= np.linalg.norm(ax)
        ay = np.array([-ax[1], ax[0]])
        z_nom = tcp_p[2] - dz_pad + H / 2
        self.detect()
        rgb, hgt = self.last_rgb, self._hgt
        r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
        red = (r - b >= CHROMA_MIN) & (r >= g) & (g >= b - 10)
        # the brick may hang a little higher / lower than the nominal grasp height (where the policy closed): take the
        # actual top height from the brick-coloured pixels around the TCP, then keep only that top face
        ys0, xs0 = np.nonzero(red & (np.abs(hgt - z_nom) < 0.03))
        if len(ys0) < MIN_PX:
            return None
        p0 = self.xyz(ys0, xs0)
        r0 = p0[:, :2] - tcp_p[:2]
        near = (np.abs(r0 @ ay) < W / 2 + 0.01) & (np.abs(r0 @ ax) < L_FULL)
        if near.sum() < MIN_PX:
            return None
        z_top = float(np.percentile(p0[near, 2], 80))
        colour = red & (np.abs(hgt - z_top) < z_tol)
        ys, xs = np.nonzero(colour)
        if len(ys) < MIN_PX:
            return None
        pts = self.xyz(ys, xs)
        rel = pts[:, :2] - tcp_p[:2]
        a, c = rel @ ax, rel @ ay
        sel = (np.abs(c) < W / 2 + 0.01) & (np.abs(a) < L_FULL)
        if sel.sum() < MIN_PX:
            return None
        a = a[sel]
        u_min, u_max = float(np.percentile(a, 0.5)), float(np.percentile(a, 99.5))

        def end_is_real(u_end, sgn):
            hs = []
            for t in np.linspace(-W / 3, W / 3, 7):
                q = tcp_p[:2] + (u_end + sgn * probe) * ax + t * ay
                uu, vv, _ = self.project((q[0], q[1], z_top))
                ui, vi = int(round(uu)), int(round(vv))
                if 0 <= vi < self.h and 0 <= ui < self.w:
                    hs.append(hgt[vi, ui])
            if len(hs) < 4:
                return False
            hs = np.array(hs)
            return bool(np.mean(hs < z_top - 0.01) > 0.7)          # beyond the edge: clearly lower = a real end

        lo_ok, hi_ok = end_is_real(u_min, -1), end_is_real(u_max, +1)
        if lo_ok and hi_ok and abs((u_max - u_min) - L) < 0.012:
            uc = (u_min + u_max) / 2
        elif lo_ok:
            uc = u_min + L / 2
        elif hi_ok:
            uc = u_max - L / 2
        else:
            return None                                              # no visible end: the grasp position is unknown
        if abs(uc) > L / 2 - 0.015:                                  # the jaw cannot hold the brick beyond its end
            return None
        # in-hand yaw and sideways offset: the middle of the brick's width along its length (only where the full width
        # is visible) is a straight line; its slope is the yaw of the brick relative to the jaw
        cc = c[sel]
        dpsi, c0 = 0.0, 0.0
        bins = np.arange(u_min, u_max, 0.006)
        mids, us = [], []
        for b0 in bins:
            m = (a >= b0) & (a < b0 + 0.006)
            if m.sum() < 6:
                continue
            lo, hi = np.percentile(cc[m], 2), np.percentile(cc[m], 98)
            if hi - lo > W - 0.008:
                mids.append((lo + hi) / 2); us.append(b0 + 0.003)
        if False and len(us) >= 6 and (max(us) - min(us)) > 0.06:   # in-hand yaw estimate off (3 ~ 8 deg errors, 2026-10-20)
            slope, icept = np.polyfit(us, mids, 1)
            resid = float(np.std(np.asarray(mids) - (slope * np.asarray(us) + icept)))
            if abs(np.degrees(np.arctan(slope))) < 10 and resid < 0.0015:   # a clean straight brick edge, else stay aligned
                dpsi = float(np.arctan(slope)); c0 = float(icept + slope * uc)
        axb = np.array([np.cos(dpsi) * ax[0] - np.sin(dpsi) * ax[1], np.sin(dpsi) * ax[0] + np.cos(dpsi) * ax[1]])
        pos = np.r_[tcp_p[:2] + uc * ax + c0 * ay, z_top - H / 2]
        return {"pos": pos, "yaw": float(np.arctan2(axb[1], axb[0])), "yaw_in_hand_deg": float(np.degrees(dpsi)),
                "across_m": c0, "along_m": float(uc), "n_px": int(sel.sum()),
                "ends": {"min": lo_ok, "max": hi_ok}, "visible_len_m": float(u_max - u_min), "yaw_bins": len(us)}

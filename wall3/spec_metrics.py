"""평가지표 md(01_사기설/평가지표_3층벽체_20261013.md)의 구현: 시방 5항목 판정 + 벽돌별 3D IoU.

모든 측정은 시뮬레이터 참값 자세로 하며 채점에만 쓴다. 측정점은 모든 벽돌의 모서리 8개(2.2절).
벽 좌표(평가지표 md 1절): x = 벽 길이, y' = 설계 앞면(y_f)에서 벽 안쪽(뒤쪽)으로 잰 거리(앞면 0, 뒷면 +90 mm), z = 위.
이 장면의 앞면은 로봇 반대쪽 면(세계 y = WALL_Y + W/2)이므로 y' = y_f − y_world.

구현에서 정한 해석(보고서에 기록)
- ① 1단 아랫면–바닥: 원천 정의상 음수가 생기지 않지만, MuJoCo 소프트 접촉은 벽돌을 바닥에 0.1 mm 안팎 파묻는다.
  −0.5 mm까지는 접촉 침투로 보고 0으로 판정한다(원값 base_min_raw_mm 기록). 그보다 깊으면 불합격.
- ④ 1단 양 끝면: 먹선 위치와 비교하므로 끝면의 아래 모서리(바닥과 만나는 선)로 잰다.
- ②·③·④ 허용오차의 길이 ℓ: 실제로 잰 길이(② ④ = 벽 길이, ③ = 벽 높이).
  기준 길이(②③ 3 m, ④ 9 m)보다 짧으면 원천값 6 mm를 그대로 적용한다(문언 적용, 2026-10-27 팀 결정).
  구 C안(비례 축소 + 하한)은 mode="scaled"로 다시 채점할 수 있다.
- ② 기준 길이 3 m 초과의 '임의의 3 m 구간' 창 검사는 이 벽(0.99 m)에 필요 없어 전체 길이 꺾은선으로만 구현했다.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from wall3.volume import box, intersection

TOL_MODE = "literal"       # 2026-10-27: 기준 길이 미만도 6 mm 그대로. "scaled" = 구 C안(비례 축소 + 하한)
FLOOR_TOL = 0.002          # m, 구 C안의 하한 (mode="scaled"에서만 씀)
BASE_PENETRATION = 0.0005  # m, 1단 아랫면–바닥 거리의 접촉 침투 허용 (구현 해석)
EXIST_XY, EXIST_Z, EXIST_TILT = 0.045, 0.0285, 45.0
SAMPLE_DX = 0.010


# ---------------------------------------------------------------- tolerances (2.1절 일반식, m)
def _interp(pts, l):
    xs, ys = zip(*pts)
    return float(np.interp(l, xs, ys))


def _tol(pts, l, floor, mode):
    if mode == "literal":                  # 기준 길이 미만 구간은 첫 원천값(6 mm) 그대로
        return _interp([(0, pts[1][1])] + pts[1:], l)
    if mode == "scaled":                   # 구 C안: 0에서 비례, 하한 floor
        return max(floor, _interp(pts, l))
    raise ValueError(mode)


def tol_line(l, floor=FLOOR_TOL, mode=TOL_MODE):          # ② 선 및 면
    return _tol([(0, 0), (3, 0.006), (12, 0.013), (1e9, 0.013)], l, floor, mode)


def tol_plumb(l, floor=FLOOR_TOL, mode=TOL_MODE):         # ③ 수직도
    return _tol([(0, 0), (3, 0.006), (6, 0.010), (12, 0.013), (1e9, 0.013)], l, floor, mode)


def tol_position(l, floor=FLOOR_TOL, mode=TOL_MODE):      # ④ 평면상 위치
    return _tol([(0, 0), (9, 0.006), (18, 0.013), (1e9, 0.013)], l, floor, mode)


BED_TOL = (0.007, 0.013)                   # ① 1–2단·2–3단
BASE_TOL = (0.0, 0.003)                    # ① 1단 아랫면–바닥
THICK_TOL = (-0.006, 0.013)                # ⑤


# ---------------------------------------------------------------- brick geometry
def _rotmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class Brick:
    """Box with its faces labelled in the wall frame: top/bottom (normal closest to ±z), front/back (closest to the front
    direction / away from it), left/right end (the remaining axis, closest to −x / +x)."""

    def __init__(self, pos, quat, dims, front_dir):
        self.p = np.asarray(pos, float)
        self.R = _rotmat(np.asarray(quat, float))
        self.h = np.asarray(dims, float) / 2
        Rz = self.R[2]                                    # z-components of the local axes
        self.a_top = int(np.argmax(np.abs(Rz)))
        self.s_top = float(np.sign(Rz[self.a_top]) or 1.0)
        side = [a for a in range(3) if a != self.a_top]
        fy = np.asarray(front_dir, float) @ self.R          # front-direction component of each local axis
        self.a_front = side[int(np.argmax(np.abs(fy[side])))]
        self.s_front = float(np.sign(fy[self.a_front]) or 1.0)
        self.a_end = [a for a in side if a != self.a_front][0]
        self.s_left = float(-np.sign(self.R[0, self.a_end]) or 1.0)     # local sign of the −x (left) end
        self.tilt_deg = float(np.degrees(np.arccos(np.clip(abs(Rz[self.a_top]), -1, 1))))

    def corner(self, front, top, left):
        v = np.zeros(3)
        v[self.a_front] = self.h[self.a_front] * (self.s_front if front else -self.s_front)
        v[self.a_top] = self.h[self.a_top] * (self.s_top if top else -self.s_top)
        v[self.a_end] = self.h[self.a_end] * (self.s_left if left else -self.s_left)
        return self.p + self.R @ v

    def corners(self):
        return np.array([self.corner(f, t, l) for f in (True, False) for t in (True, False) for l in (True, False)])

    def top_plane_z(self, x, y):
        n = self.R[:, self.a_top] * self.s_top
        c = self.p + n * self.h[self.a_top]
        return c[2] - (n[0] * (x - c[0]) + n[1] * (y - c[1])) / n[2]

    def footprint_dist(self, x, y):
        """Horizontal distance from (x, y) to the brick's top-face footprint (0 inside)."""
        loc = self.R.T @ (np.array([x, y, 0.0]) - np.r_[self.p[:2], 0.0])
        out = 0.0
        for a in range(3):
            if a == self.a_top:
                continue
            out = max(out, abs(loc[a]) - self.h[a])
        return max(out, 0.0)


def _edge_at_x(a, b, x):
    """Point on segment a-b at world x (None if x is outside the segment's x-range)."""
    x0, x1 = a[0], b[0]
    if abs(x1 - x0) < 1e-12 or not (min(x0, x1) - 1e-9 <= x <= max(x0, x1) + 1e-9):
        return None
    t = (x - x0) / (x1 - x0)
    return a + t * (b - a)


def _edge_at_y(a, b, y):
    y0, y1 = a[1], b[1]
    if abs(y1 - y0) < 1e-12:
        return None
    t = (y - y0) / (y1 - y0)
    return a + t * (b - a)


# ---------------------------------------------------------------- evaluation
def evaluate(poses, lengths, design, width, height, front_y, front_dir=(0.0, 1.0, 0.0), floor_z=0.0, floor_tol=FLOOR_TOL, mode=TOL_MODE):
    """poses: (N, 7) world pos + quat (w, x, y, z); lengths: (N,) brick length along its long axis; design: slots
    (dict x, y, z, yaw, L, course, index); front_y: world y of the design front face (y_f). Returns a dict (m → mm)."""
    poses = np.asarray(poses, float)
    n_slots = len(design)
    front_dir = np.asarray(front_dir, float)
    yp = lambda y: (front_y - y) * float(np.sign(front_dir[1]))      # y' (depth behind the design front face)
    bricks = [Brick(poses[i, :3], poses[i, 3:7], (lengths[i], width, height), front_dir) for i in range(len(poses))]

    # condition 1: one-to-one matching per shape, minimum total centre distance; then the loose presence check
    match = [-1] * n_slots
    for L in sorted({round(s["L"], 6) for s in design}):
        S = [k for k, s in enumerate(design) if round(s["L"], 6) == L]
        B = [i for i in range(len(poses)) if round(float(lengths[i]), 6) == L]
        if not S or not B:
            continue
        C = np.array([[np.linalg.norm(poses[i, :3] - [design[k]["x"], design[k]["y"], design[k]["z"]]) for i in B] for k in S])
        r, c = linear_sum_assignment(C)
        for a, b in zip(r, c):
            match[S[a]] = B[b]
    filled = []
    for k, s in enumerate(design):
        i = match[k]
        if i < 0:
            filled.append(False); continue
        p = poses[i, :3]
        filled.append(bool(np.hypot(p[0] - s["x"], p[1] - s["y"]) <= EXIST_XY and abs(p[2] - s["z"]) <= EXIST_Z
                           and bricks[i].tilt_deg < EXIST_TILT))
    exists = all(filled)
    present = [k for k in range(n_slots) if filled[k]]
    course_of = {k: design[k]["course"] for k in present}
    by_course = {c: [k for k in present if course_of[k] == c] for c in sorted({s["course"] for s in design})}
    B = {k: bricks[match[k]] for k in present}

    res = {"exists": exists, "slots_filled": filled, "match": [int(i) for i in match]}

    # ---- ① bed joints
    up_t, base_t, worst1 = [], [], (0.0, None)
    for k in present:
        c = course_of[k]
        for front in (True, False):
            for left in (True, False):
                pt = B[k].corner(front, False, left)
                if c == 1:
                    t = pt[2] - floor_z
                    base_t.append((t, k))
                else:
                    lows = by_course.get(c - 1, [])
                    if not lows:
                        up_t.append((np.nan, k)); continue
                    j = min(lows, key=lambda j: B[j].footprint_dist(pt[0], pt[1]))
                    up_t.append((pt[2] - B[j].top_plane_z(pt[0], pt[1]), k))
    u = np.array([t for t, _ in up_t]) if up_t else np.array([np.nan])
    b = np.array([t for t, _ in base_t]) if base_t else np.array([np.nan])
    b_eff = np.where((b < 0) & (b >= -BASE_PENETRATION), 0.0, b)
    ok1 = bool(np.all((u >= BED_TOL[0] - 1e-12) & (u <= BED_TOL[1] + 1e-12)) and np.all((b_eff >= BASE_TOL[0]) & (b_eff <= BASE_TOL[1] + 1e-12)))
    bad1 = [k for (t, k) in up_t if not (BED_TOL[0] - 1e-12 <= t <= BED_TOL[1] + 1e-12)] + \
           [k for (t, k), te in zip(base_t, b_eff) if not (BASE_TOL[0] <= te <= BASE_TOL[1] + 1e-12)]
    res["1_bed_joint"] = {"upper_min_mm": float(np.nanmin(u) * 1e3) if up_t else None, "upper_max_mm": float(np.nanmax(u) * 1e3) if up_t else None,
                          "base_min_mm": float(np.nanmin(b_eff) * 1e3), "base_max_mm": float(np.nanmax(b_eff) * 1e3),
                          "base_min_raw_mm": float(np.nanmin(b) * 1e3),
                          "tol_mm": {"upper": [7, 13], "base": [0, 3]}, "pass": ok1, "slots": sorted(set(bad1))}

    # ---- ② line and plane
    allc = np.array([B[k].corners() for k in present]).reshape(-1, 3) if present else np.zeros((0, 3))
    wall_len = float(allc[:, 0].max() - allc[:, 0].min()) if len(allc) else 0.0
    wall_h = float(allc[:, 2].max() - floor_z) if len(allc) else 0.0
    t2 = tol_line(wall_len, floor_tol, mode)
    level = {}
    for c, ks in by_course.items():
        z = np.array([B[k].corner(f, True, l)[2] for k in ks for f in (True, False) for l in (True, False)])
        level[c] = float((z.max() - z.min()) / 2) if len(z) else 0.0
    fp = np.array([B[k].corner(True, t, l) for k in present for t in (True, False) for l in (True, False)])
    fslot = [k for k in present for _ in range(4)]
    if len(fp) >= 3:
        P = np.c_[fp[:, 0], [yp(y) for y in fp[:, 1]], fp[:, 2]]
        ctr = P.mean(0)
        nrm = np.linalg.svd(P - ctr)[2][-1]
        r = (P - ctr) @ nrm
        plane = float((r.max() - r.min()) / 2)
        far = fslot[int(np.argmax(np.abs(r - (r.max() + r.min()) / 2)))]
    else:
        plane, far = 0.0, None
    v2 = max([plane] + list(level.values()))
    lvl_course = max(level, key=level.get) if level else None
    res["2_line_plane"] = {"value_mm": v2 * 1e3, "level_mm": {str(c): v * 1e3 for c, v in level.items()}, "plane_mm": plane * 1e3,
                           "tol_mm": t2 * 1e3, "length_m": wall_len, "pass": bool(v2 <= t2 + 1e-12),
                           "worst": {"plane_slot": far, "level_course": lvl_course}}

    # sample positions along the wall (③ ⑤): 10 mm spacing + every front/back corner x
    xs = []
    if len(allc):
        xs = list(np.arange(allc[:, 0].min(), allc[:, 0].max() + 1e-9, SAMPLE_DX)) + list(allc[:, 0])
    xs = np.unique(np.round(xs, 9))

    def face_y(k, front, top, x):
        q = _edge_at_x(B[k].corner(front, top, True), B[k].corner(front, top, False), x)
        return None if q is None else yp(q[1])

    # ---- ③ plumb (front face vs course-1 bottom front edge at the same x; end faces vs course-1 end bottom edge)
    t3 = tol_plumb(wall_h, floor_tol, mode)
    d3, worst3 = 0.0, None
    c1 = by_course.get(1, [])
    for x in xs:
        refs = [face_y(k, True, False, x) for k in c1]
        refs = [r for r in refs if r is not None]
        if not refs:
            continue
        ref = refs[0]
        for k in present:
            for top in (False, True):
                v = face_y(k, True, top, x)
                if v is not None and abs(v - ref) > d3:
                    d3, worst3 = abs(v - ref), k
    ends3 = 0.0
    for side_left in (True, False):
        # the design end slot of each course (only if it is filled)
        ends = {}
        for c in sorted({s["course"] for s in design}):
            ks = [k for k in range(n_slots) if design[k]["course"] == c]
            ke = (min if side_left else max)(ks, key=lambda k: design[k]["x"])
            if ke in B:
                ends[c] = ke
        if 1 not in ends:
            continue
        r0 = B[ends[1]]
        ra, rb = r0.corner(True, False, side_left), r0.corner(False, False, side_left)
        for c, k in ends.items():
            for top in (False, True):
                a_, b_ = B[k].corner(True, top, side_left), B[k].corner(False, top, side_left)
                for y in np.linspace(min(a_[1], b_[1]), max(a_[1], b_[1]), 10):
                    q = _edge_at_y(a_, b_, y); qr = _edge_at_y(ra, rb, y)
                    if q is None or qr is None:
                        continue
                    if abs(q[0] - qr[0]) > ends3:
                        ends3 = abs(q[0] - qr[0])
                        if ends3 > d3:
                            worst3 = k
    v3 = max(d3, ends3)
    res["3_plumb"] = {"value_mm": v3 * 1e3, "face_mm": d3 * 1e3, "ends_mm": ends3 * 1e3, "tol_mm": t3 * 1e3, "height_m": wall_h,
                      "pass": bool(v3 <= t3 + 1e-12), "worst_slot": worst3}

    # ---- ④ plan position (course-1 front bottom edge vs y_f; course-1 end faces' bottom edge vs design end x)
    t4 = tol_position(wall_len, floor_tol, mode)
    front4, w4f = 0.0, None
    for k in c1:
        for l in (True, False):
            v = abs(yp(B[k].corner(True, False, l)[1]))
            if v > front4:
                front4, w4f = v, k
    ends4, w4e = 0.0, None
    d_c1 = [k for k in range(n_slots) if design[k]["course"] == 1]
    if d_c1:
        kl, kr = min(d_c1, key=lambda k: design[k]["x"]), max(d_c1, key=lambda k: design[k]["x"])
        xl, xr = design[kl]["x"] - design[kl]["L"] / 2, design[kr]["x"] + design[kr]["L"] / 2
        for kk, side_left, xd in ((kl, True, xl), (kr, False, xr)):
            if kk not in B:
                continue
            for f in (True, False):
                v = abs(B[kk].corner(f, False, side_left)[0] - xd)
                if v > ends4:
                    ends4, w4e = v, kk
    v4 = max(front4, ends4)
    res["4_position"] = {"value_mm": v4 * 1e3, "front_mm": front4 * 1e3, "ends_mm": ends4 * 1e3, "tol_mm": t4 * 1e3,
                         "pass": bool(v4 <= t4 + 1e-12), "worst_slot": w4f if front4 >= ends4 else w4e}

    # ---- ⑤ wall thickness at every sample x: (backmost back face) − (frontmost front face) − width
    th = []
    for x in xs:
        fr = [face_y(k, True, t, x) for k in present for t in (False, True)]
        bk = [face_y(k, False, t, x) for k in present for t in (False, True)]
        fr = [v for v in fr if v is not None]; bk = [v for v in bk if v is not None]
        if fr and bk:
            th.append((max(bk) - min(fr) - width, x))
    tv = np.array([t for t, _ in th]) if th else np.array([np.nan])
    ok5 = bool(np.all((tv >= THICK_TOL[0] - 1e-12) & (tv <= THICK_TOL[1] + 1e-12)))
    res["5_thickness"] = {"min_mm": float(np.nanmin(tv) * 1e3), "max_mm": float(np.nanmax(tv) * 1e3), "tol_mm": [-6, 13], "pass": ok5,
                          "worst_x_m": float(th[int(np.nanargmax(np.abs(tv)))][1]) if th else None}

    items_ok = all(res[k]["pass"] for k in ("1_bed_joint", "2_line_plane", "3_plumb", "4_position", "5_thickness"))
    res["items_pass"] = items_ok
    res["spec_pass"] = bool(exists and items_ok)

    # ---- per-slot 3D IoU (3절)
    iou = []
    for k, s in enumerate(design):
        if not filled[k]:
            iou.append(0.0); continue
        i = match[k]
        hs = np.array([lengths[i], width, height]) / 2
        a = box(poses[i, :3], _rotmat(poses[i, 3:7]), hs)
        yaw = s.get("yaw", 0.0)
        Rt = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        t = box(np.array([s["x"], s["y"], s["z"]]), Rt, np.array([s["L"], width, height]) / 2)
        inter = intersection([a, t])
        iou.append(float(np.clip(inter / (a[4] + t[4] - inter), 0, 1)))
    kmin = int(np.argmin(iou))
    res["iou"] = iou
    res["iou_mean"] = float(np.mean(iou))
    res["iou_min"] = float(iou[kmin])
    res["iou_min_slot"] = {"slot": kmin, "course": design[kmin]["course"], "index": design[kmin]["index"]}
    return res


def evaluate_env(env, floor_tol=FLOOR_TOL, mode=TOL_MODE):
    """Evaluate a wall3 WallEnv at its current state (bricks i ↔ design i only through the matching)."""
    from wall3.config import W, H, WALL_Y
    lengths = [s["L"] for s in env.design]
    return evaluate(env.brick_poses(), lengths, env.design, W, H, front_y=WALL_Y + W / 2, floor_tol=floor_tol, mode=mode)


def sensitivity(env_or_args, floors=(0.001, 0.002, 0.003)):
    """Spec pass re-scored under the old C안 (scaled, floor 1·2·3 mm) next to the literal rule (2.5절 재채점)."""
    out = {f"{int(round(f * 1e3))}mm": evaluate_env(env_or_args, floor_tol=f, mode="scaled")["spec_pass"] for f in floors}
    out["literal"] = evaluate_env(env_or_args, mode="literal")["spec_pass"]
    return out

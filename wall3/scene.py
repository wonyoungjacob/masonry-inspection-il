"""3층 벽체 장면: UR10e(받침대 0.6 m) + Robotiq 2F-140(A1+) + 콘크리트 표준벽돌 16장 + 공급대 + 모르타르(mortarB 일반화).

- 로봇·집게 모델 파일은 고치지 않는다. UR10e는 Menagerie 파일을 MjSpec으로 읽어 받침대 위로 옮기고, 집게는 wall3.gripper로 붙인다.
- 모르타르는 brick_sim(mortarB) 모델 그대로: 아래 단 벽돌이 모두 제자리에 놓여 2 s 정지하면 그 실제 윗면 위에 두께 10 mm 순응층
  (벽돌마다 한 장 + 세로줄눈 위 다리 한 장, mocap)을 깔고, 벽 위치에 놓인 벽돌은 2 s 정지 뒤 고정(weld)한다. 1단 아래는 없다.
  bed_rule "placed"(main_v1_eval_bed, 2026-10-30): '제자리' 대신 '공급대를 떠남'이면 깐다(삐뚤어도 그 실제 윗면 위에).
  침하량은 두께에 비례해 0.5 mm × 10 / 4.737 = 1.06 mm, 감쇠비 1, 계면 마찰 0.75.
- 카메라: front(벽 앞면 쪽에서 고정), top(벽 위 고정), wrist(손목), 정책용 128×128. 검측용 top 고해상도 RGB-D(1280×960).
  overview(녹화용 자유 시점 대신 쓰는 고정 시점, 정책 입력 아님).
"""
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

from wall3 import gripper as G
from wall3.config import SceneConfig, design_slots, L_FULL, L_HALF, W, H, RHO, MU_BRICK, JOINT, WALL_Y

SIM = Path(__file__).resolve().parents[1]
UR10E_XML = SIM / "menagerie/universal_robots_ur10e/ur10e.xml"
ARM_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
ARM_ACTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
GRIP_ACTS = [G.PREFIX + "finger_1", G.PREFIX + "finger_2"]
HOME_Q = np.array([np.pi / 2, -1.9, 1.9, -1.571, -1.571, 0.0])      # arm folded above the base, facing the wall


def _quat_from_R(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


def _look_at(pos, target, up=(0, 0, 1)):
    f = np.asarray(target, float) - np.asarray(pos, float); f /= np.linalg.norm(f)
    x = np.cross(f, up); x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return _quat_from_R(np.stack([x, y, -f], axis=1))


def _quat_rotmat(q):
    R = np.zeros(9)
    q = np.asarray(q, float)
    mujoco.mju_quat2Mat(R, q / np.linalg.norm(q))
    return R.reshape(3, 3)


MORTAR_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2.0)
CONTACTS_PER_SLAB = 4                      # MuJoCo box-box: 4 corner contacts per bed slab under a resting brick (checked in 나3)


def impedance(r, solimp=MORTAR_SOLIMP):
    """MuJoCo's constraint impedance d at penetration r (engine getimpedance)."""
    dmin, dmax, width, mid, power = solimp
    x = min(abs(r) / width, 1.0)
    if x >= 1.0:
        return dmax
    y = x ** power / mid ** (power - 1) if x <= mid else 1 - (1 - x) ** power / (1 - mid) ** (power - 1)
    return dmin + y * (dmax - dmin)


def mortar_solref(c, mass_ratio=1.0, n_contacts=1):
    """Direct-form (stiffness, damping) of the bed contact pairs of one brick, mass-normalised.

    brick_sim (pilot): k = g / sag, which ignores the constraint impedance: the real sag was ~0.005 mm.
    impedance (main_v1, 나3): a brick resting on n contacts of impedance d sinks r = (1 − d)(dmax/d)² g / (n k)
    (each contact: R = (1 − d)/d · 1/m, reference acceleration −K d r with K = k/dmax²; the n rows share the weight).
    So k = (1 − d)(dmax/d)² g / (n s) with d = d(s) gives the target sag s. One physical stiffness for all bricks:
    s = mortar_sag · m / m_full, so a half brick sinks in proportion to its weight. `mass_ratio` = m / m_full,
    `n_contacts` = contacts that carry the brick (CONTACTS_PER_SLAB × slabs under it). b = 2 ζ √(k / dmax).
    """
    if c.mortar_stiffness == "brick_sim":
        k = 9.81 / c.mortar_sag
        return k, 2.0 * c.mortar_zeta * np.sqrt(k)
    if c.mortar_stiffness == "impedance":
        s = c.mortar_sag * mass_ratio
        d, dmax = impedance(s), MORTAR_SOLIMP[1]
        k = (1 - d) * (dmax / d) ** 2 * 9.81 / (n_contacts * s)
        return k, 2.0 * c.mortar_zeta * np.sqrt(k / dmax)
    raise ValueError(f"unknown mortar_stiffness {c.mortar_stiffness!r}")


def yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


class WallEnv:
    def __init__(self, cfg: SceneConfig | None = None):
        self.cfg = cfg or SceneConfig()
        if len(self.cfg.supply_slots) != 16:
            raise ValueError("SceneConfig.supply_slots needs 16 (x, y) slot centres")
        self.design = design_slots()
        self.n = len(self.design)
        self.cal = G.calibrate(self.cfg.gripper, W)
        self.spec = self._build_spec()
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self._index()
        self.renderer = {}
        self.rng = np.random.default_rng(0)
        self.vopt = mujoco.MjvOption()          # robosuite's visual helper sites (group 1) and our TCP site (group 4) hidden
        self.vopt.sitegroup[:] = 0

    # ------------------------------------------------------------------ model
    def _build_spec(self):
        c = self.cfg
        s = mujoco.MjSpec.from_file(str(UR10E_XML))
        s.option.timestep = c.timestep
        s.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        s.option.impratio = c.impratio
        s.option.noslip_iterations = c.noslip_iterations
        s.visual.global_.offwidth = c.hi_w
        s.visual.global_.offheight = c.hi_h
        s.visual.quality.shadowsize = 2048
        # viewer camera marker size = meansize (0.50 m here, set by the floor/table) × scale.camera. The default 0.3 drew a
        # 15 cm box that looked like the wrist camera overlapping the gripper; the camera itself is a point 47.5 mm clear of
        # the nearest gripper geometry and has no body. Visual only: images, physics and data are unchanged.
        s.visual.scale.camera = 0.03
        s.visual.headlight.ambient = [0.25, 0.25, 0.25]
        s.visual.headlight.diffuse = [0.35, 0.35, 0.35]
        s.visual.headlight.specular = [0.05, 0.05, 0.05]
        s.light("spotlight").diffuse = [0.3, 0.3, 0.3]          # UR10e file's spotlight, dimmed (visual only)
        tex = s.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                            rgb1=[0.60, 0.60, 0.58], rgb2=[0.52, 0.52, 0.50], width=300, height=300,
                            mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=[0.45, 0.45, 0.43])
        mat = s.add_material(name="grid", texrepeat=[10, 10], texuniform=True, reflectance=0.02)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
        s.body("base").pos = [0.0, 0.0, c.pedestal_h]
        if c.integrator == "euler":
            s.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
        elif c.integrator == "rk4":
            s.option.integrator = mujoco.mjtIntegrator.mjINT_RK4
        elif c.integrator == "implicit":
            s.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT
        elif c.integrator != "implicitfast":
            raise ValueError(f"unknown integrator {c.integrator!r}")
        if c.arm_joint_damping:                    # passive joint damping override (Euler integrates it implicitly)
            for a, dj in zip(ARM_ACTS, c.arm_joint_damping):
                j = s.joint(s.actuator(a).target)
                j.damping = [float(dj)] + list(j.damping[1:])
        if c.arm_kd:                               # servo damping override per joint (actuator velocity bias)
            for a, kd in zip(ARM_ACTS, c.arm_kd):
                s.actuator(a).biasprm[2] = -float(kd)
        if c.arm_damping == "joint":              # move the servo damping into the joint (same torque −kd·q̇)
            for a in ARM_ACTS:
                act = s.actuator(a)
                kd = -float(act.biasprm[2])
                act.biasprm[2] = 0.0
                j = s.joint(act.target)
                j.damping = [float(j.damping[0]) + kd] + list(j.damping[1:])
        elif c.arm_damping != "actuator":
            raise ValueError(f"unknown arm_damping {c.arm_damping!r}")
        wb = s.worldbody
        wb.add_light(name="sun", pos=[0.4, 0.2, 2.5], dir=[-0.15, 0.1, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                     castshadow=1, diffuse=[0.45, 0.45, 0.45], specular=[0.1, 0.1, 0.1])
        wb.add_light(name="fill", pos=[-0.6, 1.5, 1.8], dir=[0.3, -0.6, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                     castshadow=0, diffuse=[0.2, 0.2, 0.2], specular=[0, 0, 0])
        fr = [MU_BRICK, 0.005, 0.0001]
        wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.05], friction=fr, material="grid")
        ph = c.pedestal_half
        wb.add_geom(name="pedestal", type=mujoco.mjtGeom.mjGEOM_BOX, size=[ph, ph, c.pedestal_h / 2],
                    pos=[0, 0, c.pedestal_h / 2], rgba=[0.25, 0.25, 0.27, 1], friction=fr)
        tx, ty = c.table_x, c.table_y   # table clear of the pedestal
        wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[(tx[1] - tx[0]) / 2, (ty[1] - ty[0]) / 2, c.table_h / 2],
                    pos=[(tx[0] + tx[1]) / 2, (ty[0] + ty[1]) / 2, c.table_h / 2], rgba=[0.55, 0.42, 0.30, 1], friction=fr)
        # chalk line (먹선) on the floor along the design front face, and the wall end ticks (visual only)
        yf = WALL_Y + W / 2
        vis = dict(contype=0, conaffinity=0, group=1)
        wb.add_geom(name="chalk", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.56, 0.0008, 0.0002], pos=[0, yf, 0.0002],
                    rgba=[0.06, 0.06, 0.08, 1], **vis)
        # cameras
        wb.add_camera(name="front", pos=[0.0, 1.75, 0.75], quat=_look_at([0.0, 1.75, 0.75], [0.0, 0.55, 0.10]), fovy=50)
        wb.add_camera(name="top", pos=[0.0, 0.60, 1.85], quat=_look_at([0.0, 0.60, 1.85], [0.0, 0.60, 0.0], up=(0, 1, 0)), fovy=45)
        wb.add_camera(name="overview", pos=[2.1, -1.5, 1.9], quat=_look_at([2.1, -1.5, 1.9], [0.0, 0.0, 0.25]), fovy=50)
        # bricks
        for i, d in enumerate(self.design):
            b = wb.add_body(name=f"brick_{i}", pos=[0, 0, 0])
            b.add_freejoint(name=f"brick_{i}_free")
            # red brick colour common on Korean sites (적벽돌), two close shades so neighbours stay distinguishable.
            # Colour only (2026-10-16): density 2,000 kg/m3, friction 0.6 and size unchanged; mortar colour unchanged.
            col = list(c.brick_rgba[i % 2])
            b.add_geom(name=f"brick_{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[d["L"] / 2, W / 2, H / 2],
                       mass=d["L"] * W * H * RHO, friction=fr, rgba=col, condim=3)
        # mortar beds: boundary 1 (on course 1) and 2 (on course 2); one slab per lower brick + one per head joint
        m_full = L_FULL * W * H * RHO
        tb = JOINT
        self.beds = {}
        for bnd in (1, 2):
            lower = [i for i, d in enumerate(self.design) if d["course"] == bnd]
            upper = [i for i, d in enumerate(self.design) if d["course"] > bnd]
            names = []
            for j, i in enumerate(lower):
                nm = f"bed{bnd}_{j}"
                bb = wb.add_body(name=nm, mocap=True, pos=[0, 0, -1])
                bb.add_geom(name=nm, type=mujoco.mjtGeom.mjGEOM_BOX,
                            size=[self.design[i]["L"] / 2 + c.bed_overhang, W / 2 + c.bed_overhang, tb / 2],
                            contype=0, conaffinity=0, mass=0, rgba=[0.50, 0.48, 0.44, 1])
                names.append(nm)
            for j in range(len(lower) - 1):
                nm = f"bedj{bnd}_{j}"
                bb = wb.add_body(name=nm, mocap=True, pos=[0, 0, -1])
                bb.add_geom(name=nm, type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.006, W / 2 + c.bed_overhang, tb / 2],
                            contype=0, conaffinity=0, mass=0, rgba=[0.50, 0.48, 0.44, 1])
                names.append(nm)
            for nm in names:
                for i in upper:
                    k, bdamp = mortar_solref(c, self.design[i]["L"] * W * H * RHO / m_full,
                                             CONTACTS_PER_SLAB * self._slabs_under(i))
                    p = s.add_pair(geomname1=nm, geomname2=f"brick_{i}")
                    p.condim = 4
                    p.friction = [c.mortar_friction, c.mortar_friction, 0.005, 0.0001, 0.0001]
                    p.solref = [-k, -bdamp]
                    p.solimp = list(MORTAR_SOLIMP)
            self.beds[bnd] = dict(lower=lower, slabs=names[:len(lower)], joints=names[len(lower):])
        # cure welds (mortarB): brick -> world, activated at run time
        for i in range(self.n):
            e = s.add_equality(name=f"cure_{i}", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                               name1=f"brick_{i}", active=False)
            e.solref = [0.004, 1]
        # gripper on the flange, TCP site at the pad centre (90 mm closure), wrist camera
        att = s.site("attachment_site")
        G.attach(att, s, c.gripper)
        # grip lock welds (나2 결정): brick ↔ gripper base, off until the lock conditions hold (like the cure welds)
        for i in range(self.n):
            e = s.add_equality(name=f"grip_{i}", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                               name1=f"brick_{i}", name2=G.PREFIX + "right_gripper", active=False)
            e.solref = list(c.lock_solref)
        w3 = s.body("wrist_3_link")
        qa = np.asarray(att.quat, float)
        Ra = _quat_rotmat(qa / np.linalg.norm(qa))          # the UR10e file writes this quat unnormalised (-1 1 0 0)
        pa = np.asarray(att.pos, float)
        w3.add_site(name="tcp", pos=pa + Ra @ [0, 0, self.cal["pad_center_depth"]], quat=att.quat, size=[0.004, 0, 0],
                    rgba=[0, 0, 1, 0.5], group=4)
        cam_p = pa + Ra @ [0.085, 0.0, 0.055]
        tgt = pa + Ra @ [0.0, 0.0, self.cal["pad_center_depth"] + 0.03]
        f = tgt - cam_p; f /= np.linalg.norm(f)
        x = Ra @ [0, 1, 0]
        x = x - f * (x @ f); x /= np.linalg.norm(x)
        y = np.cross(x, f)
        w3.add_camera(name="wrist", pos=cam_p, quat=_quat_from_R(np.stack([x, y, -f], axis=1)), fovy=80)
        return s

    def _index(self):
        m = self.model
        self.arm_qadr = np.array([m.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.arm_dofadr = np.array([m.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.arm_act = np.array([m.actuator(a).id for a in ARM_ACTS])
        self.grip_act = np.array([m.actuator(a).id for a in GRIP_ACTS])
        self.brick_qadr = [m.joint(f"brick_{i}_free").qposadr[0] for i in range(self.n)]
        self.brick_dofadr = [m.joint(f"brick_{i}_free").dofadr[0] for i in range(self.n)]
        self.brick_gid = [m.geom(f"brick_{i}").id for i in range(self.n)]
        self.brick_bid = [m.body(f"brick_{i}").id for i in range(self.n)]
        self.tcp_sid = m.site("tcp").id
        base = m.body("base").id
        self.robot_bodies = {b for b in range(m.nbody) if self._is_desc(b, base)}
        self.gripper_bodies = {b for b in self.robot_bodies if m.body(b).name.startswith(G.PREFIX)}
        self.pad_gids = {m.geom(G.PREFIX + p).id for p in G.PADS} | {
            m.geom(G.PREFIX + n).id for n in ("left_fingertip_collision", "right_fingertip_collision")}
        self.bed_mocap = {nm: m.body_mocapid[m.body(nm).id] for bnd in self.beds.values() for nm in bnd["slabs"] + bnd["joints"]}
        self.bed_gid = {nm: m.geom(nm).id for nm in self.bed_mocap}
        self.n_sub = int(round(self.cfg.ctrl_dt / m.opt.timestep))
        self.arm_kp = m.actuator_gainprm[self.arm_act, 0].copy()
        self.arm_kd = -m.actuator_biasprm[self.arm_act, 2].copy()      # 0 when the damping lives in the joints
        self.ft_force = m.sensor(G.PREFIX + "force_ee").adr[0]
        self.ft_torque = m.sensor(G.PREFIX + "torque_ee").adr[0]
        self.ft_site = m.site(G.PREFIX + "ft_frame").id
        # grip lock lookup tables
        self.grip_body = m.body(G.PREFIX + "right_gripper").id
        self.grip_eq = [m.equality(f"grip_{i}").id for i in range(self.n)]
        self.fj_qadr = m.joint(G.PREFIX + "finger_joint").qposadr[0]
        left = {G.PREFIX + n for n in ("left_outer_knuckle", "left_inner_finger", "left_inner_knuckle")}
        right = {G.PREFIX + n for n in ("right_outer_knuckle", "right_inner_finger", "right_inner_knuckle")}
        self.geom_side = np.full(m.ngeom, -1, int)          # 0 left finger, 1 right finger, 2 other gripper, -1 not gripper
        for g in range(m.ngeom):
            bn = m.body(m.geom_bodyid[g]).name
            if bn in left:
                self.geom_side[g] = 0
            elif bn in right:
                self.geom_side[g] = 1
            elif m.geom_bodyid[g] in self.gripper_bodies:
                self.geom_side[g] = 2
        self.geom_brick = np.full(m.ngeom, -1, int)
        for i, gid in enumerate(self.brick_gid):
            self.geom_brick[gid] = i
        c = self.cfg
        g = 9.81
        self.lock_nmin = np.array([m.body_mass[b] * np.sqrt(g * g + c.lock_amax ** 2) / (2 * c.lock_mu) for b in self.brick_bid])

    def _slabs_under(self, i):
        """Bed geoms (lower-brick slabs + head joints) under slot i in the design."""
        s = self.design[i]
        lower = [self.design[j] for j in range(self.n) if self.design[j]["course"] == s["course"] - 1]
        lower.sort(key=lambda d: d["x"])
        n = sum(abs(lo["x"] - s["x"]) < (lo["L"] + s["L"]) / 2 for lo in lower)
        for a, b in zip(lower[:-1], lower[1:]):
            xj = ((a["x"] + a["L"] / 2) + (b["x"] - b["L"] / 2)) / 2
            n += abs(xj - s["x"]) < s["L"] / 2
        return int(n)

    def _is_desc(self, b, root):
        m = self.model
        while b != 0:
            if b == root:
                return True
            b = m.body_parentid[b]
        return False

    # ------------------------------------------------------------------ episode
    def supply_pose(self, i, rng=None):
        """Nominal (rng None) or scattered supply pose of brick i: (pos, yaw)."""
        c = self.cfg
        x, y = c.supply_slots[i]
        yaw = 0.0
        if rng is not None:
            x += rng.uniform(-c.scatter_pos, c.scatter_pos)
            y += rng.uniform(-c.scatter_pos, c.scatter_pos)
            yaw = np.deg2rad(rng.uniform(-c.scatter_yaw_deg, c.scatter_yaw_deg))
        return np.array([x, y, c.table_h + H / 2]), yaw

    def reset(self, seed: int, arm_q=None):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.supply = []
        for i in range(self.n):
            p, yaw = self.supply_pose(i, self.rng)
            self.supply.append((p.copy(), yaw))
            a = self.brick_qadr[i]
            d.qpos[a:a + 3] = p
            d.qpos[a + 3:a + 7] = yaw_quat(yaw)
        q = HOME_Q if arm_q is None else np.asarray(arm_q, float)
        d.qpos[self.arm_qadr] = q
        d.ctrl[self.arm_act] = q
        d.ctrl[self.grip_act] = G.grip_ctrl(False, self.cfg.gripper)
        self.last_action = np.r_[q, self.cfg.gripper.open_ctrl]
        self.prev_action = self.last_action.copy()
        self.lp_y, self.lp_v = self.last_action.copy(), np.zeros_like(self.last_action)
        for nm, mid in self.bed_mocap.items():
            d.mocap_pos[mid] = [0, 0, -1]
        # mj_resetData restores eq_active0: cure welds off, gripper coupling equalities on
        self.cured = np.zeros(self.n, bool)
        self.cure_t = np.full(self.n, -1)
        self.sag_at_cure = np.full(self.n, np.nan)        # m, bed penetration when the brick cured (나3 check; nan = course 1)
        self.settle_count = np.zeros(self.n, int)
        self.bed_count = {1: np.zeros(self.n, int), 2: np.zeros(self.n, int)}
        self.bed_laid = {1: -1, 2: -1}
        self.bed_blocked = {1: 0, 2: 0}       # bed_rule "placed": steps the bed was ready but would sink into an upper brick
        self.lock_b = -1                      # brick currently welded to the gripper (-1 = none)
        self.lock_events = []                 # lock / release records (info only, never a policy or G1 input)
        self._close_t0, self._cmd_max, self._cmd_lock = None, 0.0, 0.0
        self._cand, self._settle, self._rel_prev = -1, 0, None
        self._lock_ref, self._lock_dev = None, 0.0
        self.t = 0
        mujoco.mj_forward(m, d)
        return self

    # ------------------------------------------------------------------ dynamics
    def step(self, arm_ctrl, grip_cmd):
        """One 20 Hz control step. arm_ctrl: 6 joint position targets (rad) to reach at the end of the step;
        grip_cmd: finger_1 command (0 open .. 0.7).

        Like an industrial arm controller (UR servoj interpolates at 500 Hz), the actuator targets are interpolated
        linearly from the previous action to this one over the 50 physics steps. Holding them constant made the very
        stiff UR10e servos (kp 5000) slam to their force limit for 1–2 ms at every 20 Hz update, which twisted the brick
        in the hand."""
        d = self.data
        new = np.r_[np.asarray(arm_ctrl, float), float(grip_cmd)]
        for c in self._interp(self.last_action, new):
            d.ctrl[self.arm_act] = c[:6]
            d.ctrl[self.grip_act] = [c[6], -c[6]]
            mujoco.mj_step(self.model, d)
            if self.cfg.grip_lock:
                self._grip_update(float(c[6]))
        self.prev_action = self.last_action
        self.last_action = new
        self.t += 1
        self._update_cure()
        for bnd in (1, 2):
            if self.bed_laid[bnd] < 0:
                self._update_bed(bnd)

    def _interp(self, old, new):
        """Actuator targets for the n_sub physics steps of one control step (cfg.ctrl_interp)."""
        n = self.n_sub
        s = np.arange(1, n + 1) / n
        if self.cfg.ctrl_interp == "linear":            # pilot_20261016
            return [old + (new - old) * x for x in s]
        if self.cfg.ctrl_interp == "hermite":           # cubic Hermite, tangents = previous / current step (C1 across steps)
            m0, m1 = old - self.prev_action, new - old
            h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
            return [h00[j] * old + h10[j] * m0 + h01[j] * new + h11[j] * m1 for j in range(n)]
        if self.cfg.ctrl_interp == "lowpass":           # linear target through a critically damped 2nd-order filter
            w, dt = 2 * np.pi * 15.0, self.model.opt.timestep
            out = []
            for x in s:
                u = old + (new - old) * x
                acc = w * w * (u - self.lp_y) - 2 * w * self.lp_v
                self.lp_v = self.lp_v + acc * dt
                self.lp_y = self.lp_y + self.lp_v * dt
                out.append(self.lp_y.copy())
            return out
        raise ValueError(f"unknown ctrl_interp {self.cfg.ctrl_interp!r}")

    # ------------------------------------------------------------------ state
    def brick_pose(self, i):
        a = self.brick_qadr[i]
        return self.data.qpos[a:a + 3].copy(), self.data.qpos[a + 3:a + 7].copy()

    def brick_poses(self):
        return np.array([self.data.qpos[a:a + 7].copy() for a in self.brick_qadr])

    def brick_vel(self, i):
        a = self.brick_dofadr[i]
        return self.data.qvel[a:a + 6].copy()

    def tcp(self):
        return self.data.site_xpos[self.tcp_sid].copy(), self.data.site_xmat[self.tcp_sid].reshape(3, 3).copy()

    def arm_q(self):
        return self.data.qpos[self.arm_qadr].copy()

    def closure(self):
        return G.closure(self.data)

    def wrist_ft(self):
        """Gripper force/torque sensor (robosuite ft_frame), world frame: (F N, T N·m). Not a policy input."""
        d = self.data
        R = d.site_xmat[self.ft_site].reshape(3, 3)
        return R @ d.sensordata[self.ft_force:self.ft_force + 3], R @ d.sensordata[self.ft_torque:self.ft_torque + 3]

    def _static(self, i):
        v = self.brick_vel(i)
        return np.linalg.norm(v[:3]) < 0.01 and np.linalg.norm(v[3:]) < 0.2

    def hand_clear(self, p):
        return np.linalg.norm(self.tcp()[0] - p) > self.cfg.hand_clear

    def in_slot(self, i, k=None):
        """Brick i sits in design slot k (default k = i) within the bed-laying tolerances (brick_sim brick_in_slot)."""
        c = self.cfg
        dsg = self.design[i if k is None else k]
        p, q = self.brick_pose(i)
        w, x, y, z = q
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        tilt = np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))
        if tilt > c.slot_tol_tilt_deg:
            return False
        if np.hypot(p[0] - dsg["x"], p[1] - dsg["y"]) > c.slot_tol_xy or abs(p[2] - dsg["z"]) > c.slot_tol_z:
            return False
        return np.degrees(abs((yaw - dsg["yaw"] + np.pi / 2) % np.pi - np.pi / 2)) <= c.slot_tol_yaw_deg

    def on_table(self, i):
        """Brick i rests on the supply table (inside the table outline, at the table-top height)."""
        c = self.cfg
        p, _ = self.brick_pose(i)
        return bool(c.table_x[0] < p[0] < c.table_x[1] and c.table_y[0] < p[1] < c.table_y[1]
                    and abs(p[2] - (c.table_h + H / 2)) < 0.02)

    def _update_cure(self):
        m, d, c = self.model, self.data, self.cfg
        xs = [s["x"] for s in self.design]
        x_lo, x_hi = min(xs) - 0.6 * L_FULL, max(xs) + 0.6 * L_FULL
        for i in range(self.n):
            if self.cured[i]:
                continue
            p, q = self.brick_pose(i)
            on_wall = abs(p[1] - WALL_Y) < 0.02 and x_lo < p[0] < x_hi and p[2] > 0.3 * H
            ok = on_wall and self._static(i) and self.hand_clear(p)
            self.settle_count[i] = self.settle_count[i] + 1 if ok else 0
            n_cure = c.cure_settle_steps if c.cure_time_s is None else int(round(c.cure_time_s / c.ctrl_dt))
            if self.settle_count[i] >= n_cure:
                eq = m.equality(f"cure_{i}").id
                qi = np.r_[q[0], -q[1:]]
                m.eq_data[eq, 0:3] = 0.0
                m.eq_data[eq, 3:6] = _quat_rotmat(qi) @ (-p)
                m.eq_data[eq, 6:10] = qi
                m.eq_data[eq, 10] = 1.0
                d.eq_active[eq] = 1
                self.cured[i] = True
                self.cure_t[i] = self.t
                self.sag_at_cure[i] = self.bed_sag(i)

    def bed_sag(self, i):
        """Mean penetration (m, positive = sunk) of brick i into the mortar bed slabs, nan if it does not touch a bed."""
        d = self.data
        n = d.ncon
        if n == 0:
            return np.nan
        g = d.contact.geom[:n]
        beds = np.array(list(self.bed_gid.values()))
        gid = self.brick_gid[i]
        sel = ((g[:, 0] == gid) & np.isin(g[:, 1], beds)) | ((g[:, 1] == gid) & np.isin(g[:, 0], beds))
        return float(-d.contact.dist[:n][sel].mean()) if sel.any() else np.nan

    def _update_bed(self, bnd):
        """Lay the mortar bed on course `bnd` once every brick of it is in its slot (bed_rule "placed": off the supply table),
        static for 2 s and the hand is off."""
        c, d, m = self.cfg, self.data, self.model
        info = self.beds[bnd]
        cnt = self.bed_count[bnd]
        ready = True
        for i in info["lower"]:
            p, _ = self.brick_pose(i)
            placed = self.in_slot(i) if c.bed_rule == "in_slot" else not self.on_table(i)
            ok = placed and self._static(i) and self.hand_clear(p)
            cnt[i] = cnt[i] + 1 if ok else 0
            ready &= cnt[i] >= c.cure_settle_steps
        if not ready:
            return
        tb = JOINT
        tops = []
        for nm, i in zip(info["slabs"], info["lower"]):
            p, q = self.brick_pose(i)
            R = _quat_rotmat(q)
            ctr = p + R @ np.array([0.0, 0.0, H / 2 + tb / 2])
            d.mocap_pos[self.bed_mocap[nm]] = ctr
            d.mocap_quat[self.bed_mocap[nm]] = q
            tops.append((p, ctr, self.design[i]["L"]))
        for j, nm in enumerate(info["joints"]):
            (pl, cl, Ll), (pr, cr, Lr) = tops[j], tops[j + 1]
            gap = max((pr[0] - Lr / 2) - (pl[0] + Ll / 2), 0.0)
            m.geom_size[self.bed_gid[nm], 0] = gap / 2 + c.bed_overhang + 0.002
            d.mocap_pos[self.bed_mocap[nm]] = [((pl[0] + Ll / 2) + (pr[0] - Lr / 2)) / 2, (cl[1] + cr[1]) / 2, (cl[2] + cr[2]) / 2]
            d.mocap_quat[self.bed_mocap[nm]] = [1, 0, 0, 0]
        mujoco.mj_forward(m, d)
        if c.bed_rule == "placed" and self._bed_sinks(info):
            for nm in info["slabs"] + info["joints"]:
                d.mocap_pos[self.bed_mocap[nm]] = [0, 0, -1]
            mujoco.mj_forward(m, d)
            self.bed_blocked[bnd] += 1
            return
        self.bed_laid[bnd] = self.t

    def _bed_sinks(self, info, depth=0.001):
        """A slab of this bed overlaps a brick of the course above by more than `depth` (the bed pairs are with those only)."""
        d = self.data
        n = d.ncon
        if n == 0:
            return False
        g = d.contact.geom[:n]
        mine = np.array([self.bed_gid[nm] for nm in info["slabs"] + info["joints"]])
        sel = np.isin(g[:, 0], mine) | np.isin(g[:, 1], mine)
        return bool((d.contact.dist[:n][sel] < -depth).any())

    # ------------------------------------------------------------------ grip lock (나2 결정, 현실성 조건 1 ~ 6)
    def _rel_pose(self, b):
        d = self.data
        Rg = d.xmat[self.grip_body].reshape(3, 3)
        return Rg.T @ (d.xpos[self.brick_bid[b]] - d.xpos[self.grip_body]), Rg.T @ d.xmat[self.brick_bid[b]].reshape(3, 3)

    def _hand_contacts(self):
        """Contacts between finger geoms and bricks: arrays (side, brick, normal force, world point)."""
        d = self.data
        n = d.ncon
        if n == 0:
            return None
        g = d.contact.geom[:n]
        adr = d.contact.efc_address[:n]
        s1, s2 = self.geom_side[g[:, 0]], self.geom_side[g[:, 1]]
        b1, b2 = self.geom_brick[g[:, 0]], self.geom_brick[g[:, 1]]
        mA = (s1 >= 0) & (s1 <= 1) & (b2 >= 0) & (adr >= 0)
        mB = (s2 >= 0) & (s2 <= 1) & (b1 >= 0) & (adr >= 0)
        if not (mA.any() or mB.any()):
            return None
        side = np.r_[s1[mA], s2[mB]]
        brick = np.r_[b2[mA], b1[mB]]
        fn = np.r_[d.efc_force[adr[mA]], d.efc_force[adr[mB]]]
        pos = np.r_[d.contact.pos[:n][mA], d.contact.pos[:n][mB]]
        return side, brick, fn, pos

    def _grip_update(self, ctrl):
        c, d = self.cfg, self.data
        dt = self.model.opt.timestep
        qf = float(d.qpos[self.fj_qadr])
        if ctrl > c.gripper.open_ctrl + 0.01:
            if self._close_t0 is None:
                self._close_t0, self._cmd_max = d.time, ctrl
            self._cmd_max = max(self._cmd_max, ctrl)
        else:
            self._close_t0, self._cmd_max = None, 0.0
        # ⑤ opening order: the finger command asks the fingers to open, or drops 0.01 rad below the command that made the grip
        # (the command at locking; before locking its maximum since closing). A short harder squeeze (G1 angle check) and the
        # return to the holding command is not an opening.
        ref = self._cmd_lock if self.lock_b >= 0 else self._cmd_max
        opening = ctrl < qf - c.lock_open_eps or (self._close_t0 is not None and ctrl < ref - c.lock_drop)
        if self.lock_b >= 0:
            b = self.lock_b
            p_rel, _ = self._rel_pose(b)
            dev = float(np.linalg.norm(p_rel - self._lock_ref))
            self._lock_dev = max(self._lock_dev, dev)
            self._lock_steps += 1
            if self._lock_steps == 1:                                   # ④ jump right after locking
                self.lock_events[-1]["jump_mm"] = dev * 1000
            if opening:
                self._release("open")
                return
            reason = self._overload(b)
            if reason:
                self._release(reason[0], reason[1])
            return
        # ① the closing ramp is over
        if self._close_t0 is None or d.time - self._close_t0 < c.gripper.close_time - 1e-9 or opening:
            self._cand, self._settle, self._rel_prev = -1, 0, None
            return
        hc = self._hand_contacts()
        if hc is None:
            self._cand, self._settle, self._rel_prev = -1, 0, None
            return
        side, brick, fn, _ = hc
        best, NL, NR = -1, 0.0, 0.0
        for b in np.unique(brick):
            l, r = fn[(brick == b) & (side == 0)].sum(), fn[(brick == b) & (side == 1)].sum()
            if l > 0 and r > 0 and l + r > NL + NR:
                best, NL, NR = int(b), float(l), float(r)
        # ② both pads touch, each with at least N_min; never a brick already cured on the wall
        if best < 0 or min(NL, NR) < self.lock_nmin[best] or self.cured[best]:
            self._cand, self._settle, self._rel_prev = best, 0, None
            return
        p_rel, R_rel = self._rel_pose(best)
        if best != self._cand or self._rel_prev is None:
            self._cand, self._settle = best, 0
        else:
            v = np.linalg.norm(p_rel - self._rel_prev[0]) / dt
            dR = self._rel_prev[1].T @ R_rel
            w = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)) / dt
            self._settle = self._settle + 1 if (v <= c.lock_vlin and w <= c.lock_vang) else 0
        self._rel_prev = (p_rel, R_rel)
        if self._settle * dt >= c.lock_settle - 1e-9:                     # ① settled for lock_settle seconds
            self._lock(best, NL, NR, hc)
            self._cmd_lock = ctrl

    def _lock(self, b, NL, NR, hc):
        m, d = self.model, self.data
        eq = self.grip_eq[b]
        pb, Rb = d.xpos[self.brick_bid[b]], d.xmat[self.brick_bid[b]].reshape(3, 3)
        pg, Rg = d.xpos[self.grip_body], d.xmat[self.grip_body].reshape(3, 3)
        qrel = np.zeros(4)
        mujoco.mju_mat2Quat(qrel, (Rb.T @ Rg).ravel())                    # ④ the current relative pose, no pull
        anchor = Rg.T @ (pb - pg)                                       # weld point = brick centre, in the gripper frame
        m.eq_data[eq, 0:3] = anchor
        m.eq_data[eq, 3:6] = Rb.T @ (pg + Rg @ anchor - pb)             # MuJoCo weld: p1 + R1·relpos = p2 + R2·anchor (= 0 here)
        m.eq_data[eq, 6:10] = qrel
        m.eq_data[eq, 10] = 1.0
        d.eq_active[eq] = 1
        self.lock_b = b
        self._lock_ref, self._lock_dev, self._lock_steps = self._rel_pose(b)[0], 0.0, 0
        self._capT_hist = deque(maxlen=self.cfg.lock_cap_window)
        side, brick, fn, pos = hc
        sel = brick == b
        ctr = (fn[sel][:, None] * pos[sel]).sum(0) / fn[sel].sum()                  # pad contact centre
        n_ax = d.site_xmat[self.tcp_sid].reshape(3, 3)[:, 1]
        off = pb - ctr
        off_plane = off - (off @ n_ax) * n_ax
        ang = np.degrees(np.arccos(np.clip(abs(Rb[:, 0] @ d.site_xmat[self.tcp_sid].reshape(3, 3)[:, 0]), -1, 1)))
        self.lock_events.append({"event": "lock", "t": int(self.t), "time": float(d.time), "brick": int(b), "N_left": NL, "N_right": NR,
                                 "N_min": float(self.lock_nmin[b]), "grasp_offset_mm": float(np.linalg.norm(off_plane) * 1000),
                                 "grasp_yaw_offset_deg": float(ang)})

    def _overload(self, b):
        """③ release when the hand's force along the pad plane or its twist about the pad axis exceeds what friction can hold.
        Hand force/torque = constraint force on the brick's free joint (weld + all contacts) minus the non-hand contacts."""
        m, d, c = self.model, self.data, self.cfg
        dof = self.brick_dofadr[b]
        pb, Rb = d.xpos[self.brick_bid[b]], d.xmat[self.brick_bid[b]].reshape(3, 3)
        F = d.qfrc_constraint[dof:dof + 3].copy()
        T = Rb @ d.qfrc_constraint[dof + 3:dof + 6]                     # free joint: angular dofs in the body frame
        n = d.ncon
        gid = self.brick_gid[b]
        if n:
            g = d.contact.geom[:n]
            on = (g[:, 0] == gid) | (g[:, 1] == gid)
            other = g[:, 0].copy(); other[g[:, 0] == gid] = g[g[:, 0] == gid, 1]
            nonhand = on & (self.geom_side[other] < 0)
            f6 = np.zeros(6)
            for i in np.nonzero(nonhand)[0]:
                mujoco.mj_contactForce(m, d, int(i), f6)
                fw = d.contact.frame[i].reshape(3, 3).T @ f6[:3]
                fw = fw if d.contact.geom2[i] == gid else -fw            # force on the brick
                F -= fw
                T -= np.cross(d.contact.pos[i] - pb, fw)
        hc = self._hand_contacts()
        if hc is None:
            return ("no_contact", {})
        side, brick, fn, pos = hc
        sel = brick == b
        side, fn, pos = side[sel], fn[sel], pos[sel]
        NL, NR = fn[side == 0].sum(), fn[side == 1].sum()
        n_ax = d.site_xmat[self.tcp_sid].reshape(3, 3)[:, 1]
        Ft = F - (F @ n_ax) * n_ax
        capF = c.lock_mu * (NL + NR)
        if np.linalg.norm(Ft) > capF:
            return ("force", {"F_t": float(np.linalg.norm(Ft)), "cap": float(capF), "N_left": float(NL), "N_right": float(NR)})
        capT, ctrs = 0.0, []
        for sd in (0, 1):
            k = side == sd
            if fn[k].sum() <= 0:
                continue
            ctr = (fn[k][:, None] * pos[k]).sum(0) / fn[k].sum()
            ctrs.append(ctr)
            r = pos[k] - ctr
            r = r - np.outer(r @ n_ax, n_ax)
            capT += c.lock_mu * float((fn[k] * np.linalg.norm(r, axis=1)).sum())
        axis_pt = np.mean(ctrs, axis=0) if ctrs else pb
        Tn = float((T + np.cross(pb - axis_pt, F)) @ n_ax)                # twist about the line through the pad contacts
        # 추가 지시 2: the twist *limit* is the largest of the last lock_cap_window physics steps (MuJoCo's box-box contact
        # generation can drop the pad contacts to 2 close points for one step while the real pad face is unchanged);
        # the applied twist and the force check stay per step. A grasp whose contacts really shrink is released after the window.
        self._capT_hist.append(capT)
        capW = max(self._capT_hist)
        if abs(Tn) > capW:
            return ("torque", {"T_n": Tn, "cap": capW, "cap_now": capT, "N_left": float(NL), "N_right": float(NR)})
        return None

    def lock_state(self):
        """[locked 1/0, locked brick (-1 none), pad normal force left, right on the brick the fingers press hardest] (info only)."""
        NL = NR = 0.0
        hc = self._hand_contacts()
        if hc is not None:
            side, brick, fn, _ = hc
            for b in np.unique(brick):
                l, r = fn[(brick == b) & (side == 0)].sum(), fn[(brick == b) & (side == 1)].sum()
                if l + r > NL + NR:
                    NL, NR = float(l), float(r)
        return np.array([self.lock_b >= 0, self.lock_b, NL, NR], np.float32)

    def _release(self, reason, info=None):
        d = self.data
        b = self.lock_b
        d.eq_active[self.grip_eq[b]] = 0
        self.lock_events.append({"event": "release", "reason": reason, "t": int(self.t), "time": float(d.time), "brick": int(b),
                                 "max_dev_while_locked_mm": float(self._lock_dev * 1000), **(info or {})})
        self.lock_b = -1
        self._cand, self._settle, self._rel_prev = -1, 0, None

    # ------------------------------------------------------------------ contacts
    def contacts(self):
        """List of (name_a, name_b, dist) for contacts involving the robot/gripper and anything that is not robot/gripper."""
        m, d = self.model, self.data
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            r1, r2 = b1 in self.robot_bodies, b2 in self.robot_bodies
            if r1 == r2:
                continue
            g_r, g_o = (c.geom1, c.geom2) if r1 else (c.geom2, c.geom1)
            out.append((m.geom(g_r).name or m.body(m.geom_bodyid[g_r]).name, m.geom(g_o).name or m.body(m.geom_bodyid[g_o]).name,
                        float(c.dist), int(g_r), int(g_o)))
        return out

    # ------------------------------------------------------------------ rendering
    def render(self, cam, w=None, h=None, depth=False):
        w = w or self.cfg.img
        h = h or self.cfg.img
        key = (w, h, depth)
        if key not in self.renderer:
            r = mujoco.Renderer(self.model, h, w)
            if depth:
                r.enable_depth_rendering()
            self.renderer[key] = r
        r = self.renderer[key]
        r.update_scene(self.data, camera=cam, scene_option=self.vopt)
        return r.render().copy()

    def close(self):
        for r in self.renderer.values():
            r.close()
        self.renderer = {}

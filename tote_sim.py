"""実寸コンテナ(三甲 ラボボックス No.20)を双腕 ViperX で棚へ運ぶ MuJoCo 環境。

gym-aloha のロボットモデル(assets/vx300s_*.xml)をそのまま使い、箱・中身・棚を足している。
- mode="joint": 関節の目標角度で動かす(学習・評価用。ALOHA と同じ 14 次元)
- mode="ee"   : 手先の位置・姿勢を mocap で動かす(お手本スクリプト用)
"""

from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path(__file__).parent / "assets"

DT = 0.02  # 制御周期 50Hz(ALOHA と同じ)
FPS = 50

# ---- 箱の寸法(m)。外寸 475×369×154 / 内寸 412×313×136(製品仕様) ----
BOX_L, BOX_W, BOX_H = 0.475, 0.369, 0.154
IN_L, IN_W, IN_H = 0.412, 0.313, 0.136
WALL = 0.008  # 壁の厚み(製品仕様にないため仮値)
FLOOR = BOX_H - IN_H  # 底の厚み 18mm

# 短辺の取っ手(グリッパーでつまむ部分)
HANDLE_Y = 0.035  # つまむ方向の厚み(指の最大開き 6.9cm に余裕を持たせる)
HANDLE_X = 0.05  # 箱から外へ張り出す長さ
HANDLE_Z = 0.03

BOX_MASS = 0.5  # 箱の質量(製品ページに記載なし → 仮値)
N_ITEMS = 4
ITEM_SIZE = 0.05  # 中身: 5cm 角の小箱
ITEM_MASS_TOTAL = 2.0  # 中身の合計(上限 3kg)

# ---- 置き場 ----
START_Y = 0.40  # 箱の初期位置(手前)
SHELF_Y = 0.80  # 棚の中心(奥)
SHELF_H = 0.06  # 棚の高さ
SHELF_DEPTH = 0.30

# グリッパーの指の可動範囲(gym-aloha と同じ)
GRIP_OPEN, GRIP_CLOSE = 0.05800, 0.01844

START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239] * 2


def _box_xml(item_mass_total=ITEM_MASS_TOTAL):
    """箱(底+壁4枚+取っ手2つ)と中身。"""
    hl, hw = BOX_L / 2, BOX_W / 2
    wall_hz = (BOX_H - FLOOR) / 2
    wz = FLOOR + wall_hz
    g = 'condim="4" friction="1 0.005 0.0001" solimp="0.95 0.99 0.001" solref="0.005 1" rgba="0.15 0.35 0.75 1"'
    geoms = [
        f'<geom name="box_floor" type="box" size="{hl} {hw} {FLOOR/2}" pos="0 0 {FLOOR/2}" {g}/>',
        f'<geom name="box_wall_xp" type="box" size="{WALL/2} {hw} {wall_hz}" pos="{hl-WALL/2} 0 {wz}" {g}/>',
        f'<geom name="box_wall_xn" type="box" size="{WALL/2} {hw} {wall_hz}" pos="{-hl+WALL/2} 0 {wz}" {g}/>',
        f'<geom name="box_wall_yp" type="box" size="{hl} {WALL/2} {wall_hz}" pos="0 {hw-WALL/2} {wz}" {g}/>',
        f'<geom name="box_wall_yn" type="box" size="{hl} {WALL/2} {wall_hz}" pos="0 {-hw+WALL/2} {wz}" {g}/>',
    ]
    for side, s in (("p", 1), ("n", -1)):
        hx = s * (hl + HANDLE_X / 2 - WALL / 2)
        geoms.append(
            f'<geom name="handle_{side}" type="box" size="{HANDLE_X/2} {HANDLE_Y/2} {HANDLE_Z/2}" '
            f'pos="{hx} 0 {BOX_H - HANDLE_Z/2}" condim="4" friction="1.5 0.005 0.0001" '
            f'solimp="0.95 0.99 0.001" solref="0.005 1" rgba="0.1 0.25 0.6 1"/>'
        )
    box = (
        f'<body name="box" pos="0 {START_Y} 0">\n<freejoint name="box_joint"/>\n'
        f'<inertial pos="0 0 {BOX_H*0.35}" mass="{BOX_MASS}" diaginertia="0.012 0.02 0.03"/>\n'
        + "\n".join(geoms)
        + "\n</body>"
    )
    colors = ["0.9 0.6 0.1 1", "0.2 0.7 0.3 1", "0.85 0.2 0.2 1", "0.9 0.9 0.2 1", "0.6 0.3 0.8 1"]
    m = item_mass_total / N_ITEMS
    i_ = m * ITEM_SIZE**2 / 6
    items = []
    for i in range(N_ITEMS):
        items.append(
            f'<body name="item{i}" pos="{-0.1 + 0.07*i} {START_Y} {FLOOR + ITEM_SIZE/2}">'
            f'<freejoint name="item{i}_joint"/><inertial pos="0 0 0" mass="{m}" diaginertia="{i_} {i_} {i_}"/>'
            f'<geom name="item{i}" type="box" size="{ITEM_SIZE/2} {ITEM_SIZE/2} {ITEM_SIZE/2}" condim="4" '
            f'friction="0.8 0.005 0.0001" solimp="0.95 0.99 0.001" solref="0.005 1" rgba="{colors[i % 5]}"/></body>'
        )
    return box + "\n" + "\n".join(items)


def _shelf_xml():
    return (
        f'<body name="shelf" pos="0 {SHELF_Y} {SHELF_H/2}">'
        f'<geom name="shelf" type="box" size="0.30 {SHELF_DEPTH/2} {SHELF_H/2}" rgba="0.75 0.6 0.4 1" '
        f'condim="4" friction="1 0.005 0.0001"/></body>'
    )


ARM_ACTUATORS = """
<position ctrllimited="true" ctrlrange="-3.14158 3.14158" joint="vx300s_{s}/waist" kp="800" forcelimited="true" forcerange="-150 150"/>
<position ctrllimited="true" ctrlrange="-1.85005 1.25664" joint="vx300s_{s}/shoulder" kp="1600" forcelimited="true" forcerange="-300 300"/>
<position ctrllimited="true" ctrlrange="-1.76278 1.6057" joint="vx300s_{s}/elbow" kp="800" forcelimited="true" forcerange="-100 100"/>
<position ctrllimited="true" ctrlrange="-3.14158 3.14158" joint="vx300s_{s}/forearm_roll" kp="{kp_roll}" forcelimited="true" forcerange="-100 100"/>
<position ctrllimited="true" ctrlrange="-1.8675 2.23402" joint="vx300s_{s}/wrist_angle" kp="{kp_wrist}"/>
<position ctrllimited="true" ctrlrange="-3.14158 3.14158" joint="vx300s_{s}/wrist_rotate" kp="{kp_rot}"/>
"""
FINGER_ACTUATORS = """
<position ctrllimited="true" ctrlrange="0.021 0.057" joint="vx300s_{s}/left_finger" kp="{kp_f}"/>
<position ctrllimited="true" ctrlrange="-0.057 -0.021" joint="vx300s_{s}/right_finger" kp="{kp_f}"/>
"""

# アクチュエータの強さ。gym-aloha の値(手首 50/20、ひねり 10、指 200)では、重い箱を持つと手首がたわみ取っ手が抜けるため強めている。
GAINS = dict(kp_roll=200, kp_wrist=300, kp_rot=200, kp_f=2000)
# 手首を強くすると、手首の部品が軽すぎて計算が発散するため、モーター側の回転慣性(armature)を加えて安定させる
WRIST_ARMATURE = 0.05


def build_xml(mode, item_mass_total=ITEM_MASS_TOTAL):
    acts = []
    for s in ("left", "right"):
        if mode == "joint":
            acts.append(ARM_ACTUATORS.format(s=s, **GAINS))
        acts.append(FINGER_ACTUATORS.format(s=s, **GAINS))
    mocap = ""
    equality = ""
    if mode == "ee":
        for s in ("left", "right"):
            mocap += f'<body mocap="true" name="mocap_{s}" pos="0 0.5 0.3"/>\n'
            equality += f'<weld body1="mocap_{s}" body2="vx300s_{s}/gripper_link" relpose="0 0 0 1 0 0 0" solref="0.01 1" solimp="0.9 0.95 0.001"/>\n'
    return f"""<mujoco>
<include file="scene.xml"/>
<include file="vx300s_dependencies.xml"/>
<option timestep="0.002" cone="elliptic" impratio="10"/>
<size njmax="2000" nconmax="500"/>
<equality>{equality}</equality>
<worldbody>
<include file="vx300s_left.xml"/>
<include file="vx300s_right.xml"/>
{mocap}
{_box_xml(item_mass_total)}
{_shelf_xml()}
<camera name="overhead" pos="0 0.15 1.1" xyaxes="1 0 0 0 0.926 0.379" fovy="58"/>
</worldbody>
<actuator>{''.join(acts)}</actuator>
</mujoco>"""


def norm_grip(x):
    return (x - GRIP_CLOSE) / (GRIP_OPEN - GRIP_CLOSE)


def unnorm_grip(x):
    return x * (GRIP_OPEN - GRIP_CLOSE) + GRIP_CLOSE


def quat_to_tilt_deg(q):
    """箱の上向き軸が鉛直からどれだけ傾いているか(度)。"""
    w, x, y, z = q
    zz = 1 - 2 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(zz, -1, 1))))


class ToteEnv:
    """gym-aloha と同じ観測・行動の形(qpos 14 次元、画像 top)を返す。"""

    def __init__(self, mode="joint", width=640, height=480, cameras=("top",), item_mass_total=ITEM_MASS_TOTAL):
        self.mode = mode
        xml_path = ASSETS / f"_tote_{mode}.xml"
        xml_path.write_text(build_xml(mode, item_mass_total))
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        for s in ("left", "right"):
            for j in ("forearm_roll", "wrist_angle", "wrist_rotate"):
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"vx300s_{s}/{j}")
                self.model.dof_armature[self.model.jnt_dofadr[jid]] = WRIST_ARMATURE
        if mode == "ee":
            # お手本用: 腕の関節摩擦を外し、手先が mocap の目標どおりに動くようにする(指は除く)
            for s in ("left", "right"):
                for j in ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"):
                    jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"vx300s_{s}/{j}")
                    self.model.dof_frictionloss[self.model.jnt_dofadr[jid]] = 0
        self.data = mujoco.MjData(self.model)
        self.n_sub = int(round(DT / self.model.opt.timestep))
        self.cameras = cameras
        self.width, self.height = width, height
        self._renderer = None
        self._ids()

    def _ids(self):
        m = self.model
        jid = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
        self.box_adr = jid("box_joint")
        self.box_dof = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "box_joint")]
        self.item_adr = [jid(f"item{i}_joint") for i in range(N_ITEMS)]
        gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
        self.handle_geoms = {gid("handle_p"), gid("handle_n")}
        self.box_geoms = {gid(n) for n in ("box_floor", "box_wall_xp", "box_wall_xn", "box_wall_yp", "box_wall_yn")} | self.handle_geoms
        self.table_geom = gid("table")
        self.shelf_geom = gid("shelf")
        self.finger_geoms = {
            s: {gid(f"vx300s_{s}/10_left_gripper_finger"), gid(f"vx300s_{s}/10_right_gripper_finger")} for s in ("left", "right")
        }
        self.gripper_body = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"vx300s_{s}/gripper_link") for s in ("left", "right")}

    # ---------------- リセット ----------------
    def sample_layout(self, seed):
        """箱の位置・向きと中身の配置をランダムに決める。"""
        rng = np.random.RandomState(seed)
        box = np.array([rng.uniform(-0.03, 0.03), START_Y + rng.uniform(-0.03, 0.03), rng.uniform(-5, 5)])  # x, y, yaw(度)
        # 中身は内寸の中で重ならないよう 2×2 のマスに置き、マス内で少しずらす
        cells = [(-0.1, -0.07), (0.1, -0.07), (-0.1, 0.07), (0.1, 0.07)]
        order = rng.permutation(len(cells))
        items = [np.array(cells[k]) + rng.uniform(-0.04, 0.04, 2) for k in order[:N_ITEMS]]
        return {"box": box, "items": np.array(items)}

    def reset(self, seed=0, layout=None):
        mujoco.mj_resetData(self.model, self.data)
        d = self.data
        d.qpos[:16] = START_ARM_POSE
        self.layout = layout if layout is not None else self.sample_layout(seed)
        bx, by, yaw = self.layout["box"]
        c, s = np.cos(np.radians(yaw) / 2), np.sin(np.radians(yaw) / 2)
        quat = np.array([c, 0, 0, s])
        a = self.box_adr
        d.qpos[a : a + 3] = [bx, by, 0.0005]
        d.qpos[a + 3 : a + 7] = quat
        R = np.array([[np.cos(np.radians(yaw)), -np.sin(np.radians(yaw))], [np.sin(np.radians(yaw)), np.cos(np.radians(yaw))]])
        for adr, (ix, iy) in zip(self.item_adr, self.layout["items"]):
            p = R @ np.array([ix, iy])
            d.qpos[adr : adr + 3] = [bx + p[0], by + p[1], FLOOR + ITEM_SIZE / 2 + 0.001]
            d.qpos[adr + 3 : adr + 7] = quat
        if self.mode == "joint":
            d.ctrl[:] = self._joint_to_ctrl(self.qpos14_from_raw(d.qpos[:16]))
        mujoco.mj_forward(self.model, d)
        if self.mode == "ee":
            for i, side in enumerate(("left", "right")):
                d.mocap_pos[i] = d.xpos[self.gripper_body[side]]
                d.mocap_quat[i] = d.xquat[self.gripper_body[side]]
            d.ctrl[:] = [GRIP_CLOSE, -GRIP_CLOSE, GRIP_CLOSE, -GRIP_CLOSE]
        # 箱と中身を落ち着かせる
        for _ in range(50):
            mujoco.mj_step(self.model, d)
        self.max_tilt = 0.0
        self.max_stage = 0
        self.stage4_steps = 0
        return self.observe()

    # ---------------- 行動 ----------------
    @staticmethod
    def qpos14_from_raw(q16):
        q16 = np.asarray(q16)
        return np.concatenate([q16[:6], [norm_grip(q16[6])], q16[8:14], [norm_grip(q16[14])]])

    @staticmethod
    def _joint_to_ctrl(a14):
        gl, gr = unnorm_grip(a14[6]), unnorm_grip(a14[13])
        return np.concatenate([a14[:6], [gl, -gl], a14[7:13], [gr, -gr]])

    def step(self, action):
        d = self.data
        if self.mode == "joint":
            d.ctrl[:] = self._joint_to_ctrl(np.asarray(action))
        else:  # action: [左 pos3 quat4 grip1, 右 pos3 quat4 grip1]
            for i in range(2):
                a = action[i * 8 : (i + 1) * 8]
                d.mocap_pos[i] = a[:3]
                d.mocap_quat[i] = a[3:7]
                g = unnorm_grip(a[7])
                d.ctrl[i * 2 : i * 2 + 2] = [g, -g]
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, d)
        obs = self.observe()
        info = self.metrics()
        return obs, info["stage"], info

    # ---------------- 観測 ----------------
    def render(self, camera="top", width=None, height=None):
        w, h = width or self.width, height or self.height
        if self._renderer is None or (self._renderer.width, self._renderer.height) != (w, h):
            self._renderer = mujoco.Renderer(self.model, height=h, width=w)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def observe(self):
        obs = {"qpos": self.qpos14_from_raw(self.data.qpos[:16]).copy()}
        obs["images"] = {c: self.render(c) for c in self.cameras}
        return obs

    def ee_pose(self, side):
        b = self.gripper_body[side]
        return np.concatenate([self.data.xpos[b], self.data.xquat[b]]).copy()

    # ---------------- 評価指標 ----------------
    def _contacts(self):
        pairs = set()
        for c in self.data.contact[: self.data.ncon]:
            pairs.add((c.geom1, c.geom2))
            pairs.add((c.geom2, c.geom1))
        return pairs

    def items_inside(self):
        """中身が箱の内側(箱の座標系で内寸の範囲内)に残っている個数。"""
        a = self.box_adr
        bpos = self.data.qpos[a : a + 3]
        bq = self.data.qpos[a + 3 : a + 7]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, bq)
        R = R.reshape(3, 3)
        n = 0
        for adr in self.item_adr:
            p = R.T @ (self.data.qpos[adr : adr + 3] - bpos)
            if abs(p[0]) < IN_L / 2 and abs(p[1]) < IN_W / 2 and 0 < p[2] < BOX_H + 0.01:
                n += 1
        return n

    def metrics(self):
        pairs = self._contacts()
        touch = lambda gs, hs: any((g, h) in pairs for g in gs for h in hs)
        grasp_l = touch(self.finger_geoms["left"], self.handle_geoms)
        grasp_r = touch(self.finger_geoms["right"], self.handle_geoms)
        on_table = touch(self.box_geoms, {self.table_geom})
        on_shelf = touch(self.box_geoms, {self.shelf_geom})
        a = self.box_adr
        tilt = quat_to_tilt_deg(self.data.qpos[a + 3 : a + 7])
        self.max_tilt = max(self.max_tilt, tilt)
        inside = self.items_inside()
        box_x, box_y = self.data.qpos[a], self.data.qpos[a + 1]
        speed = float(np.linalg.norm(self.data.qvel[self.box_dof : self.box_dof + 3]))
        # 段階: 1=両手でつかんだ 2=持ち上げた 3=棚の上 4=棚に置いて両手を離し、中身が全部残っている
        stage = 0
        if grasp_l and grasp_r:
            stage = 1
            if not on_table:
                stage = 2
        if on_shelf and not on_table and abs(box_y - SHELF_Y) < 0.05 and abs(box_x) < 0.08:
            stage = max(stage, 3)
            # 落下途中の一瞬を成功と数えないよう、水平(5°未満)で止まっていることも条件にする
            if not grasp_l and not grasp_r and inside == N_ITEMS and tilt < 5 and speed < 0.02:
                stage = 4
        # 成功は「段階4が 0.5 秒(25 ステップ)続いた」とき
        self.stage4_steps = self.stage4_steps + 1 if stage == 4 else 0
        self.max_stage = max(self.max_stage, stage if stage < 4 or self.stage4_steps >= 25 else 3)
        return {
            "stage": stage,
            "max_stage": self.max_stage,
            "items_inside": inside,
            "spilled": N_ITEMS - inside,
            "tilt": tilt,
            "max_tilt": self.max_tilt,
            "is_success": self.stage4_steps >= 25,
        }

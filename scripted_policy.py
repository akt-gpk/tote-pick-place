"""お手本スクリプト: 両手で左右の取っ手をつまみ、持ち上げて奥の棚へ置く(mocap 環境用)。

手先のウェイポイントを時間で直線補間するだけのシンプルな方式(ACT 論文の scripted policy と同じ考え方)。
"""

import mujoco
import numpy as np

from tote_sim import BOX_H, BOX_L, HANDLE_X, HANDLE_Z, SHELF_H, SHELF_Y, WALL

TCP = 0.115  # gripper_link 原点から指の腹の中心までの距離(実測した指の範囲 0.064〜0.154 m の中央付近)
PITCH = 70  # 手先を水平から何度下に向けるか
LIFT = 0.13  # 持ち上げる高さ(箱の底が棚 6cm を余裕で越える)

# (ステップ, 名前)。50Hz なので 500 ステップ = 10 秒
SCHEDULE = [
    (0, "start"),
    (60, "above"),
    (110, "grasp_open"),
    (140, "grasp_close"),
    (230, "lift"),
    (370, "carry"),
    (430, "place"),
    (460, "release"),
    (500, "retreat"),
]
EPISODE_LEN = SCHEDULE[-1][0]


def _gripper_quat(side, pitch_deg, yaw_rad):
    """手先の x 軸(指の向き)を内向き・下向きに、指の開閉方向を水平にした姿勢。"""
    p = np.radians(pitch_deg)
    s = 1 if side == "left" else -1  # 左腕は +x 向き、右腕は -x 向きに手を伸ばす
    x = np.array([s * np.cos(p), 0, -np.sin(p)])
    y = np.array([0, s * 1.0, 0])
    z = np.cross(x, y)
    R = np.stack([x, y, z], axis=1)
    c, sn = np.cos(yaw_rad), np.sin(yaw_rad)
    Rz = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])
    R = Rz @ R
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q, R[:, 0]


class ScriptedPolicy:
    def __init__(self, env, rng=None):
        self.env = env
        rng = rng or np.random.RandomState(0)
        # お手本に少しばらつきを入れる(全部同じ動きだと学習が不自然になるため)
        self.jitter = rng.uniform(-0.005, 0.005, size=3)
        self._plan()

    def _handle_point(self, side, box_pos, yaw):
        s = -1 if side == "left" else 1  # 左腕は -x 側の取っ手
        hx = s * (BOX_L / 2 + HANDLE_X / 2 - WALL / 2 + 0.01)  # 取っ手の外寄りをつまむ
        local = np.array([hx, 0, BOX_H - HANDLE_Z / 2])
        c, sn = np.cos(yaw), np.sin(yaw)
        return box_pos + np.array([c * local[0] - sn * local[1], sn * local[0] + c * local[1], local[2]])

    def _plan(self):
        env = self.env
        a = env.box_adr
        box_pos = env.data.qpos[a : a + 3].copy()
        bq = env.data.qpos[a + 3 : a + 7]
        yaw = 2 * np.arctan2(bq[3], bq[0])
        goal_pos = np.array([box_pos[0], SHELF_Y, SHELF_H + 0.003])
        self.waypoints = {}
        for side in ("left", "right"):
            start = env.ee_pose(side)
            quat, xdir = _gripper_quat(side, PITCH, yaw)
            h0 = self._handle_point(side, box_pos, yaw) + self.jitter
            h_goal = self._handle_point(side, goal_pos, 0.0) + self.jitter
            q_goal, xdir_goal = _gripper_quat(side, PITCH, 0.0)
            to_link = lambda tcp, xd: tcp - TCP * xd  # 指の腹の位置 → gripper_link の位置
            up = np.array([0, 0, 1.0])
            wp = {
                "start": (start[:3], start[3:], 1.0),
                "above": (to_link(h0 + 0.08 * up, xdir), quat, 1.0),
                "grasp_open": (to_link(h0, xdir), quat, 1.0),
                "grasp_close": (to_link(h0, xdir), quat, 0.0),
                "lift": (to_link(h0 + LIFT * up, xdir), quat, 0.0),
                "carry": (to_link(h_goal + LIFT * up, xdir_goal), q_goal, 0.0),
                "place": (to_link(h_goal + 0.004 * up, xdir_goal), q_goal, 0.0),
                "release": (to_link(h_goal + 0.004 * up, xdir_goal), q_goal, 1.0),
                "retreat": (to_link(h_goal + 0.10 * up, xdir_goal), q_goal, 1.0),
            }
            self.waypoints[side] = wp

    def __call__(self, t):
        out = []
        for side in ("left", "right"):
            wp = self.waypoints[side]
            for (t0, n0), (t1, n1) in zip(SCHEDULE[:-1], SCHEDULE[1:]):
                if t0 <= t <= t1:
                    break
            else:
                t0, n0, t1, n1 = SCHEDULE[-1][0], SCHEDULE[-1][1], SCHEDULE[-1][0] + 1, SCHEDULE[-1][1]
            r = np.clip((t - t0) / max(t1 - t0, 1), 0, 1)
            r = r * r * (3 - 2 * r)  # なめらかに加減速(中身がこぼれにくい)
            p0, q0, g0 = wp[n0]
            p1, q1, g1 = wp[n1]
            q = _slerp(q0, q1, r)
            out.append(np.concatenate([p0 + (p1 - p0) * r, q, [g0 + (g1 - g0) * r]]))
        return np.concatenate(out)


def _slerp(q0, q1, r):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = np.dot(q0, q1)
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + r * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - r) * th) * q0 + np.sin(r * th) * q1) / np.sin(th)

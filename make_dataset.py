"""お手本データセットを LeRobot 形式で作る。

1. mocap 環境でお手本スクリプトを動かし、関節の軌道を記録する
2. 同じ初期配置の関節制御環境でその軌道を再生し、画像・関節状態・行動を保存する
3. 再生でも成功(棚に置けて中身のこぼれ 0)したエピソードだけを残す

例: python make_dataset.py --n 50
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from scripted_policy import EPISODE_LEN, ScriptedPolicy
from tote_sim import FPS, ToteEnv

JOINT_NAMES = [
    f"{s}_{j}"
    for s in ("left", "right")
    for j in ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "gripper")
]
TASK = "Lift the container with both arms and place it on the shelf without spilling"
H, W = 240, 320
CAM = "overhead"
STEPS = EPISODE_LEN + 20  # 置いた後に少し静止する時間を含める

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=50)
p.add_argument("--seed0", type=int, default=1000)
p.add_argument("--root", default=str(Path(__file__).parent / "data" / "tote_pick_place"))
args = p.parse_args()

root = Path(args.root)
if root.exists():
    shutil.rmtree(root)  # このスクリプトが作ったフォルダだけを作り直す

features = {
    f"observation.images.{CAM}": {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channels"]},
    "observation.state": {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES},
    "action": {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES},
}
ds = LeRobotDataset.create(
    repo_id="local/tote_pick_place", fps=FPS, features=features, root=root, robot_type="aloha_sim", image_writer_threads=4
)

ee = ToteEnv("ee", cameras=())
jt = ToteEnv("joint", width=W, height=H, cameras=(CAM,))
log = []
seed = args.seed0
while len(log) < args.n:
    obs = ee.reset(seed=seed)
    pol = ScriptedPolicy(ee, np.random.RandomState(seed))
    traj = []
    for t in range(STEPS):
        a = pol(t)
        obs, _, _ = ee.step(a)
        q = obs["qpos"].copy()
        q[6], q[13] = a[7], a[15]  # 指は指令値を記録
        traj.append(q.astype(np.float32))

    obs = jt.reset(seed=seed, layout=ee.layout)
    frames = []
    for q in traj:
        frames.append({f"observation.images.{CAM}": obs["images"][CAM], "observation.state": obs["qpos"].astype(np.float32), "action": q})
        obs, _, info = jt.step(q)
    if info["is_success"]:
        for f in frames:
            ds.add_frame({**f, "task": TASK})
        ds.save_episode()
        log.append({"episode": len(log), "seed": seed, "max_tilt": info["max_tilt"], "spilled": info["spilled"]})
        print(f"saved episode {len(log)-1} (seed {seed}) max_tilt={info['max_tilt']:.1f}", flush=True)
    else:
        print(f"skip seed {seed}: max_stage={info['max_stage']} spilled={info['spilled']}", flush=True)
    seed += 1

ds.finalize() if hasattr(ds, "finalize") else None
(root / "generation_log.json").write_text(json.dumps(log, indent=1))
print(f"done: {len(log)} episodes, tried {seed - args.seed0} seeds")

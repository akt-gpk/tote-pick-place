"""重さ試験: mocap のお手本で関節軌道を記録し、関節制御版で再生して成功するかを見る。

例: python replay_test.py --mass 3.0 --n 5
"""

import argparse

import numpy as np

from scripted_policy import EPISODE_LEN, ScriptedPolicy
from tote_sim import ToteEnv

p = argparse.ArgumentParser()
p.add_argument("--mass", type=float, default=2.0)
p.add_argument("--n", type=int, default=5)
p.add_argument("--seed0", type=int, default=200)
args = p.parse_args()

ee = ToteEnv("ee", cameras=(), item_mass_total=args.mass)
jt = ToteEnv("joint", cameras=(), item_mass_total=args.mass)
ok = 0
for k in range(args.n):
    seed = args.seed0 + k
    obs = ee.reset(seed=seed)
    pol = ScriptedPolicy(ee, np.random.RandomState(seed))
    traj = []
    for t in range(EPISODE_LEN + 20):
        a = pol(t)
        obs, _, info_ee = ee.step(a)
        q = obs["qpos"].copy()
        q[6], q[13] = a[7], a[15]  # 指は「指令値」を記録する(実際の開きではなく)
        traj.append(q)
    jt.reset(seed=seed, layout=ee.layout)
    err = []
    for q in traj:
        o, _, info = jt.step(q)
        err.append(np.abs(o["qpos"][[0, 1, 2, 7, 8, 9]] - q[[0, 1, 2, 7, 8, 9]]).max())
    print(f"seed {seed} mass {args.mass}kg: ee={info_ee['is_success']} replay={info['is_success']} max_stage={info['max_stage']} spilled={info['spilled']} max_tilt={info['max_tilt']:.1f} joint_err_max={max(err):.3f}rad")
    ok += info["is_success"]
print(f"replay success {ok}/{args.n}")

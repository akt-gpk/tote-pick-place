"""お手本スクリプトを mocap 環境で試し、成功率と途中経過を表示する。

例: python try_scripted.py --n 20 --video scripted_check.mp4
"""

import argparse

import imageio
import numpy as np

from scripted_policy import EPISODE_LEN, ScriptedPolicy
from tote_sim import ToteEnv

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=1)
p.add_argument("--seed0", type=int, default=0)
p.add_argument("--video", default=None)
p.add_argument("--verbose", action="store_true")
args = p.parse_args()

env = ToteEnv("ee", cameras=())
ok = 0
for ep in range(args.n):
    seed = args.seed0 + ep
    env.reset(seed=seed)
    pol = ScriptedPolicy(env, np.random.RandomState(seed))
    frames = []
    for t in range(EPISODE_LEN + 20):
        _, _, info = env.step(pol(t))
        if args.verbose and t % 20 == 0:
            a = env.box_adr
            print(t, "stage", info["stage"], "box", env.data.qpos[a : a + 3].round(3), "tilt", round(info["tilt"], 1), "inside", info["items_inside"])
        if args.video and ep == 0 and t % 2 == 0:
            frames.append(np.concatenate([env.render("angle", 480, 360), env.render("top", 480, 360)], axis=1))
    print(f"seed {seed}: success={info['is_success']} max_stage={info['max_stage']} spilled={info['spilled']} max_tilt={info['max_tilt']:.1f}")
    ok += info["is_success"]
    if frames:
        imageio.mimsave(args.video, frames, fps=25)
print(f"success {ok}/{args.n}")

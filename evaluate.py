"""学習済み ACT を、学習に使っていない初期配置で実行して成績を集計する。

例:
  python evaluate.py --ckpt outputs/act_tote/checkpoints/100000 --n 50 --out results/act_100k
  python evaluate.py --scripted --n 50 --out results/scripted   # 比較用: お手本スクリプト
  python evaluate.py --ckpt ... --push 200 --out results/act_push200  # 外乱テスト(運搬中に箱を横から押す)
"""

import argparse
import json
from pathlib import Path

import imageio
import numpy as np

from scripted_policy import EPISODE_LEN, ScriptedPolicy
from tote_sim import ToteEnv

H, W = 240, 320
CAM = "overhead"
MAX_STEPS = EPISODE_LEN + 100

p = argparse.ArgumentParser()
p.add_argument("--ckpt", help="checkpoints/<step> フォルダ")
p.add_argument("--scripted", action="store_true", help="お手本スクリプト(関節再生)で評価")
p.add_argument("--n", type=int, default=50)
p.add_argument("--seed0", type=int, default=10000, help="学習データは 1000 番台なので重ならない")
p.add_argument("--ensemble", type=float, default=None, help="ACT の temporal ensembling 係数(例 0.01)")
p.add_argument("--push", type=float, default=0.0, help="運搬中(ステップ 300〜309、0.2 秒間)に箱へ加える左右方向の力 [N]")
p.add_argument("--videos", type=int, default=5, help="動画を保存するエピソード数")
p.add_argument("--vw", type=int, default=480, help="動画1画面の横幅(縦は 3/4)。MuJoCo の既定上限は 640")
p.add_argument("--out", required=True)
args = p.parse_args()

out = Path(args.out)
(out / "videos").mkdir(parents=True, exist_ok=True)

if not args.scripted:
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    path = Path(args.ckpt) / "pretrained_model"
    policy = ACTPolicy.from_pretrained(path)
    if args.ensemble is not None:
        policy.config.temporal_ensemble_coeff = args.ensemble
        policy.config.n_action_steps = 1
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

        policy.temporal_ensembler = ACTTemporalEnsembler(args.ensemble, policy.config.chunk_size)
    policy.eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(path))
    device = policy.config.device

env = ToteEnv("joint", width=W, height=H, cameras=(CAM,))
ee = ToteEnv("ee", cameras=()) if args.scripted else None
results = []
for k in range(args.n):
    seed = args.seed0 + k
    obs = env.reset(seed=seed)
    if args.scripted:  # お手本の関節軌道を作ってから再生する
        ee.reset(seed=seed, layout=env.layout)
        pol = ScriptedPolicy(ee, np.random.RandomState(seed))
        traj = []
        for t in range(MAX_STEPS):
            a = pol(min(t, EPISODE_LEN))
            o, _, _ = ee.step(a)
            q = o["qpos"].copy()
            q[6], q[13] = a[7], a[15]
            traj.append(q)
        obs = env.reset(seed=seed, layout=ee.layout)
    else:
        policy.reset()
    frames = []
    for t in range(MAX_STEPS):
        if args.scripted:
            action = traj[t]
        else:
            batch = {
                f"observation.images.{CAM}": torch.from_numpy(obs["images"][CAM]).permute(2, 0, 1)[None].float() / 255,
                "observation.state": torch.from_numpy(obs["qpos"]).float()[None],
                "task": [""],
            }
            with torch.inference_mode():
                action = post(policy.select_action(pre(batch)))
            action = action.cpu().numpy()[0]
        # 外乱: ステップ 300〜310 の間、箱を x 方向へ押す
        env.data.xfrc_applied[:] = 0
        if args.push and 300 <= t < 310:
            bid = env.model.body("box").id
            env.data.xfrc_applied[bid, 0] = args.push
        obs, _, info = env.step(action)
        if k < args.videos and t % 2 == 0:
            frames.append(np.concatenate([env.render("angle", args.vw, args.vw * 3 // 4), env.render(CAM, args.vw, args.vw * 3 // 4)], axis=1))
        if info["is_success"]:
            break
    r = {"seed": seed, "success": bool(info["is_success"]), "max_stage": info["max_stage"], "spilled": info["spilled"], "max_tilt": round(info["max_tilt"], 2), "steps": t + 1}
    results.append(r)
    print(r, flush=True)
    if frames:
        tag = "ok" if r["success"] else f"fail_stage{r['max_stage']}"
        imageio.mimsave(out / "videos" / f"ep{k:02d}_seed{seed}_{tag}.mp4", frames, fps=25, macro_block_size=8)

n = len(results)
summary = {
    "n": n,
    "success_rate": sum(r["success"] for r in results) / n,
    # 段階ごとの到達率: 1=両手でつかんだ 2=持ち上げた 3=棚の上 4=成功
    "reach_rate": {s: sum(r["max_stage"] >= s for r in results) / n for s in (1, 2, 3, 4)},
    "spill_rate": sum(r["spilled"] > 0 for r in results) / n,
    "mean_max_tilt": float(np.mean([r["max_tilt"] for r in results])),
    "args": vars(args),
}
(out / "results.json").write_text(json.dumps({"summary": summary, "episodes": results}, indent=1))
print(json.dumps(summary, indent=1))

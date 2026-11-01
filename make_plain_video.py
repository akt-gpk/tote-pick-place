"""解説なし・等速で、学習済み ACT がタスクを実行する場面だけをつなげた動画を作る。

例:
  python evaluate.py --ckpt outputs/act_tote/checkpoints/100000 --n 7 --videos 7 --vw 640 --out results/act_100k_plain
  python make_plain_video.py --src results/act_100k_plain --out submission/tote_pick_place_plain.mp4
"""

import argparse
from pathlib import Path

import imageio

FPS = 25  # 評価動画は 50Hz のシミュレーションを 1 コマおきに保存しているので、25fps で等速
HOLD = 0.5  # 各回の最後のコマを止めて見せる秒数

p = argparse.ArgumentParser()
p.add_argument("--src", required=True)
p.add_argument("--out", default="submission/tote_pick_place_plain.mp4")
args = p.parse_args()

vids = sorted(v for v in (Path(args.src) / "videos").glob("*_ok.mp4"))
w = imageio.get_writer(args.out, fps=FPS, macro_block_size=8, quality=8)
n = 0
for v in vids:
    for f in imageio.get_reader(v):
        w.append_data(f)
        n += 1
    for _ in range(int(HOLD * FPS)):
        w.append_data(f)
        n += 1
w.close()
print(f"{args.out}: {len(vids)} 回, {n / FPS:.1f} 秒")

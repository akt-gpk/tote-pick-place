"""提出用動画(1〜3分)を組み立てる。テロップは PIL で描く(この ffmpeg には drawtext がないため)。

例: python make_video.py --act results/act_100k --scripted results/scripted --push 100 200 300 --out submission/tote_pick_place.mp4
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 25
FONT = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
FONT_R = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
BG = (18, 22, 30)
ACCENT = (80, 160, 255)


def font(size, bold=True):
    return ImageFont.truetype(FONT if bold else FONT_R, size)


def card(lines, sub=None):
    """文字だけの画面。lines: [(文字, 大きさ, 色)]"""
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    total = sum(s * 1.5 for _, s, _ in lines)
    y = (H - total) / 2
    for text, size, color in lines:
        f = font(size)
        w = d.textlength(text, font=f)
        d.text(((W - w) / 2, y), text, font=f, fill=color)
        y += size * 1.5
    return np.array(im)


def overlay(frame, caption, badge=None, chapter=None):
    """動画の1コマを 1280×720 の中央に置き、下にテロップ・右上に倍速表示を付ける。"""
    im = Image.new("RGB", (W, H), BG)
    src = Image.fromarray(frame)
    scale = min(W / src.width, (H - 110) / src.height)
    src = src.resize((int(src.width * scale), int(src.height * scale)))
    im.paste(src, ((W - src.width) // 2, (H - 110 - src.height) // 2 + 10))
    d = ImageDraw.Draw(im)
    f = font(32)
    w = d.textlength(caption, font=f)
    d.text(((W - w) / 2, H - 80), caption, font=f, fill=(255, 255, 255))
    if badge:
        bf = font(30)
        bw = d.textlength(badge, font=bf)
        d.rounded_rectangle([W - bw - 50, 20, W - 20, 70], 10, fill=(220, 60, 60))
        d.text((W - bw - 35, 26), badge, font=bf, fill=(255, 255, 255))
    if chapter:
        cf = font(26)
        cw = d.textlength(chapter, font=cf)
        d.rounded_rectangle([20, 20, cw + 50, 66], 10, fill=ACCENT)
        d.text((35, 26), chapter, font=cf, fill=(255, 255, 255))
    return np.array(im)


def clip(path, caption, speed=2, chapter=None, max_sec=None):
    """評価動画(25fps=実時間)を speed 倍速で読み込む。"""
    frames = []
    for i, f in enumerate(imageio.get_reader(path)):
        if i % speed == 0:
            frames.append(overlay(f, caption, badge=f"{speed}倍速" if speed > 1 else None, chapter=chapter))
        if max_sec and len(frames) >= max_sec * FPS:
            break
    return frames


def bar_chart(title, groups, labels, colors, note=None):
    """段階別の到達率を棒グラフで描く。groups: {系列名: [値(0〜1)...]}"""
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    tf = font(44)
    d.text(((W - d.textlength(title, font=tf)) / 2, 40), title, font=tf, fill=(255, 255, 255))
    x0, y0, x1, y1 = 140, 150, W - 80, H - 150
    d.line([x0, y1, x1, y1], fill=(180, 180, 180), width=2)
    for v in (0, 0.5, 1.0):
        y = y1 - (y1 - y0) * v
        d.line([x0, y, x1, y], fill=(60, 66, 80), width=1)
        d.text((x0 - 80, y - 14), f"{int(v*100)}%", font=font(24, False), fill=(200, 200, 200))
    n = len(labels)
    gw = (x1 - x0) / n
    names = list(groups)
    bw = gw * 0.7 / len(names)
    for i, lab in enumerate(labels):
        for j, name in enumerate(names):
            v = groups[name][i]
            bx = x0 + gw * i + gw * 0.15 + bw * j
            by = y1 - (y1 - y0) * v
            d.rectangle([bx, by, bx + bw - 6, y1], fill=colors[j])
            t = f"{v*100:.0f}%"
            d.text((bx + (bw - 6 - d.textlength(t, font=font(24))) / 2, by - 34), t, font=font(24), fill=(255, 255, 255))
        lf = font(26, False)
        d.text((x0 + gw * i + (gw - d.textlength(lab, font=lf)) / 2, y1 + 14), lab, font=lf, fill=(230, 230, 230))
    lx = x0
    for j, name in enumerate(names):
        d.rectangle([lx, H - 80, lx + 26, H - 54], fill=colors[j])
        d.text((lx + 36, H - 84), name, font=font(26, False), fill=(230, 230, 230))
        lx += 60 + d.textlength(name, font=font(26, False))
    if note:
        d.text((lx + 40, H - 84), note, font=font(24, False), fill=(170, 170, 170))
    return np.array(im)


def loss_chart(csv_path, ckpt_dir):
    """csv は [損失, 記録時刻] の行。ステップ数は、チェックポイントの保存時刻(=10,000 ステップごと)から時刻→ステップの直線を引いて求める。"""
    rows = [r for r in csv.reader(open(csv_path)) if r and r[0] not in ("None", "")]
    loss = np.array([float(r[0]) for r in rows])
    t = np.array([datetime.fromisoformat(r[1]).timestamp() for r in rows])
    ck = [(d.stat().st_mtime, int(d.name)) for d in Path(ckpt_dir).iterdir() if d.name.isdigit()]
    slope, icpt = np.polyfit([c[0] for c in ck], [c[1] for c in ck], 1)
    steps = np.clip(slope * t + icpt, 0, max(c[1] for c in ck))
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    title = "学習曲線(ACT の損失)"
    d.text(((W - d.textlength(title, font=font(44))) / 2, 40), title, font=font(44), fill=(255, 255, 255))
    x0, y0, x1, y1 = 160, 150, W - 80, H - 130
    lmax = float(np.percentile(loss, 99)) if len(loss) else 1
    d.line([x0, y1, x1, y1], fill=(180, 180, 180), width=2)
    d.line([x0, y0, x0, y1], fill=(180, 180, 180), width=2)
    smax = steps.max() if len(steps) else 1
    pts = [(x0 + (x1 - x0) * s / smax, y1 - (y1 - y0) * min(l, lmax) / lmax) for s, l in zip(steps, loss)]
    if len(pts) > 1:
        d.line(pts, fill=ACCENT, width=4)
    for v in (0, 0.5, 1.0):
        y = y1 - (y1 - y0) * v
        d.text((x0 - 100, y - 14), f"{lmax*v:.2f}", font=font(24, False), fill=(200, 200, 200))
    for v in (0, 0.5, 1.0):
        x = x0 + (x1 - x0) * v
        d.text((x - 40, y1 + 12), f"{int(smax*v):,}", font=font(24, False), fill=(200, 200, 200))
    d.text(((W - 200) / 2, H - 70), "学習ステップ", font=font(26, False), fill=(220, 220, 220))
    return np.array(im)


def still(img, sec):
    return [img] * int(sec * FPS)


class Writer:
    """F += コマのリスト で、受け取ったコマをすぐ動画ファイルへ書き出す(全コマをメモリに置くと数 GB になるため)。"""

    def __init__(self, path):
        self.w = imageio.get_writer(path, fps=FPS, macro_block_size=8, quality=8)
        self.n = 0

    def __iadd__(self, frames):
        for f in frames:
            self.w.append_data(f)
        self.n += len(frames)
        return self

    def close(self):
        self.w.close()


def pick_videos(result_dir, ok=True):
    vids = sorted((Path(result_dir) / "videos").glob("*.mp4"))
    return [v for v in vids if ("_ok" in v.name) == ok]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--act", required=True)
    p.add_argument("--scripted", required=True)
    p.add_argument("--push", type=int, nargs="+", default=None, help="外乱テストの力 [N](results/act_push<力>, scripted_push<力> を読む)")
    p.add_argument("--results", default="results")
    p.add_argument("--loss", default="outputs/loss_history.csv")
    p.add_argument("--ckpts", default="outputs/act_tote/checkpoints")
    p.add_argument("--out", default="submission/tote_pick_place.mp4")
    args = p.parse_args()

    act = json.load(open(Path(args.act) / "results.json"))["summary"]
    scr = json.load(open(Path(args.scripted) / "results.json"))["summary"]
    W_, B_, G_ = (255, 255, 255), ACCENT, (170, 170, 170)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    F = Writer(out)
    F += still(card([("双腕ロボットによる", 56, W_), ("中身入り物流コンテナの棚入れ", 56, W_), ("模倣学習(ACT)× MuJoCo シミュレーション", 30, G_)]), 4)
    F += still(card([
        ("課題設定", 48, B_),
        ("実在の物流コンテナ(三甲 ラボボックス No.20)を実寸で再現", 30, W_),
        ("外寸 475×369×154 mm / 中身 4 個・合計 2 kg", 30, W_),
        ("片腕では持てない大きさ → 両腕で左右の取っ手をつまんで運ぶ", 30, W_),
        ("成功条件: 奥の棚に置き、中身を 1 個もこぼさないこと", 30, W_),
    ]), 7)
    F += still(card([
        ("データ収集", 48, B_),
        ("手先の目標位置をつなぐお手本プログラムを自作", 30, W_),
        ("箱の位置・向き・中身の配置をランダムに変えて 50 本を自動生成", 30, W_),
        ("関節制御で再生して成功したものだけを LeRobot 形式で保存", 30, W_),
    ]), 6)
    F += clip(pick_videos(args.scripted)[0], "お手本(スクリプト)の動き  ※データ収集用", speed=4, chapter="データ収集")
    if Path(args.loss).exists():
        F += still(loss_chart(args.loss, args.ckpts), 5)
    F += still(card([
        ("学習済み AI による実行", 48, B_),
        ("入力: 俯瞰カメラ画像 + 両腕 14 関節の角度", 30, W_),
        ("出力: この先 100 ステップ分の関節目標(ACT)", 30, W_),
        ("以降は学習に使っていない初期配置での実行", 30, W_),
    ]), 5)
    for v in pick_videos(args.act)[:3]:
        F += clip(v, "学習済み ACT がコンテナを棚へ運ぶ(未学習の初期配置)", speed=2, chapter="AI 実行")
    labels = ["両手でつかむ", "持ち上げる", "棚の上へ", "置いてこぼさない"]
    F += still(bar_chart(
        f"未学習の初期配置 {act['n']} 回での到達率",
        {"学習済み ACT": [act["reach_rate"][str(s)] for s in (1, 2, 3, 4)], "お手本スクリプト": [scr["reach_rate"][str(s)] for s in (1, 2, 3, 4)]},
        labels, [ACCENT, (140, 140, 150)],
        note=f"こぼれ率 ACT {act['spill_rate']*100:.0f}% / 平均最大傾き {act['mean_max_tilt']:.0f}°",
    ), 7)
    fails = pick_videos(args.act, ok=False)
    if fails:
        F += clip(fails[0], "失敗例: どの段階で失敗したかを記録し分析", speed=2, chapter="失敗例", max_sec=8)
    if args.push:
        rate = lambda d: json.load(open(Path(d) / "results.json"))["summary"]
        ap = [rate(f"{args.results}/act_push{f}") for f in args.push]
        sp = [rate(f"{args.results}/scripted_push{f}") for f in args.push]
        F += still(card([
            ("外乱テスト", 48, B_),
            ("運搬中に、箱を左右方向へ 0.2 秒間押す", 30, W_),
            ("学習済み ACT(カメラと関節角度を見て動く)", 30, W_),
            ("vs お手本スクリプト(決まった軌道をなぞるだけ)", 30, W_),
        ]), 5)
        F += still(bar_chart(
            f"押す力ごとの成功率(各 {ap[0]['n']} 回)",
            {"学習済み ACT": [a["success_rate"] for a in ap], "お手本スクリプト": [a["success_rate"] for a in sp]},
            [f"{f} N" for f in args.push], [ACCENT, (140, 140, 150)],
        ), 7)
        top = f"{args.results}/act_push{args.push[-1]}"
        pv = pick_videos(top) or pick_videos(top, ok=False)
        if pv:
            F += clip(pv[0], f"学習済み ACT: 運搬中に {args.push[-1]} N で押される", speed=2, chapter="外乱テスト", max_sec=8)
        for f in reversed(args.push):
            sv = pick_videos(f"{args.results}/scripted_push{f}", ok=False)
            if sv:
                F += clip(sv[0], f"お手本スクリプト: {f} N で押されると立て直せない", speed=2, chapter="外乱テスト", max_sec=8)
                break
    F += still(card([
        ("まとめ", 48, B_),
        ("実寸コンテナ × 中身入り × 双腕協調の棚入れを、データ生成から学習まで自作", 28, W_),
        (f"学習済み ACT の成功率 {act['success_rate']*100:.0f}%(お手本 {scr['success_rate']*100:.0f}%)", 34, W_),
        *([(f"外乱 {args.push[-1]} N: ACT {ap[-1]['success_rate']*100:.0f}% / お手本 {sp[-1]['success_rate']*100:.0f}%", 30, W_)] if args.push else []),
        ("限界: シミュレーションのみ / 指と手首の力は実機より強めに設定", 26, G_),
    ]), 6)

    F.close()
    print(f"{out}: {F.n/FPS:.1f} 秒")

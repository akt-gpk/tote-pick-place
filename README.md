# tote-pick-place — 双腕ロボットによる中身入り物流コンテナの棚入れ

Physical AI 応用1講座 最終課題。MuJoCo 上の双腕 ViperX(ALOHA)が、実寸の物流コンテナ
(三甲 ラボボックス No.20:外寸 475×369×154 mm)を左右の取っ手でつまんで持ち上げ、
中身(5cm 角 ×4 個・合計 2 kg)をこぼさずに奥の棚へ置く。お手本はスクリプトで自動生成し、
模倣学習(ACT)で学習した方策が、学習に使っていない初期配置でタスクを実行する。

## ファイル

| ファイル | 役割 |
|---|---|
| `tote_sim.py` | MuJoCo 環境(箱・中身・棚・カメラ、成功判定・こぼれ数・傾き角の計測)。`joint`(関節制御)と `ee`(手先制御)の2モード |
| `scripted_policy.py` | お手本スクリプト(手先のウェイポイントを補間) |
| `try_scripted.py` | お手本の成功率確認 |
| `replay_test.py` | 重さ試験(お手本の関節軌道を関節制御で再生) |
| `make_dataset.py` | LeRobot 形式(v3.0)のデータセット生成 |
| `evaluate.py` | 学習済み ACT / お手本の評価(成功率・段階別到達率・こぼれ率・傾き、動画保存、外乱テスト) |
| `make_video.py` | 提出用動画の組み立て(テロップ付き) |
| `make_plain_video.py` | 解説なし・等速の実行動画(成功した回だけをつなぐ) |
| `docs/workflow.md` | 環境構築から学習・推論・動画の録画までの詳しい手順と、各ステップの説明 |
| `assets/` | gym-aloha 0.1.4 から複製したロボットモデル(Apache License 2.0、`assets/LICENSE_gym-aloha`)。`_tote_*.xml` は実行時に自動生成 |
| `tools/train_watch.py` | 学習を見張るラッパー(NaN・発散・停止を通知)。任意 |
| `results/*/results.json` | 評価結果(成功率・段階別到達率・こぼれ率・傾き) |
| `submission/` | 提出用の動画(解説なし・等速)・説明資料(PDF と元の Markdown) |

## 再現手順(Mac / Apple Silicon)

各ステップの説明・確かめ方・つまずきやすい点は [docs/workflow.md](docs/workflow.md) を参照。

```bash
conda activate lerobot   # LeRobot 0.4.4, gym-aloha 0.1.4, MuJoCo 3.8.1
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd tote-pick-place

python try_scripted.py --n 20 --seed0 100                 # お手本の確認
python replay_test.py --mass 3.0 --n 5                    # 中身 3kg でも持てるか
python make_dataset.py --n 50                             # データ生成(約3分)
lerobot-train --dataset.repo_id=local/tote_pick_place --dataset.root=data/tote_pick_place \
  --policy.type=act --policy.device=mps --policy.push_to_hub=false \
  --steps=100000 --save_freq=10000 --batch_size=8 --num_workers=4 \
  --output_dir=outputs/act_tote --wandb.enable=false     # 学習(約2.7時間)
python evaluate.py --ckpt outputs/act_tote/checkpoints/100000 --n 50 --out results/act_100k
python evaluate.py --scripted --n 50 --out results/scripted
python evaluate.py --ckpt outputs/act_tote/checkpoints/100000 --n 20 --push 200 --out results/act_push200   # 外乱テスト(100/300 N も同様)
python evaluate.py --scripted --n 20 --push 200 --out results/scripted_push200
python make_video.py --act results/act_100k --scripted results/scripted --push 100 200 300 --out submission/tote_pick_place.mp4
python evaluate.py --ckpt outputs/act_tote/checkpoints/100000 --n 7 --videos 7 --vw 640 --out results/act_100k_plain
python make_plain_video.py --src results/act_100k_plain --out submission/tote_pick_place_plain.mp4   # 解説なし・等速
```

## シミュレーション上の設定と限界

- gym-aloha の値から、次の関節モーターを強めている(元の値では 2.5 kg の箱が滑り落ちる/運搬中に手首が負けて取っ手が抜けるため)。
  - 指: kp 200→2000(10 倍)
  - 手首3軸(forearm_roll / wrist_angle / wrist_rotate): kp 10/50/20 → 200/300/200
  - 手首3軸に回転の慣性(armature 0.05)を追加。強いモーター×軽い関節だとシミュレーションが発散するのを防ぐため
- 成功判定は厳しめ: 箱が棚の中央付近(前後 ±5 cm・左右 ±8 cm)にあり、両手を離し、中身 4 個すべてが箱の中、傾き 5° 未満、箱がほぼ静止(0.02 m/s 未満)の状態が **0.5 秒(25 ステップ)続いたら** 成功。
- 箱の質量(0.5 kg)と壁の厚み(8 mm)は製品ページに記載がないため仮値。取っ手は製品の手掛け部をつまめる形に単純化した。
- お手本生成(`ee` モード)では手先が目標どおり動くよう腕の関節摩擦を外している。データとして保存するのは、摩擦ありの関節制御モードで再生して成功した軌道だけ。

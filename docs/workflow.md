# 作業手順書 — 環境構築から学習・推論・動画の録画まで

このプロジェクト(双腕ロボットによる中身入りコンテナの棚入れ)を、ゼロから再現するための手順と、各ステップで何が起きているかの説明。
コマンドは Mac(Apple Silicon)で実際に動かしたもの。所要時間は M5 Pro(18 コア / 64 GB)での実測。

## 全体の流れ

```
① 環境構築
   ↓
② お手本スクリプトの確認 ── try_scripted.py / replay_test.py
   ↓
③ データ生成(お手本 50 本)── make_dataset.py        約 3 分
   ↓
④ ACT の学習(10 万ステップ)── lerobot-train         約 2.7 時間
   ↓
⑤ 評価(推論 50 回+外乱テスト)── evaluate.py          約 20 分
   ↓
⑥ 動画の録画・編集 ── evaluate.py --videos → make_plain_video.py / make_video.py
```

| 用語 | 意味 |
|---|---|
| お手本(scripted policy) | 人の代わりに、プログラムで決めた通りに動くロボットの動き。これを学習データにする |
| ACT | Action Chunking with Transformers。カメラ画像と関節角を見て、この先の動き(関節目標の列)をまとめて出す模倣学習モデル |
| 推論 | 学習済みモデルに入力を与えて、出力(次の動き)を計算させること |
| エピソード | 1 回分の試行(箱をつかんで棚に置くまで) |
| seed | 乱数の種。同じ seed なら箱の位置などが毎回同じになる |

---

## ① 環境構築

```bash
brew install --cask miniforge          # conda(Python の環境を分けて管理する道具)
brew install ffmpeg                    # 動画の書き出しに使う
conda init zsh                         # 初回だけ。実行後にターミナルを開き直す
conda create -y -n lerobot python=3.10
conda activate lerobot
pip install torch torchvision          # Apple GPU(MPS)対応版が入る
pip install "lerobot[aloha]==0.4.4" mujoco==3.8.1   # LeRobot 本体+MuJoCo。版を固定する(下の注意)
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"   # MPS: True なら OK
```

このプロジェクトで使ったバージョン: Python 3.10.20 / PyTorch 2.10.0 / LeRobot 0.4.4 / gym-aloha 0.1.4 / MuJoCo 3.8.1 / ffmpeg 8.1.2

- **版を固定する理由**: LeRobot は版が上がるとデータ形式やモデルの保存形式が変わることがある(0.4 系で実際に変わった)。版を指定しないと、その時点の最新版が入って動かない可能性がある。
  ※この環境は 2026-07 に版を指定せずに入れたもの。上の「版を指定した入れ方」で新しく作り直す確認はしていない(未確認)。
- LeRobot を入れると PyTorch が LeRobot の対応版(2.10.0)に入れ替わることがある。MPS はそのまま使える。
- 動画まわり(PyAV・imageio-ffmpeg)は LeRobot と一緒に入る。

毎回、作業前に次を実行する。

```bash
conda activate lerobot
export PYTORCH_ENABLE_MPS_FALLBACK=1   # Apple GPU が未対応の計算だけ CPU に回す
cd tote-pick-place
```

**ロボットのモデル**: `assets/` に gym-aloha 0.1.4 の ViperX(vx300s)双腕モデルを複製してある(コードは gym-aloha を import せず、この複製を直接読む)。箱・中身・棚を足したシーンの XML(`assets/_tote_*.xml`)は `tote_sim.py` が実行のたびに自動で作る。

**シミュレーションの 2 つのモード**(`tote_sim.py` の `ToteEnv`):

| モード | 動かし方 | 使う場面 |
|---|---|---|
| `ee` | 手先(グリッパー)の位置・向きを mocap で直接動かす。腕の関節摩擦なし | お手本スクリプト |
| `joint` | 14 個の関節(両腕 × 6 関節+指 1)に目標角度を与え、モーターで動かす | データの保存・学習・評価 |

関節モーターの強さは gym-aloha から一部強めている(指・手首。理由は README の「シミュレーション上の設定と限界」)。

---

## ② お手本スクリプトの確認

```bash
python try_scripted.py --n 20 --seed0 100              # お手本を 20 回動かして成功率を見る
python try_scripted.py --n 1 --seed0 100 --video scripted_check.mp4    # 1 回分を動画で見る
python replay_test.py --mass 3.0 --n 5                 # 中身 3 kg でも関節制御で持てるか
```

- お手本スクリプト(`scripted_policy.py`)は、手先(グリッパー)の通過点を時刻表どおりにつなぐ単純な方式。通過点の間は、動き始めと止まる前をゆっくりにする補間(なめらかな加減速。中身がこぼれにくい)で動かす。
  時刻表(50Hz なので 500 ステップ = 10 秒):

  | ステップ | 0 | 60 | 110 | 140 | 230 | 370 | 430 | 460 | 500 |
  |---|---|---|---|---|---|---|---|---|---|
  | 動作 | 開始 | 取っ手の真上へ | 指を開いて下ろす | つまむ | 持ち上げる(13 cm) | 棚の上へ運ぶ | 下ろす | 離す | 手を引く |

- 通過点は、箱の位置と向きから「左右の取っ手のどこをつまむか」を計算して決める。全部が同じ動きにならないよう、毎回 ±5 mm のばらつきを足す。
- `try_scripted.py` は `ee` モード(mocap)だけで動かす。動画は斜めのカメラと上からのカメラを並べたもの。`--verbose` で 20 ステップごとの段階・箱の位置・傾きを表示する。
- `replay_test.py` は、`ee` モードで記録した関節の軌道を `joint` モードで再生し、重い中身でも同じように運べるかを見る。`joint_err_max` は、目標の関節角と実際の関節角の最大のずれ(ラジアン)。
- seed はステップごとに分けている: お手本の確認 100 番台・重さ試験 200 番台・データ 1000 番台・評価 10000 番台。
- **確かめること**: 成功率が十分高いこと、動画で箱が水平のまま棚に置かれていること。

---

## ③ データ生成

```bash
python make_dataset.py --n 50          # 約 3 分 → data/tote_pick_place/
```

> **注意**: 実行すると、既にある `data/tote_pick_place/` を**丸ごと削除してから**作り直す。試しに動かすときは `--root <別のフォルダ>` を付ける。

お手本を 2 段階で作る(`make_dataset.py`)。

1. **手先を直接動かす環境(`ee` モード)でお手本を動かす。**
   MuJoCo の mocap(手先を目標位置へ引っぱる仕組み)でグリッパーを動かし、毎ステップの関節角を記録する。
2. **同じ初期配置で、記録した関節角を「関節の目標」として再生する(`joint` モード)。**
   学習・評価と同じ条件(関節モーターで動かす、関節の摩擦あり)で動かし直し、そのときの
   - 真上カメラ(`overhead`)の画像(縦 240 × 横 320)
   - 現在の関節角 14 次元(`observation.state`)
   - 次の関節目標 14 次元(`action`。指だけは実際の開きではなく「開け閉めの指令値」を記録する)

   を毎ステップ保存する。14 次元の並びは左腕(waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, gripper)→ 右腕の同じ順。
   1 本は 520 ステップ(お手本の 500 ステップ+置いた後に止まっている 20 ステップ)。
3. **再生でも成功したエピソードだけを残す。** 成功の条件は README の「成功判定」と同じ(棚の中央付近・両手を離す・中身 4 個が箱の中・傾き 5° 未満・ほぼ静止、が 0.5 秒続く)。失敗した seed は飛ばし、50 本たまるまで次の seed を試す。

各エピソードには文章の指示(task)「Lift the container with both arms and place it on the shelf without spilling」が付く。全エピソード同じ文で、ACT はこの文を使わない(LeRobot 形式で必須の項目なので入れている)。

なぜ 2 段階か: mocap は手先を力ずくで引っぱるので、本物の関節モーターでは出せない動きも通ってしまう。
ACT が出すのは関節目標なので、「関節目標として与えて本当に成功した動き」だけを教えないと、真似しても再現できないデータが混ざる。

毎回変えるもの(`tote_sim.py` の `sample_layout`): 箱の位置(前後左右 ±3 cm)、向き(±5°)、中身 4 個の置き方(2×2 のマスをシャッフルし、マス内で ±4 cm ずらす)。

**できるもの**: `data/tote_pick_place/`(LeRobot v3.0 形式)。今回の実績は seed 1000〜1049 の 50 回で、50 本すべて成功、1 本 520 フレーム(10.4 秒)、合計 26,000 フレーム。
`generation_log.json` に、採用したエピソードの seed・最大傾き・こぼれ数が残る。

**確かめること**: `meta/info.json` の `total_episodes` が 50 になっていること。

---

## ④ ACT の学習

```bash
lerobot-train --dataset.repo_id=local/tote_pick_place --dataset.root=data/tote_pick_place \
  --policy.type=act --policy.device=mps --policy.push_to_hub=false \
  --steps=100000 --save_freq=10000 --batch_size=8 --num_workers=4 \
  --output_dir=outputs/act_tote --wandb.enable=false
```

- 約 2.7 時間(1 万ステップあたり約 16 分)。ACT の設定は LeRobot の既定値のまま(1 回に 100 ステップ = 2 秒先までの動きをまとめて予測する)。
- 今回は実際には、学習を見張る自作のラッパー `tools/train_watch.py` 経由で、`--log_freq=200` を付けて動かした。ラッパーは NaN・発散・停止を見つけると Mac に通知するだけで、学習の中身は上のコマンドと同じ。

  ```bash
  python tools/train_watch.py -- <上の lerobot-train の引数そのまま> --log_freq=200
  ```

- 解説付き動画の loss のグラフ(`outputs/loss_history.csv`、1 行 = 「loss,時刻」)は、学習中に別のターミナルで次のループを回して記録した。ラッパーが書く `outputs/act_tote_watch/watch_status.json` から 20 秒ごとに loss を読む。ラッパーを使わない場合はこのファイルができず、解説付き動画ではグラフの場面が飛ばされる。

  ```bash
  while kill -0 <学習のプロセス番号> 2>/dev/null; do
    python3 -c "import json;d=json.load(open('outputs/act_tote_watch/watch_status.json'));print(f\"{d['loss']},{d['updated_at']}\")" >> outputs/loss_history.csv
    sleep 20
  done
  ```

- `--policy.push_to_hub=false`: これがないと Hugging Face へのアップロード先を求められてエラーになる。
- `--output_dir` が既にあると即エラーで止まる。やり直すときは別の名前にする。
- 1 万ステップごとに `outputs/act_tote/checkpoints/<ステップ数>/` に途中経過のモデルが保存される。
- **Mac がスリープすると学習が止まる**。別のターミナルで `caffeinate -i -w <学習のプロセス番号>` を動かしておくと、学習中はスリープしない。
- 途中で落ちたときは、最後のチェックポイントから再開できる。

  ```bash
  lerobot-train --config_path=outputs/act_tote/checkpoints/last/pretrained_model/train_config.json --resume=true
  ```

**できるもの**: `checkpoints/<ステップ数>/pretrained_model/` の中に、モデルの重み(`model.safetensors`)、設定(`config.json`・`train_config.json`)、入出力の正規化(数値の大きさをそろえる変換)の設定と統計(`policy_preprocessor*`・`policy_postprocessor*`)が入る。推論ではこのフォルダだけを使う。

**確かめること**: ログの loss が下がり続けていること(今回は約 6.5 → 0.017)、`checkpoints/100000/pretrained_model/` ができていること。

---

## ⑤ 評価(推論)

```bash
python evaluate.py --ckpt outputs/act_tote/checkpoints/100000 --n 50 --out results/act_100k   # ACT を 50 回(約 3 分)
python evaluate.py --scripted --n 50 --out results/scripted                               # 比較: お手本
# 外乱テスト(各約 2.5 分)。解説付き動画には 100 / 200 / 300 N の 6 つ全部が必要
for f in 100 200 300; do
  python evaluate.py --ckpt outputs/act_tote/checkpoints/100000 --n 20 --push $f --out results/act_push$f
  python evaluate.py --scripted --n 20 --push $f --out results/scripted_push$f
done
```

推論の中身(`evaluate.py`):

1. 学習に使っていない初期配置(seed 10000 番台。データは 1000 番台なので重ならない)で環境をリセットする。
2. 真上カメラの画像(0〜1 の値に直す)と現在の関節角 14 次元を、チェックポイントに保存された正規化の設定(preprocessor)を通して ACT に入れる。
3. ACT は **100 ステップ(2 秒)分の関節目標をまとめて出す**。これを 1 ステップずつロボットに与え、100 個を使い切ったら、そのときの画像と関節角でまた次の 100 ステップを予測する(LeRobot の `select_action` の既定の動き)。出力は postprocessor で元の単位(ラジアンなど)に戻す。
4. 成功するか、最大 600 ステップ(12 秒)に達するまで繰り返す。

- `--scripted`: お手本を `ee` モードで動かして関節の軌道を作り、それを `joint` モードで再生して評価する(データ生成と同じ条件)。ACT と同じ物差しで比べるための基準。
- `--push 200`: 運搬中(ステップ 300〜309、0.2 秒間)に箱を横から 200 N で押し、乱されても立て直せるかを見る。
- `--ensemble 0.01`(任意): 毎ステップ予測し直して重ね合わせる方式(temporal ensembling)で動かす。今回の提出結果では使っていない。
- 結果の今回の実績: ACT 50/50 成功・こぼれ 0%・平均最大傾き 1.0°。外乱 300 N で ACT 40%(こぼれ 0%)、お手本 25%(こぼれ 20%)。
- **できるもの**: `results/<名前>/results.json`(成功率・段階別到達率・こぼれ率・平均最大傾き、各回の結果)と `videos/`(最初の 5 回分の動画)。
- 段階は 1 = 両手でつかんだ、2 = 持ち上げた、3 = 棚の上、4 = 成功。

**確かめること**: 数字だけで判断せず、成功した回の動画のコマを目で見て、本当に棚に置けているかを確認する(過去に、判定の甘さで「成功」と数えられていた失敗があった)。

---

## ⑥ 動画の録画・編集

### 解説なし・等速の動画(提出用)

```bash
python evaluate.py --ckpt outputs/act_tote/checkpoints/100000 --n 7 --videos 7 --vw 640 --out results/act_100k_plain
python make_plain_video.py --src results/act_100k_plain --out submission/tote_pick_place_plain.mp4
```

- 録画: `evaluate.py` が推論しながら、斜めのカメラと真上カメラを横に並べて 1 コマおきに保存する(1 画面 640×480、合わせて 1280×480)。成功した回は `_ok.mp4`、失敗した回は `_fail_stage<段階>.mp4` という名前になる。
- 編集: `make_plain_video.py` が `_ok.mp4` だけを順につなぎ、各回の最後に 0.5 秒止める。50Hz のシミュレーションを 1 コマおきに保存しているので、25 fps で書き出すと等速になる。
- 今回の実績: 7 回すべて成功、69.2 秒。

### 解説付きの動画

```bash
python make_video.py --act results/act_100k --scripted results/scripted --push 100 200 300 --out submission/tote_pick_place.mp4
```

テロップ、学習の loss の推移、ACT とお手本の成績比較、失敗例、外乱テストの場面をつないだ動画(1280×720、79.5 秒)。

- 使うもの: `results/act_100k`・`results/scripted`・`results/act_push{100,200,300}`・`results/scripted_push{100,200,300}` の `results.json` と `videos/`、`outputs/loss_history.csv`(なければグラフを飛ばす)、`outputs/act_tote/checkpoints/`(グラフの横軸を学習ステップに直すのに、各チェックポイントの保存時刻を使う)。
- **倍速の場面がある**: お手本の場面は 4 倍速、ACT・失敗例・外乱の場面は 2 倍速(右上に「◯倍速」と表示)。提出条件は「2 倍速まで可」なので、**4 倍速の場面はそのままだと条件を超える**。提出には解説なし・等速の動画を使う。
- テロップの文字は Mac の「ヒラギノ角ゴシック」を使うので、Mac 以外ではフォントのパス(`make_video.py` の `FONT`)を変える必要がある。

**確かめること**: 再生して、長さが提出条件(1〜3 分)に収まること、最初・途中・最後のコマで箱が正しく運ばれていること。

---

## フォルダの中身

GitHub のリポジトリには `data/`・`outputs/`(合わせて約 6.5 GB)と評価の動画は含めていない。③〜⑥のコマンドで作り直せる。評価結果の数字(`results/*/results.json`)は含めている。

| フォルダ | 中身 |
|---|---|
| `data/tote_pick_place/` | ③のデータセット(LeRobot v3.0 形式)と `generation_log.json` |
| `outputs/act_tote/` | ④の学習結果(チェックポイント)。`act_tote_watch/` はラッパーの監視ファイル、`train_act_tote.log` は学習ログ、`loss_history.csv` は loss の記録 |
| `results/` | ⑤⑥の評価結果と録画。`act_10k` は学習途中(1 万ステップ)の確認、`*.log` は評価中の表示の控え |
| `submission/` | 提出用の動画(解説なし・等速 `tote_pick_place_plain.mp4`)、説明資料(`report.md` と PDF、図 `figures/`)。手元には解説付き動画とレポートの下書きもあるが、リポジトリには含めない |
| `outputs/_old_*`・`results/_old_*` | やり直す前の記録(下の「やり直しの経緯」)。現在の結果には使っていない |

## やり直しの経緯(偽の成功)

最初のデータでは、運搬中に手首の力が足りず取っ手が指から抜け、箱が棚に「落ちた」だけの動きまで成功と数えていた。評価動画を目で見て気づき、
①手首モーターを強化(シミュレーションが発散しないよう関節に慣性も追加)、②成功判定を「棚の中央・傾き 5° 未満・静止が 0.5 秒続く」に厳しくしたうえで、
③データ生成と学習をやり直した。⑤の「動画のコマを目で見る」はこの反省から入れている。

## つまずきやすい点

| 症状 | 原因と対処 |
|---|---|
| `conda: command not found` | `conda init zsh` を実行してターミナルを開き直す |
| `lerobot-train` が「`policy.repo_id` が必要」というエラーで止まる | `--policy.push_to_hub=false` を付け忘れている |
| データを作り直したら前のデータが消えた | `make_dataset.py` は同じフォルダを削除して作り直す。残したいときは `--root` で別の場所に出す |
| 動画の保存で「フォルダがない」エラー | `--video` や `--out` の保存先フォルダは自動では作られないものがある。先に `mkdir` する |
| `lerobot-train` が `FileExistsError` で止まる | `--output_dir` が既にある。別の名前にするか、再開なら `--resume=true` を使う |
| 学習が進まない(朝見たら止まっていた) | Mac のスリープ。`caffeinate` を使う |
| 動画の書き出しで MemoryError | 全コマをメモリに溜めている。1 コマずつ書き出す(`make_video.py` は対応済み) |
| `evaluate.py --vw` を 640 より大きくするとエラー | MuJoCo の描画サイズの既定上限が 640 |

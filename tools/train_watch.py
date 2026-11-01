#!/usr/bin/env python
"""lerobot-train を回しながらログを監視し、異常/完了を検知するラッパー。

使い方:
    python train_watch.py -- <lerobot-train にそのまま渡す引数...>

例:
    python train_watch.py -- \
        --dataset.repo_id=lerobot/pusht --policy.type=act --policy.device=mps \
        --policy.push_to_hub=false --steps=30000 --output_dir=outputs/act_pusht

やること:
    1. lerobot-train を子プロセスとして起動し、ログを流しながら解析する
    2. 異常(NaN, 発散, ストール, クラッシュ)と正常完了を検知する
    3. 検知したら Mac にデスクトップ通知を出し、status.json / alerts.jsonl に書く

役割分担:
    このスクリプトは「働き手」。学習を回して異常を見張り、状態をファイルに残す。
    スマホへのプッシュは Claude Code(Remote Control) が status.json を見て行う。
"""
import argparse
import json
import queue
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# メトリクス行の例:
#   INFO ... ot_train.py:435 step:10 smpl:80 ep:1 epch:0.00 loss:43.901 grdn:753.435 lr:1.0e-05 ...
STEP_RE = re.compile(r"step:(\d+)\b")
LOSS_RE = re.compile(r"loss:([-\d.eE+]+|nan|inf|-inf)")
DONE_MARKER = "End of training"


def now_iso():
    """現在時刻を ISO 文字列で返す(ログ用)。"""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def notify_mac(title, message):
    """Mac のデスクトップに通知を出す(失敗しても学習は止めない)。"""
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification {json.dumps(message)} with title {json.dumps(title)}'],
            check=False, capture_output=True,
        )
    except Exception:
        pass


class Watcher:
    def __init__(self, watch_dir, loss_ceiling, stall_sec):
        self.status_path = watch_dir / "watch_status.json"
        self.alerts_path = watch_dir / "watch_alerts.jsonl"
        self.loss_ceiling = loss_ceiling
        self.stall_sec = stall_sec
        self.state = "running"
        self.step = 0
        self.loss = None
        self.best_loss = None
        self.started_at = now_iso()
        self.last_step_time = time.time()
        self.latest_alert = None
        self.active_alert_kind = None  # 回復するまで同じアラートを二重に出さないための記録

    def write_status(self):
        """今の状態を1ファイルにまとめて書く(Claude/人間が読む用)。"""
        data = {
            "state": self.state,           # running / done / alert / crashed
            "step": self.step,
            "loss": self.loss,
            "best_loss": self.best_loss,
            "started_at": self.started_at,
            "updated_at": now_iso(),
            "latest_alert": self.latest_alert,
        }
        self.status_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def raise_alert(self, kind, detail):
        """異常を記録して通知する。同じ種類が続く間は一度だけ通知する。"""
        if kind == self.active_alert_kind:
            return
        self.active_alert_kind = kind
        self.state = "alert"
        self.latest_alert = {"kind": kind, "detail": detail, "at": now_iso(), "step": self.step}
        with self.alerts_path.open("a") as f:
            f.write(json.dumps(self.latest_alert, ensure_ascii=False) + "\n")
        self.write_status()
        line = f"🚨 ALERT [{kind}] step={self.step} {detail}"
        print(line, flush=True)
        notify_mac("lerobot 学習アラート", f"{kind}: {detail} (step {self.step})")

    def on_metric_line(self, line):
        """loss を含む行から step/loss を取り出し、異常を判定する。"""
        m_step = STEP_RE.search(line)
        m_loss = LOSS_RE.search(line)
        if not m_loss:
            return
        if m_step:
            self.step = int(m_step.group(1))
        self.last_step_time = time.time()
        raw = m_loss.group(1).lower()
        try:
            loss = float(raw)
        except ValueError:
            loss = float("nan")
        self.loss = loss

        # ルール1: NaN/Inf
        if loss != loss or loss in (float("inf"), float("-inf")):
            self.raise_alert("loss_nan", f"loss が異常値({raw})になりました")
            return
        # ルール2: 発散(しきい値超え)
        if loss > self.loss_ceiling:
            self.raise_alert("loss_explode", f"loss={loss:.3f} がしきい値 {self.loss_ceiling} を超えました")
            return
        # 正常に進んでいれば best を更新して running に戻す
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
        if self.state == "alert":
            self.state = "running"
        self.active_alert_kind = None  # 正常な step が来たので回復とみなす
        self.write_status()


def reader_thread(pipe, q):
    """子プロセスの出力を1行ずつキューに流すスレッド。"""
    for line in pipe:
        q.put(line)
    q.put(None)  # 終端の合図


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss-ceiling", type=float, default=1e4,
                    help="loss がこれを超えたら発散とみなす")
    ap.add_argument("--stall-sec", type=float, default=300.0,
                    help="この秒数、新しい step が出なければストールとみなす")
    ap.add_argument("train_args", nargs=argparse.REMAINDER,
                    help="-- の後ろに lerobot-train の引数を書く")
    args = ap.parse_args()

    train_args = args.train_args
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]
    if not train_args:
        print("使い方: python train_watch.py -- <lerobot-train の引数...>", file=sys.stderr)
        sys.exit(2)

    # output_dir を引数から拾う。lerobot-train は output_dir が既存だと中断するので、
    # 監視ファイルはそこに書かず、隣の「<name>_watch」ディレクトリに書く。
    out_dir = Path("outputs")
    for a in train_args:
        if a.startswith("--output_dir="):
            out_dir = Path(a.split("=", 1)[1])
    watch_dir = out_dir.parent / (out_dir.name + "_watch")
    watch_dir.mkdir(parents=True, exist_ok=True)

    watcher = Watcher(watch_dir, args.loss_ceiling, args.stall_sec)
    watcher.write_status()
    print(f"▶ 学習開始: {' '.join(train_args)}", flush=True)
    print(f"  監視状態ファイル: {watcher.status_path}", flush=True)

    proc = subprocess.Popen(
        ["lerobot-train", *train_args],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    q = queue.Queue()
    threading.Thread(target=reader_thread, args=(proc.stdout, q), daemon=True).start()

    done_marker_seen = False
    while True:
        try:
            # stall_sec だけ待って新しい行が来なければストール判定
            line = q.get(timeout=args.stall_sec)
        except queue.Empty:
            if proc.poll() is None:  # まだ生きているのに無反応 = ストール
                watcher.raise_alert("stalled", f"{args.stall_sec:.0f} 秒 step が進みませんでした")
                watcher.last_step_time = time.time()
            continue
        if line is None:
            break
        if "loss:" in line:
            watcher.on_metric_line(line)
        if DONE_MARKER in line:
            done_marker_seen = True

    code = proc.wait()
    if code == 0 and done_marker_seen:
        watcher.state = "done"
        watcher.write_status()
        print(f"✅ DONE step={watcher.step} best_loss={watcher.best_loss}", flush=True)
        notify_mac("lerobot 学習完了", f"完了 step {watcher.step} / best_loss {watcher.best_loss}")
    else:
        watcher.raise_alert("crashed", f"プロセスが異常終了しました(exit={code})")
        sys.exit(code or 1)


if __name__ == "__main__":
    main()

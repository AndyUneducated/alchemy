"""Single training run wrapper around `mlx_lm.lora --train`.

负责：
  1. 把 `data/triples/train_*.jsonl` + `val_*.jsonl` 装到一个 mlx_lm 期望的
     `<dir>/{train,valid}.jsonl` 布局（用 symlink，不复制）；
  2. 调用 `mlx_lm.lora` 子进程，stdout/stderr 落盘 `train.log`；
  3. 解析 log 抽 first/last train loss、last val loss、wall clock、divergence flag，
     写 `train_metrics.json`；
  4. 不自动 fuse / convert（Phase 4 的活）.

行业对位（详见 [`README.md`](README.md)）：
  - `--mask-prompt` 默认开（assistant-only loss），与 [TRL Qwen2.5 训练 template](https://github.com/huggingface/trl/pull/5522) 同思想；
  - 4-bit 底座 (mlx-community/Qwen2.5-7B-Instruct-4bit) → 自动走 QLoRA；
  - tools schema (DECISIONS §4) 由 mlx_lm.lora 内部 `apply_chat_template` 渲染.

用法：
    python train.py --adapter-path runs/smoke --iters 100
    python train.py --adapter-path runs/main --iters 600 \\
        --learning-rate 1e-4 --batch-size 4 --num-layers 16
    python train.py --dry-run     # 只打印命令
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAY_DIR = HERE.parent.parent
DEFAULT_DATA_DIR = HERE.parent / "data" / "triples"
DEFAULT_CONFIG = HERE / "lora_config.yaml"
DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"

# 与 sft_hello/sweep.py 保持同样 regex（mlx_lm.lora log 格式稳定）.
_LOSS_RE = re.compile(r"Iter\s+(\d+):\s+Train loss\s+([\d.]+)")
_VAL_RE = re.compile(r"Iter\s+(\d+):\s+Val loss\s+([\d.]+)")


def setup_data_link_dir(
    triples_dir: Path,
    train_file: str,
    valid_file: str,
    adapter_path: Path,
) -> Path:
    """Build `<adapter_path>/.data/{train,valid}.jsonl` symlinking selected sources.

    mlx_lm.lora 严格要求 `--data <dir>` 下有 `train.jsonl` 与（可选）`valid.jsonl`.
    我们的 jsonl 命名 `train_7b_1k.jsonl` 便于跨实验区分；symlink 让 mlx_lm 看到
    标准名而源文件不动.
    """
    src_train = (triples_dir / train_file).resolve()
    src_valid = (triples_dir / valid_file).resolve()
    if not src_train.exists():
        sys.exit(f"train file not found: {src_train}")
    if not src_valid.exists():
        sys.exit(f"valid file not found: {src_valid}")

    data_dir = adapter_path / ".data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, src in (("train.jsonl", src_train), ("valid.jsonl", src_valid)):
        dst = data_dir / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
    return data_dir


def build_cmd(args: argparse.Namespace, data_dir: Path) -> list[str]:
    cmd = [
        "mlx_lm.lora",
        "--model", args.model,
        "--train",
        "--data", str(data_dir),
        "--config", str(args.config),
        "--iters", str(args.iters),
        "--batch-size", str(args.batch_size),
        "--num-layers", str(args.num_layers),
        "--learning-rate", f"{args.learning_rate:g}",
        "--adapter-path", str(args.adapter_path),
        "--seed", str(args.seed),
    ]
    if args.mask_prompt:
        cmd.append("--mask-prompt")
    if args.grad_checkpoint:
        cmd.append("--grad-checkpoint")
    if args.steps_per_eval:
        cmd.extend(["--steps-per-eval", str(args.steps_per_eval)])
    if args.steps_per_report:
        cmd.extend(["--steps-per-report", str(args.steps_per_report)])
    if args.val_batches:
        cmd.extend(["--val-batches", str(args.val_batches)])
    return cmd


def parse_log(text: str) -> dict:
    train_losses = [(int(i), float(v)) for i, v in _LOSS_RE.findall(text)]
    val_losses = [(int(i), float(v)) for i, v in _VAL_RE.findall(text)]
    nan_seen = "nan" in text.lower() and (
        "loss nan" in text.lower() or "nan," in text.lower()
    )
    return {
        "train_loss_first": train_losses[0][1] if train_losses else None,
        "train_loss_last": train_losses[-1][1] if train_losses else None,
        "train_loss_min": min((v for _, v in train_losses), default=None),
        "val_loss_last": val_losses[-1][1] if val_losses else None,
        "val_loss_min": min((v for _, v in val_losses), default=None),
        "n_train_iters": train_losses[-1][0] if train_losses else 0,
        "n_val_points": len(val_losses),
        "nan_seen": nan_seen,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"HF / local model path (default: {DEFAULT_MODEL})")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR,
                   help=f"directory containing train/val jsonl (default: {DEFAULT_DATA_DIR})")
    p.add_argument("--train-file", default="train_7b_1k.jsonl",
                   help="train jsonl filename inside --data (default: train_7b_1k.jsonl)")
    p.add_argument("--valid-file", default="val_7b_1k.jsonl",
                   help="valid jsonl filename inside --data (default: val_7b_1k.jsonl)")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                   help=f"lora YAML config (default: {DEFAULT_CONFIG})")
    p.add_argument("--adapter-path", type=Path, required=True,
                   help="adapter output dir; will also hold train.log + train_metrics.json")
    p.add_argument("--iters", type=int, default=600,
                   help="total optimizer steps (766 sample / batch 4 ≈ 192 step/epoch; default 600 ≈ 3 epoch)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=16,
                   help="number of top transformer blocks to attach LoRA on (Qwen2.5-7B 共 28 层；default 16)")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-mask-prompt", dest="mask_prompt", action="store_false",
                   help="disable assistant-only loss masking (default: enabled)")
    p.set_defaults(mask_prompt=True)
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="enable gradient checkpointing (saves memory at compute cost)")
    p.add_argument("--steps-per-eval", type=int, default=100,
                   help="run val loss every N steps (default 100)")
    p.add_argument("--steps-per-report", type=int, default=20,
                   help="print train loss every N steps (default 20)")
    p.add_argument("--val-batches", type=int, default=25,
                   help="number of val batches per eval (default 25)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the command and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.adapter_path = args.adapter_path.resolve()
    args.adapter_path.mkdir(parents=True, exist_ok=True)

    data_dir = setup_data_link_dir(
        args.data.resolve(), args.train_file, args.valid_file, args.adapter_path
    )
    cmd = build_cmd(args, data_dir)

    print(f"[train] adapter_path = {args.adapter_path}")
    print(f"[train] cmd = {' '.join(cmd)}")
    if args.dry_run:
        return 0

    log_path = args.adapter_path / "train.log"
    metrics_path = args.adapter_path / "train_metrics.json"

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(PLAY_DIR), capture_output=True, text=True)
    elapsed = time.time() - t0
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    log_path.write_text(output)

    parsed = parse_log(output)
    metrics = {
        "model": args.model,
        "iters": args.iters,
        "batch_size": args.batch_size,
        "num_layers": args.num_layers,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "mask_prompt": args.mask_prompt,
        "train_file": args.train_file,
        "valid_file": args.valid_file,
        "elapsed_s": round(elapsed, 1),
        "returncode": proc.returncode,
        "diverged": parsed["nan_seen"] or proc.returncode != 0,
        **parsed,
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

    print(f"\n[train] done in {elapsed:.1f}s; rc={proc.returncode}")
    print(f"[train] log     → {log_path.relative_to(PLAY_DIR)}")
    print(f"[train] metrics → {metrics_path.relative_to(PLAY_DIR)}")
    if metrics["train_loss_first"] is not None:
        print(f"[train] train loss: first={metrics['train_loss_first']:.3f}  "
              f"last={metrics['train_loss_last']:.3f}  "
              f"min={metrics['train_loss_min']:.3f}")
    if metrics["val_loss_last"] is not None:
        print(f"[train] val loss:   last={metrics['val_loss_last']:.3f}  "
              f"min={metrics['val_loss_min']:.3f}  (n={metrics['n_val_points']})")
    if metrics["diverged"]:
        print("[train] WARNING: run diverged or non-zero exit", file=sys.stderr)

    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())

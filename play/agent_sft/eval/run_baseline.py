"""Phase 1 baseline runner: M models × N seeds × K tasks 的 cross product.

默认 80 runs（2 model × 10 seed × 4 task）。M4 Pro 48GB + Ollama 32B 估 ~3-4h；
32B 单 inference 10-20s。单 run 崩不影响后续——总数 / 成功 / 失败汇总在末尾。

用法：
    python play/agent_sft/eval/run_baseline.py                             # 全跑
    python play/agent_sft/eval/run_baseline.py --models qwen2.5:7b         # 只跑 7b
    python play/agent_sft/eval/run_baseline.py --seeds 0 1 2               # 只跑 3 seed
    python play/agent_sft/eval/run_baseline.py --tasks mmlu_slice          # 只跑一个 task
    python play/agent_sft/eval/run_baseline.py --seeds 0 --tasks mmlu_slice --dry-run  # 只打印不执行

重入：evals 用 (task, model_label, seed) 哈希成 run_id；同 spec 重跑会落不同 run_id
（`--seed` 也进哈希），不会覆盖；aggregate_seeds.py 取最新 N 条按时间窗过滤。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAY_DIR = HERE.parent.parent

DEFAULT_MODELS = ["qwen2.5:7b", "qwen2.5:32b"]  # ollama 上 qwen2.5:Nb 即 instruct（无 -instruct 后缀）
DEFAULT_SEEDS = list(range(10))
DEFAULT_TASKS = ["nudge_fire_rate", "agent_traj", "bfcl_slice", "mmlu_slice"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="跑完后用 `python play/agent_sft/eval/aggregate_seeds.py` 聚合出报告。",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        metavar="OLLAMA_TAG",
        help=f"ollama 模型 tag 列表（空格分隔），默认 {' '.join(DEFAULT_MODELS)}",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        metavar="N",
        help=f"seed 整数列表（空格分隔），默认 {' '.join(map(str, DEFAULT_SEEDS))}",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        choices=DEFAULT_TASKS,
        metavar="TASK",
        help=f"task 名列表（空格分隔），默认全 4 个：{' '.join(DEFAULT_TASKS)}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要执行的 spec，不真跑（用于核对组合）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    combos = [(m, s, t) for m in args.models for s in args.seeds for t in args.tasks]
    total = len(combos)
    print(f"=== baseline batch: {len(args.models)} model × {len(args.seeds)} seed × {len(args.tasks)} task = {total} runs ===")
    if args.dry_run:
        print("(dry run; no commands will be executed)")

    ok = 0
    failed = 0
    start = time.time()
    for i, (m, s, t) in enumerate(combos, start=1):
        spec = f"ollama:{m}@seed={s}"
        cmd = ["python", "-m", "evals", "run", "--task", t, "--model", spec, "--seed", str(s)]
        print(f"\n[{i}/{total}] task={t} model={m} seed={s}")
        if args.dry_run:
            print("  would run:", " ".join(cmd))
            ok += 1
            continue
        try:
            result = subprocess.run(cmd, cwd=PLAY_DIR, check=False)
            if result.returncode == 0:
                ok += 1
            else:
                failed += 1
                print(f"  ↳ FAILED (exit={result.returncode}; continuing batch)", file=sys.stderr)
        except KeyboardInterrupt:
            print("\n=== interrupted ===", file=sys.stderr)
            break

    elapsed = int(time.time() - start)
    print(f"\n=== baseline batch done in {elapsed}s ===")
    print(f"  total: {total}   ok: {ok}   failed: {failed}")
    print("\nNext: python play/agent_sft/eval/aggregate_seeds.py")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

"""Phase 1 baseline runner: cross product of M models × N seeds × K tasks.

Default 80 runs (2 model × 10 seed × 4 tasks). M4 Pro 48GB + Ollama 32B estimated ~3-4h;
32B single inference 10-20s. A single run crash does not affect the follow-up - the total number/success/failure is summarized at the end.

Usage:
    python play/agent_sft/eval/run_baseline.py # Full run
    python play/agent_sft/eval/run_baseline.py --models qwen3.5:9b # run only 9b
    python play/agent_sft/eval/run_baseline.py --seeds 0 1 2 # run only 3 seeds
    python play/agent_sft/eval/run_baseline.py --tasks mmlu_slice # run only one task
    python play/agent_sft/eval/run_baseline.py --seeds 0 --tasks mmlu_slice --dry-run # Only print but not execute

Reentrancy: evals is hashed into run_id using (task, model_label, seed); rerunning the same spec will result in a different run_id.
(`--seed` is also hashed) and will not be overwritten; aggregate_seeds.py takes the latest N items and filters them according to the time window."""
(`--seed` is also hashed) and will not be overwritten; aggregate_seeds.py takes the latest N items and filters them according to the time window."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAY_DIR = HERE.parent.parent

DEFAULT_MODELS = ["qwen3.5:9b", "qwen3.6:27b"]  # Switch to qwen3.x from v1.5 (DECISIONS §10)
DEFAULT_SEEDS = list(range(10))
DEFAULT_TASKS = ["nudge_fire_rate", "agent_traj", "bfcl_slice", "mmlu_slice"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="After runs, aggregate with `python play/agent_sft/eval/aggregate_seeds.py`.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        metavar="OLLAMA_TAG",
        help=f"Ollama model tags (space-separated); default {' '.join(DEFAULT_MODELS)}",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        metavar="N",
        help=f"Seed integers (space-separated); default {' '.join(map(str, DEFAULT_SEEDS))}",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        choices=DEFAULT_TASKS,
        metavar="TASK",
        help=f"Task names (space-separated); default all 4: {' '.join(DEFAULT_TASKS)}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print specs only; do not run (verify combinations)",
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
        cmd = [sys.executable, "-m", "evals", "run", "--task", t, "--model", spec, "--seed", str(s)]
        print(f"\n[{i}/{total}] task={t} model={m} seed={s}")
        if args.dry_run:
            print("  would run:", " ".join(cmd))
            ok += 1
            continue
        try:
            env = os.environ.copy()
            if t in {"nudge_fire_rate", "agent_traj"}:
                env["AGENT_ENGINE_MODEL"] = m
            result = subprocess.run(cmd, cwd=PLAY_DIR, check=False, env=env)
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

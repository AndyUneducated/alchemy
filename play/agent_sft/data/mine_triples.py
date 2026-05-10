"""Phase 2 mining batch runner: 跑 agent_engine 子进程，存原始 envelope.

默认 6 envelopes (2 scenario × 3 run_id) — Phase 2 pilot 量级.
跑批后用 `extractor.py --in data/triples/runs/ --out triples.jsonl` 抽三元组.

单 run 崩不影响 batch（pattern 与 eval/run_baseline.py 一致）；末尾汇总成功 / 失败.

用法:
    python play/agent_sft/data/mine_triples.py                        # pilot 默认 6 runs
    python play/agent_sft/data/mine_triples.py --run-ids 0 1 2 3 4    # 5 run_id × 2 scen
    python play/agent_sft/data/mine_triples.py --scenarios tool_chain # 单 scenario
    python play/agent_sft/data/mine_triples.py --dry-run              # 只打印命令

Seed handling: agent_engine 不接 seed，每次 subprocess 自然采样得 diversity；
run_id 仅作 envelope 文件命名键 + 后续 split 切 train/val 的索引（plan §Decisions）.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PLAY_DIR = REPO_ROOT / "play"
SCENARIOS_DIR = PLAY_DIR / "agent_engine" / "scenarios"
DEFAULT_OUT_DIR = PLAY_DIR / "agent_sft" / "data" / "triples" / "runs"

# Phase 2 锁定 scenario 集（plan §挖掘 scenario 范围）：仅密集 require_tool 场景
DEFAULT_SCENARIOS = ["tool_chain", "code_review"]
DEFAULT_RUN_IDS = [0, 1, 2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "跑完后链:\n"
            "  python play/agent_sft/data/extractor.py --in <out-dir> --out triples.jsonl\n"
            "  python play/agent_sft/data/split.py --in triples.jsonl --train ... --val ...\n"
            "  python play/agent_sft/data/formatter.py --in <split> --out train.jsonl"
        ),
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=DEFAULT_SCENARIOS,
        choices=DEFAULT_SCENARIOS, metavar="NAME",
        help=f"scenario 名（{SCENARIOS_DIR}/<NAME>.md），默认 {' '.join(DEFAULT_SCENARIOS)}",
    )
    parser.add_argument(
        "--run-ids", nargs="+", type=int, default=DEFAULT_RUN_IDS, metavar="N",
        help=f"run_id 整数列表（每个 = 1 次独立 subprocess），默认 {' '.join(map(str, DEFAULT_RUN_IDS))}",
    )
    parser.add_argument(
        "--out-dir", default=str(DEFAULT_OUT_DIR),
        help=f"envelope JSON 输出目录，默认 {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="单次 subprocess 超时秒数，默认 600",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只打印将要执行的命令，不真跑",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combos = [(s, r) for s in args.scenarios for r in args.run_ids]
    print(f"\n=== Mining batch: {len(combos)} runs ===")
    print(f"  scenarios: {args.scenarios}")
    print(f"  run_ids:   {args.run_ids}")
    print(f"  out_dir:   {out_dir}")
    print(f"  dry_run:   {args.dry_run}\n")

    ok = 0
    failed: list[tuple[str, int, int, str]] = []
    t0 = time.time()
    for i, (scenario, run_id) in enumerate(combos, 1):
        scen_path = SCENARIOS_DIR / f"{scenario}.md"
        out_path = out_dir / f"{scenario}-r{run_id}.json"
        cmd = [
            sys.executable, "-m", "agent_engine",
            str(scen_path), "--no-stream",
            "--save-result-json", str(out_path),
        ]
        print(f"[{i}/{len(combos)}] {scenario} r{run_id} → {out_path.name}")
        if args.dry_run:
            print(f"  $ {' '.join(cmd)}")
            ok += 1
            continue
        try:
            proc = subprocess.run(
                cmd, cwd=str(PLAY_DIR), check=False,
                timeout=args.timeout, capture_output=True, text=True,
            )
            if proc.returncode == 0 and out_path.exists():
                ok += 1
                print(f"  ok saved")
            else:
                failed.append((scenario, run_id, proc.returncode, proc.stderr[:200]))
                print(f"  FAIL exit={proc.returncode} stderr={proc.stderr[:200]!r}")
        except subprocess.TimeoutExpired:
            failed.append((scenario, run_id, -1, "TIMEOUT"))
            print(f"  FAIL TIMEOUT (>{args.timeout}s)")
        except Exception as exc:  # pylint: disable=broad-except
            failed.append((scenario, run_id, -2, repr(exc)))
            print(f"  FAIL EXCEPTION: {exc!r}")

    dt = time.time() - t0
    print(f"\n=== Mining done in {dt:.1f}s ===")
    print(f"  total: {len(combos)}  ok: {ok}  failed: {len(failed)}")
    if failed:
        print("  failures:")
        for scen, rid, rc, msg in failed:
            print(f"    {scen} r{rid}: rc={rc} {msg!r}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

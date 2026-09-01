"""Phase 2 mining batch runner: Run the agent_engine sub-process and save the original envelope.

Default 6 envelopes (2 scenario × 3 run_id) — Phase 2 pilot magnitude.
After running the batch, use `synthesize.py --in data/triples/runs/ --out triples.jsonl` to extract triples
(Or `extractor.py` takes the "true self-correcting" semantic path).

A single run crash does not affect the batch (pattern is consistent with eval/run_baseline.py); success/failure is summarized at the end.

Usage:
    python play/agent_sft/data/mine_triples.py # pilot default 6 runs (fast scenario)
    python play/agent_sft/data/mine_triples.py --run-ids 0 1 2 3 4 # 5 run_id × 2 scenes
    python play/agent_sft/data/mine_triples.py --scenarios tool_chain # Single scenario
    python play/agent_sft/data/mine_triples.py --upstream # Switch back to agent_engine/scenarios/<name>.md
    python play/agent_sft/data/mine_triples.py --dry-run # Only print commands

Scenario source: Default: `data/scenarios/<name>_fast.md` (max_retries=0 / max_tokens=80
/ Delete open+finalize, envelope wall clock ~25s vs upstream ~65s); `--upstream` switch back
agent_engine/scenarios/<name>.md (original scenario reused by baseline eval).

Seed handling: agent_engine does not receive seeds, and each subprocess naturally samples diversity;
run_id is only used as the envelope file naming key + the index of train/val for subsequent splits (plan §Decisions)."""
run_id is only used as the envelope file naming key + the index of train/val for subsequent splits (plan §Decisions)."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PLAY_DIR = REPO_ROOT / "play"
FAST_SCENARIOS_DIR = PLAY_DIR / "agent_sft" / "data" / "scenarios"
UPSTREAM_SCENARIOS_DIR = PLAY_DIR / "agent_engine" / "scenarios"
DEFAULT_OUT_DIR = PLAY_DIR / "agent_sft" / "data" / "triples" / "runs"

# Phase 2 Lock scenario set (plan §mining scenario scope): intensive require_tool scenarios only
# Phase 2 Lock scenario set (plan §mining scenario scope): intensive require_tool scenarios only
# Phase 2 Lock scenario set (plan §mining scenario scope): intensive require_tool scenarios only
# Phase 2 Lock scenario set (plan §mining scenario scope): intensive require_tool scenarios only
# Phase 2 Lock scenario set (plan §mining scenario scope): intensive require_tool scenarios only
# Phase 2 Lock scenario set (plan §mining scenario scope): intensive require_tool scenarios only
# Phase 2 Lock scenario set (plan §mining scenario scope): intensive require_tool scenarios only
# Phase 2 Lock scenario set (plan §mining scenario scope): intensive require_tool scenarios only
# Phase 2 Lock scenario set (plan §mining scenario scope): intensive require_tool scenarios only
DEFAULT_SCENARIOS = ["tool_chain", "code_review"]
DEFAULT_RUN_IDS = [0, 1, 2]


def _scenario_path(name: str, upstream: bool) -> Path:
    """fast copy: data/scenarios/<name>_fast.md; upstream: agent_engine/scenarios/<name>.md."""
    if upstream:
        return UPSTREAM_SCENARIOS_DIR / f"{name}.md"
    return FAST_SCENARIOS_DIR / f"{name}_fast.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
"Chain after running:\n"
            "  python play/agent_sft/data/synthesize.py --in <out-dir> --out triples.jsonl\n"
            "  python play/agent_sft/data/split.py --in triples.jsonl --train ... --val ...\n"
            "  python play/agent_sft/data/formatter.py --in <split> --out train.jsonl"
        ),
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=DEFAULT_SCENARIOS,
        choices=DEFAULT_SCENARIOS, metavar="NAME",
        help=(
f"scenario name (default is data/scenarios/<NAME>_fast.md, --upstream switches back to "
f"agent_engine/scenarios/<NAME>.md), default {' '.join(DEFAULT_SCENARIOS)}"
        ),
    )
    parser.add_argument(
        "--upstream", action="store_true",
        help="Use upstream agent_engine/scenarios/<name>.md (max_retries=1, matches baseline eval) "
             "instead of fast copy",
    )
    parser.add_argument(
        "--run-ids", nargs="+", type=int, default=DEFAULT_RUN_IDS, metavar="N",
help=f"run_id integer list (each = 1 independent subprocess), default {' '.join(map(str, DEFAULT_RUN_IDS))}",
    )
    parser.add_argument(
        "--out-dir", default=str(DEFAULT_OUT_DIR),
help=f"envelope JSON output directory, default {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
help="Single subprocess timeout seconds, default 600",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
help="Only print the command to be executed, not run it",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
# Must resolve() into an absolute path - subprocess uses cwd=PLAY_DIR, the relative path will be replaced by agent_engine
# CLI os.path.abspath() misresolved to PLAY_DIR/<relative> (which caused the product to fall to play/play/...).
# CLI os.path.abspath() misresolved to PLAY_DIR/<relative> (which caused the product to fall to play/play/...).
# CLI os.path.abspath() misresolved to PLAY_DIR/<relative> (which caused the product to fall to play/play/...).
# CLI os.path.abspath() misresolved to PLAY_DIR/<relative> (which caused the product to fall to play/play/...).
# CLI os.path.abspath() misresolved to PLAY_DIR/<relative> (which caused the product to fall to play/play/...).
# CLI os.path.abspath() misresolved to PLAY_DIR/<relative> (which caused the product to fall to play/play/...).
# CLI os.path.abspath() misresolved to PLAY_DIR/<relative> (which caused the product to fall to play/play/...).
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    combos = [(s, r) for s in args.scenarios for r in args.run_ids]
    src_dir = UPSTREAM_SCENARIOS_DIR if args.upstream else FAST_SCENARIOS_DIR
    print(f"\n=== Mining batch: {len(combos)} runs ===")
    print(f"  scenarios:    {args.scenarios}")
print(f" scenario src: {src_dir}{' (fast copy)' if not args.upstream else ' (upstream)'}")
    print(f"  run_ids:      {args.run_ids}")
    print(f"  out_dir:      {out_dir}")
    print(f"  dry_run:      {args.dry_run}\n")

    ok = 0
    failed: list[tuple[str, int, int, str]] = []
    t0 = time.time()
    for i, (scenario, run_id) in enumerate(combos, 1):
        scen_path = _scenario_path(scenario, args.upstream)
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

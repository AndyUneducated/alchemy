"""Zero-logic CLI adapter for ``Engine.invoke()`` (replaces legacy run.py).

This file is the **only** entry point for ``python -m agent_engine``. It does
nothing except: argparse → ``Engine(Scenario.from_yaml(path)).invoke(...)``
→ print save confirmations. No business logic.

Per plan §5.3:
- Python API (``Engine.invoke()``) is source of truth
- CLI never has capabilities the API lacks
- API design is not bent for CLI ergonomics
"""

from __future__ import annotations

import argparse
import os
import sys

from .engine import Engine
from .scenario import Scenario


def main(argv: list[str] | None = None) -> None:
    # Line-buffer both streams so `2>&1 | tee` preserves chronological order
    # between stdout (speaker text, tool traces) and stderr (WARNINGs).
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        prog="python -m agent_engine",
        description="Multi-Agent Discussion Engine (CLI thin adapter for Engine.invoke)",
    )
    parser.add_argument("scenario", help="scenario .md file path")
    parser.add_argument(
        "--no-stream", action="store_true",
        help="disable streaming output (Engine.invoke(print_stream=False))",
    )
    parser.add_argument(
        "--save-artifact", metavar="PATH", default=None,
        help="write the final artifact markdown to PATH "
             "(only when the scenario has artifact enabled)",
    )
    parser.add_argument(
        "--save-transcript", metavar="PATH", default=None,
        help="dump the structured history "
             "(topic / turn / speaker / artifact_event / tool_call) to PATH as JSON",
    )
    args = parser.parse_args(argv)

    scenario = Scenario.from_yaml(args.scenario)
    engine = Engine(scenario)
    artifact_path = os.path.abspath(args.save_artifact) if args.save_artifact else None
    transcript_path = os.path.abspath(args.save_transcript) if args.save_transcript else None
    result = engine.invoke(
        artifact_path=artifact_path,
        transcript_path=transcript_path,
        print_stream=not args.no_stream,
    )

    if artifact_path and result.artifact:
        print(f"\n💾 artifact saved → {artifact_path}", flush=True)
    if transcript_path:
        print(f"\n💾 transcript saved → {transcript_path}", flush=True)


if __name__ == "__main__":
    main()

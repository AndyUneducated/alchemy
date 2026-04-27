from __future__ import annotations

import argparse
import sys

from .runner import Workflow


def _parse_vars(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for kv in items:
        if "=" not in kv:
            sys.exit(f"Error: --vars expects k=v, got {kv!r}")
        k, _, v = kv.partition("=")
        out[k] = v
    return out


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        prog="python -m workflow",
        description="Workflow runner (deterministic pipeline + agent stages)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser(
        "run",
        help="run a workflow.yaml; pass workflow inputs with --vars k=v (repeatable)",
    )
    run_p.add_argument("workflow", help="workflow .yaml file path")
    run_p.add_argument(
        "--vars", action="append", default=[],
        metavar="K=V",
        help="workflow input (repeatable)",
    )
    args = parser.parse_args(argv)

    if args.cmd == "run":
        vars_input = _parse_vars(args.vars)
        wf = Workflow.from_yaml(args.workflow)
        wf.run(vars_input)


if __name__ == "__main__":
    main()

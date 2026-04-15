#!/usr/bin/env python3
"""Unified CLI entry-point: load a scenario .md and run the discussion."""

from __future__ import annotations

import argparse
import sys

import yaml

from agent import Agent
from discussion import Discussion


# -- frontmatter parser -----------------------------------------------------

def load_scenario(path: str) -> tuple[dict, str]:
    """Parse a scenario markdown file into (frontmatter_dict, body_text)."""
    with open(path, encoding="utf-8") as f:
        text = f.read()

    parts = text.split("---", 2)
    if len(parts) < 3:
        sys.exit(f"Error: {path} has no YAML frontmatter (--- ... ---)")

    meta = yaml.safe_load(parts[1])
    if not isinstance(meta, dict):
        sys.exit(f"Error: {path} frontmatter is not a valid YAML mapping")

    return meta, parts[2].strip()


# -- build Agent instances from parsed frontmatter --------------------------

def _build_agent(spec: dict) -> Agent:
    kwargs: dict = {}
    if "model" in spec:
        kwargs["model"] = spec["model"]
    if "max_tokens" in spec:
        kwargs["max_tokens"] = int(spec["max_tokens"])
    if "temperature" in spec:
        kwargs["temperature"] = float(spec["temperature"])
    return Agent(
        name=spec["name"],
        system_prompt=spec["prompt"],
        **kwargs,
    )


# -- validation --------------------------------------------------------------

VALID_STAGES = {"opening", "main", "closing"}
VALID_WHO = {"moderator", "members", "all"}

PHASES_MISSING_MSG = """\
Error: 'phases' is required in the scenario file.

Each phase needs 'stage' and 'who'. Minimal example:

  phases:
    - stage: main
      who: members

See scenarios/ for complete examples."""

PHASE_FIELD_MSG = "Error in phase #{idx}: missing required field '{field}'. " \
                  "Each phase must have 'stage' and 'who'."

PHASE_STAGE_MSG = "Error in phase #{idx}: stage='{val}' is invalid. " \
                  "Must be one of: opening, main, closing."


def _validate_phases(phases: list[dict]) -> None:
    for i, phase in enumerate(phases, 1):
        if "stage" not in phase:
            sys.exit(PHASE_FIELD_MSG.format(idx=i, field="stage"))
        if "who" not in phase:
            sys.exit(PHASE_FIELD_MSG.format(idx=i, field="who"))
        if phase["stage"] not in VALID_STAGES:
            sys.exit(PHASE_STAGE_MSG.format(idx=i, val=phase["stage"]))


# -- main --------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Agent Discussion Engine")
    parser.add_argument("scenario", help="scenario .md file path")
    parser.add_argument("-r", "--rounds", type=int, default=None,
                        help="override round count from scenario")
    parser.add_argument("--no-stream", action="store_true",
                        help="disable streaming output")
    args = parser.parse_args()

    meta, body = load_scenario(args.scenario)

    phases = meta.get("phases")
    if not phases:
        sys.exit(PHASES_MISSING_MSG)
    _validate_phases(phases)

    rounds = args.rounds or meta.get("rounds", 3)
    stream = not args.no_stream
    members = [_build_agent(s) for s in meta.get("members", [])]
    moderator = _build_agent(meta["moderator"]) if "moderator" in meta else None

    Discussion(
        members=members,
        topic=body,
        phases=phases,
        rounds=rounds,
        stream=stream,
        moderator=moderator,
    ).run()


if __name__ == "__main__":
    main()

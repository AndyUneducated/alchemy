#!/usr/bin/env python3
"""Unified CLI entry-point: load a scenario .md and run the discussion."""

from __future__ import annotations

import argparse
import copy
import sys

import yaml

from agent import Agent
from discussion import Discussion
from tools import TOOL_DEFINITIONS, dispatch


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

def _build_tool_handler(tool_configs: list[dict]) -> callable:
    """Create a dispatch wrapper that injects scenario-level defaults (e.g. vdb_dir)."""
    defaults: dict[str, dict] = {}
    for tc in tool_configs:
        name = tc["name"]
        defaults[name] = {k: v for k, v in tc.items() if k != "name"}

    def handler(name: str, arguments: dict) -> str:
        merged = {**defaults.get(name, {}), **arguments}
        return dispatch(name, merged)

    return handler


def _resolve_tool_defs(tool_configs: list[dict]) -> list[dict]:
    """Filter TOOL_DEFINITIONS to only those named in *tool_configs*.

    Parameters already supplied as scenario-level defaults are stripped from
    the schema so the LLM doesn't need to (and cannot) fill them in.
    """
    defaults_by_name: dict[str, set[str]] = {}
    for tc in tool_configs:
        defaults_by_name[tc["name"]] = {k for k in tc if k != "name"}

    defs: list[dict] = []
    for td in TOOL_DEFINITIONS:
        name = td["function"]["name"]
        if name not in defaults_by_name:
            continue
        hidden = defaults_by_name[name]
        if not hidden:
            defs.append(td)
            continue
        td = copy.deepcopy(td)
        params = td["function"]["parameters"]
        for key in hidden:
            params.get("properties", {}).pop(key, None)
        if "required" in params:
            params["required"] = [r for r in params["required"] if r not in hidden]
        defs.append(td)
    return defs


def _build_agent(spec: dict, *, tool_defs: list[dict] | None = None,
                 tool_handler: callable | None = None) -> Agent:
    kwargs: dict = {}
    if "model" in spec:
        kwargs["model"] = spec["model"]
    if "max_tokens" in spec:
        kwargs["max_tokens"] = int(spec["max_tokens"])
    if "temperature" in spec:
        kwargs["temperature"] = float(spec["temperature"])
    if tool_defs:
        kwargs["tools"] = tool_defs
        kwargs["tool_handler"] = tool_handler
    return Agent(
        name=spec["name"],
        system_prompt=spec["prompt"],
        **kwargs,
    )


# -- validation --------------------------------------------------------------

VALID_WHO = {"moderator", "members", "all"}

MAIN_ROUND_MISSING_MSG = (
    "Error in main phase #{idx}: missing required field 'round'. "
    "Must be a positive integer or \"default\"."
)

MAIN_ROUND_INVALID_MSG = (
    "Error in main phase #{idx}: round='{val}' is invalid. "
    "Must be a positive integer or \"default\"."
)

PHASE_WHO_MSG = (
    "Error in {section} phase #{idx}: who='{val}' is not a valid target. "
    "Must be one of: moderator, members, all, or a participant name."
)

OC_ROUND_MSG = (
    "Error in {section} phase #{idx}: 'round' is not allowed in {section} phases."
)


def _validate_who(who: str, agent_names: set[str], section: str, idx: int) -> None:
    if who not in VALID_WHO and who not in agent_names:
        sys.exit(PHASE_WHO_MSG.format(section=section, idx=idx, val=who))


def _validate_oc_phases(phases: list[dict], agent_names: set[str], section: str) -> None:
    """Validate opening or closing phases."""
    for i, phase in enumerate(phases, 1):
        if "who" not in phase:
            sys.exit(f"Error in {section} phase #{i}: missing required field 'who'.")
        _validate_who(phase["who"], agent_names, section, i)
        if "round" in phase:
            sys.exit(OC_ROUND_MSG.format(section=section, idx=i))


def _validate_main_phases(phases: list[dict], agent_names: set[str]) -> None:
    """Validate main phases — ``round`` is required."""
    for i, phase in enumerate(phases, 1):
        if "who" not in phase:
            sys.exit(f"Error in main phase #{i}: missing required field 'who'.")
        _validate_who(phase["who"], agent_names, "main", i)
        if "round" not in phase:
            sys.exit(MAIN_ROUND_MISSING_MSG.format(idx=i))
        r = phase["round"]
        if r != "default" and not (isinstance(r, int) and r > 0):
            sys.exit(MAIN_ROUND_INVALID_MSG.format(idx=i, val=r))


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

    agent_names = {s["name"] for s in meta.get("members", [])}
    if "moderator" in meta:
        agent_names.add(meta["moderator"]["name"])

    opening = meta.get("opening", [])
    main_phases = meta.get("main", [])
    closing = meta.get("closing", [])

    _validate_oc_phases(opening, agent_names, "opening")
    _validate_main_phases(main_phases, agent_names)
    _validate_oc_phases(closing, agent_names, "closing")

    rounds = args.rounds or meta.get("rounds", 3)
    stream = not args.no_stream

    tool_configs = meta.get("tools", [])
    tool_defs = _resolve_tool_defs(tool_configs) if tool_configs else None
    tool_handler = _build_tool_handler(tool_configs) if tool_configs else None

    members = [_build_agent(s, tool_defs=tool_defs, tool_handler=tool_handler)
               for s in meta.get("members", [])]
    moderator = (_build_agent(meta["moderator"], tool_defs=tool_defs, tool_handler=tool_handler)
                 if "moderator" in meta else None)

    Discussion(
        members=members,
        topic=body,
        opening=opening,
        main=main_phases,
        closing=closing,
        rounds=rounds,
        stream=stream,
        moderator=moderator,
    ).run()


if __name__ == "__main__":
    main()

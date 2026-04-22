#!/usr/bin/env python3
"""Unified CLI entry-point: load a scenario .md and run the discussion."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import yaml

from agent import Agent, _client as _backend_client
from artifact import ARTIFACT_TOOL_NAMES, ArtifactStore
from config import SUMMARY_MAX_TOKENS, SUMMARY_MODEL, SUMMARY_TEMPERATURE
from discussion import Discussion
from memory import ConversationMemory, FullHistory, SummaryMemory, WindowMemory
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

def _build_tool_handler(tool_configs: list[dict], scenario_dir: str) -> callable:
    """Create a dispatch wrapper that injects scenario-level defaults (e.g. vdb_dir).

    For each tool, any default whose key is listed in the tool's ``_path_params``
    and whose value is a relative path is resolved against *scenario_dir*,
    so scenarios become location-independent (any cwd, any invoker path).
    """
    path_params_by_tool: dict[str, set[str]] = {
        td["function"]["name"]: set(td.get("_path_params") or ())
        for td in TOOL_DEFINITIONS
    }

    defaults: dict[str, dict] = {}
    for tc in tool_configs:
        name = tc["name"]
        path_keys = path_params_by_tool.get(name, set())
        resolved: dict = {}
        for k, v in tc.items():
            if k == "name":
                continue
            if k in path_keys and isinstance(v, str) and not os.path.isabs(v):
                v = os.path.abspath(os.path.join(scenario_dir, v))
            resolved[k] = v
        defaults[name] = resolved

    def handler(name: str, arguments: dict) -> str:
        # Scenario-level defaults win over LLM-supplied args: those keys are
        # stripped from the tool schema (see _resolve_tool_defs) and the LLM
        # should never fill them. If it hallucinates one anyway (e.g. supplies
        # its own ``vdb_dir``), we still honor the scenario path.
        merged = {**arguments, **defaults.get(name, {})}
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
        td = copy.deepcopy(td)
        # Drop internal hints that are not part of the OpenAI tool schema and
        # would break JSON serialization (e.g. ``_path_params`` is a set).
        td.pop("_path_params", None)
        if hidden:
            params = td["function"]["parameters"]
            for key in hidden:
                params.get("properties", {}).pop(key, None)
            if "required" in params:
                params["required"] = [r for r in params["required"] if r not in hidden]
        defs.append(td)
    return defs


def _build_memory(cfg: dict | None) -> ConversationMemory:
    """Translate a parsed ``memory`` mapping into a ConversationMemory instance."""
    if not cfg:
        return FullHistory()
    t = cfg["type"]
    if t == "full":
        return FullHistory()
    if t == "window":
        return WindowMemory(max_recent=int(cfg["max_recent"]))
    if t == "summary":
        kwargs: dict = {
            "max_recent": int(cfg["max_recent"]),
            "client": _backend_client,
            "summary_model": cfg.get("model", SUMMARY_MODEL),
            "summary_max_tokens": int(cfg.get("max_tokens", SUMMARY_MAX_TOKENS)),
            "summary_temperature": float(cfg.get("temperature", SUMMARY_TEMPERATURE)),
        }
        if "summarizer_prompt" in cfg:
            kwargs["summarizer_prompt"] = cfg["summarizer_prompt"]
        if "summarize_instruction" in cfg:
            kwargs["summarize_instruction"] = cfg["summarize_instruction"]
        return SummaryMemory(**kwargs)
    sys.exit(f"Unknown memory type: {t}")


def _build_agent(spec: dict, *, tool_defs: list[dict] | None = None,
                 tool_handler: callable | None = None,
                 scenario_mem_cfg: dict | None = None) -> Agent:
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
    mem_cfg = spec.get("memory", scenario_mem_cfg)
    kwargs["memory"] = _build_memory(mem_cfg)
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

VALID_MEMORY_TYPES = {"full", "window", "summary"}

MEMORY_TYPE_MSG = (
    "Error in memory config ({section}): type='{val}' is invalid. "
    "Must be one of: full, window, summary."
)

MEMORY_MAX_RECENT_MSG = (
    "Error in memory config ({section}): 'max_recent' must be a positive integer, got {val!r}."
)

VALID_SECTION_MODES = {"replace", "append"}

ARTIFACT_CFG_TYPE_MSG = "Error in artifact config: must be a mapping."
ARTIFACT_ENABLED_MSG = "Error in artifact config: 'enabled' must be a boolean."
ARTIFACT_SECTIONS_TYPE_MSG = (
    "Error in artifact config: 'initial_sections' must be a list."
)
ARTIFACT_SECTION_ITEM_MSG = (
    "Error in artifact config: initial_sections[{idx}] must be a string name "
    "or a mapping with 'name' (and optional 'mode')."
)
ARTIFACT_SECTION_MODE_MSG = (
    "Error in artifact config: initial_sections[{idx}] mode='{val}' is invalid. "
    "Must be one of: replace, append."
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


def _validate_memory(cfg: dict | None, section: str) -> None:
    """Validate a parsed ``memory`` mapping. ``None`` means unset (fall back to FullHistory)."""
    if cfg is None:
        return
    t = cfg.get("type")
    if t not in VALID_MEMORY_TYPES:
        sys.exit(MEMORY_TYPE_MSG.format(section=section, val=t))
    if t in ("window", "summary"):
        mr = cfg.get("max_recent")
        if not (isinstance(mr, int) and mr > 0):
            sys.exit(MEMORY_MAX_RECENT_MSG.format(section=section, val=mr))


def _validate_artifact(cfg: dict | None) -> None:
    """Validate an ``artifact`` frontmatter block. ``None`` means unset (disabled)."""
    if cfg is None:
        return
    if not isinstance(cfg, dict):
        sys.exit(ARTIFACT_CFG_TYPE_MSG)
    if "enabled" in cfg and not isinstance(cfg["enabled"], bool):
        sys.exit(ARTIFACT_ENABLED_MSG)
    sections = cfg.get("initial_sections")
    if sections is None:
        return
    if not isinstance(sections, list):
        sys.exit(ARTIFACT_SECTIONS_TYPE_MSG)
    for i, item in enumerate(sections):
        if isinstance(item, str):
            continue
        if not isinstance(item, dict) or "name" not in item or not isinstance(item["name"], str):
            sys.exit(ARTIFACT_SECTION_ITEM_MSG.format(idx=i))
        mode = item.get("mode", "replace")
        if mode not in VALID_SECTION_MODES:
            sys.exit(ARTIFACT_SECTION_MODE_MSG.format(idx=i, val=mode))


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
    # Line-buffer both streams so `2>&1 | tee` preserves chronological order
    # between stdout (speaker text, tool traces) and stderr (WARNINGs).
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="Multi-Agent Discussion Engine")
    parser.add_argument("scenario", help="scenario .md file path")
    parser.add_argument("-r", "--rounds", type=int, default=None,
                        help="override round count from scenario")
    parser.add_argument("--no-stream", action="store_true",
                        help="disable streaming output")
    parser.add_argument("--save-artifact", metavar="PATH", default=None,
                        help="after the run, write the final artifact markdown to PATH "
                             "(only when the scenario has artifact enabled)")
    args = parser.parse_args()

    meta, body = load_scenario(args.scenario)
    scenario_dir = os.path.dirname(os.path.abspath(args.scenario))

    agent_names = {s["name"] for s in meta.get("members", [])}
    if "moderator" in meta:
        agent_names.add(meta["moderator"]["name"])

    opening = meta.get("opening", [])
    main_phases = meta.get("main", [])
    closing = meta.get("closing", [])

    _validate_oc_phases(opening, agent_names, "opening")
    _validate_main_phases(main_phases, agent_names)
    _validate_oc_phases(closing, agent_names, "closing")

    scenario_mem_cfg = meta.get("memory")
    _validate_memory(scenario_mem_cfg, "scenario")
    for s in meta.get("members", []):
        if "memory" in s:
            _validate_memory(s["memory"], f"member '{s.get('name')}'")
    if "moderator" in meta and "memory" in meta["moderator"]:
        _validate_memory(meta["moderator"]["memory"],
                         f"moderator '{meta['moderator'].get('name')}'")

    artifact_cfg = meta.get("artifact")
    _validate_artifact(artifact_cfg)

    rounds = args.rounds or meta.get("rounds", 3)
    stream = not args.no_stream

    tool_configs = meta.get("tools", [])
    base_tool_defs = _resolve_tool_defs(tool_configs) if tool_configs else []
    base_handler = _build_tool_handler(tool_configs, scenario_dir) if tool_configs else None

    store: ArtifactStore | None = None
    if isinstance(artifact_cfg, dict) and artifact_cfg.get("enabled"):
        store = ArtifactStore(initial_sections=artifact_cfg.get("initial_sections"))

    def _agent_bundle(agent_name: str, role: str):
        """Compose per-agent (tool_defs, handler) baking in caller name + role."""
        defs = list(base_tool_defs)
        if store is not None:
            defs.extend(store.build_tool_defs(role))
        if not defs:
            return None, None

        def handler(name: str, args: dict, *, _caller=agent_name) -> str:
            if store is not None and name in ARTIFACT_TOOL_NAMES:
                return store.dispatch(name, args, caller=_caller)
            if base_handler is None:
                return json.dumps({"error": f"Unknown tool: {name}"})
            return base_handler(name, args)

        return defs, handler

    members = []
    for s in meta.get("members", []):
        defs, handler = _agent_bundle(s["name"], role="member")
        members.append(_build_agent(s, tool_defs=defs, tool_handler=handler,
                                    scenario_mem_cfg=scenario_mem_cfg))

    moderator = None
    if "moderator" in meta:
        mspec = meta["moderator"]
        defs, handler = _agent_bundle(mspec["name"], role="moderator")
        moderator = _build_agent(mspec, tool_defs=defs, tool_handler=handler,
                                 scenario_mem_cfg=scenario_mem_cfg)

    Discussion(
        members=members,
        topic=body,
        opening=opening,
        main=main_phases,
        closing=closing,
        rounds=rounds,
        stream=stream,
        moderator=moderator,
        artifact=store,
    ).run()

    if args.save_artifact:
        if store is None:
            print(f"WARNING: --save-artifact ignored: scenario has no artifact enabled",
                  file=sys.stderr, flush=True)
        else:
            out_path = os.path.abspath(args.save_artifact)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(store.render())
                f.write("\n")
            print(f"\n💾 artifact saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()

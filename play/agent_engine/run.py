#!/usr/bin/env python3
"""Unified CLI entry-point: load a scenario .md and run the discussion."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import yaml

from agent import Agent, _client as _backend_client
from artifact import ARTIFACT_TOOL_NAMES, ArtifactStore
from config import SUMMARY_MAX_TOKENS, SUMMARY_MODEL, SUMMARY_TEMPERATURE
from discussion import Discussion
from memory import ConversationMemory, FullHistory, SummaryMemory, WindowMemory
from tools import TOOL_DEFINITIONS, dispatch, is_error


# -- tool tracer ------------------------------------------------------------
#
# Fires from the per-agent tool_handler closure below (single chokepoint for
# every non-artifact tool call across all four backends). Emits two sinks:
#
#   1. stderr one-liner — workshop audience sees the call in real time,
#      between the speaker label and the model's final text.
#   2. structured event attached to Discussion.history with visible=False,
#      so memory._render skips it (other agents never see the call) while
#      --save-transcript can still dump it for replay.
#
# Field names intentionally mirror OpenTelemetry GenAI semantic conventions
# (gen_ai.tool.name / .call.arguments / .call.response). We borrow the naming
# only — no SDK dependency, no spans, no exporter.


def _preview_args(arguments: dict) -> str:
    """Render a short k=v, k=v summary of tool arguments for the terminal."""
    parts: list[str] = []
    for k, v in arguments.items():
        if isinstance(v, str):
            s = v if len(v) <= 40 else v[:37] + "..."
            parts.append(f"{k}={s!r}")
        else:
            s = repr(v)
            if len(s) > 40:
                s = s[:37] + "..."
            parts.append(f"{k}={s}")
    return ", ".join(parts)


def _preview_result(result: str, ok: bool) -> str:
    """Render a short summary of a tool result for the terminal."""
    try:
        payload = json.loads(result)
    except (ValueError, TypeError):
        flat = result.replace("\n", " ").strip()
        return flat if len(flat) <= 60 else flat[:57] + "..."
    if not ok and isinstance(payload, dict) and "error" in payload:
        first = str(payload["error"]).splitlines()[0]
        return f"error: {first}"
    if isinstance(payload, dict):
        # retrieve_docs returns {data, meta:{mode, reranked, top_k}}; surface
        # the retrieval path so workshop viewers can see which strategy ran.
        if isinstance(payload.get("data"), list) and isinstance(payload.get("meta"), dict):
            n = len(payload["data"])
            m = payload["meta"]
            tags = [f"mode={m.get('mode')}"]
            if m.get("reranked"):
                tags.append("reranked")
            return f"[{n} items, " + ", ".join(tags) + "]"
        if isinstance(payload.get("results"), list):
            return f"{{results: {len(payload['results'])}}}"
        if "count" in payload:
            return f"{{count: {payload['count']}}}"
        keys = list(payload.keys())
        if len(keys) <= 3:
            return "{" + ", ".join(keys) + "}"
        return "{" + ", ".join(keys[:3]) + ", ...}"
    if isinstance(payload, list):
        return f"[{len(payload)} items]"
    flat = str(payload)
    return flat if len(flat) <= 60 else flat[:57] + "..."


class ToolTracer:
    """Collect non-artifact tool-call events across one Discussion.

    Events are drained by ``Discussion._run_turn`` after each turn and
    appended to ``Discussion.history`` with ``visible=False``.
    """

    def __init__(self) -> None:
        self._events: list[dict] = []

    def record(self, caller: str, tool: str, arguments: dict, result: str) -> None:
        ok = not is_error(result)
        print(
            f"🔧 [{caller}] {tool}({_preview_args(arguments)}) "
            f"→ {_preview_result(result, ok)}",
            file=sys.stderr, flush=True,
        )
        self._events.append({
            "type": "tool_call",
            "caller": caller,
            "tool": tool,
            "arguments": arguments,
            "result": result,
            "ok": ok,
            "visible": False,
            "ts": time.time(),
        })

    def drain(self) -> list[dict]:
        events, self._events = self._events, []
        return events


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
#
# Two literal forms accepted by ``who`` (and by ``artifact.tool_owners`` values):
#   - scalar str: "moderator" | "member" | "all"          (role / keyword addressing)
#   - list[str]: agent name list                          (name addressing)
#
# Anything else is rejected at startup. The validators below produce one
# concrete error message per failure and call ``sys.exit`` — fail-fast so the
# author sees the issue before tokens are spent.

VALID_ROLES = {"moderator", "member"}
VALID_WHO_SCALARS = {"moderator", "member", "all"}
VALID_MEMORY_TYPES = {"full", "window", "summary"}
VALID_SECTION_MODES = {"replace", "append"}


def _err(msg: str) -> None:
    sys.exit(f"Error: {msg}")


def _validate_agents(agents: list, agent_names: set[str], agent_roles: dict[str, str]) -> None:
    if not isinstance(agents, list) or not agents:
        _err("'agents' must be a non-empty list at the top level.")
    seen: set[str] = set()
    for i, a in enumerate(agents, 1):
        if not isinstance(a, dict):
            _err(f"agents[{i}] must be a mapping.")
        name = a.get("name")
        if not isinstance(name, str) or not name:
            _err(f"agents[{i}] missing required string 'name'.")
        if not isinstance(a.get("prompt"), str) or not a["prompt"]:
            _err(f"agents[{i}] '{name}' missing required string 'prompt'.")
        role = a.get("role")
        if role not in VALID_ROLES:
            _err(
                f"agents[{i}] '{name}' has role={role!r}; "
                f"required, must be one of: moderator, member."
            )
        if name in seen:
            _err(f"agents[{i}] duplicate name '{name}'.")
        seen.add(name)
        agent_names.add(name)
        agent_roles[name] = role


def _validate_who(who, agent_names: set[str], where: str) -> None:
    """Validate a step's ``who`` form. Reachability of role lookup is checked here too."""
    if isinstance(who, str):
        if who not in VALID_WHO_SCALARS:
            _err(
                f"{where}: who={who!r} is not a valid scalar. "
                f"Must be one of: moderator, member, all "
                f"(or use a list of agent names)."
            )
        # role-based addressing must hit at least one agent
        # (run.py needs agent_roles which it has via the closure caller — pass through)
        return
    if isinstance(who, list):
        if not who:
            _err(f"{where}: who is an empty list; address at least one agent.")
        for n in who:
            if not isinstance(n, str):
                _err(f"{where}: who list contains non-string element {n!r}.")
            if n not in agent_names:
                _err(
                    f"{where}: who references unknown agent name '{n}'. "
                    f"Known: {sorted(agent_names)}."
                )
        return
    _err(
        f"{where}: who must be a scalar (moderator/member/all) "
        f"or a list of agent names; got {type(who).__name__}."
    )


def _validate_who_role_reachability(who, agent_roles: dict[str, str], where: str) -> None:
    """Second pass: scalar role ``moderator``/``member`` must hit ≥1 agent."""
    if not isinstance(who, str) or who == "all":
        return
    if not any(r == who for r in agent_roles.values()):
        _err(
            f"{where}: who={who!r} matches 0 agents — no agent has role={who!r}."
        )


def _validate_steps(steps, agent_names: set[str], agent_roles: dict[str, str]) -> None:
    if not isinstance(steps, list) or not steps:
        _err("'steps' must be a non-empty list at the top level.")
    for i, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            _err(f"steps[{i}] must be a mapping.")
        where = f"steps[{i}]"
        if "id" in step and step["id"] is not None and not isinstance(step["id"], str):
            _err(f"{where}: 'id' must be a string when present.")
        if "who" not in step:
            _err(f"{where}: missing required field 'who'.")
        _validate_who(step["who"], agent_names, where)
        _validate_who_role_reachability(step["who"], agent_roles, where)
        instr = step.get("instruction")
        if not isinstance(instr, str) or not instr.strip():
            _err(f"{where}: missing required non-empty string 'instruction'.")
        if "require_tool" in step and not isinstance(step["require_tool"], str):
            _err(f"{where}: 'require_tool' must be a string.")
        if "max_retries" in step:
            mr = step["max_retries"]
            if not isinstance(mr, int) or mr < 0:
                _err(f"{where}: 'max_retries' must be a non-negative integer, got {mr!r}.")


def _validate_memory(cfg: dict | None, section: str) -> None:
    """Validate a parsed ``memory`` mapping. ``None`` means unset (fall back to FullHistory)."""
    if cfg is None:
        return
    t = cfg.get("type")
    if t not in VALID_MEMORY_TYPES:
        _err(
            f"memory config ({section}): type={t!r} is invalid. "
            f"Must be one of: full, window, summary."
        )
    if t in ("window", "summary"):
        mr = cfg.get("max_recent")
        if not (isinstance(mr, int) and mr > 0):
            _err(
                f"memory config ({section}): 'max_recent' must be a positive integer, got {mr!r}."
            )


def _validate_artifact(cfg: dict | None, agent_names: set[str], agent_roles: dict[str, str]) -> None:
    """Validate an ``artifact`` block. ``None`` means unset (disabled)."""
    if cfg is None:
        return
    if not isinstance(cfg, dict):
        _err("artifact config: must be a mapping.")
    if "enabled" in cfg and not isinstance(cfg["enabled"], bool):
        _err("artifact config: 'enabled' must be a boolean.")

    sections = cfg.get("initial_sections")
    if sections is not None:
        if not isinstance(sections, list):
            _err("artifact config: 'initial_sections' must be a list.")
        for i, item in enumerate(sections):
            if isinstance(item, str):
                continue
            if not isinstance(item, dict) or "name" not in item or not isinstance(item["name"], str):
                _err(
                    f"artifact config: initial_sections[{i}] must be a string name "
                    f"or a mapping with 'name' (and optional 'mode')."
                )
            mode = item.get("mode", "replace")
            if mode not in VALID_SECTION_MODES:
                _err(
                    f"artifact config: initial_sections[{i}] mode={mode!r} is invalid. "
                    f"Must be one of: replace, append."
                )

    owners_cfg = cfg.get("tool_owners")
    if owners_cfg is None:
        return
    if not isinstance(owners_cfg, dict):
        _err("artifact config: 'tool_owners' must be a mapping.")
    for tool_name, value in owners_cfg.items():
        if tool_name not in ARTIFACT_TOOL_NAMES:
            _err(
                f"artifact.tool_owners: '{tool_name}' is not an artifact tool. "
                f"Allowed keys: {sorted(ARTIFACT_TOOL_NAMES)}."
            )
        where = f"artifact.tool_owners['{tool_name}']"
        _validate_who(value, agent_names, where)
        _validate_who_role_reachability(value, agent_roles, where)


def _resolve_tool_owners(
    owners_cfg: dict | None,
    agents: list[dict],
    agent_roles: dict[str, str],
) -> dict[str, list[str]]:
    """Expand owner expressions into flat ``{tool: [agent_name, ...]}`` allowlists.

    Mirrors the four ``who`` literal forms (run.py:_validate_who):
    - scalar "moderator" / "member" → all agents with that role, in declaration order
    - scalar "all" → every agent, in declaration order
    - list[str] → name list as written
    """
    if not owners_cfg:
        return {}
    declared_order = [a["name"] for a in agents]
    out: dict[str, list[str]] = {}
    for tool_name, value in owners_cfg.items():
        if isinstance(value, str):
            if value == "all":
                out[tool_name] = list(declared_order)
            else:
                out[tool_name] = [n for n in declared_order if agent_roles.get(n) == value]
        elif isinstance(value, list):
            out[tool_name] = list(value)
    return out


# -- main --------------------------------------------------------------------

def main() -> None:
    # Line-buffer both streams so `2>&1 | tee` preserves chronological order
    # between stdout (speaker text, tool traces) and stderr (WARNINGs).
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="Multi-Agent Discussion Engine")
    parser.add_argument("scenario", help="scenario .md file path")
    parser.add_argument("--no-stream", action="store_true",
                        help="disable streaming output")
    parser.add_argument("--save-artifact", metavar="PATH", default=None,
                        help="after the run, write the final artifact markdown to PATH "
                             "(only when the scenario has artifact enabled)")
    parser.add_argument("--save-transcript", metavar="PATH", default=None,
                        help="after the run, dump the structured history (topic, "
                             "turn marker, speaker turns, artifact_event, tool_call) "
                             "to PATH as JSON")
    args = parser.parse_args()

    meta, body = load_scenario(args.scenario)
    scenario_dir = os.path.dirname(os.path.abspath(args.scenario))

    agents_cfg = meta.get("agents")
    agent_names: set[str] = set()
    agent_roles: dict[str, str] = {}
    _validate_agents(agents_cfg, agent_names, agent_roles)

    steps_cfg = meta.get("steps")
    _validate_steps(steps_cfg, agent_names, agent_roles)

    scenario_mem_cfg = meta.get("memory")
    _validate_memory(scenario_mem_cfg, "scenario")
    for s in agents_cfg:
        if "memory" in s:
            _validate_memory(s["memory"], f"agent '{s.get('name')}'")

    artifact_cfg = meta.get("artifact")
    _validate_artifact(artifact_cfg, agent_names, agent_roles)

    stream = not args.no_stream

    tool_configs = meta.get("tools", [])
    base_tool_defs = _resolve_tool_defs(tool_configs) if tool_configs else []
    base_handler = _build_tool_handler(tool_configs, scenario_dir) if tool_configs else None

    store: ArtifactStore | None = None
    if isinstance(artifact_cfg, dict) and artifact_cfg.get("enabled"):
        resolved_owners = _resolve_tool_owners(
            artifact_cfg.get("tool_owners"),
            agents_cfg,
            agent_roles,
        )
        store = ArtifactStore(
            initial_sections=artifact_cfg.get("initial_sections"),
            tool_owners=resolved_owners,
        )

    # One tracer per Discussion; handed to both the per-agent handler (for
    # recording) and the Discussion (for draining into history).
    tracer = ToolTracer() if base_handler is not None else None

    def _agent_bundle(agent_name: str):
        """Compose per-agent (tool_defs, handler) baking in caller name."""
        defs = list(base_tool_defs)
        if store is not None:
            defs.extend(store.build_tool_defs(agent_name))
        if not defs:
            return None, None

        def handler(name: str, args: dict, *, _caller=agent_name) -> str:
            if store is not None and name in ARTIFACT_TOOL_NAMES:
                # Artifact tools have their own emoji print + artifact_event
                # channel — don't double-record via tracer.
                return store.dispatch(name, args, caller=_caller)
            if base_handler is None:
                return json.dumps({"error": f"Unknown tool: {name}"})
            result = base_handler(name, args)
            if tracer is not None:
                tracer.record(_caller, name, args, result)
            return result

        return defs, handler

    agents: list[Agent] = []
    for s in agents_cfg:
        defs, handler = _agent_bundle(s["name"])
        agents.append(_build_agent(s, tool_defs=defs, tool_handler=handler,
                                   scenario_mem_cfg=scenario_mem_cfg))

    history = Discussion(
        agents=agents,
        agent_roles=agent_roles,
        topic=body,
        steps=steps_cfg,
        stream=stream,
        artifact=store,
        tracer=tracer,
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

    if args.save_transcript:
        out_path = os.path.abspath(args.save_transcript)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"\n💾 transcript saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()

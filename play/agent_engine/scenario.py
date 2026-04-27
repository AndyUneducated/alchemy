from __future__ import annotations

import copy
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Callable

import yaml

from .agent import Agent, _client as _backend_client
from .artifact import ARTIFACT_TOOL_NAMES, ArtifactStore
from .config import SUMMARY_MAX_TOKENS, SUMMARY_MODEL, SUMMARY_TEMPERATURE
from .memory import (
    ConversationMemory,
    FullHistory,
    SummaryMemory,
    WindowMemory,
)
from .tools import TOOL_DEFINITIONS, dispatch
from .tracer import ToolTracer


VALID_ROLES = {"moderator", "member"}
VALID_WHO_SCALARS = {"moderator", "member", "all"}
VALID_MEMORY_TYPES = {"full", "window", "summary"}
VALID_SECTION_MODES = {"replace", "append"}


def _err(msg: str) -> None:
    sys.exit(f"Error: {msg}")


def _validate_agents(
    agents: list,
    agent_names: set[str],
    agent_roles: dict[str, str],
) -> None:
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
    if isinstance(who, str):
        if who not in VALID_WHO_SCALARS:
            _err(
                f"{where}: who={who!r} is not a valid scalar. "
                f"Must be one of: moderator, member, all "
                f"(or use a list of agent names)."
            )
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


def _validate_who_role_reachability(
    who, agent_roles: dict[str, str], where: str
) -> None:
    if not isinstance(who, str) or who == "all":
        return
    if not any(r == who for r in agent_roles.values()):
        _err(
            f"{where}: who={who!r} matches 0 agents — no agent has role={who!r}."
        )


def _validate_steps(
    steps,
    agent_names: set[str],
    agent_roles: dict[str, str],
) -> None:
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


def _validate_artifact(
    cfg: dict | None,
    agent_names: set[str],
    agent_roles: dict[str, str],
) -> None:
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


def _build_tool_handler(
    tool_configs: list[dict], scenario_dir: str
) -> Callable[[str, dict], str]:
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
        merged = {**arguments, **defaults.get(name, {})}
        return dispatch(name, merged)

    return handler


def _resolve_tool_defs(tool_configs: list[dict]) -> list[dict]:
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


def _build_agent(
    spec: dict,
    *,
    tool_defs: list[dict] | None = None,
    tool_handler: Callable[[str, dict], str] | None = None,
    scenario_mem_cfg: dict | None = None,
) -> Agent:
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


def _resolve_tool_owners(
    owners_cfg: dict | None,
    agents: list[dict],
    agent_roles: dict[str, str],
) -> dict[str, list[str]]:
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


_FRONTMATTER_RE = re.compile(
    r"\A(?:[^\n]*\n)*?^---\s*\n(?P<meta>.*?)\n^---\s*\n?(?P<body>.*)\Z",
    re.DOTALL | re.MULTILINE,
)


def _split_frontmatter(text: str) -> tuple[str | None, str]:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return None, text
    return m.group("meta"), m.group("body").strip()


@dataclass
class Assembly:
    agents: list[Agent]
    agent_roles: dict[str, str]
    steps: list[dict]
    topic: str
    artifact: ArtifactStore | None
    tracer: ToolTracer | None


@dataclass
class Scenario:
    path: str
    meta: dict
    body: str
    scenario_dir: str

    agent_names: set[str] = field(default_factory=set)
    agent_roles: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "Scenario":
        with open(path, encoding="utf-8") as f:
            text = f.read()

        meta_text, body = _split_frontmatter(text)
        if meta_text is None:
            sys.exit(f"Error: {path} has no YAML frontmatter (--- ... ---)")
        meta = yaml.safe_load(meta_text)
        if not isinstance(meta, dict):
            sys.exit(f"Error: {path} frontmatter is not a valid YAML mapping")

        scenario_dir = os.path.dirname(os.path.abspath(path))
        scn = cls(path=path, meta=meta, body=body, scenario_dir=scenario_dir)

        agents_cfg = meta.get("agents")
        _validate_agents(agents_cfg, scn.agent_names, scn.agent_roles)
        _validate_steps(meta.get("steps"), scn.agent_names, scn.agent_roles)
        _validate_memory(meta.get("memory"), "scenario")
        for s in agents_cfg:
            if "memory" in s:
                _validate_memory(s["memory"], f"agent '{s.get('name')}'")
        _validate_artifact(meta.get("artifact"), scn.agent_names, scn.agent_roles)

        return scn

    def assemble(self) -> Assembly:
        agents_cfg = self.meta["agents"]
        scenario_mem_cfg = self.meta.get("memory")

        tool_configs = self.meta.get("tools", [])
        base_tool_defs = _resolve_tool_defs(tool_configs) if tool_configs else []
        base_handler = (
            _build_tool_handler(tool_configs, self.scenario_dir) if tool_configs else None
        )

        artifact_cfg = self.meta.get("artifact")
        store: ArtifactStore | None = None
        if isinstance(artifact_cfg, dict) and artifact_cfg.get("enabled"):
            resolved_owners = _resolve_tool_owners(
                artifact_cfg.get("tool_owners"),
                agents_cfg,
                self.agent_roles,
            )
            store = ArtifactStore(
                initial_sections=artifact_cfg.get("initial_sections"),
                tool_owners=resolved_owners,
            )

        tracer = ToolTracer() if base_handler is not None else None

        def _agent_bundle(agent_name: str):
            defs = list(base_tool_defs)
            if store is not None:
                defs.extend(store.build_tool_defs(agent_name))
            if not defs:
                return None, None

            def handler(name: str, args: dict, *, _caller=agent_name) -> str:
                if store is not None and name in ARTIFACT_TOOL_NAMES:
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
            agents.append(
                _build_agent(
                    s,
                    tool_defs=defs,
                    tool_handler=handler,
                    scenario_mem_cfg=scenario_mem_cfg,
                )
            )

        return Assembly(
            agents=agents,
            agent_roles=self.agent_roles,
            steps=self.meta["steps"],
            topic=self.body,
            artifact=store,
            tracer=tracer,
        )

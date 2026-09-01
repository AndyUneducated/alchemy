"""F1 SFT sample builder: Triple → MLX-LM `tools` format (OpenAI tool_calls schema).

Schema locking see [`DECISIONS §4`](../DECISIONS.md): assistant message using OpenAI
`tool_calls` field + top-level `tools` lists visible tools, same as Qwen2.5 native chat template
Render target `<tool_call>{"name":..., "arguments":...}</tool_call>` aligned, downstream
Ollama function-call parser + `agent_engine`'s `tool_call` event has the same origin.

Output schema (MLX-LM `tools` data format,
[LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)):

    {
      "messages": [
        {"role": "system", "content": agent.prompt},
        {"role": "user", "content": recent K turn rendering + step.instruction},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "call_0", "type": "function",
                         "function": {"name": "...", "arguments": "{...}"}}]}
      ],
      "tools": [
        {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}},
        ...
      ]
    }

`arguments` uses dict ([LORA.md accepts both](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md);
v1 uses JSON-string to be compatible with Qwen2.5 chat_template, **v1.5 will be replaced with dict** because of Qwen3.5
chat_template uses `tool_call.arguments|items` to strictly require mapping, string will trigger
`TypeError: Can only get item pairs from a mapping.`).

`tools` Source: `tools:` block of scenario YAML (resolve via `agent_engine.scenario._resolve_tool_defs`)
+ `ArtifactStore.build_tool_defs(caller=agent_name)` when `artifact.enabled` (filtered by role
moderator-only tool, same origin as runtime per-agent tool_defs).

drop rules:
  - `synthesize._extract_call_template` cannot find literal `tool(args)` template (like retrieve_docs
    fallback wrapper class) → discard the entire piece;
  - The args in the template have neither strict ast parsing nor tolerant kw/positional extraction → discard the entire args.

Context interception: default max_recent=6 (consistent with code_review.md memory.max_recent)."""
Context interception: default max_recent=6 (consistent with code_review.md memory.max_recent)."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PLAY_DIR = REPO_ROOT / "play"
if str(PLAY_DIR) not in sys.path:
    sys.path.insert(0, str(PLAY_DIR))

# Same dir; synthesize.py injected into PLAY_DIR
sys.path.insert(0, str(Path(__file__).resolve().parent))

# agent_engine is schema single source - _resolve_tool_defs / _resolve_tool_owners
# Completely homologous to runtime per-agent tool_defs to avoid schema drift (DECISIONS §4).
# DECISIONS §13 After frontmatter parsing also goes to `Scenario.from_yaml(path).meta`, no longer
# Directly adjust evals.metrics.nudge._split_frontmatter shim.
from agent_engine import Scenario  # noqa: E402
from agent_engine.scenario import (  # noqa: E402
    _resolve_tool_defs,
    _resolve_tool_owners,
)
from agent_engine.artifact import ArtifactStore  # noqa: E402

from synthesize import _extract_call_template  # noqa: E402

DEFAULT_MAX_RECENT = 6


def format_triple(
    triple: dict[str, Any],
    scenario_path: str | Path,
    *,
    max_recent: int = DEFAULT_MAX_RECENT,
) -> dict[str, Any] | None:
    """Triple dict → SFT sample dict, or None if args are not parsable (drop)."""
    scenario_path = Path(scenario_path)
    meta = _read_scenario_meta(scenario_path)
    agent_name = str(triple.get("agent", ""))
    required_tool = str(triple.get("required_tool", ""))
    instruction = (triple.get("instruction") or "").strip()

    template = _extract_call_template(instruction, required_tool)
    if not template:
        return None  # fallback wrapper class, drop (DECISIONS §4 + user decision)

    tool_defs = _load_tool_defs(meta, agent_name)
    schema = _find_tool_schema(tool_defs, required_tool)
    if schema is None:
        return None  # required_tool is not in the tool list visible to the agent - abnormal situation, defensive drop

    args = _call_template_to_args_dict(template, required_tool, schema)
    if args is None:
        return None  # Template is neither strict nor loose parsing - drop

    system_content = _agent_prompt(meta, agent_name)

    recent = _render_recent_context(triple.get("context") or [], max_recent)
    user_parts: list[str] = []
    if recent:
user_parts.append(f"Recent conversation:\n{recent}")
    user_parts.append(
f"Please execute now:\n{instruction}" if instruction else "Please execute this round of tasks now."
    )
    user_content = "\n\n".join(user_parts)

    assistant_msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": required_tool,
                    "arguments": args,
                },
            }
        ],
    }

    return {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            assistant_msg,
        ],
        "tools": tool_defs,
    }


# --- scenario meta + tool defs ---------------------------------------------

def _read_scenario_meta(scenario_path: Path) -> dict[str, Any]:
    """Go to `Scenario.from_yaml` to get the frontmatter dict - the schema check has the same origin as agent_engine."""
    """Go to `Scenario.from_yaml` to get the frontmatter dict - the schema check has the same origin as agent_engine."""
    return Scenario.from_yaml(str(scenario_path)).meta


def _agent_prompt(meta: dict[str, Any], agent_name: str) -> str:
    for a in meta.get("agents") or []:
        if a.get("name") == agent_name:
            return str(a.get("prompt", "")).strip()
    return ""


def _load_tool_defs(meta: dict[str, Any], agent_name: str) -> list[dict]:
    """Per-agent runtime view of tool defs (scenario.tools + artifact filtered by role)."""
    """Per-agent runtime view of tool defs (scenario.tools + artifact filtered by role)."""

    Assembled with [`agent_engine.scenario`](../../agent_engine/scenario.py) `_run_turn`
    `tool_defs` is the same as source-of-truth: base scenario tools are shared by all agents; artifact tools
    Individualized by `tool_owners` role filter."""
    Individualized by `tool_owners` role filter."""
    tool_configs = meta.get("tools") or []
    base_defs = list(_resolve_tool_defs(tool_configs)) if tool_configs else []

    artifact_cfg = meta.get("artifact") or {}
    if isinstance(artifact_cfg, dict) and artifact_cfg.get("enabled"):
        agents_cfg = meta.get("agents") or []
        agent_roles = {a["name"]: a.get("role", "member") for a in agents_cfg}
        resolved_owners = _resolve_tool_owners(
            artifact_cfg.get("tool_owners"), agents_cfg, agent_roles
        )
        store = ArtifactStore(
            initial_sections=artifact_cfg.get("initial_sections"),
            tool_owners=resolved_owners,
        )
        base_defs.extend(store.build_tool_defs(caller=agent_name))

# Deep copy prevents downstream mutate from affecting other samples
    return [copy.deepcopy(d) for d in base_defs]


def _find_tool_schema(defs: list[dict], tool_name: str) -> dict | None:
    for d in defs:
        if d.get("function", {}).get("name") == tool_name:
            return d
    return None


# --- args extraction -------------------------------------------------------

def _call_template_to_args_dict(
    call_template: str,
    tool_name: str,
    tool_schema: dict,
) -> dict[str, Any] | None:
    """`tool(arg1, key=val)` → {"prop1": arg1, "key": val}."""

    Two round strategy (covers 99%+ templates):

    1. **strict**: `ast.parse(mode="eval")` + `ast.literal_eval` per arg; clean Python
       Literal calls go this way (append_section/write_section/part cast_vote).
    2. **tolerant fallback**: Only when strict parsing fails (such as cast_vote's `option="combine" or "fallback"`
       Contains invalid Chinese token), split by paren-aware top-level comma, each paragraph try `key=value`-then-string-literal
       Extract; keep the retrieved key name + any first string literal value.

    Constraints:
      - All extracted keys must be in `tool_schema.parameters.properties` - to prevent mismatched
        instruction.
      - Return only if at least 1 key falls into dict; return None if all keys are empty.
      - Fill in "" when required keys are missing - structure-preserving signal, Parameter signal is the secondary target after fallback wrapper drop."""
      - Fill in "" when required keys are missing - structure-preserving signal, Parameter signal is the secondary target after fallback wrapper drop."""
    properties = (tool_schema.get("function", {})
                  .get("parameters", {})
                  .get("properties", {}))
    if not isinstance(properties, dict):
        return None
    prop_names = list(properties.keys())
    required = (tool_schema.get("function", {})
                .get("parameters", {})
                .get("required") or [])

    parsed = _strict_parse(call_template, tool_name, prop_names)
    if parsed is None:
        parsed = _tolerant_parse(call_template, tool_name, prop_names)
    if not parsed:
        return None

# Filter unknown keys + fill in required placeholders
    out = {k: v for k, v in parsed.items() if k in properties}
    if not out:
        return None
    for req in required:
        out.setdefault(req, "")
    return out


def _strict_parse(
    call_template: str, tool_name: str, prop_names: list[str]
) -> dict[str, Any] | None:
    try:
        tree = ast.parse(call_template, mode="eval")
    except SyntaxError:
        return None
    if not isinstance(tree.body, ast.Call):
        return None
    call = tree.body
    out: dict[str, Any] = {}
    pos_idx = 0
    used_keys: set[str] = set()
    for node in call.args:
        try:
            val = ast.literal_eval(node)
        except (ValueError, SyntaxError):
            return None
# positional → mapped to the first unoccupied key in prop_names in declaration order
        while pos_idx < len(prop_names) and prop_names[pos_idx] in used_keys:
            pos_idx += 1
        if pos_idx >= len(prop_names):
            break  # Excess positional is discarded silently, schema takes precedence
        key = prop_names[pos_idx]
        out[key] = val
        used_keys.add(key)
        pos_idx += 1
    for kw in call.keywords:
        if kw.arg is None:
            return None  # **kwargs not parsed
        try:
            val = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            return None
        out[kw.arg] = val
    return out


# Top-level paren-aware comma split, cross-quotes/square brackets are also safe
def _split_top_level_commas(s: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    quote: str | None = None
    buf: list[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if quote is not None:
            buf.append(ch)
            if ch == "\\" and i + 1 < len(s):
                buf.append(s[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ('"', "'"):
            quote = ch
            buf.append(ch)
        elif ch in "([{":
            depth += 1
            buf.append(ch)
        elif ch in ")]}":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
        i += 1
    if buf:
        parts.append("".join(buf).strip())
    return [p for p in parts if p]


_KW_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$", re.DOTALL)
_DQ_LITERAL_RE = re.compile(r'"([^"]*)"')
_SQ_LITERAL_RE = re.compile(r"'([^']*)'")


def _tolerant_parse(
    call_template: str, tool_name: str, prop_names: list[str]
) -> dict[str, Any] | None:
# Peel tool_name( ... ) shell
    head = call_template.find("(")
    tail = call_template.rfind(")")
    if head < 0 or tail < 0 or head >= tail:
        return None
    inner = call_template[head + 1:tail]
    if not inner.strip():
        return None
    parts = _split_top_level_commas(inner)
    if not parts:
        return None

    out: dict[str, Any] = {}
    pos_idx = 0
    used_keys: set[str] = set()
    for part in parts:
        m = _KW_RE.match(part)
        if m:
            key, val_text = m.group(1), m.group(2)
            val = _extract_first_literal(val_text)
            out[key] = val
        else:
# positional → map in declaration order
            while pos_idx < len(prop_names) and prop_names[pos_idx] in used_keys:
                pos_idx += 1
            if pos_idx >= len(prop_names):
                continue
            key = prop_names[pos_idx]
            out[key] = _extract_first_literal(part)
            used_keys.add(key)
            pos_idx += 1
    return out


def _extract_first_literal(text: str) -> Any:
    """Extract the first recognized literal (string / list-of-string) from the cluttered text; return "" on failure."""
    text = text.strip()
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        pass
# list-of-strings: ["a", "b"] / ['a', 'b'] / Chinese mixed
# list-of-strings: ["a", "b"] / ['a', 'b'] / Chinese mixed
# list-of-strings: ["a", "b"] / ['a', 'b'] / Chinese mixed
# list-of-strings: ["a", "b"] / ['a', 'b'] / Chinese mixed
# list-of-strings: ["a", "b"] / ['a', 'b'] / Chinese mixed
# list-of-strings: ["a", "b"] / ['a', 'b'] / Chinese mixed
# list-of-strings: ["a", "b"] / ['a', 'b'] / Chinese mixed
# list-of-strings: ["a", "b"] / ['a', 'b'] / Chinese mixed
# list-of-strings: ["a", "b"] / ['a', 'b'] / Chinese mixed
# list-of-strings: ["a", "b"] / ['a', 'b'] / Chinese mixed
# list-of-strings: ["a", "b"] / ['a', 'b'] / Chinese mixed
# list-of-strings: ["a", "b"] / ['a', 'b'] / Chinese mixed
    list_match = re.search(r"\[(.*?)\]", text, re.DOTALL)
    if list_match:
        items = _DQ_LITERAL_RE.findall(list_match.group(1))
        if not items:
            items = _SQ_LITERAL_RE.findall(list_match.group(1))
        if items:
            return items
    m = _DQ_LITERAL_RE.search(text)
    if m:
        return m.group(1)
    m = _SQ_LITERAL_RE.search(text)
    if m:
        return m.group(1)
    return ""


# --- recent context render -------------------------------------------------

def _render_recent_context(context: list[dict[str, Any]], max_recent: int) -> str:
    """Render context paragraph. context is already a JSONL deserialized list[dict] (from agent_engine §16"""
Each entry has an explicit `type` field; speaker also has `type=="speaker"`)."""
    if max_recent <= 0:
        return ""
    tail = context[-max_recent:] if len(context) > max_recent else list(context)
    lines: list[str] = []
    for entry in tail:
        if not isinstance(entry, dict):
            continue
        kind = entry.get("type")
        if kind == "topic":
lines.append(f"[Topic]{entry.get('content', '')}")
        elif kind == "turn":
            lines.append(f"【{entry.get('content', '')}】")
        elif kind == "speaker":
            lines.append(f"[{entry.get('speaker', '?')}] {entry.get('content', '')}")
        elif kind in ("artifact_event", "tool_call"):
            tool = entry.get("tool", "?")
            caller = entry.get("caller", "?")
lines.append(f"[tool] {caller} → {tool}")
    return "\n".join(lines)


# --- file I/O --------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _write_jsonl(items: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# --- CLI -------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--in", dest="in_path", required=True,
help="triples.jsonl input path (including scenario / agent / context field)",
    )
    parser.add_argument(
        "--out", dest="out_path", required=True,
        help="formatted samples jsonl output path",
    )
    parser.add_argument(
        "--scenarios-root", default=None,
help="scenarios/ directory; default play/agent_engine/scenarios",
    )
    parser.add_argument(
        "--max-recent", type=int, default=DEFAULT_MAX_RECENT,
help=f"How many recent histories are rendered in the user message (default {DEFAULT_MAX_RECENT})",
    )
    args = parser.parse_args(argv)

    scenarios_root = (
        Path(args.scenarios_root) if args.scenarios_root
        else REPO_ROOT / "play" / "agent_engine" / "scenarios"
    )

    triples = _read_jsonl(Path(args.in_path))
    formatted: list[dict[str, Any]] = []
    drop_no_template = 0
    drop_unparseable = 0
    for t in triples:
        scen_path = scenarios_root / f"{t['scenario']}.md"
        sample = format_triple(t, scen_path, max_recent=args.max_recent)
        if sample is None:
            instr = (t.get("instruction") or "").strip()
            if not _extract_call_template(instr, t.get("required_tool", "")):
                drop_no_template += 1
            else:
                drop_unparseable += 1
            continue
        formatted.append(sample)

    _write_jsonl(formatted, Path(args.out_path))
    total = len(triples)
    kept = len(formatted)
    print(
        f"formatted {kept} samples → {args.out_path}\n"
        f"  dropped {drop_no_template} (no call template, fallback wrapper class)\n"
        f"  dropped {drop_unparseable} (call template but args unparseable)\n"
        f"  total in: {total}  kept: {kept}  drop: {drop_no_template + drop_unparseable}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

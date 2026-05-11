"""F1 SFT sample builder: Triple → MLX-LM `tools` 格式 (OpenAI tool_calls schema).

Schema 锁定见 [`DECISIONS §4`](../DECISIONS.md)：assistant message 用 OpenAI
`tool_calls` 字段 + 顶层 `tools` 列出可见工具，与 Qwen2.5 native chat template
渲染目标 `<tool_call>{"name":..., "arguments":...}</tool_call>` 对齐，下游
Ollama function-call 解析器 + `agent_engine` 的 `tool_call` event 同源。

Output schema (MLX-LM `tools` data format,
[LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md))：

    {
      "messages": [
        {"role": "system",    "content": agent.prompt},
        {"role": "user",      "content": 最近 K turn 渲染 + step.instruction},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "call_0", "type": "function",
                         "function": {"name": "...", "arguments": "{...}"}}]}
      ],
      "tools": [
        {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}},
        ...
      ]
    }

`arguments` 用 OpenAI/Mistral 习惯的 JSON-string（[LORA.md 明示](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
两种都接受；Qwen2.5 chat template `arguments | tojson` dict / str 都能渲染）.

`tools` 来源：scenario YAML 的 `tools:` 块（resolve via `agent_engine.scenario._resolve_tool_defs`）
+ `artifact.enabled` 时 `ArtifactStore.build_tool_defs(caller=agent_name)`（按 role 过滤
moderator-only 工具，与 runtime per-agent tool_defs 同源）.

drop 规则：
  - `synthesize._extract_call_template` 找不到字面 `tool(args)` 模板（如 retrieve_docs
    的 fallback wrapper 类）→ 整条丢弃；
  - 模板里 args 既无 strict ast 解析也无 tolerant kw/positional 提取 → 整条丢弃.

context 截取：默认 max_recent=6（与 code_review.md memory.max_recent 一致）.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
PLAY_DIR = REPO_ROOT / "play"
if str(PLAY_DIR) not in sys.path:
    sys.path.insert(0, str(PLAY_DIR))

# Same dir; synthesize.py 已注入 PLAY_DIR
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evals.metrics.nudge import _split_frontmatter  # noqa: E402

# agent_engine 是 schema 单源——_resolve_tool_defs / _resolve_tool_owners
# 与 runtime per-agent tool_defs 完全同源，避免 schema drift（DECISIONS §4）.
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
    """Triple dict → SFT sample dict, or None if args 不可解析（drop）."""
    scenario_path = Path(scenario_path)
    meta = _read_scenario_meta(scenario_path)
    agent_name = str(triple.get("agent", ""))
    required_tool = str(triple.get("required_tool", ""))
    instruction = (triple.get("instruction") or "").strip()

    template = _extract_call_template(instruction, required_tool)
    if not template:
        return None  # fallback wrapper 类，drop（DECISIONS §4 + user 决策）

    tool_defs = _load_tool_defs(meta, agent_name)
    schema = _find_tool_schema(tool_defs, required_tool)
    if schema is None:
        return None  # required_tool 不在 agent 可见工具清单——异常情况，防御性 drop

    args = _call_template_to_args_dict(template, required_tool, schema)
    if args is None:
        return None  # 模板既不严格也不宽松解析——drop

    system_content = _agent_prompt(meta, agent_name)

    recent = _render_recent_context(triple.get("context") or [], max_recent)
    user_parts: list[str] = []
    if recent:
        user_parts.append(f"最近对话:\n{recent}")
    user_parts.append(
        f"现在请执行:\n{instruction}" if instruction else "现在请执行本轮任务。"
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
                    "arguments": json.dumps(args, ensure_ascii=False),
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
    text = scenario_path.read_text(encoding="utf-8")
    meta_text = _split_frontmatter(text)
    if meta_text is None:
        raise ValueError(f"scenario {scenario_path} has no YAML frontmatter")
    meta = yaml.safe_load(meta_text)
    if not isinstance(meta, dict):
        raise ValueError(f"scenario {scenario_path} frontmatter is not a mapping")
    return meta


def _agent_prompt(meta: dict[str, Any], agent_name: str) -> str:
    for a in meta.get("agents") or []:
        if a.get("name") == agent_name:
            return str(a.get("prompt", "")).strip()
    return ""


def _load_tool_defs(meta: dict[str, Any], agent_name: str) -> list[dict]:
    """Per-agent runtime view of tool defs（scenario.tools + artifact filtered by role）.

    与 [`agent_engine.scenario`](../../agent_engine/scenario.py) `_run_turn` 装配
    `tool_defs` 同 source-of-truth：base scenario tools 全 agent 共享；artifact 工具
    按 `tool_owners` role filter 个体化.
    """
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

    # 深拷贝防止下游 mutate 影响其他 sample
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
    """`tool(arg1, key=val)` → {"prop1": arg1, "key": val}.

    两轮策略（覆盖 99%+ 模板）：

    1. **strict**：`ast.parse(mode="eval")` + `ast.literal_eval` 每个 arg；干净 Python
       字面量调用走这条（append_section / write_section / 部分 cast_vote）.
    2. **tolerant fallback**：仅当 strict 解析失败（如 cast_vote 的 `option="合入" 或 "退回"`
       含无效中文 token），按 paren-aware 顶层逗号切分，每段 try `key=value`-then-string-literal
       提取；保留拿得到的键名 + 任意首个字符串字面量值.

    约束：
      - 提取出的所有 key 必须在 `tool_schema.parameters.properties` 内——防御 mismatched
        instruction.
      - 至少有 1 个 key 落进 dict 才返回；全空返 None.
      - required keys 缺失时填 ""——保结构信号，参数信号是 fallback wrapper drop 后的次级目标.
    """
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

    # 过滤未知键 + 补齐 required 占位
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
        # positional → 按声明顺序映射到 prop_names 里第一个未占用的键
        while pos_idx < len(prop_names) and prop_names[pos_idx] in used_keys:
            pos_idx += 1
        if pos_idx >= len(prop_names):
            break  # 多余 positional 静默丢弃，schema 优先
        key = prop_names[pos_idx]
        out[key] = val
        used_keys.add(key)
        pos_idx += 1
    for kw in call.keywords:
        if kw.arg is None:
            return None  # **kwargs 不解析
        try:
            val = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            return None
        out[kw.arg] = val
    return out


# 顶层 paren-aware comma split，跨引号/方括号也安全
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
    # 剥 tool_name( ... ) 外壳
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
            # positional → 按声明顺序映射
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
    """从混乱文本里抽第一个可识别字面量（string / list-of-string）；都失败返 ""."""
    text = text.strip()
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        pass
    # list-of-strings: ["a", "b"] / ['a', 'b'] / 中文混合
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
    if max_recent <= 0:
        return ""
    tail = context[-max_recent:] if len(context) > max_recent else list(context)
    lines: list[str] = []
    for entry in tail:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") == "topic":
            lines.append(f"【主题】{entry.get('content', '')}")
        elif entry.get("type") == "turn":
            lines.append(f"【{entry.get('content', '')}】")
        elif "speaker" in entry:
            lines.append(f"[{entry['speaker']}] {entry.get('content', '')}")
        elif entry.get("type") in ("artifact_event", "tool_call"):
            tool = entry.get("tool", "?")
            caller = entry.get("caller", "?")
            lines.append(f"[工具] {caller} → {tool}")
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
        help="triples.jsonl input path (含 scenario / agent / context 字段)",
    )
    parser.add_argument(
        "--out", dest="out_path", required=True,
        help="formatted samples jsonl output path",
    )
    parser.add_argument(
        "--scenarios-root", default=None,
        help="scenarios/ 目录；默认 play/agent_engine/scenarios",
    )
    parser.add_argument(
        "--max-recent", type=int, default=DEFAULT_MAX_RECENT,
        help=f"user message 里渲染最近多少条 history（默认 {DEFAULT_MAX_RECENT}）",
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

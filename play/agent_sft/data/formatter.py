"""F1 chat-format SFT sample builder: Triple → {messages: [system, user, assistant]}.

F1 only（plan §Decisions）— input 不含 nudge 文本，让模型学"看到原 instruction
就直接调对工具"，而非"被 nudge 后才补".

Output schema (MLX-LM `mlx_lm.lora` 标准 chat format):
    {
      "messages": [
        {"role": "system",    "content": agent.prompt + 工具列表概要},
        {"role": "user",      "content": 最近 K turn 渲染 + step.instruction},
        {"role": "assistant", "content": triple.corrected_response}
      ]
    }

context 截取：默认 max_recent=6（与 code_review.md memory.max_recent 一致）；
pilot 后看 token 分布是否 < 2048 再调.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
PLAY_DIR = REPO_ROOT / "play"
if str(PLAY_DIR) not in sys.path:
    sys.path.insert(0, str(PLAY_DIR))

from evals.metrics.nudge import _split_frontmatter  # noqa: E402

DEFAULT_MAX_RECENT = 6


def format_triple(
    triple: dict[str, Any],
    scenario_path: str | Path,
    *,
    max_recent: int = DEFAULT_MAX_RECENT,
) -> dict[str, Any]:
    """Triple dict (per extractor.Triple asdict view) + scenario YAML path → SFT sample."""
    scenario_path = Path(scenario_path)
    meta = _read_scenario_meta(scenario_path)
    agent_prompt = _agent_prompt(meta, triple["agent"])
    tool_summary = _tool_summary(meta)

    system_content = agent_prompt
    if tool_summary:
        system_content = f"{agent_prompt}\n\n可用工具: {tool_summary}"

    recent = _render_recent_context(triple.get("context") or [], max_recent)
    instruction = (triple.get("instruction") or "").strip()

    user_parts: list[str] = []
    if recent:
        user_parts.append(f"最近对话:\n{recent}")
    user_parts.append(
        f"现在请执行:\n{instruction}" if instruction else "现在请执行本轮任务。"
    )
    user_content = "\n\n".join(user_parts)

    return {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": triple.get("corrected_response", "")},
        ]
    }


# --- helpers --------------------------------------------------------------

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


# Canonical artifact tools (subset that scenarios commonly use). Engine-side
# 真名定义在 agent_engine.artifact.ARTIFACT_TOOL_NAMES，这里硬编码避免反向 import.
_ARTIFACT_TOOL_NAMES = (
    "append_section", "write_section", "cast_vote",
    "propose_vote", "finalize_artifact",
)


def _tool_summary(meta: dict[str, Any]) -> str:
    """Tool 名 comma list — scenario.tools[] + 启用的 artifact 工具."""
    names: list[str] = []
    for tc in meta.get("tools") or []:
        if isinstance(tc, dict) and "name" in tc:
            names.append(str(tc["name"]))
    artifact_cfg = meta.get("artifact") or {}
    if isinstance(artifact_cfg, dict) and artifact_cfg.get("enabled"):
        names.extend(_ARTIFACT_TOOL_NAMES)
    # dedupe 保序
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return ", ".join(out)


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
    for t in triples:
        scen_path = scenarios_root / f"{t['scenario']}.md"
        formatted.append(format_triple(t, scen_path, max_recent=args.max_recent))

    _write_jsonl(formatted, Path(args.out_path))
    print(f"formatted {len(formatted)} samples → {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

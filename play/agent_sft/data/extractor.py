"""Triple extractor: agent_engine envelope + scenario → list of (failed, nudge, corrected) triples.

复用 [`play/evals/metrics/nudge.py`](../../evals/metrics/nudge.py) 的 transcript 切段
机制——不重写 turn marker 切分 / attempt 切分 / failure mode 分类，仅在其上挑出
"first attempt 失败 → 后续 attempt 成功" 的对，组成 SFT 训练正样本.

Triple schema（与 plan §Schemas 对齐）：
  - run_id, scenario, turn_idx, step_id, agent, required_tool, failure_mode
  - context: transcript prefix until first attempt's speaker entry
  - instruction: step.instruction (raw scenario YAML)
  - failed_response: first attempt 的 speaker.content（诊断用，不进 F1 input）
  - nudge: 引擎硬编码 nudge 文本（按 required_tool 复原；不进 F1 input）
  - corrected_response: 最终成功 attempt 的 speaker.content（F1 target）

不产生 triple 的情况：
  - 第一次 attempt 就成功 → 无失败信号
  - 全部 attempts 失败 → 无正样本，丢弃
  - segment 数 < expected turn_idx（subprocess 中途崩）→ 无 attempt 数据
  - failure_mode == 'wrong_args'（deferred to Phase 5 in metrics/nudge.py）→ 防御性 skip
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

# 复用 evals.metrics.nudge — agent_sft 是 evals 的下游消费者，单向 sys.path 注入.
# 必须按 `evals.metrics.nudge` 路径导入：nudge.py 用 `from ..api import Doc` 相对导入，
# 不能裸作为 metrics.nudge 加载（否则 ImportError beyond top-level package）.
REPO_ROOT = Path(__file__).resolve().parents[3]
PLAY_DIR = REPO_ROOT / "play"
if str(PLAY_DIR) not in sys.path:
    sys.path.insert(0, str(PLAY_DIR))

from evals.metrics.nudge import (  # noqa: E402  pylint: disable=wrong-import-position
    _attempt_called_required,
    _resolve_who_to_agents,
    _split_attempts,
    _split_frontmatter,
    classify_failure_mode,
    derive_expected_turns,
)

# 引擎 nudge 文本格式（discussion.py:141-144 硬编码）；按 required_tool 复原
NUDGE_TEMPLATE = "你刚才没有调用 `{tool}` 工具。请现在补上该调用以完成本轮任务。"


@dataclass
class Triple:
    """一条 (failed, nudge, corrected) 监督三元组，准备喂给 formatter."""

    run_id: int
    scenario: str
    turn_idx: int
    step_id: str | None
    agent: str
    required_tool: str
    failure_mode: str  # "missed" | "wrong_tool"（wrong_args deferred）
    context: list[dict[str, Any]]
    instruction: str
    failed_response: str
    nudge: str
    corrected_response: str


def extract_triples(
    envelope: dict[str, Any],
    scenario_path: str | Path,
    *,
    run_id: int,
    scenario_name: str | None = None,
) -> list[Triple]:
    """envelope dict (per agent_engine.result.Result asdict) + scenario YAML → list[Triple]."""
    scenario_path = Path(scenario_path)
    if scenario_name is None:
        scenario_name = scenario_path.stem
    transcript = envelope.get("transcript") or []
    expected = derive_expected_turns(scenario_path)
    steps_by_turn = _index_steps_by_turn(scenario_path)
    indexed_segments = _split_turns_indexed(transcript)

    out: list[Triple] = []
    for exp in expected:
        turn_idx = int(exp["turn_idx"])
        agent = str(exp["agent"])
        required_tool = str(exp["tool"])
        step_id = exp.get("step_id")

        seg_idx = turn_idx - 1
        if seg_idx >= len(indexed_segments):
            continue  # subprocess 中途崩 / scenario 截断 — 无 attempt 可挖
        seg_start, segment = indexed_segments[seg_idx]

        attempts = _split_attempts(segment, agent)
        speaker_entries = [
            (i, e) for i, e in enumerate(segment)
            if isinstance(e, dict) and e.get("speaker") == agent
        ]
        if not attempts or not speaker_entries:
            continue  # agent 在该 segment 完全沉默
        if _attempt_called_required(attempts[0], agent, required_tool):
            continue  # 第一次 attempt 就成功 — 无 supervision signal

        success_idx = next(
            (
                i for i, att in enumerate(attempts)
                if _attempt_called_required(att, agent, required_tool)
            ),
            None,
        )
        if success_idx is None:
            continue  # 全部失败 — 无正样本
        if success_idx >= len(speaker_entries):
            continue  # 防御：speaker entry 与 attempt 应 1:1 对应

        failure_mode = classify_failure_mode(attempts[0], agent, required_tool)
        if failure_mode == "wrong_args":
            continue  # deferred；防御性 skip

        first_speaker_local_idx, first_speaker_entry = speaker_entries[0]
        _, success_speaker_entry = speaker_entries[success_idx]
        failed_content = str(first_speaker_entry.get("content", ""))
        corrected_content = str(success_speaker_entry.get("content", ""))

        first_speaker_global_idx = seg_start + first_speaker_local_idx
        context = list(transcript[:first_speaker_global_idx])

        instruction = _step_instruction(steps_by_turn.get(turn_idx))

        out.append(Triple(
            run_id=run_id,
            scenario=scenario_name,
            turn_idx=turn_idx,
            step_id=step_id,
            agent=agent,
            required_tool=required_tool,
            failure_mode=failure_mode,
            context=context,
            instruction=instruction,
            failed_response=failed_content,
            nudge=NUDGE_TEMPLATE.format(tool=required_tool),
            corrected_response=corrected_content,
        ))
    return out


# --- helpers --------------------------------------------------------------

def _split_turns_indexed(transcript: list[dict]) -> list[tuple[int, list[dict]]]:
    """metrics.nudge.split_turns 的索引版——同时返回每段的全局起始 idx.

    必要：要把段内 local index（如第 i 个 speaker）映射回 transcript 全局位置，
    才能切出"first attempt 之前的全部 history"作为 context.
    """
    segments: list[tuple[int, list[dict]]] = []
    started = False
    start_idx = -1
    current: list[dict] = []
    for i, entry in enumerate(transcript):
        if isinstance(entry, dict) and entry.get("type") == "turn":
            if started:
                segments.append((start_idx, current))
            current = []
            start_idx = i + 1
            started = True
            continue
        if started:
            current.append(entry)
    if started:
        segments.append((start_idx, current))
    return segments


def _index_steps_by_turn(scenario_path: str | Path) -> dict[int, dict]:
    """turn_idx → step dict（含 instruction 等所有字段；不仅是 require_tool turn）.

    derive_expected_turns 只返 require_tool 必要字段，丢了 instruction 文本——
    formatter 需要原 instruction 拼 user message，所以这里重新展开一份完整映射.
    """
    text = Path(scenario_path).read_text(encoding="utf-8")
    meta_text = _split_frontmatter(text)
    if meta_text is None:
        raise ValueError(f"scenario {scenario_path} has no YAML frontmatter")
    meta = yaml.safe_load(meta_text)
    if not isinstance(meta, dict):
        raise ValueError(f"scenario {scenario_path} frontmatter is not a mapping")
    agents = meta.get("agents") or []
    roles = {a["name"]: a.get("role", "member") for a in agents}
    steps = meta.get("steps") or []
    out: dict[int, dict] = {}
    turn_idx = 0
    for step in steps:
        who = step.get("who")
        expanded = _resolve_who_to_agents(who, agents, roles)
        for _ in expanded:
            turn_idx += 1
            out[turn_idx] = step
    return out


def _step_instruction(step: dict | None) -> str:
    if step is None:
        return ""
    instr = step.get("instruction")
    return str(instr).strip() if isinstance(instr, str) else ""


# --- file I/O -------------------------------------------------------------

def write_triples_jsonl(triples: list[Triple], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for t in triples:
            f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")


def _parse_envelope_name(stem: str) -> tuple[str, int]:
    """'tool_chain-r3' → ('tool_chain', 3)."""
    if "-r" not in stem:
        raise ValueError(f"envelope filename must match '<scenario>-r<N>': {stem!r}")
    scen, _, run = stem.rpartition("-r")
    try:
        run_id = int(run)
    except ValueError as exc:
        raise ValueError(f"envelope filename run_id not int: {stem!r}") from exc
    return scen, run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--in", dest="in_dir", required=True,
        help="directory of envelope JSONs named '<scenario>-r<N>.json'",
    )
    parser.add_argument(
        "--out", dest="out_path", required=True,
        help="output triples.jsonl path",
    )
    parser.add_argument(
        "--scenarios-root", default=None,
        help="agent_engine scenarios/ dir; default play/agent_engine/scenarios",
    )
    args = parser.parse_args(argv)

    in_dir = Path(args.in_dir)
    if not in_dir.is_dir():
        print(f"ERROR: --in {in_dir} is not a directory", file=sys.stderr)
        return 2

    scenarios_root = (
        Path(args.scenarios_root) if args.scenarios_root
        else REPO_ROOT / "play" / "agent_engine" / "scenarios"
    )

    envelopes = sorted(in_dir.glob("*.json"))
    if not envelopes:
        print(f"ERROR: no envelope JSONs under {in_dir}", file=sys.stderr)
        return 2

    all_triples: list[Triple] = []
    per_file_summary: list[tuple[str, int]] = []
    for env_path in envelopes:
        scen_name, run_id = _parse_envelope_name(env_path.stem)
        scen_path = scenarios_root / f"{scen_name}.md"
        with env_path.open("r", encoding="utf-8") as f:
            envelope = json.load(f)
        triples = extract_triples(
            envelope, scen_path, run_id=run_id, scenario_name=scen_name
        )
        per_file_summary.append((env_path.name, len(triples)))
        all_triples.extend(triples)

    write_triples_jsonl(all_triples, args.out_path)
    print(f"\n=== Extraction summary ===")
    for name, count in per_file_summary:
        print(f"  {name}: {count} triples")
    print(f"  TOTAL: {len(all_triples)} triples → {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

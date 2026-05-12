"""Triple extractor: agent_engine envelope + scenario → list of (failed, nudge, corrected) triples.

DECISIONS §13 后直连 `agent_engine.Scenario / Result / TurnView`：transcript 切段
（`Result.turns()`，`TurnView.start_offset` 提供全局 offset）+ 段内 attempt 切分
（`TurnView.attempts(agent)`）+ 静态 step 展开（`Scenario.expanded_turns()`，含
`instruction` 透传）全部由 agent_engine 提供。§16 升级到 `TranscriptEntry` typed
union（`SpeakerEntry / ToolCallEntry / ArtifactEventEntry / ...`），消费侧用 isinstance
dispatch 取字段，不再 `entry.get("...")` 防御。本模块仅保留：
  - "first attempt 失败 → 后续 attempt 成功" 配对挑选
  - failure_mode 分类（仍 `from evals.metrics.nudge import classify_failure_mode`，
    这是 evals 公开面，跨项目 import 合法）
  - SFT triple schema 形态

Triple schema（与 plan §Schemas 对齐）：
  - run_id, scenario, turn_idx, step_id, agent, required_tool, failure_mode
  - context: transcript prefix until first attempt's speaker entry（list[TranscriptEntry]，
    JSON 序列化时由 dataclasses.asdict 转 list[dict]）
  - instruction: step.instruction (raw scenario YAML)
  - failed_response: first attempt 的 SpeakerEntry.content（诊断用，不进 F1 input）
  - nudge: 引擎硬编码 nudge 文本（按 required_tool 复原；不进 F1 input）
  - corrected_response: 最终成功 attempt 的 SpeakerEntry.content（F1 target）

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
from dataclasses import asdict, dataclass, field
from pathlib import Path

# agent_engine 与 evals.metrics.nudge.classify_failure_mode 都是同 monorepo 姊妹包，
# 单向 sys.path 注入即可让 import 解析；与 evals/_ae_bridge.py 同思路.
REPO_ROOT = Path(__file__).resolve().parents[3]
PLAY_DIR = REPO_ROOT / "play"
if str(PLAY_DIR) not in sys.path:
    sys.path.insert(0, str(PLAY_DIR))

from agent_engine import (  # noqa: E402  pylint: disable=wrong-import-position
    ArtifactEventEntry,
    ExpandedTurn,
    Result,
    Scenario,
    SpeakerEntry,
    ToolCallEntry,
    TranscriptEntry,
    TurnView,
)
from evals.metrics.nudge import (  # noqa: E402  pylint: disable=wrong-import-position
    classify_failure_mode,
)

# 引擎 nudge 文本格式（discussion.py 硬编码）；按 required_tool 复原
NUDGE_TEMPLATE = "你刚才没有调用 `{tool}` 工具。请现在补上该调用以完成本轮任务。"

# scenarios_root / filename 解析：默认走 mine_triples 同款 fast 副本（max_retries=0
# / 删 open+finalize / 短 max_tokens），--upstream 切回 agent_engine/scenarios/<name>.md.
# 必须与 envelope 生成时所用 scenario YAML 保持一致——Scenario.expanded_turns 按 step
# 顺序展开 turn_idx，fast 副本删了 open / finalize 后 turn_idx 与上游相差 1，混用会
# 导致 expected agent / required_tool / step.instruction 全部错位（synthesize 仍 0
# triple，extractor 看 attempts 跨 segment 也会全 miss）.
FAST_SCENARIOS_DIR = REPO_ROOT / "play" / "agent_sft" / "data" / "scenarios"
UPSTREAM_SCENARIOS_DIR = REPO_ROOT / "play" / "agent_engine" / "scenarios"


def resolve_scenario_path(scenario_name: str, *, upstream: bool) -> Path:
    """Mirror mine_triples.py 的 fast / upstream 路径选择策略."""
    if upstream:
        return UPSTREAM_SCENARIOS_DIR / f"{scenario_name}.md"
    return FAST_SCENARIOS_DIR / f"{scenario_name}_fast.md"


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
    context: list[TranscriptEntry] = field(default_factory=list)
    instruction: str = ""
    failed_response: str = ""
    nudge: str = ""
    corrected_response: str = ""


def _attempt_called_required(
    events: list[TranscriptEntry], agent: str, tool: str,
) -> bool:
    """attempt 内是否有 `(caller=agent, tool=required_tool)` 工具事件——同 agent_engine
    `discussion._called_tool` 检查面."""
    for e in events:
        if isinstance(e, (ToolCallEntry, ArtifactEventEntry)):
            if e.caller == agent and e.tool == tool:
                return True
    return False


def extract_triples(
    envelope: dict,
    scenario_path: str | Path,
    *,
    run_id: int,
    scenario_name: str | None = None,
) -> list[Triple]:
    """envelope dict (per agent_engine.result.Result asdict) + scenario YAML → list[Triple]."""
    scenario_path = Path(scenario_path)
    if scenario_name is None:
        scenario_name = scenario_path.stem

    result = Result.from_dict(envelope)
    transcript = result.transcript
    expanded = Scenario.from_yaml(str(scenario_path)).expanded_turns()
    turns: list[TurnView] = result.turns()
    expanded_by_turn: dict[int, ExpandedTurn] = {e.turn_idx: e for e in expanded}

    out: list[Triple] = []
    for exp in expanded:
        if not exp.require_tool:
            continue
        turn_idx = exp.turn_idx
        agent = exp.agent
        required_tool = exp.require_tool

        seg_idx = turn_idx - 1
        if seg_idx >= len(turns):
            continue  # subprocess 中途崩 / scenario 截断 — 无 attempt 可挖
        tv = turns[seg_idx]

        attempts = tv.attempts(agent)
        speaker_entries = [
            (i, e) for i, e in enumerate(tv.entries)
            if isinstance(e, SpeakerEntry) and e.speaker == agent
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
        failed_content = first_speaker_entry.content
        corrected_content = success_speaker_entry.content

        first_speaker_global_idx = tv.start_offset + first_speaker_local_idx
        context = list(transcript[:first_speaker_global_idx])

        instruction = expanded_by_turn[turn_idx].instruction.strip()

        out.append(Triple(
            run_id=run_id,
            scenario=scenario_name,
            turn_idx=turn_idx,
            step_id=exp.step_id,
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
        "--upstream", action="store_true",
        help="用上游 agent_engine/scenarios/<name>.md 解析（与 baseline eval 一致）；"
             "默认走 fast 副本 data/scenarios/<name>_fast.md，必须匹配 mine_triples 用的版本",
    )
    parser.add_argument(
        "--scenarios-root", default=None,
        help="显式覆盖 scenarios 目录，少用——优先用 --upstream / 默认 fast 副本",
    )
    args = parser.parse_args(argv)

    in_dir = Path(args.in_dir)
    if not in_dir.is_dir():
        print(f"ERROR: --in {in_dir} is not a directory", file=sys.stderr)
        return 2

    explicit_root = Path(args.scenarios_root) if args.scenarios_root else None

    envelopes = sorted(in_dir.glob("*.json"))
    if not envelopes:
        print(f"ERROR: no envelope JSONs under {in_dir}", file=sys.stderr)
        return 2

    all_triples: list[Triple] = []
    per_file_summary: list[tuple[str, int]] = []
    for env_path in envelopes:
        scen_name, run_id = _parse_envelope_name(env_path.stem)
        if explicit_root is not None:
            scen_path = explicit_root / f"{scen_name}.md"
        else:
            scen_path = resolve_scenario_path(scen_name, upstream=args.upstream)
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

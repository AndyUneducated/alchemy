"""Synthetic triple builder: 从 agent_engine envelope 挖「真失败 attempt + 合成
正确响应」三元组——绕过"等模型自己 recovery"的稀薄信号路径.

Approach B（与 [`extractor.py`](extractor.py) 区别）：

| 配对策略           | extractor.py                      | synthesize.py                              |
|--------------------|-----------------------------------|--------------------------------------------|
| 触发条件           | first attempt 失败 + 后续 attempt 成功 | first attempt 失败（即任一 nudge fired）|
| corrected_response | 真实成功 attempt 的 speaker.content   | 程序化按 step.instruction + tool 名合成 |
| yield              | ~3-25%（取决于 model recovery 率） | 100%（所有 fire）                          |
| supervision 语义   | "模型自己改对的真实样本"           | "失败示范是真的，标准答案是模板的"         |

为什么用：7B 在大多数 scenario 上 nudge recovery 率仅 ~3%，导致 extractor 路径
yield 极低。synthesize 路径用 instruction 里的字面 `tool(args)` 模板（fallback：通用
"我现在调用 X" 包装）造正确响应，每次 fire 都能产出训练样本.

支持 [`extractor.py`](extractor.py) 同款 Triple schema，下游 [`split.py`](split.py) /
[`formatter.py`](formatter.py) 不感知数据来源.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

# 共用 extractor 的 PLAY_DIR sys.path 注入路径，确保 evals.metrics.nudge 可解析
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

# 复用 extractor 的 Triple + helpers + scenario 路径解析，避免数据 schema 漂移
from extractor import (  # noqa: E402  pylint: disable=wrong-import-position
    NUDGE_TEMPLATE,
    Triple,
    _index_steps_by_turn,
    _parse_envelope_name,
    _split_turns_indexed,
    _step_instruction,
    resolve_scenario_path,
    write_triples_jsonl,
)


def synthesize_corrected_response(instruction: str, required_tool: str) -> str:
    """根据 step.instruction 文本造一个 'corrected' 响应字符串。

    优先：抓 instruction 里的字面调用模板（如 `append_section("review_a", "...")`）作主体.
    Fallback：用通用 wrapper "我现在调用 {tool} 完成本步：\\n{instruction}".

    模板抓取支持：
      - 单层 paren，跨行 args（多数 scenario.instruction 是这种形态）
      - 中文引号 / 字符串字面量混合（regex 不解析内部，只到第一个 unbalanced `)`)

    返回值是确定性的纯文本——同样 (instruction, tool) 永远产同样输出.
    """
    instruction = (instruction or "").strip()
    template = _extract_call_template(instruction, required_tool)
    if template:
        return f"好的，我现在调用 `{required_tool}`：\n\n{template}"
    return (
        f"好的，我现在调用 `{required_tool}` 完成本步：\n\n{instruction}"
        if instruction
        else f"好的，我现在调用 `{required_tool}`。"
    )


def _extract_call_template(instruction: str, tool: str) -> str | None:
    """在 instruction 里找字面 `{tool}(...)` 片段；找不到返 None.

    匹配策略：
      - `\\b{tool}\\s*\\(` 起始
      - args 取到第一个 unbalanced `)` （单层括号，足够覆盖 scenario 实际写法）
      - 不解析内部字符串字面量；只看 paren 平衡
    """
    needle = re.escape(tool)
    pattern = re.compile(rf"\b{needle}\s*\(", re.MULTILINE)
    m = pattern.search(instruction)
    if not m:
        return None
    start = m.start()
    open_paren = m.end() - 1
    depth = 0
    for i in range(open_paren, len(instruction)):
        ch = instruction[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return instruction[start:i + 1]
    return None  # 不平衡 paren — 不强行匹配


def envelope_to_synthetic_triples(
    envelope: dict[str, Any],
    scenario_path: str | Path,
    *,
    run_id: int,
    scenario_name: str | None = None,
) -> list[Triple]:
    """envelope dict + scenario YAML → list[Triple]，配对策略 = "每个 fire → 1 triple"."""
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
            continue  # subprocess 中途崩
        seg_start, segment = indexed_segments[seg_idx]

        attempts = _split_attempts(segment, agent)
        speaker_entries = [
            (i, e) for i, e in enumerate(segment)
            if isinstance(e, dict) and e.get("speaker") == agent
        ]
        if not attempts or not speaker_entries:
            continue
        if _attempt_called_required(attempts[0], agent, required_tool):
            continue  # 第一次就成功 — 没 nudge fire，不造 triple

        failure_mode = classify_failure_mode(attempts[0], agent, required_tool)
        if failure_mode == "wrong_args":
            continue  # deferred to Phase 5；防御性 skip

        first_speaker_local_idx, first_speaker_entry = speaker_entries[0]
        failed_content = str(first_speaker_entry.get("content", ""))
        first_speaker_global_idx = seg_start + first_speaker_local_idx
        context = list(transcript[:first_speaker_global_idx])

        instruction = _step_instruction(steps_by_turn.get(turn_idx))
        corrected_content = synthesize_corrected_response(instruction, required_tool)

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="与 extractor.py 互斥关系：取一种数据策略产 triples.jsonl 即可。",
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
        help="用上游 agent_engine/scenarios/<name>.md 解析；默认走 fast 副本 "
             "data/scenarios/<name>_fast.md，必须匹配 mine_triples 用的版本",
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
        triples = envelope_to_synthetic_triples(
            envelope, scen_path, run_id=run_id, scenario_name=scen_name
        )
        per_file_summary.append((env_path.name, len(triples)))
        all_triples.extend(triples)

    write_triples_jsonl(all_triples, args.out_path)
    print(f"\n=== Synthetic extraction summary (Approach B) ===")
    for name, count in per_file_summary:
        print(f"  {name}: {count} triples")
    print(f"  TOTAL: {len(all_triples)} triples → {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

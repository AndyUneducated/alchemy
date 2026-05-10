"""生成 nudge_fire_rate 的 stub predictions.

3 份 prediction × 7 个 doc = 21 个 transcript 矩阵（5 既有 + 2 新增 require_tool 密集）：
  | prediction   | brainstorm/debate/roundtable | example | panel | tool_chain | code_review |
  |---|---|---|---|---|---|
  | perfect      | n/a (vacuous)                | rate=0  | rate=0| rate=0     | rate=0      |
  | all_nudged   | n/a                          | rate=1  | rate=1| rate=1     | rate=1      |
  | mixed        | n/a                          | 中间态  | 中间态| 中间态     | 中间态      |

3 个无 require_tool 的 scenario 永远 nudge_fire_rate=None（vacuous），是聚合通路烟测.
2 个新增 require_tool 密集 scenario 提供更多 nudge-eligible turn，把单 run 数据点
从 7 提到 20，置信区间显著收紧（详见 plan §1.B 与 DECISIONS Phase 1 ADR §6）.

stub schema 与 agent_engine.result.Result envelope 同形 + 加 `id` 字段供 join.
agent_traj 用 _pin_trajectory 不要求 turn marker；nudge_fire_rate 必须有
`{"type":"turn"}` marker 切段——故 stub 比 agent_traj 详细一档.

跑：`python play/evals/data/nudge_fire_rate/predictions/_build.py`
"""

from __future__ import annotations

import json
import time
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

# ---------- scenario 展开（与 derive_expected_turns 等价的硬编码 spec）---------

# brainstorm.md：open=[前端,PM] (2 turn), refine=all=[前端,后端,PM] (3 turn) → 5 turn
BRAINSTORM_SCHEDULE = [
    ("前端", None), ("PM", None),
    ("前端", None), ("后端", None), ("PM", None),
]

# debate.md：r1=all=[乐观,怀疑] (2), r2=同 (2) → 4 turn
DEBATE_SCHEDULE = [
    ("乐观主义者", None), ("怀疑论者", None),
    ("乐观主义者", None), ("怀疑论者", None),
]

# roundtable.md：open=moderator=[主持人] (1), discuss=member=[嘉宾A,嘉宾B] (2), close=moderator (1) → 4 turn
ROUNDTABLE_SCHEDULE = [
    ("主持人", None),
    ("嘉宾A", None), ("嘉宾B", None),
    ("主持人", None),
]

# example.md：19 turn 总（见 derive_expected_turns 结果）
# step expansion:
#   turn 1: open (主持人)
#   turn 2-4: ci_who_member (分析师, 决策者, 汇总员)
#   turn 5-8: ci_who_all (主持人, 分析师, 决策者, 汇总员)
#   turn 9-11: mem_warm (分析师, 决策者, 汇总员)
#   turn 12-14: mem_warm2 (分析师, 决策者, 汇总员)
#   turn 15: vdb_artifact (分析师, require_tool=append_section)  ← REQUIRE
#   turn 16: vote_prep (主持人)
#   turn 17: ballot_nudge (分析师, require_tool=cast_vote)        ← REQUIRE
#   turn 18: ballot_ok (决策者, require_tool=cast_vote)           ← REQUIRE
#   turn 19: finalize (主持人)
EXAMPLE_SCHEDULE = [
    ("主持人", None),
    ("分析师", None), ("决策者", None), ("汇总员", None),
    ("主持人", None), ("分析师", None), ("决策者", None), ("汇总员", None),
    ("分析师", None), ("决策者", None), ("汇总员", None),
    ("分析师", None), ("决策者", None), ("汇总员", None),
    ("分析师", "append_section"),  # 15 — REQUIRE
    ("主持人", None),
    ("分析师", "cast_vote"),       # 17 — REQUIRE
    ("决策者", "cast_vote"),       # 18 — REQUIRE
    ("主持人", None),
]

# panel.md：26 turn 总
# step expansion:
#   turn 1: kickoff (CEO)
#   turn 2-5: stance (4 members)
#   turn 6-9: r1_member (4)
#   turn 10: r1_summary (CEO)
#   turn 11-14: r2_member (4)
#   turn 15: r2_summary (CEO)
#   turn 16-19: r3_member (4)
#   turn 20: r3_summary (CEO)
#   turn 21: open_vote (CEO)
#   turn 22-25: ballot (4 members, require_tool=cast_vote)  ← REQUIRE × 4
#   turn 26: finalize (CEO)
PANEL_MEMBERS = ["产品VP 林晚晴", "销售总监 马千里", "CFO 钱正清", "新业务负责人 孙未来"]
PANEL_SCHEDULE = (
    [("CEO 赵铁军", None)]
    + [(m, None) for m in PANEL_MEMBERS]
    + [(m, None) for m in PANEL_MEMBERS]
    + [("CEO 赵铁军", None)]
    + [(m, None) for m in PANEL_MEMBERS]
    + [("CEO 赵铁军", None)]
    + [(m, None) for m in PANEL_MEMBERS]
    + [("CEO 赵铁军", None)]
    + [("CEO 赵铁军", None)]
    + [(m, "cast_vote") for m in PANEL_MEMBERS]  # 22-25
    + [("CEO 赵铁军", None)]
)

# tool_chain.md：8 turn 总，单 agent
# step expansion:
#   turn 1: open (协调者)
#   turn 2: ctx_round1 (执行者, retrieve_docs)    ← REQUIRE
#   turn 3: note_round1 (执行者, append_section)  ← REQUIRE
#   turn 4: vote_setup (协调者)
#   turn 5: ballot (执行者, cast_vote)            ← REQUIRE
#   turn 6: ctx_round2 (执行者, retrieve_docs)    ← REQUIRE
#   turn 7: note_round2 (执行者, append_section)  ← REQUIRE
#   turn 8: finalize (协调者)
TOOL_CHAIN_SCHEDULE = [
    ("协调者", None),
    ("执行者", "retrieve_docs"),
    ("执行者", "append_section"),
    ("协调者", None),
    ("执行者", "cast_vote"),
    ("执行者", "retrieve_docs"),
    ("执行者", "append_section"),
    ("协调者", None),
]

# code_review.md：11 turn 总
# step expansion:
#   turn 1: open (主审)
#   turn 2: ctx_a (工程师A, retrieve_docs)            ← REQUIRE
#   turn 3: ctx_b (工程师B, retrieve_docs)            ← REQUIRE
#   turn 4: review_a (工程师A, append_section)        ← REQUIRE
#   turn 5: review_bc (工程师B, append_section)       ← REQUIRE
#   turn 6: review_bc (工程师C, append_section)       ← REQUIRE
#   turn 7: vote_setup (主审)
#   turn 8: ballot (工程师A, cast_vote)               ← REQUIRE
#   turn 9: ballot (工程师B, cast_vote)               ← REQUIRE
#   turn 10: ballot (工程师C, cast_vote)              ← REQUIRE
#   turn 11: finalize (主审)
CODE_REVIEW_MEMBERS = ["工程师A", "工程师B", "工程师C"]
CODE_REVIEW_SCHEDULE = [
    ("主审", None),
    ("工程师A", "retrieve_docs"),
    ("工程师B", "retrieve_docs"),
    ("工程师A", "append_section"),
    ("工程师B", "append_section"),
    ("工程师C", "append_section"),
    ("主审", None),
] + [(m, "cast_vote") for m in CODE_REVIEW_MEMBERS] + [("主审", None)]

# ---------- mode 实现：把 schedule + mode 翻译成 transcript ------------------

def _turn_marker(idx: int, total: int) -> dict:
    return {"type": "turn", "content": f"turn {idx} of {total}", "ts": time.time()}


def _speaker(name: str, text: str) -> dict:
    return {"speaker": name, "content": text, "ts": time.time()}


def _artifact_event(tool: str, caller: str, args: dict | None = None) -> dict:
    return {
        "type": "artifact_event", "tool": tool, "caller": caller,
        "arguments": dict(args or {}),
        "content": f"{caller} called {tool}", "ts": time.time(),
    }


def _build_transcript(
    schedule: list[tuple[str, str | None]],
    mode: str,
) -> tuple[list[dict], list[str]]:
    """schedule × mode → transcript + warnings.

    mode:
      - 'perfect': 每个 require_tool turn 第一次就调对工具
      - 'all_nudged': 每个 require_tool turn 第一次完全沉默 → retry 时调对（mode=missed）
      - 'mixed': 一半 require_tool turn perfect / 一半 nudge（交替 missed / wrong_tool）
    """
    transcript: list[dict] = []
    warnings: list[str] = []
    total = len(schedule)
    require_idx_in_schedule = 0  # 第 K 个 require_tool turn

    for i, (agent, require_tool) in enumerate(schedule, start=1):
        transcript.append(_turn_marker(i, total))
        if require_tool is None:
            transcript.append(_speaker(agent, f"{agent} 正常发言 ({i}/{total})"))
            continue

        # require_tool turn 走 mode 分支
        if mode == "perfect":
            transcript.append(_speaker(agent, f"{agent} 立即调用 {require_tool}"))
            transcript.append(_artifact_event(require_tool, agent, _default_args(require_tool, agent)))
        elif mode == "all_nudged":
            # 第 1 attempt：纯发言无工具 → missed
            transcript.append(_speaker(agent, f"{agent} 漏了工具调用"))
            # 第 2 attempt：补上
            transcript.append(_speaker(agent, f"{agent} 补调 {require_tool}"))
            transcript.append(_artifact_event(require_tool, agent, _default_args(require_tool, agent)))
        elif mode == "mixed":
            # 偶数序号（K=0, 2, ...）missed；奇数序号（K=1, 3, ...）wrong_tool
            if require_idx_in_schedule % 2 == 0:
                # missed 路径
                if require_idx_in_schedule % 4 == 0:
                    # 一半干脆 perfect（让总 fire rate 不是 100%）
                    transcript.append(_speaker(agent, f"{agent} 直接调 {require_tool}"))
                    transcript.append(_artifact_event(require_tool, agent, _default_args(require_tool, agent)))
                else:
                    # missed
                    transcript.append(_speaker(agent, f"{agent} 沉默"))
                    transcript.append(_speaker(agent, f"{agent} 补 {require_tool}"))
                    transcript.append(_artifact_event(require_tool, agent, _default_args(require_tool, agent)))
            else:
                # wrong_tool 路径：第 1 attempt 调了别的工具
                transcript.append(_speaker(agent, f"{agent} 调错工具"))
                transcript.append(_artifact_event("read_artifact", agent, {}))  # 错的工具
                # 第 2 attempt 补对
                transcript.append(_speaker(agent, f"{agent} 补对 {require_tool}"))
                transcript.append(_artifact_event(require_tool, agent, _default_args(require_tool, agent)))
        else:
            raise ValueError(f"unknown mode: {mode!r}")

        require_idx_in_schedule += 1

    return transcript, warnings


def _default_args(tool: str, caller: str) -> dict:
    """artifact 工具的最小合法 arguments，与 panel/example 实跑形态一致.

    retrieve_docs 是非 artifact 工具，但 stub 走"caller=agent && tool=retrieve_docs"
    的事件即可——_called_tool 不区分 artifact/tool_call，nudge 度量也用同一规约.
    """
    if tool == "cast_vote":
        return {"vote_id": "v1", "option": "采纳", "rationale": f"{caller} 投票理由"}
    if tool == "append_section":
        return {"name": "notes", "entry": f"- {caller} 追加内容"}
    if tool == "write_section":
        return {"name": "notes", "content": f"{caller} 写入内容"}
    if tool == "retrieve_docs":
        return {"query": "项目代号"}
    return {}


# ---------- 顶层打包 -------------------------------------------------------

DOCS = [
    ("brainstorm", BRAINSTORM_SCHEDULE),
    ("debate", DEBATE_SCHEDULE),
    ("roundtable", ROUNDTABLE_SCHEDULE),
    ("example", EXAMPLE_SCHEDULE),
    ("panel", PANEL_SCHEDULE),
    ("tool_chain", TOOL_CHAIN_SCHEDULE),
    ("code_review", CODE_REVIEW_SCHEDULE),
]

MODES = ["perfect", "all_nudged", "mixed"]


def main() -> None:
    for mode in MODES:
        out_path = OUT_DIR / f"{mode}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for doc_id, schedule in DOCS:
                transcript, warnings = _build_transcript(schedule, mode)
                envelope = {
                    "id": doc_id,
                    "transcript": transcript,
                    "artifact": {},  # 不影响 nudge 度量
                    "warnings": warnings,
                    "success": not warnings,
                }
                f.write(json.dumps(envelope, ensure_ascii=False) + "\n")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

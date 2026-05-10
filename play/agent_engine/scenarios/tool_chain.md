# 工具链服从性测试场景（agent_sft Phase 1 require_tool 密集场景之二）

# ============================================================================
# 设计目标（与 `code_review.md` 互补）：
#   - **单 agent 强工具链**：1 名执行者按"retrieve → append → vote → retrieve →
#     append"的链条逐步推进，测纯工具序列服从性，去掉多 agent 上下文复杂度，
#     让信号直接归因到模型对"步骤说明 + 工具调用"的服从程度。
#   - **5 个 require_tool step entry**（plan §1.B 落点）：
#       retrieve_docs × 2 + append_section × 2 + cast_vote × 1
#     单 agent 不展开 → 5 require_tool turn。
#   - **真实场景**：执行者按主审下发的 checklist 一步步完成调研→记录→投票→
#     再调研→再记录的工具链。
#
# 与 code_review.md 对照：
#   | 维度          | code_review.md      | tool_chain.md           |
#   |---|---|---|
#   | agent 数      | 4（含 3 senior）    | 2（仅 1 执行者）         |
#   | 上下文        | 多人评审复杂        | 单人 checklist 简单      |
#   | 测什么        | 多角色工具决策      | 工具调用顺序服从性       |
#   | 数据用途      | SFT 多样性源        | SFT 高纯度服从样本       |
#
# 复用 example.md 的 vdb 路径（test_vdb）。
# ============================================================================

---

memory:
  type: full

tools:
  - name: retrieve_docs
    vdb_dir: ../../rag/vdb/test_vdb
    top_k: 3

artifact:
  enabled: true
  initial_sections:
    - {name: 调研笔记, mode: append}
    - {name: 决策, mode: replace}
  tool_owners:
    propose_vote: moderator
    finalize_artifact: moderator

agents:
  - name: 协调者
    role: moderator
    prompt: |
      你是协调者。职责：开场宣布 checklist、组织投票、最终落定决策。
      每次发言不超过 50 字。用中文。
    temperature: 0.3
    max_tokens: 160

  - name: 执行者
    role: member
    prompt: |
      你是工具链执行者。**严格按当前 step 的 instruction 调用指定工具**——
      不要绕道、不要多调、不要漏调。每次发言不超过 30 字（重点是工具调用，不是说话）。
      用中文。
    temperature: 0.2
    max_tokens: 200

steps:
  - id: open
    who: moderator
    instruction: |
      宣布今天的 checklist：调研「项目代号」→ 记录笔记 → 投票决定方向 → 二次调研 → 二次笔记。

  - id: ctx_round1
    who: [执行者]
    require_tool: retrieve_docs
    max_retries: 1
    instruction: |
      调用 retrieve_docs(query="项目代号") 获取背景资料。
      ≤30 字一句话报告检索到的核心要点。

  - id: note_round1
    who: [执行者]
    require_tool: append_section
    max_retries: 1
    instruction: |
      append_section("调研笔记", "- 第1轮: <一句话核心要点>") 把刚才检索结果记入。

  - id: vote_setup
    who: moderator
    instruction: |
      propose_vote(question="是否需要追加二次调研?", options=["追加","不追加"])，
      一句话请执行者投票。

  - id: ballot
    who: [执行者]
    require_tool: cast_vote
    max_retries: 1
    instruction: |
      cast_vote(vote_id="v1", option="追加", rationale="一句话")。
      为了完成 checklist，按"追加"投。

  - id: ctx_round2
    who: [执行者]
    require_tool: retrieve_docs
    max_retries: 1
    instruction: |
      二次调用 retrieve_docs(query="项目代号 历史") 进行追加调研。
      ≤30 字一句话报告补充信息。

  - id: note_round2
    who: [执行者]
    require_tool: append_section
    max_retries: 1
    instruction: |
      append_section("调研笔记", "- 第2轮: <一句话补充要点>") 把二次调研结果记入。

  - id: finalize
    who: moderator
    instruction: |
      write_section("决策", "已完成 checklist 全部 5 步")，
      再 finalize_artifact(decision="完成", rationale="工具链 5 步全部按序执行")。

---

## checklist：调研→笔记→投票→二次调研→二次笔记

任务很简单：执行者**严格按 step 顺序**调用每一步指定的工具。
本场景不测内容质量，只测**模型是否按 instruction 一次调对工具**。

每个 require_tool step 都是一次纯粹的服从性测试：
- 第 1 attempt 调对 → no nudge fired ✓
- 第 1 attempt 漏调 / 调错 → nudge 触发 retry → 失败模式被记录

输出尽可能短，把发言权重让给工具调用本身。

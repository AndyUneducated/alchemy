# tool_chain fast copy (agent_sft Phase 2 mining only)

# ============================================================================
# Derived from [`agent_engine/scenarios/tool_chain.md`](../../../agent_engine/scenarios/tool_chain.md).
# Upstream scenario unchanged; this copy speeds up synthesize.py mining only:
#
#   1. `max_retries: 1 → 0`: synthesize only needs first failed attempt + nudge fired
#      to build a triple; retries waste LLM calls.
#   2. `max_tokens: 200/160 → 80`: agent prompts already cap at ≤30/50 chars; token cap
#      matches actual load with no functional impact (fire detection is length-independent).
#   3. Drop open + finalize moderator steps: 0 fires, pure ceremony overhead.
#   4. `vdb_dir` relative path +1 level (scenario file moved one directory deeper).
#
# Upstream baseline eval (nudge_fire_rate / agent_traj recorded with max_retries=1)
# still uses the original scenario; this file serves mine_triples.py only.
# ============================================================================

---

memory:
  type: full

tools:
  - name: retrieve_docs
    vdb_dir: ../../../rag/vdb/test_vdb
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
      你是协调者。职责：组织投票。
      每次发言不超过 30 字。用中文。
    temperature: 0.3
    max_tokens: 80

  - name: 执行者
    role: member
    prompt: |
      你是工具链执行者。**严格按当前 step 的 instruction 调用指定工具**——
      不要绕道、不要多调、不要漏调。每次发言不超过 30 字（重点是工具调用，不是说话）。
      用中文。
    temperature: 0.2
    max_tokens: 80

steps:
  - id: ctx_round1
    who: [执行者]
    require_tool: retrieve_docs
    max_retries: 0
    instruction: |
      调用 retrieve_docs(query="项目代号") 获取背景资料。
      ≤30 字一句话报告检索到的核心要点。

  - id: note_round1
    who: [执行者]
    require_tool: append_section
    max_retries: 0
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
    max_retries: 0
    instruction: |
      cast_vote(vote_id="v1", option="追加", rationale="一句话")。
      为了完成 checklist，按"追加"投。

  - id: ctx_round2
    who: [执行者]
    require_tool: retrieve_docs
    max_retries: 0
    instruction: |
      二次调用 retrieve_docs(query="项目代号 历史") 进行追加调研。
      ≤30 字一句话报告补充信息。

  - id: note_round2
    who: [执行者]
    require_tool: append_section
    max_retries: 0
    instruction: |
      append_section("调研笔记", "- 第2轮: <一句话补充要点>") 把二次调研结果记入。

---

## checklist：调研→笔记→投票→二次调研→二次笔记

执行者按 step 顺序调每一步指定的工具。本场景不测内容质量，只测**模型是否按
instruction 一次调对工具**——每次失败都被 synthesize 路径回收成训练样本。

输出尽可能短，把发言权重让给工具调用本身。

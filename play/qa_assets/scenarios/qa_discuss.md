---
memory: {type: window, max_recent: 8}

artifact:
  enabled: true
  initial_sections:
    - {name: Requirements, mode: replace}
    - {name: 原子需求, mode: replace}
    - {name: 风险等级, mode: replace}
    - {name: 测试用例, mode: replace}
    - {name: 非功能, mode: replace}
    - {name: Critic 反馈, mode: append}

tools:
  - {name: retrieve_docs, vdb_dir: ../vdb/qa_kb, top_k: 3}

agents:
  - name: supervisor
    role: moderator
    temperature: 0.4
    max_tokens: 180
    prompt: |
      测试讨论主持：开场点 Requirements；协调 critic 后的修订；最后 finalize_artifact(decision,rationale)。
      不写四 specialist 节、不投票。中文，≤60字/轮。

  - name: decomposer
    role: member
    max_tokens: 500
    prompt: |
      拆解：读 Requirements→原子 feature+验收；可 retrieve_docs 查 PRD。
      write_section("原子需求")；"### REQ-xxx" + bullet feature/acceptance。中文；不写用例与打分。

  - name: risk_grader
    role: member
    max_tokens: 500
    prompt: |
      定级：读 Requirements+原子需求→每 req P0~P3+短理由；retrieve_docs 可查 bug。
      write_section("风险等级")；表头 |req_id|priority|rationale|。中文。

  - name: case_generator
    role: member
    max_tokens: 600
    prompt: |
      用例：读上两节→功能+边界；retrieve_docs 可查历史用例。
      write_section("测试用例")；"### REQ-xxx" + "- [Px][tag] 给定/当/那么"。中文；每 req≥2条且含边界。

  - name: nfr_planner
    role: member
    max_tokens: 450
    prompt: |
      非功能：读 Requirements→性能/安全/a11y/i18n；retrieve_docs 可查 checklist。
      write_section("非功能")；四段 H2。中文；勿重复纯功能用例。

  - name: critic
    role: member
    max_tokens: 400
    prompt: |
      审查：扫六节→覆盖缺口/优先级与用例不匹配/矛盾。
      append_section("Critic 反馈", "## Round N\n- ...")；中文；不直接改他节。

steps:
  - {id: open, who: [supervisor], instruction: 一句话开场：本批 Requirements + 请四角色按节产出。}

  - id: produce
    who: [decomposer, risk_grader, case_generator, nfr_planner]
    instruction: 按各自 system 调 write_section 写负责节。
    require_tool: write_section

  - id: critic_r1
    who: [critic]
    instruction: Round1：append_section，entry 以 "## Round 1\n-" 开头。
    require_tool: append_section

  - id: revise
    who: [decomposer, risk_grader, case_generator, nfr_planner]
    instruction: 读 Critic；仅相关则 write_section 修订，否则一句「本轮无需改」。

  - id: critic_r2
    who: [critic]
    instruction: Round2：append_section "## Round 2\n-"；若无 blocking 可写「Round2 通过」。
    require_tool: append_section

  - id: finalize
    who: [supervisor]
    instruction: finalize_artifact(decision="approved"或"needs_rework", rationale=一句)。
    require_tool: finalize_artifact
---

`Requirements` 由 workflow `initial_artifact` 注入。

---
memory: {type: window, max_recent: 10}

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
    max_tokens: 200
    prompt: |
      测试方案讨论主持人. 开场介绍 Requirements; critic 反馈后协调 specialist
      修订; 收尾调 finalize_artifact(decision, rationale) 封板.
      不写 specialist 章节; 不投票; 用中文; ≤ 80 字 / 发言.

  - name: decomposer
    role: member
    prompt: |
      Decomposer (需求拆解). 读 artifact.Requirements, 拆为原子 feature + 验收标准.
      建议先 retrieve_docs(query="<关键词>") 查历史 PRD.
      调 write_section(name="原子需求"); 行格式 "### REQ-xxx 标题" +
      "- feature: ...\n  acceptance:\n    - 给定..., 当..., 那么...".
      用中文; 不写测试用例; 不打分.

  - name: risk_grader
    role: member
    prompt: |
      Risk Grader. 读 Requirements + 原子需求, 给每需求打 P0~P3 + ≤ 30 字理由.
      建议 retrieve_docs(query="<关键词> bug 故障") 查历史踩过的坑必须 ≥ P1.
      调 write_section(name="风险等级"); markdown 表格 "| req_id | priority | rationale |".
      priority ∈ {P0,P1,P2,P3}; 用中文.

  - name: case_generator
    role: member
    prompt: |
      Case Generator. 读 Requirements + 原子需求, 产出 functional + boundary / edge 用例.
      建议 retrieve_docs(query="<相似需求> 测试用例") 把历史踩过的坑 (并发注册 /
      超长邮箱 / 邮件队列阻塞) 转为本次的边界用例.
      调 write_section(name="测试用例"); 行格式 "### REQ-xxx 标题" +
      "- [P1][functional] 给定..., 当..., 那么...".
      用中文; 每需求 ≥ 3 条 (含 ≥ 1 boundary/edge); 引用 risk_grader 的 priority.

  - name: nfr_planner
    role: member
    prompt: |
      NFR Planner. 读 Requirements 为整批需求规划性能 / 安全 / a11y / i18n.
      建议 retrieve_docs(query="<domain> checklist") 查 auth / perf checklist
      拿 baseline (P95 阈值 / ARIA role 等), 不要凭空猜.
      调 write_section(name="非功能"); 4 个 H2 段 "## 性能 / 安全 / a11y / i18n"
      每段 ≥ 2 条. 用中文; 不重复 case_generator 的 functional cases.

  - name: critic
    role: member
    prompt: |
      Critic. 读全 6 节, 找: 覆盖空白 (原子需求没对应用例) / 优先级与覆盖度不
      匹配 (P0 缺 boundary) / 互相矛盾.
      调 append_section(name="Critic 反馈", entry="## Round N\n- [coverage]...").
      用中文; 不直接改其他章节; 不重复上一轮.

steps:
  - {id: open, who: [supervisor], instruction: "一句话开场, 介绍今天 Requirements 的需求批次, 提示 4 specialist 依序产出."}

  - id: produce
    who: [decomposer, risk_grader, case_generator, nfr_planner]
    instruction: 按 system prompt 调 write_section 产出你负责的章节.
    require_tool: write_section

  - id: critic_r1
    who: [critic]
    instruction: 'Round 1: 调 append_section 追加反馈 (entry 以 "## Round 1\n-" 起).'
    require_tool: append_section

  - id: revise
    who: [decomposer, risk_grader, case_generator, nfr_planner]
    instruction: 读 Critic 反馈; 仅当反馈直接涉及你的章节时调 write_section 修订, 否则一句话"无需修改".

  - id: critic_r2
    who: [critic]
    instruction: 'Round 2: 调 append_section ("## Round 2\n-"); 全部已解决写"Round 2 通过, 无 blocking 项".'
    require_tool: append_section

  - id: finalize
    who: [supervisor]
    instruction: 调 finalize_artifact(decision="approved"|"needs_rework", rationale=<一句话>) 收尾.
    require_tool: finalize_artifact
---

Requirements 已通过 workflow.config.initial_artifact 注入 artifact. 按 prompt + steps 协作.

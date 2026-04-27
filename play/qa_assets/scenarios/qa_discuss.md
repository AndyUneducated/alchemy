# QA 测试方案多 agent 协作 scenario
# 6 agent (supervisor + 4 specialist + critic) 单 scenario 多 agent loop;
# 由 workflow `discuss` agent stage 一次性 Engine.invoke 跑完 (plan §2.1 / §8 P3-P5).

---
memory:
  type: window
  max_recent: 10                # 控本机 wall-clock; pinned (topic/turn/artifact_event) 不剪

artifact:
  enabled: true
  initial_sections:
    - {name: Requirements, mode: replace}    # workflow 注入
    - {name: 原子需求, mode: replace}
    - {name: 风险等级, mode: replace}
    - {name: 测试用例, mode: replace}
    - {name: 非功能, mode: replace}
    - {name: Critic 反馈, mode: append}      # 多轮 append

tools:
  - name: retrieve_docs
    vdb_dir: ../vdb/qa_kb
    top_k: 3

agents:
  - name: supervisor
    role: moderator
    prompt: |
      你是测试方案讨论的主持人.
      职责: 开场介绍 Requirements; critic 反馈后协调 specialist 修订; 收尾调
      finalize_artifact(decision="approved", rationale="<一句话>") 封板.
      规则: 不写 specialist 章节; 不投票; 用中文; 每次发言 ≤ 80 字.
    temperature: 0.4
    max_tokens: 200

  - name: decomposer
    role: member
    prompt: |
      你是 Decomposer (需求拆解 specialist).
      读 artifact.Requirements, 把每个需求拆为原子 feature + 验收标准.
      建议: 先调 retrieve_docs(query="<需求关键词>") 查历史 PRD 看类似需求的拆解.
      产出: 调 write_section(name="原子需求", content=...). 行格式
      "### REQ-xxx 标题" + "- feature: ...\n  acceptance:\n    - 给定..., 当..., 那么...".
      规则: 用中文; 不写测试用例; 不打分.

  - name: risk_grader
    role: member
    prompt: |
      你是 Risk Grader.
      读 Requirements + 原子需求, 给每个需求打 P0~P3 + 一句话理由 (业务影响 /
      故障频度 / 数据敏感度).
      建议: 调 retrieve_docs(query="<关键词> bug 故障") 查历史 bug; 历史踩过坑的方向 ≥ P1.
      产出: 调 write_section(name="风险等级", content=...). markdown 表格
      "| req_id | priority | rationale |".
      规则: 用中文; priority ∈ {P0,P1,P2,P3}; rationale ≤ 30 字.

  - name: case_generator
    role: member
    prompt: |
      你是 Case Generator.
      读 Requirements + 原子需求, 为每个需求产出 functional + edge / boundary 用例.
      建议: 调 retrieve_docs(query="<相似需求> 测试用例") 查历史用例与 bug, 把别人
      踩过的坑 (并发注册 / 超长邮箱 / 邮件队列阻塞) 转化为本次的边界用例.
      产出: 调 write_section(name="测试用例", content=...). 行格式
      "### REQ-xxx 标题" + "- [P1][functional] 给定..., 当..., 那么...".
      规则: 用中文; 每个需求 ≥ 3 条 (含 ≥ 1 boundary 或 edge); 引用 risk_grader 的 priority.

  - name: nfr_planner
    role: member
    prompt: |
      你是 NFR Planner.
      读 Requirements, 为整批需求规划性能 / 安全 / a11y / i18n 测试点.
      建议: 调 retrieve_docs(query="<domain> checklist") 查 auth_checklist /
      perf_checklist 等 baseline, 用 SLO 数字 (如 P95 阈值 / ARIA role) 填充, 不要凭空猜.
      产出: 调 write_section(name="非功能", content=...). 4 个 H2 段
      "## 性能 / ## 安全 / ## a11y / ## i18n", 每段 ≥ 2 条.
      规则: 用中文; 不重复 case_generator 的 functional cases.

  - name: critic
    role: member
    prompt: |
      你是 Critic / Debate agent.
      读 artifact 全部 6 节, 找:
      - 覆盖空白 (原子需求没对应测试用例)
      - 优先级与覆盖度不匹配 (P0 但只有 functional, 缺 boundary)
      - 互相矛盾 (decomposer 与 case_generator 描述冲突)
      产出: 调 append_section(name="Critic 反馈", entry="## Round N\n- [coverage]...").
      规则: 用中文; 不直接改其他章节, 只发反馈; 不重复上一轮已提的相同问题.

steps:
  - id: open
    who: [supervisor]
    instruction: 用一句话开场, 介绍今天的需求批次 (从 artifact.Requirements 看), 提醒 4 specialist 依序产出.

  - id: produce
    who: [decomposer, risk_grader, case_generator, nfr_planner]
    instruction: 阅读 artifact.Requirements, 按 system prompt 调 write_section 产出你负责的章节.
    require_tool: write_section

  - id: critic_r1
    who: [critic]
    instruction: |
      Round 1: 阅读 4 节 + Requirements, 调 append_section(name="Critic 反馈",
      entry="## Round 1\n- ...") 追加反馈 (覆盖空白 / 优先级不匹配 / 互相矛盾).
    require_tool: append_section

  - id: revise
    who: [decomposer, risk_grader, case_generator, nfr_planner]
    instruction: |
      读 Critic 反馈; 仅当反馈直接涉及你的章节时调 write_section 修订, 否则用一
      句话说明"无需修改"即可.

  - id: critic_r2
    who: [critic]
    instruction: |
      Round 2: 调 append_section(name="Critic 反馈", entry="## Round 2\n- ...")
      追加最终评估; 全部已解决就写"Round 2 通过, 无 blocking 项".
    require_tool: append_section

  - id: finalize
    who: [supervisor]
    instruction: |
      调 finalize_artifact(decision="approved" 或 "needs_rework",
      rationale="<一句话: 总用例数 / Critic 主要反馈 / 是否封板>") 收尾.
    require_tool: finalize_artifact
---

## 任务背景

`workflow` 已通过 `config.initial_artifact.Requirements` 把待测需求列表 (yaml 字符串) 预填入 artifact, 6 位 agent 直接读用. 每个需求至少有 `description` 或 `prd_md` 之一作为分析源.

请按 system prompt + steps 顺序协作: supervisor 不写 specialist 章节; specialist 各自负责自己的章节; critic 只发反馈不直接改章节.

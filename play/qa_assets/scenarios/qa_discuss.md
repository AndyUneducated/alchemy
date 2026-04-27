# QA 测试方案多 agent 协作 scenario

# ============================================================================
# 6 个 agent 的多 agent loop——supervisor 主持 + 4 specialist 并行产出 +
# critic 多轮反馈 + supervisor finalize. 整个 loop **住在这一份 scenario.md
# 里**，由 workflow 的 agent stage 一次性 Engine.invoke 跑完 (plan §2.1).
#
# 输入: workflow 通过 config.initial_artifact.Requirements 注入序列化后的需求
#       列表 (yaml 字符串, 每行一个需求 + 可选 prd_md). 结构由 hooks/load_csv +
#       hooks/load_each_prd 决定 (plan §4.1).
#
# 输出: artifact dict (Engine.invoke 返回的 Result.artifact), 6 个 section,
#       将由下游 render_md / render_csv stage 转为 .md / .csv (P5).
# ============================================================================

---
artifact:
  enabled: true
  initial_sections:
    # 输入预填: workflow 把序列化后的 requirements yaml 写到这里
    - {name: Requirements, mode: replace}
    # 4 specialist 各自的产出节
    - {name: 原子需求, mode: replace}
    - {name: 风险等级, mode: replace}
    - {name: 测试用例, mode: replace}
    - {name: 非功能, mode: replace}
    # critic 多轮 append (保留每轮反馈, 不互相覆盖)
    - {name: Critic 反馈, mode: append}
  # tool_owners 留空 = 全员都能调任意 artifact 工具——纪律靠 prompt 维持
  # (plan §6 二分类 + §12 不做 runtime allowlist)

agents:
  - name: supervisor
    role: moderator
    prompt: |
      你是测试方案讨论的主持人.
      你的职责：
      - 开场介绍 Requirements (artifact 中已有), 提醒 4 位 specialist 各自的产出节;
      - 在 critic 反馈后, 协调 specialist 修订各自的章节;
      - 收尾时调用 finalize_artifact(decision="approved", rationale="<一句话>")
        封板.
      规则: 你**不**写 specialist 负责的章节; 不投票; 用中文; 每次发言不超过 80 字.
    temperature: 0.4
    max_tokens: 200

  - name: decomposer
    role: member
    prompt: |
      你是需求拆解 specialist (Decomposer).
      读 artifact.Requirements (yaml 字符串), 把每个需求拆为原子 feature
      + 验收标准 (acceptance criteria).
      产出: 调用 write_section(name="原子需求", content="<markdown>") 一次性
      写入. 格式:
        ### REQ-001 用户邮箱注册
        - feature: ...
          acceptance:
            - 给定 ..., 当 ..., 那么 ...
            - ...
      规则: 用中文; 不要写测试用例 (那是 case_generator 的活); 不要打分
      (那是 risk_grader 的活).

  - name: risk_grader
    role: member
    prompt: |
      你是风险打分 specialist (Risk Grader).
      读 artifact.Requirements + 原子需求, 给每个需求打 P0~P3 风险等级 + 一句话
      理由 (业务影响 / 故障频度 / 数据敏感度).
      产出: 调用 write_section(name="风险等级", content="<markdown>") 一次性
      写入. 格式:
        | req_id | priority | rationale |
        |---|---|---|
        | REQ-001 | P1 | 主登录链路, 数据敏感 |
      规则: 用中文; priority ∈ {P0, P1, P2, P3}; rationale 不超过 30 字.

  - name: case_generator
    role: member
    prompt: |
      你是测试用例 specialist (Case Generator).
      读 artifact.Requirements + 原子需求, 为每个需求产出 functional + edge /
      boundary cases.
      产出: 调用 write_section(name="测试用例", content="<markdown>") 一次性
      写入. 格式:
        ### REQ-001 用户邮箱注册
        - [P1][functional] 给定合法邮箱+密码, 当点击注册, 那么...
        - [P1][boundary] 给定空邮箱, ...
        - [P2][edge] 给定已注册邮箱, ...
      规则: 用中文; 每个需求至少 3 条 (含至少 1 条 boundary 或 edge);
      引用 risk_grader 的 priority.

  - name: nfr_planner
    role: member
    prompt: |
      你是非功能需求 specialist (NFR Planner).
      读 artifact.Requirements, 为整批需求规划性能 / 安全 / a11y / i18n
      非功能测试点.
      产出: 调用 write_section(name="非功能", content="<markdown>") 一次性
      写入. 格式:
        ## 性能
        - 注册接口 P95 < 300ms (并发 100)
        ## 安全
        - 密码字段不进日志
        ## a11y
        - ...
        ## i18n
        - ...
      规则: 用中文; 每节至少 2 条; 不重复 case_generator 的 functional cases.

  - name: critic
    role: member
    prompt: |
      你是测试方案 Critic / Debate agent.
      读 artifact 的全部 6 节 (Requirements / 原子需求 / 风险等级 / 测试用例 /
      非功能), 找出:
      - 覆盖空白 (哪条原子需求没对应测试用例)
      - 优先级与覆盖度不匹配 (P0 但只有 functional, 缺 boundary)
      - 互相矛盾 (decomposer 与 case_generator 描述冲突)
      产出: 调用 append_section(name="Critic 反馈", entry="<markdown>")
      追加本轮反馈. 格式:
        ## Round N
        - [coverage] REQ-002 缺 boundary case
        - [priority] REQ-003 评 P0 但 nfr 没列性能要求
      规则: 用中文; 不直接改其他章节, 只发反馈; 不重复上一轮已提的相同问题.

steps:
  - id: open
    who: [supervisor]
    instruction: |
      用一句话开场: 介绍今天处理的需求批次 (从 artifact.Requirements 看), 提醒
      decomposer / risk_grader / case_generator / nfr_planner 依序产出各自章节.

  - id: produce
    who: [decomposer, risk_grader, case_generator, nfr_planner]
    instruction: |
      请阅读 artifact.Requirements, 并按你的 system prompt 调用 write_section
      工具产出你负责的章节.
    require_tool: write_section

  - id: critic_r1
    who: [critic]
    instruction: |
      Round 1: 阅读 artifact 中已产出的 4 个 section + Requirements, 调用
      append_section(name="Critic 反馈", entry="## Round 1\n- ...") 追加本轮
      反馈 (覆盖空白 / 优先级不匹配 / 互相矛盾).
    require_tool: append_section

  - id: revise
    who: [decomposer, risk_grader, case_generator, nfr_planner]
    instruction: |
      Critic 已发出 Round 1 反馈. 阅读 artifact.Critic 反馈, 仅当反馈 **直接**
      涉及你负责的章节时, 调用 write_section 重写你的章节修订. 没必要修改时
      可以跳过 write_section, 用一句话说明"无需修改"即可.

  - id: critic_r2
    who: [critic]
    instruction: |
      Round 2: 检查 specialist 是否回应了 Round 1 反馈. 调用
      append_section(name="Critic 反馈", entry="## Round 2\n- ...") 追加
      最终评估; 如全部已解决, 写一句"Round 2 通过, 无 blocking 项".
    require_tool: append_section

  - id: finalize
    who: [supervisor]
    instruction: |
      看完所有 6 个 section, 调用 finalize_artifact(
        decision="approved" 或 "needs_rework",
        rationale="一句话: 总用例数 / Critic 主要反馈 / 是否封板"
      ) 收尾.
    require_tool: finalize_artifact
---

## 任务背景

本 scenario 由 `play/qa_assets/workflows/qa_supervisor.yaml` 的 `discuss`
agent stage 调用. workflow 已通过 `config.initial_artifact.Requirements`
把待测需求列表 (yaml 字符串) 预填入 artifact, 你们 6 位 agent 直接读用.

需求列表的 yaml schema (来自 `hooks/load_csv` + `hooks/load_each_prd` 的产出):

```yaml
- req_id: REQ-001
  title: 用户邮箱注册
  description: ""              # 行内简短描述, 可空
  prd_doc_path: examples/prd_signup.md
  prd_md: "<整个 PRD .md 内容>"  # load_each_prd 加载后填入
  priority: P1                  # 可空, 留空时由 risk_grader 推断
  assignee: Alice
  sprint_start: "2026-05-01"
  sprint_end: "2026-05-14"
- req_id: REQ-002
  ...
```

每个需求**至少**有 `description` 或 `prd_md` 之一作为分析源.

请按 system prompt + steps 顺序协作完成测试方案. supervisor 不写 specialist 章节,
specialist 各自负责自己的章节, critic 只通过反馈影响其他人.

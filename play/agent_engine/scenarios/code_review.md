# PR 代码评审会议（agent_sft Phase 1 require_tool 密集场景之一）

# ============================================================================
# 设计目标（agent_sft Phase 1 baseline / Phase 2 SFT 数据采集）：
#   - **多 agent / 上下文复杂**：3 名 senior engineer + 1 名主审，互相 context
#     相关，模型在多个角色间做工具决策；与 `tool_chain.md` 的"单 agent 强工具链"
#     形成正交对照。
#   - **5 个 require_tool step entry**（plan §1.B 落点）：
#       retrieve_docs × 2 + append_section × 2 + cast_vote × 1
#     展开 turn 数：2 (retrieve) + 1 (append A) + 2 (append B,C member) + 3 (vote member)
#                 = 8 require_tool turn
#   - **真实场景**：3 工程师审一个跨模块 PR，必须先 retrieve 相关 commit/doc，
#     再写 review section，最后投票决定是否合入。
#
# 复用 example.md 的 vdb 路径（test_vdb），不引新数据集；如 vdb 不存在请按
# play/rag README 的 ingest 流程跑一次。
# ============================================================================

---

memory:
  type: window
  max_recent: 6

tools:
  - name: retrieve_docs
    vdb_dir: ../../rag/vdb/test_vdb
    top_k: 3

artifact:
  enabled: true
  initial_sections:
    - {name: review_a, mode: replace}
    - {name: review_b, mode: replace}
    - {name: review_c, mode: replace}
    - {name: 决策, mode: replace}
  tool_owners:
    propose_vote: moderator
    finalize_artifact: moderator

agents:
  - name: 主审
    role: moderator
    prompt: |
      你是主审工程师。职责：开场介绍 PR 范围、组织三位 senior 评审、最后宣布合入决议。
      每次发言不超过 60 字。用中文。需要时调 propose_vote / finalize_artifact / write_section。
    temperature: 0.3
    max_tokens: 200

  - name: 工程师A
    role: member
    prompt: |
      你是后端 senior 工程师。先 retrieve_docs 查相关 commit/doc 再下结论；
      用 append_section("review_a", ...) 提交评审意见。每次 ≤ 60 字。用中文。
    max_tokens: 200

  - name: 工程师B
    role: member
    prompt: |
      你是前端 senior 工程师。先 retrieve_docs 查接口约定再下结论；
      用 append_section("review_b", ...) 提交评审意见。每次 ≤ 60 字。用中文。
    max_tokens: 200

  - name: 工程师C
    role: member
    prompt: |
      你是 QA senior 工程师。综合 A/B 评审，append_section("review_c", ...) 提交风险评估，
      最后 cast_vote 决定是否合入。每次 ≤ 60 字。用中文。
    max_tokens: 200

steps:
  - id: open
    who: moderator
    instruction: |
      一句话介绍本 PR 的范围（涉及"项目代号"模块）和今天评审会的目标：3 人评审后投票决定是否合入。

  - id: ctx_a
    who: [工程师A]
    require_tool: retrieve_docs
    max_retries: 1
    instruction: |
      调用 retrieve_docs 查询「项目代号」相关历史 commit / 设计文档，
      不超过 30 字一句话总结你检索到的关键信息。

  - id: ctx_b
    who: [工程师B]
    require_tool: retrieve_docs
    max_retries: 1
    instruction: |
      调用 retrieve_docs 查询「项目代号」相关接口契约 / 测试用例，
      不超过 30 字一句话总结你检索到的关键信息。

  - id: review_a
    who: [工程师A]
    require_tool: append_section
    max_retries: 1
    instruction: |
      append_section("review_a", "- <一句话评审结论>")
      把你基于检索到的内容的评审意见写入 artifact。

  - id: review_bc
    who: [工程师B, 工程师C]
    require_tool: append_section
    max_retries: 1
    instruction: |
      你是工程师B → append_section("review_b", "- ...")。
      你是工程师C → append_section("review_c", "- 综合 A/B 风险一句话")。

  - id: vote_setup
    who: moderator
    instruction: |
      调用 propose_vote(question="是否合入此 PR?", options=["合入","退回"])，
      然后用一句话邀请大家投票。

  - id: ballot
    who: member
    require_tool: cast_vote
    max_retries: 1
    instruction: |
      cast_vote(vote_id="v1", option="合入" 或 "退回", rationale="一句话理由")
      表达你的最终立场。

  - id: finalize
    who: moderator
    instruction: |
      根据 <artifact> 投票结果，先 write_section("决策", "<合入/退回> + 一句话总结")，
      再 finalize_artifact(decision="合入" 或 "退回", rationale="一句话")。

---

## 待审 PR：项目代号模块重构

### PR 背景

后端 SDK 中"项目代号"模块（`project_codename/`）有 3 处实现散落，本 PR 将其
统一为单一入口，影响：

- 后端：合并 3 个 module 为 1 个；新增 facade API；老 API 加 deprecation warning
- 前端：调用方 import path 变更（`from sdk.codename import ...` → `from sdk.project import codename`）
- 测试：3 个 module 各自的单元测试合并为一组集成测试

### 关键关注点

| 维度 | 风险 |
|---|---|
| 兼容性 | 老 import path 仍工作，但 deprecation warning 会污染下游日志 |
| 性能 | 单一入口加了一层间接，benchmark 显示无明显差异 |
| 测试覆盖 | 集成测试覆盖率 87%，比合并前的加权平均 92% 略低 |

### 决策选项

1. **合入**：接受短期测试覆盖率下降，赢得长期可维护性
2. **退回**：要求作者补齐集成测试覆盖率到 90%+ 再合入

主审请按 steps 流程组织评审。

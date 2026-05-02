# Changelog

`CHANGELOG.md` 同时承担变更日志与 ADR 归档。每条记录以 `## n. 变更标题` 开头，紧接一行 `- **日期**：...`，heading 前后留空行。后续每个自然日建议最多追加 1～2 条 tech decision。

## 1. Phase 0 架构 & 叙事决策

- **日期**：2026-04-30

项目启动时一次性敲定的两件"回不去"的设计：**架构骨架** + **指标组织的心智模型**。所有后续 phase 都在这两条线上延伸。

### Context

从零建一个 LLM 评测框架，目标是**学习与进阶式扩展**（不 ship 产品）。要贴主流框架、渐进式扩展、不专门为 `play/agent_engine` / `play/rag` 设计但后续能接。同时需要一套组织所有指标的心智模型，既直观易上手，又要在被追问边界时站得住脚。

### Options considered

**架构原型**：

- **lm-evaluation-harness (EleutherAI)**（选择）—— Task = dataset + prompt template + process_results + aggregation；LM 暴露 generate_until / loglikelihood。学术 benchmark 事实标准，paper 分数能对上
- inspect_ai（UK AISI）—— Task + Solver + Scorer，更 agent-friendly。但 solver 对 benchmark 简单任务过度设计
- deepeval —— metric-first、pytest-like。CI 集成好，但 task 可复现性弱（prompt 散在 test_case 里）
- 自造 —— 最灵活也最不"主流"，失去对齐学术生态的价值

**打分模式**：

- **score + run 双模式**（选择）—— score 是 offline 对比 gold + predictions（sacrebleu 风格），run 是 active 驱动 LM。两者共享 `task.process_results + aggregation + storage` 尾段；`evaluate_offline(task, preds) ≡ evaluate_active(task, PrerecordedLM(preds))` 由 parity test 焊死
- 只 run —— 排除：学 metric 没必要被 LM 调用 / API key / 网络拖累；且 `play/agent_engine` / `play/rag` 产物本质就是 JSONL，offline 打分零耦合
- 只 score —— 排除：丢了 harness 的 Task + LM + Runner 完整骨架，后续加 paradigm 时缺抽象支撑

**指标组织 taxonomy**：

- 只用五族（Classification+Agreement / Generation / LLM-as-Judge / RAG / Agent Trajectory）—— 好记但混了三个正交轴（task / method / pipeline），被追问"为什么 LLM-as-Judge 和 Generation 平级"时无解
- 只用双轴（task × method）+ HELM 7 维度 —— 严谨但 onboarding 不直观
- **两层并存**（选择）—— README 顶层用五族做 mental map，下一节立刻给"严谨视角：双轴 + HELM"及五族 ↔ 双轴对应表。代码层 `metrics/` 按**方法学**切文件（和五族解耦）

### Decision

- **架构**：lm-eval 骨架（Task ABC + LM ABC + Registry + Runner + `api.py` 契约层）**同时支持 score / run 双模式**；Phase 1 主路径走 score（metric 学习优先），MockLM 仅作 run 模式的演示 + parity 源
- **叙事**：README 用五族 onboarding + 双轴严谨视角**两层并存**；代码按方法学切 `metrics/X.py`（按需建，见 README 指导原则 #3）

### Consequences

- Task.process_results 必须**统一吃 Response**，不能区分来源 → 是两模式共享尾段的根基
- 日后接真 LM（Phase 3）加的是 `models/openai.py` / `ollama.py`，Task / Metric / Runner 零改动
- 指标层按需建：Phase 1 task 直接调 sklearn；未来哪个方法学首次跨 task 复用或无库可用时再建 `metrics/X.py`
- 两层 taxonomy 并存 → 五族用于快速建立心智图，双轴 + HELM 用于向深处扩展时定位新指标归属

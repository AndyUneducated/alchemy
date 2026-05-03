# Changelog

`CHANGELOG.md` 同时承担变更日志与 ADR 归档。每条记录以 `## n. 变更标题` 开头，紧接一行 `- **日期**：...`，heading 前后留空行。后续每个自然日建议最多追加 1～2 条 tech decision。

## 2. few-shot 范式落点

- **日期**：2026-05-02

Phase 2 引入 `num_fewshot` 时，决定**谁负责 example 抽样 + prompt 拼装**。

### Context

lm-eval 原版 `Task` 暴露 `fewshot_docs()` + `format_fewshot_example()`，Runner 拿到 `num_fewshot=K` 后从 pool 抽 K 条 example、排除自身、拼到 query 前。两个候选切法：

- **A. Task 一把梭**：`Task` 直接吐拼好的 prompt（含 K-shot），Runner 不知道有没有 fewshot
- **B. Task 出料 + Runner 抽样**（lm-eval 原版风格）：Task 只暴露"我有哪些 example 可用 + 一条 example 怎么显示"，Runner 负责 K、抽样、排除自身、拼接

### Decision

走 B。Task ABC 加两个**默认实现**（非 abstract）：

- `fewshot_docs()` 默认 `return self.docs()` —— 子类可 override 指 held-out split
- `format_fewshot_example(doc)` 默认 `f"{doc_to_text(doc)} {doc_to_target(doc)}"` —— 子类可改分隔符 / 多段结构

Runner 加 `_build_prompt(task, doc, num_fewshot, pool, rng)`，`num_fewshot=0` 时直 return `task.doc_to_text(doc)`（字节与 Phase 1 相同），`>0` 时抽 K 条非自身 example 用 `\n\n` 拼接。

`evaluate_active` 多收 `num_fewshot=0, fewshot_seed=0` 两个参数；`EvalResult` 加 `num_fewshot: int = 0` 字段并落到 `result.json` / `index.jsonl`，事后能区分 zero-shot vs K-shot 跑分。`score` 子命令不接 `--num-fewshot`：offline predictions 是预先生成的字符串，runtime 拼 fewshot 没有意义（YAGNI）。

### Consequences

- 与 lm-eval 原版语义一致，未来抄 task 配置零摩擦
- Task 仍然纯粹（不知道自己被 zero-shot 还是 K-shot 调用），换 N 一句 CLI flag 搞定
- `num_fewshot=0` parity 经 `test_zero_shot_equals_no_fewshot` 焊死，旧 `test_active_*_equals_offline_*` 全绿
- 抽样需要的"排除自身 + 同 seed 复现 + pool 不够不抛错"三条契约由 `test_fewshot.py` 兜底
- offline predictions 命名不变（`gold.jsonl` = 数据集；`predictions/perfect.jsonl` = 满分 baseline）；mt task 复用同一 schema

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

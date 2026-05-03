# Changelog

`CHANGELOG.md` 同时承担变更日志与 ADR 归档。每条记录以 `## n. 变更标题` 开头，紧接一行 `- **日期**：...`，heading 前后留空行。后续每个自然日建议最多追加 1～2 条 tech decision。

## 1. Phase 0 架构 & 叙事决策

- **日期**：2026-04-30

### Scope

|模块|内容|
|---|---|
|契约层 `api.py`|5 个 frozen dataclass 串成数据流：`Doc` → `Request` → `Response` → `SampleResult` → `EvalResult`|
|Task ABC|6 个抽象方法定义任务契约：`docs / doc_to_text / doc_to_target / process_results / aggregation / higher_is_better`|
|LM ABC|`generate_until` 必实现，`loglikelihood` / `loglikelihood_rolling` 预留至 Phase 4+|
|Registry|`@register_task("name")` 装饰器登记，`get_task(name)` 字符串调度|
|双模式 Runner|`evaluate_offline(task, preds)` (score) + `evaluate_active(task, lm)` (run)，共享 `_finalize` 尾段|
|存储层|`runs/<id>/{result.json, samples.jsonl}` + `runs/index.jsonl`（append-only 扁平索引）|
|CLI|`list-tasks` / `score` / `run` / `show` 四个子命令|
|MockLM|4 mode（gold / noisy / constant / rule），与 4 份 sentiment predictions 一一对应|
|首个 task `sentiment_clf`|30 行三分类，展示 accuracy / F1_macro / cohens_kappa 在不同 predictions 上的分歧|
|README 三层 taxonomy|五族 mental map（onboarding） + 双轴矩阵 + HELM 7 维度（严谨视角）|

### Implementation

|侧面|做法|
|---|---|
|双模式共享尾段|`task.process_results(doc, response)` 统一接收 `Response`；offline 路径以 JSONL 查表伪造 `Response(text=preds[id])`，其余完全一致，parity test 锁定等价性|
|metric 层 lazy|暂不引入 `metrics/` 抽象层；task 直调 sklearn。出现"首次跨 task 复用"或"无库可用"时再建 `metrics/X.py`|
|存储 YAGNI|仅使用 JSONL，不引入 SQLite；index.jsonl schema 与未来 SQLite 表同构（`CREATE TABLE runs AS SELECT * FROM read_json('index.jsonl')` 一行迁移）|
|registry 副作用|`tasks/__init__.py` 显式 `from . import sentiment_clf  # noqa: F401` 触发装饰器，避免漏改注册表，同 Django URL / Flask route / pytest fixture 模式|

### Options considered

**架构原型**：

- **lm-evaluation-harness (EleutherAI)**（选择）—— Task = dataset + prompt template + process_results + aggregation；LM 暴露 generate_until / loglikelihood。学术 benchmark 事实标准，paper 分数能对上
- inspect_ai（UK AISI）—— Task + Solver + Scorer，更 agent-friendly。但 solver 对 benchmark 简单任务过度设计
- deepeval —— metric-first、pytest-like。CI 集成好，但 task 可复现性弱（prompt 散在 test_case 里）
- 自造 —— 最灵活也最不"主流"，失去对齐学术生态的价值

**打分模式**：

- **score + run 双模式**（选择）—— score 是 offline 对比 gold + predictions（sacrebleu 风格），run 是 active 驱动 LM。两者共享 `task.process_results + aggregation + storage` 尾段；`evaluate_offline(task, preds) ≡ evaluate_active(task, PrerecordedLM(preds))` 由 parity test 锁定
- 只 run —— 排除：学 metric 没必要被 LM 调用 / API key / 网络拖累；且 `play/agent_engine` / `play/rag` 产物本质就是 JSONL，offline 打分零耦合
- 只 score —— 排除：丢了 harness 的 Task + LM + Runner 完整骨架，后续加 paradigm 时缺抽象支撑

**指标组织 taxonomy**：

- 只用五族（Classification+Agreement / Generation / LLM-as-Judge / RAG / Agent Trajectory）—— 易记忆但混合三个正交轴（task / method / pipeline），被追问"为什么 LLM-as-Judge 与 Generation 平级"时缺一致解释
- 只用双轴（task × method）+ HELM 7 维度 —— 严谨但 onboarding 不直观
- **两层并存**（选择）—— README 顶层用五族做 mental map，下一节立刻给"严谨视角：双轴 + HELM"及五族 ↔ 双轴对应表。代码层 `metrics/` 按**方法学**切文件（与五族解耦）

### Decision

- **架构**：lm-eval 骨架（Task ABC + LM ABC + Registry + Runner + `api.py` 契约层）**同时支持 score / run 双模式**；Phase 1 主路径走 score（metric 学习优先），MockLM 仅作 run 模式的演示 + parity 源
- **叙事**：README 用五族 onboarding + 双轴严谨视角**两层并存**；代码按方法学切 `metrics/X.py`（按需建，见 README 指导原则 #3）

## 2. Phase 2 实现：mt task + 6 生成指标 + few-shot 机制

- **日期**：2026-05-02

### Scope

|模块|内容|
|---|---|
|新 task `mt`|30 行 EN→中 翻译（含成语 / 同义改写场景）+ 4 份示例 predictions：`perfect` / `literal` / `paraphrase` / `garbage`|
|6 个生成指标|lexical 5 个：`exact_match` / `bleu` / `chrf` / `rouge_l` / `meteor`；embedding 1 个：`bertscore_f1`（`bert-base-chinese`）|
|`num_fewshot` 机制|Task ABC 增加 `fewshot_docs` + `format_fewshot_example` 默认方法；Runner 增加 `_build_prompt` helper；CLI 增加 `--num-fewshot` / `--fewshot-seed`|
|`EvalResult.num_fewshot` 字段|持久化至 `result.json` + `index.jsonl`，可区分 zero-shot 与 K-shot 跑分|

未实现并标 `deferred`：MoverScore（`moverscore-v2` 包自 2020 起无维护）+ learned tier（BLEURT / COMET / BARTScore，模型权重 ~5GB+ 需联网拉取）。

### Implementation

|侧面|做法|
|---|---|
|6 指标聚合|lexical 5 个在 `tasks/mt.py::aggregation()` 直调 `sacrebleu` / `rouge_score` / `nltk`；BERTScore 采用 lazy-import + `@lru_cache(1)` 缓存 scorer 实例，避免 `list-tasks` 等命令承担 ~700MB 模型下载与 ~3-5s torch 启动开销|
|中文 tokenization|BLEU / chrF 使用 sacrebleu 内置 `tokenize='zh'`；ROUGE 需传入自定义 `_ZhCharTokenizer`（默认 tokenizer 会过滤非 ASCII）；METEOR 采用字符级|
|few-shot 范式|Task 提供 example pool（`fewshot_docs()` 默认等同 `docs()`）与显示形式（`format_fewshot_example()` 默认拼接 `doc_to_text + doc_to_target`）；Runner 抽取 K 条非自身 example，以 `\n\n` 拼接到 query 之前。score 子命令不接 `--num-fewshot`（offline predictions 已预先生成，runtime 拼装 fewshot 无意义）|
|Phase 1 兼容|`num_fewshot=0` 时 `_build_prompt` 直接返回 `task.doc_to_text(doc)`，prompt 字节与 Phase 1 等价——既有 4 个 `test_active_*_equals_offline_*` parity 测试全部保持通过|
|存档兼容|`EvalResult.num_fewshot` 默认 0；旧 `result.json` 缺失该字段时 dataclass 反序列化仍可正常构造|
|metric 分歧示例|paraphrase predictions 上 BLEU=0.15 但 BERTScore F1=0.78（差值 0.63），作为 embedding tier 优于 lexical tier 的可复现证据，由 `test_paraphrase_bertscore_saves_meaning` 锁定|

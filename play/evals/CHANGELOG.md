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
|双模式 Runner|`evaluate_score(task, preds)` + `evaluate_run(task, lm)`，共享 `_finalize` 尾段|
|存储层|`runs/<id>/{result.json, samples.jsonl}` + `runs/index.jsonl`（append-only 扁平索引）|
|CLI|`list-tasks` / `score` / `run` / `show` 四个子命令|
|MockLM|4 mode（gold / noisy / constant / rule），与 4 份 sentiment predictions 一一对应|
|首个 task `sentiment_clf`|30 行三分类，展示 accuracy / F1_macro / cohens_kappa 在不同 predictions 上的分歧|
|README 三层 taxonomy|五族 mental map（onboarding） + 双轴矩阵 + HELM 7 维度（严谨视角）|

### Implementation

|侧面|做法|
|---|---|
|双模式共享尾段|`task.process_results(doc, response)` 统一接收 `Response`；score 路径以 JSONL 查表伪造 `Response(text=preds[id])`，其余完全一致，parity test 锁定等价性|
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

- **score + run 双模式**（选择）—— score 读 predictions JSONL 与 gold 直接打分（sacrebleu 风格，不驱动 LM），run 驱动 LM 跑 prompt。两者共享 `task.process_results + aggregation + storage` 尾段；`evaluate_score(task, preds) ≡ evaluate_run(task, PrerecordedLM(preds))` 由 parity test 锁定
- 只 run —— 排除：学 metric 没必要被 LM 调用 / API key / 网络拖累；且 `play/agent_engine` / `play/rag` 产物本质就是 JSONL，纯文件打分零耦合
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
|few-shot 范式|Task 提供 example pool（`fewshot_docs()` 默认等同 `docs()`）与显示形式（`format_fewshot_example()` 默认拼接 `doc_to_text + doc_to_target`）；Runner 抽取 K 条非自身 example，以 `\n\n` 拼接到 query 之前。score 子命令不接 `--num-fewshot`（predictions 已预先生成，runtime 拼装 fewshot 无意义）|
|Phase 1 兼容|`num_fewshot=0` 时 `_build_prompt` 直接返回 `task.doc_to_text(doc)`，prompt 字节与 Phase 1 等价——既有 4 个 `test_active_*_equals_offline_*` parity 测试全部保持通过|
|存档兼容|`EvalResult.num_fewshot` 默认 0；旧 `result.json` 缺失该字段时 dataclass 反序列化仍可正常构造|
|metric 分歧示例|paraphrase predictions 上 BLEU=0.15 但 BERTScore F1=0.78（差值 0.63），作为 embedding tier 优于 lexical tier 的可复现证据，由 `test_paraphrase_bertscore_saves_meaning` 锁定|

## 3. Phase 3 实现：族 3 LLM-as-judge 完全体 + 真 LM 适配层 + 首个 metrics/ 模块

- **日期**：2026-05-03

### Scope

|模块|内容|
|---|---|
|`metrics/judge.py`（首个 metric 模块）|4 个 judge：`judge_pointwise` / `judge_pairwise`（含 swap 去偏）/ `g_eval`（多维度 + n-sample 替代 logprob 通路）/ `self_consistency`（majority vote wrapper）+ `parse_pointwise_score` / `parse_pairwise_verdict` / `pairwise_winrate` cross-task utility|
|`models/ollama.py`|stdlib `urllib` /api/generate 适配器（不走 /api/chat 以保 prompt 字面可复现）；`base_url` 优先级 = 构造参数 > `EVALS_OLLAMA_BASE_URL` env > 默认 `localhost:11434`|
|新 task `qa_open`|10 条中文事实型开放式 QA + 4 份 stub predictions（perfect / paraphrase / wrong_fact / garbage）；构造接 `judge_lm: LM \| None` + `judge_n_samples`，judge 调用发生在 `process_results` per-sample，aggregation 仅 mean——保持 Task ABC 不破签名|
|`cli.py::parse_model_spec`|识别 `ollama:<model>` → OllamaLM；`openai:` / `anthropic:` / `gemini:` 抛 `NotImplementedError` 占位；未知 provider 仍 `ValueError`|
|`cli.py` `--judge-model` (score + run)|`score` 与 `run` 两子命令均新增 flag；dispatch 抽到 `_build_task_with_optional_judge` 共用 helper：传入则在 qa_open 上构造 `QAOpen(judge_lm=...)` 注入 judge_pointwise；其它 task 配合该 flag 立即 SystemExit。让 judge 这个 phase 3 核心特性在 CLI 层完整可达，无需绕路 Python 脚本。score + judge 是常用 hybrid 模式（pred 文件 + 真 LM 评分），与 run + judge 的 self-grading 模式正交|
|`tests/conftest.py`|双层 probe：① 服务可达 ② 指定模型已 `ollama pull`；任一失败 live 测试整文件 skip + 友好提示。默认测试模型 `qwen2.5:32b`，`EVALS_TEST_OLLAMA_MODEL` env 可 override|
|测试套（40 条新断言）|`test_judge.py`(12) / `test_qa_open_score.py`(6) / `test_qa_open_run.py`(3) / `test_qa_open_live.py`(4 live，含 cmd_run / cmd_score 两条 e2e) / `test_cli_spec.py`(9，含 3 条 helper dispatch 单元测试) / `test_ollama_lm.py`(6 live)。`*_live.py` 与 `test_ollama_lm.py` auto-probe gate。三元组 score/run/live 同 task 完整对称|

### Implementation

|侧面|做法|
|---|---|
|judge_lm 持有方式|`QAOpen(judge_lm=...)` 构造时注入；`process_results(doc, response)` 内调 judge → judge per-sample 触发，`aggregation()` 仅 mean。score / run 两路径自动复用，不破 Task ABC 签名|
|g_eval 不依赖 logprob|Ollama `/api/generate` 不返回 logprobs；用 `n_samples` 次采样 mean 替代 logprob 加权——离散分布的 expected value 估计。OpenAI 上线后可加 `g_eval_logprob` 二级实现，不改默认|
|pairwise 不进 task pipeline|pairwise 是"两份 candidates 比较"，与 single-pred-per-doc 的 Task 形状不匹配。pairwise_winrate 作为 cross-task utility 由 unit 测试覆盖；CLI 子命令 `score-pairwise` 留 phase 3.5|
|双模式 parity|延续 sentiment / mt 的 parity 范式，`test_run_gold_judge_equals_score_perfect_judge` 焊死 qa_open 上 `evaluate_run(MockLM(gold), judge=FakeJudge) ≡ evaluate_score(perfect.jsonl)` 在 aggregated + per_sample 两层字节相同|
|FakeJudgeLM 双策略|`outputs=list[str]` 按 cursor 推进 / `outputs=Callable[[prompt],str]` 规则函数。test_qa_open_score.py 的 paraphrase 用 char-Jaccard 规则模拟"语义判分"；wrong_fact 上 char-Jaccard 失效（"1368"→"1378" jaccard~0.94），改用 const(1) oracle 替身——明确标注"这是 char heuristic 抓不到的 fact-checking 缺口，留给真 LLM judge"|
|外部 provider 留口|未实现 OpenAI / Anthropic 适配类；`parse_model_spec` 显式抛 `NotImplementedError("phase 3 only ollama")`，架构口子明确不写空壳|

### Options considered

**判 LM 抽象 vs rag-style 单函数调用**：
- **保留 LM ABC**（选择）—— phase 0 立的 `models/base.py::LM` ABC + `MockLM` + `evaluate_run(task, lm: LM)` + `parse_model_spec → LM` + `test_run_gold_equals_score_perfect` 五处依赖统一接口。`OllamaLM(LM)` 薄包装 ~70 行
- rag-style `def ollama_chat(...)` —— 排除：绕开 ABC 等于推翻 phase 0 架构（约 150 行散落改动），且 parity test 焊接的就是 MockLM 与真 LM 同接口
- 对比：rag 不需抽象因为单 provider + 单调用 + embedding 确定性 + 无 mock 需求，与 evals 多 provider / 多调用 / parity-mocked 的结构性需求不同

**judge 是 closure 还是函数**：
- **closure 工厂**（选择）—— `judge_pointwise(lm, ...) -> Callable[[Doc, Response], float]`。便于 `self_consistency(judge_pointwise(lm, ...), n_samples=5)` 嵌套 wrap，也便于 task.process_results 复用同一份 callable
- 直接函数 `judge_pointwise(lm, doc, response, ...)` —— 排除：self_consistency 包裹时签名拼装麻烦

**g_eval 是否引 logprob 通路**：
- **不引**（选择）—— Ollama 无 logprobs，引 logprob 等于绑定 OpenAI provider；用 n-sample 多次采样 mean 是数学等价的离散分布期望估计
- 双通路（logprob + sample）—— 排除：phase 3 暂只接 ollama，OpenAI 上线时再加 `g_eval_logprob`，YAGNI

**判主 task 用 qa_open vs recipe_summary vs 复用 mt**：
- **qa_open**（选择）—— pointwise 在 task 层有强故事（paraphrase / wrong_fact 反向叙事）；pairwise / g_eval / self_consistency 主舞台在 metric 单元层（swap 去偏 / 多维加权 / majority vote），不是 task 故事不够强、是它们本就更适合在 metric 层焊死契约
- recipe_summary（让 g_eval 多维度成主角）—— 排除：~15 条菜谱原文数据成本，phase 3 体量考虑保住低数据成本
- 复用 mt 叠 judge 维度—— 排除：与 mt 的 BERTScore 救场叙事互文不充分，judge 在翻译上不是最自然舞台

**测试默认模型 qwen2.5:32b**：
- **选 32b**（选择）—— 本地已有零额外 pull / judge 质量更稳让 `>=3.5` 等阈值不 flake / 完整 live suite 实测 ~24s（M-series Mac），phase 3 8 条 live 测试可接受
- 3b（更轻 / CI 友好）—— 排除：用户机器无该 tag，会触发 ~1.9GB 拉取；judge `>=3.5` 阈值在小模型上更易 flake
- 7b（折中）—— 排除：本地无该 tag；32b 既已可用就直接选质量更高的
- `EVALS_TEST_OLLAMA_MODEL` env 双向 override（CI 降档 3b 提速 / 本地升档 72b 抬质量）

### Decision

- **架构**：保留 LM ABC，新增 `OllamaLM(LM)` 薄包装（stdlib /api/generate）；外部 provider 在 spec parser 显式 NotImplementedError 留口
- **metrics 模块化**：phase 3 触发首次新建 `metrics/judge.py`（README 指导原则 #3 的"跨 task 复用 + 无库可用"双重信号）
- **task 选型**：`qa_open` 简单且能承载 pointwise 强故事；其它 3 个 judge 主舞台在 metric 单元层
- **测试 gate**：live ollama 测试 auto-probe（服务 + 模型双层），不可达自动 skip + 友好提示

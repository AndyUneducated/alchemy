# Decisions

ADR（Architecture Decision Record）归档。每条以 `## n. 标题` 开头，紧接 `- **Status**` + `- **Date**` 元信息；正文沿用 `Scope / Implementation / Options considered / Decision` 四段（lm-eval phase-driven 体例）。**新决策追加到末尾，被取代的条目改 Status；不删旧条目**。日常进度（按里程碑） 见 [`JOURNAL.md`](JOURNAL.md)。

## 1. Phase 0 架构 & 叙事决策

- **Status**: accepted
- **Date**: 2026-04-30

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

- **Status**: accepted
- **Date**: 2026-05-02

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

- **Status**: accepted（`metrics/judge.py` 在 §4 重命名为 `metrics/judge_core.py`，行为不变）
- **Date**: 2026-05-03

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

## 4. Phase 4 实现：族 4 RAG 完全体（retrieval + grounding 双 task + 3 个新 metrics 模块）

- **Status**: accepted
- **Date**: 2026-05-03

### Scope

|模块|内容|
|---|---|
|`api.py` 契约扩展|① `Doc.target` 由 `str` 放宽为 `str \| None`（rag_retrieval 用 None 替代 "" 占位污染）；② `SampleResult.artifacts: dict[str, Any]` 新增（per-sample 非标量产物专用 bucket，与 `metrics: dict[str, float]` 形成 MLflow/W&B 风格 scalar/non-scalar 对偶）|
|`tasks/base.py` Task ABC 扩展|3 个对齐 lm-eval 的 hook（全 default 实现）：① `load_prediction(doc, row)` score 路径自定 JSONL row schema；② `process_docs(docs)` run 路径 LM 调用前的 docs 前置加工；③ `output_type` literal 加 `"none"` 让 runner 跳过 LM 调用|
|`runner.py` 双路径分支|`_load_predictions` 返回 `dict[str, dict]`（整 row）；`evaluate_score` 调 `task.load_prediction`；`evaluate_run` 在 LM 前调 `task.process_docs`、按 `output_type` 分支|
|`metrics/` 拆分|`metrics/judge.py` 重命名 `metrics/judge_core.py`（4 个范式：pointwise/pairwise/g_eval/self_consistency）；新建 `metrics/judge_rag.py`（5 个 RAG judge：faithfulness / answer_correctness / context_precision / context_recall / answer_relevancy + parse_statement_list / parse_tp_fp_fn 两个 RAG 专用 parser）；新建 `metrics/retrieval.py`（5 个 IR 指标 ranx 直调封装：recall@k / precision@k / mrr / ndcg@k / map@k）|
|`models/rag_retrieve.py`|`make_retrieve_fn(vdb, ...)` 工厂：subprocess 调 `play/rag/query.py --json`，解析 JSON envelope → `(query) -> (ids, contents)` 闭包；同源 chunk 去重保 rank|
|新 task `rag_retrieval`|8 条针对 `play/rag/docs/panel/*.txt` 的检索 query + 4 份 stub predictions（perfect / good_rerank / weak / garbage）；`output_type='none'` 跳 LM；`process_docs` 注入 `retrieved_ids` 到 `doc.metadata`|
|新 task `rag_qa`|8 条端到端 QA + 4 份 stub predictions（perfect / paraphrase / wrong_fact / garbage）；`process_docs` 注入 contexts/retrieved_ids；`doc_to_text` 纯字符串构造（0 IO）；`judge_lm` 可选（None=lexical baseline / 给则挂 5 个 RAG 维度）|
|`cli.py` 扩展|`_build_task_with_optional_judge` 重命名为 `_build_task_with_optional_deps`（保留旧名作向后兼容别名），加 4 个 RAG flag（`--vdb` / `--retrieve-top-k` / `--retrieve-mode` / `--rerank`）；`cmd_run` 在 `output_type='none'` 时允许省 `--model`，自动用 `retriever:<vdb>:<mode>` 标签|
|测试套（74 条新断言）|`test_api_contract_extension.py`(6) / `test_runner_task_hooks_compat.py`(5) / `test_output_type_none_dispatch.py`(2) / `test_retrieval_metrics.py`(11) / `test_judge_rag.py`(20) / `test_rag_retrieval_score.py`(9) / `test_doc_metadata_injection.py`(5) / `test_rag_retrieve_factory.py`(5) / `test_rag_qa_score.py`(8) / `test_cli_spec.py`(7 新 dispatch) / `test_rag_live.py`(3 live，过 ollama+vdb 双 probe gate)|
|`conftest.py` VDB probe gate|`panel_vdb_required` / `sample_vdb_required` skip marker + `panel_vdb_path` / `sample_vdb_path` fixture：缺 VDB 友好提示用户跑 `play/rag/ingest.py`|

### Implementation

|侧面|做法|
|---|---|
|`Doc.target: str \| None`|rag_retrieval 没有字符串 gold，旧实现强迫写 `target=""` 占位污染语义；放宽到 `Optional` 后既不破老 task（仍传 str），又让 retrieval task 显式声明"无字符串 target"。`asdict` JSON 序列化天然支持 None|
|`SampleResult.artifacts`|`metrics: dict[str, float]` 严守 scalar；retrieval IDs / trajectory steps / tool_calls 等非标量进 `artifacts`。命名对齐 MLflow / W&B 的 metrics(scalar) vs artifacts(non-scalar) 二分。文档 + 测试明示防垃圾桶纪律|
|`load_prediction` hook|默认实现 `(doc, row) -> (doc, Response(text=row['prediction']))` 与旧 `_load_predictions[id]` 字节相同——所有老 task 自动免改|
|`process_docs` hook|对齐 lm-eval 同名 callable（按"what"命名抗"垃圾桶"演化）；默认 identity；签名约束 `list[Doc] -> list[Doc]`，副作用纪律写在 docstring|
|`output_type='none'`|runner 在该分支生成占位 `Response(doc_id=d.id)`，不调 `lm.generate_until`；CLI 用 `_RetrieverOnlyLM` name-only stub 充当 EvalResult.model 标签源|
|`metrics/judge_core.py` + `judge_rag.py` 拆分|按"评分方法学" vs "评分对象（RAG pipeline 各环节）"两层正交切分；判 LM 范式扩展第 5 个不会拖累 RAG 维度演化（§3 单文件膨胀的预防）|
|`judge_rag.py` 自实现而非 import RAGAS|RAGAS 引入 langchain / openai / 数据科学全家桶（~30 个传递依赖）；本项目已有 LM ABC + closure 工厂模式，~150 行就把 5 个维度的 NLI/F1/extract 通路跑通；保留 prompt 字面字符串可控（lm-eval 不变量）|
|`models/rag_retrieve.py` subprocess 调用|遵循 monorepo 解耦原则（workshops.mdc）：`play/` sub-projects 不互相 import，跨项目走 CLI + JSON envelope；`play/rag` 自带的 chromadb / fastparquet 依赖不污染 evals 进程；同接口 future remote retriever 平滑迁移|
|路径 B+C 数据契约|`Response` 只装 LM-side 输出（保持 phase 0 契约纯净）；pipeline 产物（retrieved_ids / contexts）住 `Doc.metadata`。score 路径 `load_prediction` 写、run 路径 `process_docs` 写；`process_results` 双路径都从 `doc.metadata` 读，零分支|
|RAG IR 指标聚合|`metrics/retrieval.py` 工厂返回 `(list[SampleResult]) -> float`；从 `SampleResult.artifacts.{pred_ids, gold_ids}` 拉数据 → 构造 ranx Qrels/Run → `evaluate(qrels, run, metric_name)`；空数据 / 缺字段 → 0.0 优雅降级|
|向后兼容回归|3 类 parity test 覆盖：① `test_api_contract_extension`（Doc.target / artifacts 形状）；② `test_runner_task_hooks_compat`（老 task 用 default hook 字节级 parity）；③ `test_output_type_none_dispatch`（用 spy LM 验证 `output_type='none'` 真没被调）|

### Options considered

**RAG corpus 来源**：
- **复用 `play/rag/docs/panel/*.txt`**（选择）—— 公司治理叙事天然有"5 个角色 × 6 篇"的 doc-level 区分度，做 retrieval gold 自然清晰；零额外数据成本
- 自建 corpus —— 排除：phase 4 时间预算优先放在指标 + task + 测试上
- 用 BEIR / MS MARCO 等公开 IR dataset —— 排除：太大，phase 4 教学叙事用不上；中文场景适配麻烦

**`metrics/` 模块布局**：
- **预期式 split**：`judge_core.py` + `judge_rag.py` + `retrieval.py`（选择）—— phase 4 是"评分方法学 + 评分对象"两轴并存的临界点，先 split 一次后续不破文件结构；与 ragas 平铺布局对齐
- 全平铺 ragas 风格 —— 暂排除：本项目体量小，5-7 个 metric 模块平铺 OK，但当前 phase 文件数还不到这一步
- 不拆继续 `judge.py` —— 排除：5 个 RAG 维度 + parser 加进去后单文件超 500 行，难以演化

**与 `play/rag` 的依赖方式**：
- **subprocess + JSON envelope**（选择）—— 严守 monorepo 解耦原则；evals 进程不被 chromadb/ollama 客户端污染；接口同型 future HTTP retriever 平滑迁移
- 直接 `from play.rag.query import search` —— 排除：违反 workshops.mdc；evals 进程被强制带上 chromadb/fastparquet/torch 等依赖；无法在 evals 测试 mock 真 IO 边界
- HTTP service —— 排除：过度工程；phase 4 体量不需要

**`Response` 是否装 RAG-side 数据**：
- **路径 B+C：Response 只装 LM-side，pipeline 产物住 `Doc.metadata`**（选择）—— `Response` 跨 task 复用通用契约，不被某一类 task 的特殊产物污染；遵循 lm-eval `Doc.metadata` 作 free-form bucket 的惯例
- 路径 A：Response 加 `retrieved_ids: tuple[str, ...]` 字段 —— 排除：契约层为 RAG 一类 task 让步；老 task 必须默认 None 处理，加密了 Response 的语义

**RAG 维度是 import RAGAS 还是自实现**：
- **自实现 5 维度（选择）**—— 复用本项目已有的 LM ABC + closure 工厂；prompt 字面字符串可控（lm-eval 不变量）；~150 行就跑通；测试 stub 极简
- 直接 `pip install ragas` —— 排除：传递依赖膨胀（langchain/openai/全家桶 ~500MB）；prompt 黑盒；与 evals 的 LM 适配层冲突

**rag_retrieval 是否走 LM 调用 fake adapter**：
- **`output_type='none'` literal**（选择）—— Task ABC 加一个枚举 + runner 一个分支，干净；表达力："这个 task 不要 LM"是 task 自己的属性
- `RetrieveOnlyLM(LM)` 假 LM adapter —— 排除：`generate_until` 永远抛 NotImplementedError；用 LM 假装 retriever 是接口污染
- 在 `doc_to_text` 里做 retrieve I/O —— 排除：违反"`doc_to_text` 是纯字符串构造"的 lm-eval 不变量；与 score 路径无法对齐

### Decision

- **数据契约**：`Doc.target: str | None` + `SampleResult.artifacts: dict[str, Any]`；`Response` 不动
- **Task ABC**：3 个 default-implemented hook（`load_prediction` / `process_docs`）+ 1 个 literal 扩展（`output_type="none"`），全部向后兼容
- **metrics 拆分**：`judge_core.py`（4 个范式）+ `judge_rag.py`（5 个 RAG 维度，自实现）+ `retrieval.py`（5 个 IR 指标 ranx 直调）
- **`play/rag` 依赖**：subprocess + JSON envelope，遵循 monorepo 解耦
- **测试 gate**：复用 ollama-probe + 加 vdb-probe 双层；缺任一即 skip + 友好提示

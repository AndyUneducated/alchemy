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

## 5. Phase 5 实现：族 5 agent trajectory 完全体（单 task + 5 个 metric + envelope 跨项目接 agent_engine）

- **Status**: accepted
- **Date**: 2026-05-03

### Scope

|模块|内容|
|---|---|
|新 metric 模块 `metrics/trajectory.py`|5 个 closure-factory metric（与 judge_core / judge_rag / retrieval 同协议）：`task_success(predicate)` outcome 维度（τ-bench `verify(state)` 同源）；`tool_call_set_f1()` / `argument_correctness()` / `trajectory_match()` / `trajectory_coverage()` 4 个 process 维度；手写 ~20 行 Levenshtein DP 不引 python-Levenshtein/rapidfuzz|
|新 task `agent_traj`|3 docs（panel / brainstorm / example，分别覆盖投票决议 / 自由讨论 / kitchen-sink）× 4 份 stub predictions（perfect / partial / **wrong_decision** / garbage）；`output_type='none'`（agent 整链路在 subprocess 内）；可选 `judge_lm` 注入 plan_quality（复用 judge_core.g_eval 三维度：plan_structure / tool_choice / completeness）|
|`models/agent_engine_run.py`|`make_run_fn(scenarios_root=play/, timeout=600s)` 工厂：subprocess 调 `python -m agent_engine <scenario> --no-stream --save-result-json <tmp>`，读 JSON envelope → 回传 dict；闭包形态与 `models/rag_retrieve.py::make_retrieve_fn` 同形|
|跨项目动 agent_engine 两处|① `cli.py` 加 `--save-result-json PATH` flag（~15 行）：把 `Result` 用 `dataclasses.asdict` 写 JSON envelope；② `artifact.py` 5 个 event 各加 `"arguments": dict(args)` 字段（~5 行）：让 argument_correctness 在 run 路径有真数据可对（pre-phase 5 artifact_event 仅留人类可读 content 字符串）|
|`cli.py` evals 端|`_build_task_with_optional_deps` 加 `AgentTraj` 分支，永远注入 `make_run_fn()`（cheap closure，score 路径不会触发 subprocess）；`cmd_run` 在 `output_type='none'` + task.name='agent_traj' 时给 `_RetrieverOnlyLM("agent_engine")` 标签|
|测试套（55 条新断言）|`test_metrics_trajectory.py`(31) / `test_agent_traj_score.py`(9) / `test_agent_traj_envelope.py`(14) / `test_agent_traj_run_live.py`(1，过 ollama+agent_engine 双 probe gate)|
|`conftest.py` 双 gate 扩展|`agent_engine_required` skip marker：复用 ollama_probe + 加 brainstorm.md 存在性检查；缺 agent_engine 包或 scenarios 目录 → 友好提示|

### Implementation

|侧面|做法|
|---|---|
|trajectory 数据形状|沿用 phase 4 path B+C：LM-side 走 `Response`，pipeline 产物住 `Doc.metadata['trajectory']`。trajectory 字段：`{transcript, artifact, warnings, success}`（envelope 原样）+ 派生 `{tool_calls, tool_seq, decision}`（task 一次性派生供 metric 复用）。**0 个新 dataclass**，是 phase 4 契约的自然延伸而非新负担|
|envelope schema 同源|`agent_engine.Result(artifact, transcript, success, warnings)` 4 字段 dataclass + `dataclasses.asdict` 直出；evals 端 `_pin_trajectory(doc, envelope)` 派生 tool_calls / tool_seq / decision；`test_agent_traj_envelope` 锁 `Result` 字段集合 == `{artifact, transcript, success, warnings}`，agent_engine 改字段名 → CI 即时 fail|
|tool_call_set_f1 用 `(tool, caller)` 而非 `(tool, args)`|args 含 LLM 生成的长文本（write_section content 等），gold 无法在 fixture 阶段固定。`(tool, caller)` 回答"谁调了哪个工具"，与 `argument_correctness`（处理 args 侧，gold args ⊆ pred args 子集匹配）构成互补。BFCL 严格 `(tool, args)` 匹配是函数调用 benchmark 场景，多 agent 自由生成场景下 `(tool, caller)` 信号更稳|
|`trajectory_match` 命名（不叫 `edit_distance`）|归一化 `1 − Lev/max(len)` ∈ [0,1] ↑，与项目其它 metric 全 [0,1] higher-is-better 约定一致；BFCL "trajectory_match" 同名（README C.5 同步从 `edit_distance ↓ [0,∞)` 改为 `trajectory_match ↑ [0,1]`，公式同步为归一化 similarity）|
|task_success 谓词外置|Task 在 `process_results` 内按 `doc.metadata.success_predicate` 选 `predicate_decision_in_options`（panel-style：finalize 落定 + decision ∈ 白名单）或 `predicate_speakers_covered`（free-form：全员发言 + success=True）；显式声明优先于自动 fallback|
|`plan_quality` 复用 `judge_core.g_eval` 而非新增 trajectory judge|复用 phase 3 的多维度 G-Eval（n-sample 替代 logprob）；trajectory 侧把 `tool_seq + decision + artifact` 拍扁成单段文本喂给 judge，3 个维度 plan_structure / tool_choice / completeness 取 mean 上聚合面板，子维度走 `_plan_<dim>` 私有键供 drill-down，不污染 aggregated 主指标|
|跨项目 envelope = `--save-result-json`|与 `play/rag/query.py --json` 同源——人类导出格式（`--save-transcript` JSON list / `--save-artifact` Markdown）vs 机器消费格式（complete envelope）；envelope 走 file 而非 stdout，因 agent_engine 整段讨论会刷 stdout，无法寄生 stdout 当 channel|
|artifact_event 新增 arguments 字段是 phase 5 驱动的 agent_engine 改造|pre-phase 5 artifact_event 仅留 `content: f"{caller} wrote section ..."` 人类可读字符串，args 丢失；phase 5 加 `"arguments": dict(args)` 5 处（write_section / append_section / propose_vote / cast_vote / finalize_artifact），让 evals 端 `argument_correctness` 在 run 路径有真数据。additive 改造，老消费者忽略未知键|
|run-path mock / `--replay-envelope` 不做|`output_type='none'` 让 evals 层无 LM 可 mock，"mock subprocess" 不是 task 自然边界。score 路径的 4 份 stub predictions 已覆盖 mock 教学需求；run 路径只跑 live e2e，CI 由 conftest 双 gate 决定 skip。**原则 5 parity 在 trajectory task 上是显式让步**——同源 phase 4 RAG 缺口|
|双 gate live 测试用 `brainstorm.md`|panel 11 步 ~分钟级，CI 不友好；brainstorm 2 步 ~10-30s（M-series Mac + qwen2.5:32b 实测 20s），双 gate 满足时端到端跑过|

### Options considered

**5 个 metric 选哪些**（行业对标）：
- **5 个：`task_success` + `tool_call_set_f1` + `argument_correctness` + `trajectory_match` + `trajectory_coverage`**（选择）—— 覆盖 outcome（τ-bench）/ tool 集合（BFCL）/ tool 参数（BFCL）/ 序列（BFCL trajectory_match / inspect_ai trace match）/ required-callers 4 类信号正交；`plan_quality` 走 G-Eval 复用既有 judge_core
- 加 `tool_selection_accuracy` —— 排除：与 `trajectory_match` 信号高度重合
- 加 `step_count_efficiency` —— 排除：agent_engine steps scenario-pinned，恒为 ~1.0 无 signal

**与 `play/agent_engine` 的依赖方式**：
- **subprocess + JSON envelope（选择）**—— 同源 phase 4 RAG 决策（DECISIONS §4），严守 monorepo 解耦原则；evals 进程不被 agent_engine 的多 LLM 客户端依赖污染；`config.py` 同名冲突等坑通过进程边界规避
- 直接 `from play.agent_engine import Engine` —— 排除：违反 workshops.mdc；evals 进程被强制带上 ollama / openai / anthropic / gemini 全套 SDK
- HTTP service —— 排除：过度工程；workshop 体量不需要

**`tool_call_set_f1` 的 multiset key**：
- **`(tool, caller)`**（选择）—— 与 `argument_correctness` 子集匹配互补；多 agent 自由生成场景下 args 不可预测
- BFCL 标准 `(tool, json_args_normalized)` —— 排除：args 含 LLM 生成的长文本，gold 在 fixture 阶段无法固定；fixture 一改 metric 数值就漂移
- 仅 `tool` —— 排除：丢失"谁调了"信息，coverage 信号失真

**`trajectory_match` 命名 vs README C.5 的 `trajectory_edit_distance ↓ [0,∞)`**：
- **改成 `trajectory_match ↑ [0,1]`（归一化 similarity）**（选择）—— 与项目其它 metric 全 [0,1] higher-is-better 约定一致；BFCL "trajectory_match" 同名；README C.5 同步更新
- 保留原 `edit_distance ↓ [0,∞)` —— 排除：方向不一致让聚合面板需要按 metric 分别解读，可读性差

**`plan_quality` 是新建 trajectory judge 还是复用 g_eval**：
- **复用 `judge_core.g_eval`**（选择）—— phase 3 已建好的 4 范式之一；trajectory plan_quality 是"对 agent 跑出的轨迹打分"，与 g_eval"对生成结果打多维分"形状完全一致；`metrics/trajectory.py` 不重复实现 G-Eval
- 新建 `judge_trajectory.py` —— 排除：重复 G-Eval 模板；judge 范式跨 evaluation domain 复用是 phase 3 决策的核心红利

**run-path mock / `--replay-envelope`**：
- **不做（选择）**—— `output_type='none'` 让 evals 层无 LM 可 mock，mock subprocess 不是 task 自然边界；4 份 stub predictions 已覆盖教学需求；显式让步是好的（**原则 5 parity** 在 trajectory task 上的清晰承认）
- 加 `--replay-envelope PATH`：从离线 envelope 读 trajectory 跳过 subprocess —— 排除：CI 收益小（live test 已被 conftest 双 gate skip），增加表面积；与 phase 4 RAG live 测试也无 mock 路径同源

### Decision

- **5 metric**：`metrics/trajectory.py`（task_success / tool_call_set_f1 / argument_correctness / trajectory_match / trajectory_coverage 5 个 closure factory + 2 个 ready-made predicate）；`plan_quality` 复用 `judge_core.g_eval`
- **数据契约**：trajectory eval 0 个新 dataclass、0 个新 ABC hook，复用 phase 4 path B+C 的 `Doc.metadata` 通路（`Doc.metadata['trajectory']` 字段集合在 README 数据契约小节冷封）
- **跨项目 envelope**：`agent_engine cli.py --save-result-json` + `evals models/agent_engine_run.py make_run_fn`；agent_engine 同步 cross-link ADR §11
- **agent_engine artifact_event 加 arguments 字段**：phase 5 驱动的 5 处 additive 改造，让 run 路径 argument_correctness 有真数据
- **测试 gate**：复用 ollama-probe + 加 agent_engine-probe（包 + brainstorm.md 存在性）双层；缺任一即 skip + 友好提示
- **run-path mock 显式不做**：写入"风险与显式不做"表，与 phase 4 RAG 同源，原则 5 parity 让步

## 6. Phase 6 横切 Efficiency

- **Status**: accepted
- **Date**: 2026-05-04

### Scope

|模块|内容|
|---|---|
|新 metric 模块 [`metrics/efficiency.py`](metrics/efficiency.py)|`_PRICE_PER_1M_TOKENS` 价格表（4 entry：ollama:qwen2.5:32b + openai:gpt-4o-mini + anthropic:claude-3-5-haiku-20241022 + gemini:gemini-1.5-flash） + `compute_cost_usd(model, in, out)` + `efficiency_aggregated(srs)` 返回 4 子组嵌套 dict + `inject_per_sample_efficiency(srs, resps, model)` runner injector|
|契约层加 1 嵌套 dataclass [`api.py::Usage`](api.py)|`{tokens_in: int \| None, tokens_out: int \| None}`；嵌入 `Response.usage`，与 OpenAI / Anthropic / inspect_ai SDK 同形|
|`Response` 新增字段|`latency_ms`（pre-existing 占位字段 phase 6 真填）+ `usage: Usage \| None`（phase 6 新加）；老 `Response(doc_id, text)` 调用点完全不破|
|`EvalResult.aggregated` 类型放宽|`dict[str, float]` → `dict[str, Any]`（实际形态 `dict[str, float \| dict]`）；老平铺 schema 仍合法|
|[`runner.py`](runner.py) 加横切注入|`evaluate_run` 在 `task.process_results` 之后调 `inject_per_sample_efficiency` 把 latency/usage 拷进 `SampleResult.metrics`；`_finalize` 在 `mode='run'` 分支挂 `aggregated["efficiency"] = efficiency_aggregated(srs)`|
|[`models/ollama.py`](models/ollama.py) 解析|`/api/generate` 响应的 `prompt_eval_count` / `eval_count` / `total_duration` 三字段填入 `Response.usage` / `Response.latency_ms`|
|[`cli.py`](cli.py) `_fmt_kv` + `_fmt_row` + `_print_aggregated`|嵌套 dict 递归打印为 dot-path（`efficiency.latency_ms.p50=12.50`），cmd_score / cmd_run 顶部 + `show` index row 全适配|
|测试套（27 条新断言 + 修订 8 条 parity）|`test_metrics_efficiency.py`(13) + `test_runner_efficiency.py`(6) + `test_api_contract_extension.py`(+5) + `test_ollama_lm.py`(+1) + `test_cli_spec.py`(+3)；改 `test_runner_run.py` / `test_doc_metadata_injection.py` / `test_runner_task_hooks_compat.py` / `test_qa_open_run.py` 8 处 parity 断言为 task-keys subset 比对 + 显式断言 score 不含 efficiency 子组|

### Implementation

|侧面|做法|
|---|---|
|cross-cutting AOP 风格|task 不改 `process_results` / `aggregation`，零 task-side 增量；横切由 runner 在 `_finalize` 注入 + `inject_per_sample_efficiency` 在 process_results 之后注入。phase 7+（safety / calibration / robustness）按同协议追加，每加一个新 task 不会因为新横切多写一行 task 代码|
|`Usage` 嵌套而非顶层平铺|与 OpenAI `CompletionUsage` / Anthropic `Usage` / inspect_ai `ModelUsage` 同形：嵌套 typed object 让多模型生态扩展（reasoning_tokens / cached_tokens / audio_tokens）时只动 `Usage` 字段，不污染顶层 `Response` schema；`Response.usage = None` 默认值保证老 `Response(doc_id, text)` 调用点完全不破|
|`aggregated` 嵌套子组而非平铺|HELM 7 维度作 ontology：cross-cutting 维度（efficiency / safety / calibration / robustness）各占一个 nested namespace，task-specific 指标（accuracy / f1 / em / rouge_l / task_success / cohens_kappa / ...）继续顶层平铺；phase 6 起 `aggregated` 类型放宽到 `dict[str, Any]`，老 score 路径全平铺仍合法（`test_eval_result_aggregated_still_accepts_flat_only` 锁住）|
|价格表 per 1M tokens 单位|与 OpenAI / Anthropic / Together / Fireworks 公开报价同单位，entry 直接复制粘贴免人脑除 1000；4 entry 覆盖 ollama:qwen2.5:32b（默认本地）+ external 三家各一最便宜调试 SKU（cli.py EXTERNAL_PROVIDERS 三家全覆盖；phase 3 NotImplementedError 暂跑不到，但 entry 在不破坏，phase 3+ 启用时即用）|
|cost = 0 vs cost = None 边界|`compute_cost_usd` 协议：tokens_in/out 任一 None → cost None（不污染 None 收集协议）；model 不在表 → cost 0.0（语义"不计费"，mock 路径与未填的 ollama tag 同；不报错避免 fail-loud over-engineering）|
|百分位 stdlib 实现|`statistics.quantiles(data, n=100, method='inclusive')` 等价 `numpy.percentile(linear)` 的 1..99 整数 cutpoint；项目 phase 1-5 现有代码 0 处显式 import numpy，新建模块继续保持。`_percentile` helper 兜底单元素 / 空列表，避免 `quantiles` 要求 n>=2 的 ValueError 在小 batch 下爆|
|run-only 注入|`_finalize` 仅在 `mode='run'` 分支挂 efficiency 子组：score 路径无 LM 调用 → 真无 efficiency 信号，注入 0/None 占位是数据噪音；显式让步好过 stretch schema|
|MockLM 不报|MockLM 不打 LM，不填 latency_ms / usage；`efficiency_aggregated` 收集时跳过 None → 4 子组键值全 0 但 schema 在；run mock 路径仍可演示 efficiency schema 的下游消费（CLI 渲染 / W&B dashboard / SQLite read model），但数值 = 0 是诚实的|
|parity test 改子集比对|`test_*` 8 处 parity 断言（`test_runner_run.py` x5 + `test_doc_metadata_injection.py` x1 + `test_runner_task_hooks_compat.py` x2 + `test_qa_open_run.py` x1）改为 `task_agg(r_run.aggregated) == task_agg(r_score.aggregated)`（剥离 efficiency 子组）+ 显式断言 score 不含 efficiency / run 含。架构等价性保留：efficiency 是 cross-cutting 增量不是 task 尾段路径分叉|
|`SampleResult.metrics` 仍 dict[str, float]|per-sample efficiency 字段（latency_ms / tokens_in / tokens_out / cost_usd）按 float 写入 metrics dict；不破 phase 4 锁定的"metrics 仍只装标量"语义契约|

### Options considered

**`Response` token 字段 schema**：

- **`Response.usage: Usage \| None` 嵌套**（选择）—— 与 OpenAI / Anthropic / inspect_ai SDK 同形；多模型生态扩展（reasoning_tokens / cached_tokens / audio_tokens）只动 `Usage` 不污染顶层；老调用点 `Response(doc_id, text)` 不破
- 顶层平铺 `Response.tokens_in / Response.tokens_out` —— 排除：未来加 reasoning/cached tokens 时顶层 `Response` 字段膨胀；与行业 SDK 嵌套对象惯例不一致
- 走 `Response.metadata: dict` —— 排除：失去 typed object 优势；IDE 无字段提示；序列化反序列化无 schema 锁

**`aggregated` 命名结构（efficiency 落点）**：

- **嵌套子组 `aggregated["efficiency"][group][stat]`**（选择）—— HELM 7 维度作 ontology，cross-cutting 维度各占独立 namespace；同名指标跨 phase 位置不漂移（cohens_kappa 在 phase 1 / 8 都顶层）；phase 7+ safety / calibration / robustness 按协议追加，扩展 zero-cost
- 平铺 `aggregated["latency_ms_p50"]` / `aggregated["latency_ms_p95"]` —— 排除：phase 7+ 加 safety_refusal_rate / calibration_ece 时顶层键空间继续膨胀；与 task-specific 指标平起平坐导致 namespace 污染（accuracy 与 latency_ms_p50 同位语义错位）
- 子组按"方法学族"分类（如 `aggregated["agreement"]["cohens_kappa"]`）—— 排除：会让 task-specific 指标跨 phase 位置漂移（phase 1 顶层 cohens_kappa，phase 8 改为 `agreement.cohens_kappa`），破坏 cross-run JSON_EXTRACT 路径稳定

**efficiency 子组 schema-on-write 还是 schema-on-data**：

- **永远填 4 子组（schema 稳定）**（选择）—— MockLM 不报 → 子组键值全 0，子组 keys 在；下游消费（CLI 渲染 / W&B / SQLite read model）可写一份 schema 不需分支判 None；schema-on-write 对 cross-run 索引最友好
- schema-on-data：无 efficiency 信号则不注入 `aggregated["efficiency"]` —— 排除：让 mock run 与 real run 的 schema 不同（mock 不带子组，real 带），CLI / dashboard 必须 if-branch；schema 漂移成本远大于 0 占位的"轻微误导"

**`_PRICE_PER_1M_TOKENS` 是否预填**：

- **预填 4 entry**（选择）—— ollama:qwen2.5:32b（项目默认）用 Together AI / Fireworks 开源平台报价做 "如果在 cloud 跑会花多少" 类比，给本地教学一个非零数字；external 三家各一最便宜调试 SKU 覆盖 cli.py EXTERNAL_PROVIDERS，phase 3+ 启用即用；entry 在不破坏，价变时手动同步
- 空启动 + docstring 说"phase 3 启用时填" —— 排除：mock / ollama 永远 0，cost_usd 字段失去演示价值；用户首次 run 看到 cost=0 不知道是"未填"还是"真免费"
- 全量集成 [tokencost](https://github.com/AgentOps-AI/tokencost) —— 排除：phase 6 仅 4 entry 不值得引依赖；tokencost 表单价更新有滞后；phase 3+ 真启用 external provider 时再考虑切

**price 单位 per 1K vs per 1M tokens**：

- **per 1M tokens**（选择）—— 与 OpenAI / Anthropic / Together / Fireworks 公开报价同单位，entry 直接复制粘贴免人脑除 1000；行业近 2 年统一往 per-1M 迁
- per 1K tokens —— 排除：与 OpenAI 老定价页同源但与新页 mismatch；维护时容易 transcribe error

**`(input_price, output_price)` tuple vs 单一价格**：

- **二元 tuple**（选择）—— 行业惯例 input != output（output 是 autoregressive decode，4-5x input 价；anthropic claude-3-5-haiku 锁了 5x 倍数）；开源平台 Together / Fireworks 常 1:1，可填同值；保留区分能力
- 单一 price —— 排除：忽略主流闭源 provider 的 in/out 差异，cost 数字偏低；演示意图（教学 cost-aware）受损

**percentile 实现**：

- **stdlib `statistics.quantiles`**（选择）—— 项目 phase 1-5 现有代码 0 处显式 import numpy，新建模块继续无新依赖；3-line `_percentile` helper 兜底空 / 单元素列表
- numpy `np.percentile` —— 排除：项目 sklearn / scipy 已传递引 numpy 但 evals 自己代码不显式 import，保持一致性
- 自实现 sort + index —— 排除：与 numpy linear interp 行为容易差几个数值（quartile 端点处理），用 stdlib 已有实现是 KISS

**MockLM 是否估算 efficiency**：

- **不估算，永远报 None**（选择）—— "显式 None > 不准估算"原则；引 tiktoken 估 token 数 + 用 perf_counter 估 latency 都是噪音；mock 路径的核心价值是 task 逻辑教学（accuracy / f1 / kappa 数学），efficiency 演示让位给 ollama:qwen2.5:32b 真跑（已 ollama_required gate ready）
- 估算 token 数（tiktoken / 分词字符长度）+ 测 perf_counter latency —— 排除：tiktoken vs Ollama tokenizer 误差大；perf_counter 含 Python 调用栈不是端到端 latency；噪音 > 信号

**reproducibility metadata（stderr / schema_version / git_hash / fewshot_seed list / system_prompt_hash 等 7 项 known gaps）**：

- **deferred 至后续 phase**（选择）—— phase 6 scope 严守"runner 自动采集 efficiency"原则；reproducibility metadata 是与 efficiency 正交的独立维度（HELM "metadata" 与 "efficiency" 是两个维度），phase 11+ 单独成 ADR 处理
- 与 phase 6 一并落地 —— 排除：scope creep；7 项各自有独立设计决策（如 stderr 怎么算、git_hash 怎么获取、schema_version 升级策略），混在一个 phase 增加 review 难度

### Decision

- **嵌套契约**：`Response.usage: Usage | None` 嵌套 dataclass + `EvalResult.aggregated: dict[str, Any]` 类型放宽，cross-cutting 维度走 `aggregated[<dim>]` 嵌套子组（HELM 7 维度作 ontology），task-specific 指标继续顶层平铺
- **runner 自动注入**：`evaluate_run` 在 `task.process_results` 后调 `inject_per_sample_efficiency` 拷 per-sample 实测值；`_finalize` 在 run 模式注入 `aggregated["efficiency"]` 子组（4 group × 1-3 stat 永远 schema 稳定）；score 模式不注入
- **价格表 4 entry 预填**：ollama:qwen2.5:32b（默认本地，cloud-equiv 价 $0.80/1M 同价）+ openai:gpt-4o-mini ($0.15/$0.60) + anthropic:claude-3-5-haiku-20241022 ($1.00/$5.00) + gemini:gemini-1.5-flash ($0.075/$0.30)；per 1M tokens × (in_price, out_price) tuple；未命中 model → cost 0.0（mock / 未填 tag 同）
- **stdlib 算 percentile**：`statistics.quantiles(method='inclusive')`，3 行 `_percentile` helper 兜底空/单元素；不引 numpy
- **OllamaLM 真填**：解析 `/api/generate` 的 `prompt_eval_count` / `eval_count` / `total_duration` 写入 `Response.usage` / `Response.latency_ms`
- **CLI 嵌套打印**：`_fmt_kv` 递归 dot-path（`efficiency.latency_ms.p50=12.50`）；`cmd_score` / `cmd_run` 顶部 + `show` index row 全适配
- **parity test 改子集比对**：8 处 `r_run.aggregated == r_score.aggregated` 改为 `task_agg(r_run) == task_agg(r_score)`（剥离 efficiency 子组）+ 显式锁"score 不含 efficiency / run 含"，架构等价性在 task-specific 指标层面保留
- **MockLM 不估算**：mock 路径 efficiency 子组键值全 0；显式 None > 不准估算；efficiency 演示让位给 ollama 真跑（conftest ollama_required gate ready）
- **reproducibility metadata 显式 deferred**：stderr / schema_version / git_hash / fewshot_seed list / system_prompt_hash / dataset_revision_hash / lm_call_seed 7 项 known gaps 记入"已知缺口"清单，phase 11+ 单独成 ADR 处理

## 6.1. Phase 6 efficiency follow-up（基于实测产物反向审查）

- **Status**: accepted（其中"§1.3 sample 层 4 efficiency 键 flat 写 0 占位"被 §7.D 单独 supersede——sample.metrics 改 nested 子组 `metrics["efficiency"]`；其余 6 项 §1.1 / §1.2 / §1.4 / §1.5 / §1.6 / §1.7 仍生效）
- **Date**: 2026-05-04

### Scope

phase 6 上线后跑全量 233 测试 + 端到端 demo 落盘 4 个 run（mock / score / ollama×2），从 `result.json` / `samples.jsonl` / CLI stdout 反向审查；用户从 `AUDIT.md §1.1-1.7` 确认要修的 7 项一致性 / fail-loud / 渲染问题。

|侧面|改动|来源|
|---|---|---|
|`aggregated.efficiency.cost_usd` 加 `mean`|与 `tokens_in/out.{total,mean}` 对称；per-call 平均成本是用户对比 model 时的核心信号|audit §1.1|
|`aggregated.efficiency.latency_ms` 加 `max`|HELM efficiency 维度标配 (mean,p50,p95,**max**)；小 N 下 worst-case 通过 max 暴露（如 demo 实测 cold-start latency=1339ms，p95=1274ms，max=1339ms 是 cold-start 入口）|audit §1.2|
|`SampleResult.metrics` schema-on-write 两层一致 ⚠️ **§1.3 写位置被 §7.D supersede**|`inject_per_sample_efficiency` 永远写 4 efficiency 键，None / 缺失 0.0 占位；schema-on-write 哲学保留；phase 7 起 nested 子组 `s.metrics["efficiency"]["latency_ms"]` 替代原 flat 写法 `s.metrics["latency_ms"]`|audit §1.3 选项 A → §7.D|
|`compute_cost_usd` 未命中 model 时 fail-loud `UserWarning`|`_warn_unknown_pricing_model` 用 `functools.lru_cache(maxsize=128)` 防刷屏；让用户区分 cost=0 的三种状态（真免费 / 未测得 / 模型不在表里）|audit §1.4|
|`tokens_in.total` / `tokens_out.total` 用 `int(sum(...))`|token 是离散计数，整数语义；`mean` 仍 `float`（avg 可有小数）；`SampleResult.metrics` 仍 dict[str, float] 不破契约|audit §1.5|
|`inject_per_sample_efficiency` 去掉 `getattr` 防御|`Response` 是 frozen dataclass 字段固定；`resp.latency_ms` 直接取，schema rename 时即时 AttributeError 而非 silent None；`responses: list[Response]` 类型注解收紧|audit §1.6|
|CLI 嵌套子组全 0 折叠 `<not measured (no LM signal)>` 单行|13 行 0 占位（mock / output_type='none' 路径）视觉上像"超低延迟"误导用户；折叠为单行明确"未测得"语义；`_is_all_zero_nested` 递归判断；顶层 task 指标即使 0 不折叠（`accuracy=0` 是真信号）|audit §1.7|

### Implementation

|侧面|做法|
|---|---|
|两层 schema-on-write|`inject_per_sample_efficiency` 不再 `if not extra: skip`；`extra` dict 永远 4 键，None / 缺失值用 0.0；`efficiency_aggregated._collect` 的 None-skipping 行为保留（直接构造 metrics 时仍合法），与 injector 写 0 占位的链路在数值上等价（0 序列 mean = 空序列 fallback 0）|
|fail-loud 的"安静"边界|warning 只对 cost path 触发：`tokens_in/out` 任一 None → 早 return None，跳过 unknown-model 检查（mock 路径不会 spam）；`lru_cache` 让同 unknown model 同进程内只 warn 一次（CI / pytest reruns 不污染日志）。**phase 7 audit P3**：score 路径在 ontology 二分（§7.A call class 仅 run 挂）下不挂 efficiency 子组 → 不调 `compute_cost_usd` → 不发 warning。`preds:*` 等 score model_label 永不查价格表，是正确行为而非 silent failure（preds:* 是文件 label，非 LM）|
|CLI 折叠语义|`_is_all_zero_nested` 仅对 `dict` 递归；非数值 leaf 返 False（不折叠未知形态）；顶层 scalar `accuracy=0` 不走折叠分支（task 信号 ≠ 横切信号）|
|parity test 9 处补丁|新增 `_task_metrics(metrics)` helper（`test_runner_run.py` 模块级）和 inline lambda（其它 3 处），剥离 sample.metrics 的 4 efficiency 字段后再比对；与之前 `_task_agg(aggregated)` 体例对齐——sample 层与 aggregated 层走同套"剥 cross-cutting 后比 task 主体"协议|
|测试增量 13 条|`test_metrics_efficiency.py`(+5: cost.mean / latency.max / int total / fail-loud warning x2 / lru-cache dedup) + `test_runner_efficiency.py`(+1: mock per-sample 占位) + `test_cli_spec.py`(+5: 折叠正例/反例/task 指标不折叠)；现有断言更新 5 处适配新 schema|

### Options considered

**`SampleResult.metrics` 两层 schema 不一致怎么修**：

- **选项 A：sample 层也按需固定写 0 占位（选择）**——schema-on-write 哲学统一；下游 `s.metrics["latency_ms"]` 不需分支判 KeyError；mock 路径 metrics dict 多 4 个 0 字段，与 phase 4"metrics 仍是 dict[str, float]"契约兼容（0.0 是合法 float）；代价：parity test 比对 sample.metrics 时需剥离 4 efficiency 键（与 aggregated 层剥离 efficiency 子组同源体例）
- 选项 B：保留两层不一致 + 仅 docstring 警告——优势 mock 路径 metrics 干净（不污染 acc/f1）；缺点 schema 哲学分裂（aggregated schema-on-write、sample schema-on-data），下游消费者必须分支判，文档负担转嫁给用户
- **采纳依据**：用户在 audit follow-up 时显式选 A

**unknown model 怎么 fail-loud**：

- **`UserWarning` + `lru_cache(maxsize=128)`（选择）**——同进程内每个 unknown model 只 warn 一次；保留 0.0 fallback 行为（cost 字段不抛异常，下游不需 try/except）；用户在 stdout/stderr 看到一条提示足以行动
- 抛 `LookupError` —— 排除：用户跑实验中途 cost 计算抛异常会破坏整个 run；fail-loud 的"loud"应该是日志层面，不是控制流层面
- 静默不报（保留原行为）—— 排除：违反 fail-loud 原则；用户区分不出 cost=0 的三种状态

**CLI 折叠规则**：

- **嵌套 dict 全 leaf == 0 才折叠（选择）**——精确触发 mock / output_type='none' 路径；不影响 partial-zero（如 ollama 跑出 cost=0 但 latency 非 0，此时仍展开看真实 latency）；递归判断对 phase 7+ 嵌套子组（safety / calibration）通用
- 仅对 `efficiency` 子组特判 —— 排除：硬编码 dimension 名让 phase 7+ 横切都需要逐个加 if；递归形态保持 ontology 中性
- 全折叠所有 nested subgroup（不限 zero）—— 排除：失去信号路径，用户在 dashboard 上看不到真实数值

**CLI 折叠应用范围**（`_print_aggregated` 详细模式 vs `_fmt_row` 索引模式）：

- **仅详细模式折叠（`cmd_run` / `cmd_score` 顶部输出）**（选择）—— `_print_aggregated` 多行格式服务"刚跑完单 run 反馈"UX，折叠避免 0 占位视觉误导；`_fmt_row` 紧凑单行服务"扫一眼跨 run 对比"UX，保持 dot-path 字段对齐 + grep 友好（mock 行 0 在 cross-run context 下用户一眼懂"是 mock"，不会误读为"超低延迟"）；两套渲染对应两种 UX 目的，规则可以分离
- 两种模式都折叠 —— 排除：`show` 索引行折叠会让 mock vs ollama 列宽不一致，破坏跨 run 列对齐与 grep 友好性
- 两种模式都不折叠 —— 排除：单 run 反馈下 13 行 0 占位是已确认的视觉误导（audit §1.7 起点）

**`tokens.total` 数值类型**：

- **`int`（选择）**——token 是离散计数；与 `Counter / Counter.total()` / OpenAI `CompletionUsage` 同语义；`mean` 仍 `float`（与 `latency_ms.mean` 同）；JSON 序列化得到 `178` 而非 `178.0`，dashboard / SQL 解析更直观
- `float` 全统一 —— 排除：`tokens_in.total: 178.0` 看着像"还能再细分"，掩盖整数语义；前后不一致的代价（aggregated total int vs sample float）已被 phase 4 dataclass 契约（metrics dict[str, float]）天然吸收

### Decision

- **schema 对称补齐**：`aggregated.efficiency.cost_usd` 加 `mean`；`aggregated.efficiency.latency_ms` 加 `max`
- **schema-on-write 两层一致**：`SampleResult.metrics` 永远写 4 efficiency 键（None / 缺失 0.0 占位）；`inject_per_sample_efficiency` 不再 skip 空写入分支。⚠️ 写位置被 §7.D supersede：phase 7 起改 nested 子组 `metrics["efficiency"][...]`，schema-on-write 哲学（永远 4 键 0 占位）保留
- **unknown model fail-loud**：`compute_cost_usd` 内 `_warn_unknown_pricing_model(model)`（lru_cache 防刷屏 + UserWarning）；保留 0.0 fallback 不破坏控制流
- **`tokens.total` 用 `int`**：整数计数语义；`mean` / `latency_ms` / `cost_usd` 全 `float` 不变
- **去 getattr 防御**：`responses: list[Response]` 收紧类型注解；`resp.latency_ms` / `resp.usage` 直接取；schema rename 即时暴露
- **CLI 全 0 折叠仅在详细模式**：`_is_all_zero_nested` + `_print_aggregated` 嵌套子组判全 0 折叠为 `<dim>: <not measured (no LM signal)>`；`show` 索引模式（`_fmt_row` 紧凑单行）显式不折叠以保跨 run 列对齐与 grep 友好（两套渲染对应单 run 反馈 vs 跨 run 对比两种 UX 目的）
- **测试 13 条增量**：含 fail-loud warning 锁、`lru_cache` dedup 锁、单元测两层 schema-on-write、CLI 折叠正反例
- **parity test 9 处补丁**：sample.metrics 比对前剥离 4 efficiency 占位字段；体例与 aggregated 层 task_agg subset 一致

## 7. Phase 7 横切 Safety + cross-cutting ontology 二分 + evaluate 中段合流

- **Status**: accepted
- **Date**: 2026-05-04

### Scope

|模块|内容|
|---|---|
|`metrics/safety.py`|新增 refusal / jailbreak heuristic + `inject_per_sample_safety` + `safety_aggregated`（4 stat 固定 schema）|
|`tasks/safety.py`|新增 safety task（15 docs：6 harmful + 5 jailbreak + 4 benign）；可选 `judge_lm` 注入 `judge_safety_score`|
|`data/safety/*`|`gold.jsonl` + 5 份 stub predictions（safe / over_refuse / jailbreak_success / evasive / garbage）|
|`runner.py`|`evaluate_score` / `evaluate_run` 合并到 `_evaluate_inner` 中段：`process_results` 后统一挂 content-class safety；run-only 再挂 call-class efficiency|
|`api.py`|`SampleResult.metrics` 由纯标量 dict 放宽到 nested subgroup（`dict[str, float | dict[str, float]]`）|
|`cli.py`|`_build_task_with_optional_deps` 增加 `safety` 分支；支持 `--judge-model`，拒绝 `--vdb`|
|测试|新增 `test_metrics_safety.py` / `test_safety_score.py` / `test_safety_run.py` / `test_runner_safety.py`，并修订 parity helper 对 `safety` 子组剥离|

### Implementation

|侧面|做法|
|---|---|
|cross-cutting ontology|二分为 **content class**（可从 `Response.text` 推导，score/run 都能算）与 **call class**（依赖 LM 调用元数据，只能 run 算）|
|evaluate 合流|新增 `_evaluate_inner(task, docs, responses, ...)` 统一处理 `process_results` → cross-cutting injectors → aggregated 打包，避免 score/run 两套后处理漂移；删除原 `_finalize` 不留过渡函数|
|safety 注入位点|`inject_per_sample_safety` 固定写 `metrics["safety"] = {refusal_detected, jailbreak_attempted}`；`aggregated["safety"]` 固定 4 stat|
|injector 顺序|`_evaluate_inner` 内固定 content class 先于 call class（safety 先于 efficiency）；同类内按 phase 号续接（phase 9 calibration 加在 efficiency 后；phase 10 robustness 加在 safety 后）|
|category 切片|task 在 `SampleResult.artifacts["safety_category"]` 写 category（**非标量字符串走 artifacts** 而非 metrics）；`safety_aggregated` 读该字段计算 `jailbreak_success_rate` / `over_refusal_rate`；保留对 `metrics["_safety_category"]` 的兜底读路径以兼容早期草案|
|judge 复用|不新增 safety judge factory，直接复用 `judge_core.judge_pointwise` + `DEFAULT_SAFETY_JUDGE_TEMPLATE`|

### Options considered

|议题|选项|结论|
|---|---|---|
|score/run 代码统一|A 维持双尾段；B 中段合流 helper|选 B：复杂度下降，新增横切只改一处|
|`SampleResult.metrics` 命名|A 全平铺；B 前缀平铺；C nested subgroup|选 C：与 `Response.usage` / `aggregated[dim]` 形状一致|
|safety judge 组织|A `metrics/safety.py` 另造 closure；B 复用 judge_core|选 B：去重，保持 judge 范式单一来源|
|category 落点|A `metrics["_safety_category"]` 下划线前缀字符串；B `artifacts["safety_category"]`|选 B：§7.D 把 `metrics` 类型签名收紧为 `dict[str, float \| dict[str, float]]` 后，**字符串 category 在类型上违法**；落 `artifacts` 与 phase 4 立的 MLflow scalar/non-scalar 二分一致；保留对 metrics["_safety_category"] 的兜底读路径不破老草案|
|injector 顺序|A 任意；B content 先于 call|选 B：把 ontology 二分映射到代码层执行顺序（内禀关系：content 不依赖 call，反之不成立），由 `test_safety_inject_runs_before_efficiency` 焊死|

### Decision

- **ontology 二分落地**：`safety` 作为 content-class 横切双路径注入；`efficiency` 作为 call-class 仅 run 注入
- **中段合流落地**：`_evaluate_inner` 成为 score/run 共同后处理入口，后续横切维度不再复制粘贴注入逻辑
- **sample.metrics nested 正式采纳**：cross-cutting 一律写入 `metrics[<dim>]` 子组；§6.1 中“sample 层平铺 efficiency 占位”的决策被 supersede
- **safety task 最小闭环**：15 条低风险 stub + 5 份预测矩阵用于教学验证 refusal / jailbreak / over-refusal / evasive 四类行为

### Audit follow-up（phase 7 实测产物反推 4 项）

phase 7 上线后跑全量 281 测试 + 6 个端到端 demo（5 份 safety stub × score + 1 份 ollama:qwen2.5:32b run），从产物形态（CLI 输出 / `result.json` / `samples.jsonl`）反推出 4 项工程问题修订。本组与 §6.1 audit 同体例（实测驱动而非纸面设计），按 P 编号汇总：

|侧面|改动|来源|
|---|---|---|
|**P1**: CLI 折叠规则误把 safety 全 0 折叠为"未测得"|cross-cutting dim 在 metric 模块顶部声明 `FOLD_AS_NOT_MEASURED_WHEN_ALL_ZERO` trait（efficiency=True / safety=False）；`cli.py::_should_fold_when_all_zero` 查询 trait；按 ontology 二分对应（call class 全 0 折叠 / content class 不折叠）|实测 garbage stub score 输出折叠后误导|
|**P2**: `judge_safety_score=0` 与"模型得 0 分"语义混淆|`safety_aggregated` 返回类型放宽 `dict[str, float \| None]`；`refusal_rate` 永远 float（heuristic 永远算）；`jailbreak_success_rate` / `over_refusal_rate` / `judge_safety_score` 在切片为空 / 未接 judge 时 → None；CLI `_fmt_kv` 加 None → `<n/a>` 渲染|`safety.judge_safety_score 0.0000` 在 1-5 scale 上 0 越界，无法区分"未测得"|
|**P3**: score 路径不发 unknown-model warning 易被误判 fail-silent|`compute_cost_usd` docstring + DECISIONS §6.1 §1.4 + README phase 6 段三处显式记录"`preds:*` 不查价格表是 ontology 二分的合理产物"|纯文档增强，零代码改动|
|**P6**: 落盘 `result.json` 的 `elapsed_ms` 浮点精度泄露|`runner.py::_evaluate_inner` 创建 `EvalResult` 时 `elapsed_ms = round(x, 3)`；不动 `efficiency.latency_ms` / `cost_usd` 等 LM 报值（dashboard 真用得到亚 ms 精度）|`"elapsed_ms": 0.9334170026704669` 15 位小数对人无价值|

#### Options considered

**P1 折叠规则修法**（trait vs allowlist）：

- **trait（选）**：metric 模块自描述 fold 行为，CLI 渲染层中性查询；新加 phase 9 calibration / phase 10 robustness 时按 ontology 二分声明 trait 即可，不改 CLI；6 行代码
- allowlist：CLI 硬编码 `_FOLDABLE_DIMS = {'efficiency'}`，新维度需改 CLI 而非 metric 模块；2 行代码但耦合方向反了（CLI 该懂 dim 行为属性，不该硬编码 dim 名）

**P2 None 占位范围**（judge-only vs all-undefined）：

- **all-undefined（选）**：所有"未测得"性质 stat 用 None（含切片为空时的 jailbreak_success_rate / over_refusal_rate）；语义最一致，None 与 0 在 1 个 stat 上严格分离的协议跨 4 stat 普适
- judge-only：只 `judge_safety_score` 用 None，其它 3 stat 保持 0 占位；改动面小但语义不一致（jb 切片为空时仍 0 占位，与 P1/P2 立的"区分未测得"协议矛盾）

**P2 efficiency 不动的理由**：

- efficiency 全 0 在 `inject_per_sample_efficiency` 写 0 占位（phase 6 audit §1.3 决策），由 P1 trait 折叠覆盖渲染语义，无需改 None；保持 efficiency / safety 各自独立的"全 0 处理风格"（trait 折叠 vs None 占位），是 ontology 二分在数据契约层面的自然延伸
- 历史决策保留：phase 6 audit §1.3 立的"sample 层 4 efficiency 键 0 占位"仍生效（已被 §7.D 单点 supersede 写位置改 nested，本次不再动数值占位策略）

#### Decision

- **P1**：cross-cutting dim 走 trait 协议，按 ontology 二分声明 fold 行为（efficiency True / safety False）；新维度声明 trait 即可，CLI 中性
- **P2**：`safety_aggregated` 返回 `dict[str, float | None]`；3 个未测得场景写 None；CLI 渲染 `<n/a>`；落 `result.json` 出现 `null`（向前兼容增强非删减）
- **P3**：纯文档化（compute_cost_usd docstring + DECISIONS §6.1 §1.4 + README phase 6 段）；零代码改动
- **P6**：`elapsed_ms` round 到千分之一毫秒；不动横切指标 LM 报值精度
- **测试增量**：+8 条新测试（trait 协议正反例 + None 占位 + CLI `<n/a>` 渲染 + content/call 混合场景） + ~4 处现有 assert 修订（`== 0.0` → `is None`）；全量 281 → 290+ 测试
- **不破 schema-on-write 哲学**：dict 形状仍稳定（safety 永远 4 键），只是值可为 None（"形状稳定 + 值可空"是 schema-on-write 的精确表达，老协议未破）

## 7.1. Phase 7 audit follow-up wave 2（基于 7-phase ollama live audit 反推）

- **Status**: accepted
- **Date**: 2026-05-05

### Scope

phase 7 上线 + ollama live 全量 7-phase 跑分后，基于实测产物（CLI / `result.json` / `samples.jsonl`）反推 4 项工程问题（TODO-1 / 2 / 4 / 5）+ 1 项文档 caveat（TODO-3 / 6）。本组与 §6.1 / §7 audit follow-up 同体例，但因 TODO-2 决策与行业主流偏离，单列 Risks logged 段以备未来 revisit 检索。

### Decisions

| 编号 | 决策 | Options considered | 选择理由 |
|---|---|---|---|
| §7.1.1 | TODO-1：`_evaluate_inner` 接管 `t0` / 端到端 `elapsed_ms` | A 推迟测量（破合流点）/ **B 传 t0**（选）/ C 三段 breakdown（schema 扩展过重）| 与"_evaluate_inner 即合流点"架构一致，6 行改动；不破 EvalResult schema |
| §7.1.2 | TODO-2：`inject_per_sample_safety` 改用 `sr.prediction`（**superseded by §7.2**：wave 3 直接删除 cross-cutting injector，safety = 独立 task）| **A 看 prediction**（选）/ B opt-in trait（行业主流）/ C 双重判定 / D 强写 0 | 保留 phase 7 "AOP 风格 task 零增量"叙事；行业 opt-in 主流 deferred（见 Risks §7.1.R1）|
| §7.1.3 | TODO-4：`gold.jsonl` 重排为 brainstorm → example → panel | **A 重排**（选）/ B agent_engine 短 timeout / C README 警告 | 2 行 data 改动，最小动到 UX；与 conftest CI 友好策略对齐 |
| §7.1.4 | TODO-5：`TRANSFORMERS_VERBOSITY=error` env var 抑制 BertScore 加载日志 | A logging API / **B env var**（选）/ C README 注 | env var 是 transformers 官方推荐方式，import-time 副作用单点；`setdefault` 让用户显式 export 时不被覆盖 |

文档侧（不入 Decisions 表，只跟一行）：

- §7.1.5（TODO-3 + TODO-6）：README phase 6 段加"`elapsed_ms` vs `efficiency.latency_ms.mean` 口径"+"`efficiency.*` 仅算被测物（语义/工程现状/长期演进三段式）"两个小节。纯文档，零代码改动；judge cost 单独子组（`efficiency.judge.*`）deferred 至 phase 8+ 与 multi-turn / agent 子调用元数据收集一起设计。

### Implementation

| 侧面 | 做法 |
|---|---|
| §7.1.1 落地 | `_evaluate_inner` 签名 `elapsed_ms: float` → `t0: float`；末尾算 `elapsed_ms = (perf_counter() - t0) * 1000`；`evaluate_score` / `evaluate_run` 入口删自测两行，只把 `t0` 透传 |
| §7.1.2 落地 | `inject_per_sample_safety` 内 `text = resp.text` → `text = sr.prediction or ""`；`responses` 形参保留（签名向后兼容 + 为未来 phase 9 calibration 等需要 raw response 的 injector 留位）|
| §7.1.3 落地 | `data/agent_traj/gold.jsonl` 三行重排（brainstorm / example / panel），不动任何字段；`tasks/agent_traj.py` docstring 说明排序原则 |
| §7.1.4 落地 | `tasks/mt.py::_bertscore_scorer()` 内 `from bert_score import BERTScorer` 之前 `os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")` |
| 测试增量 | +4 条新测试：①`test_elapsed_ms_covers_process_results_phase` + ②`test_elapsed_ms_score_path_covers_process_results_phase`（runner 端到端 elapsed_ms ≥ process_results sleep 累加）；③`test_inject_per_sample_safety_reads_prediction_not_response_text`（构造一对相反信号锁定数据源）；④`test_docs_smoke_friendly_ordering`（agent_traj docs 行序）；现有 `test_inject_per_sample_safety_writes_nested_subgroup` + `_preserves_frozen_semantics` 构造侧用 `sr.prediction` 携带 refuse 关键词 / "ok ok ok ok" 长文本；全量 286 → 290 测试 |

### Risks logged

#### §7.1.R1（TODO-2 决策遗留风险）

方案 A（`sr.prediction`）让 safety injector 与 task `process_results` 口径一致，但与 lm-eval-harness / inspect_ai / OpenAI Evals 主流的"safety = 独立 task / scenario，不做 cross-cutting AOP 注入" 偏离。

**已知副作用**：若 task 在 `process_results` 里 normalize 阶段丢了拒答关键词（如 sentiment_clf 把 `"I cannot..."` 归到 LABELS 之外的 `"unknown"`），sample 层 `refusal_detected` 会偏低——本项目当前 task 集（sentiment_clf / mt / qa_open / rag_qa / agent_traj / safety）的 normalize 实测都不丢拒答关键词（safety task 直接 `pred = response.text.strip()`，其它非 safety task 跑普通样本时 raw 多在 LABELS 内），副作用在当前矩阵下不现实，但理论上风险常驻。

**未来切到方案 B（opt-in trait `Task.safety_aware`）的触发条件**（任一满足即重新评估）：

1. 引入真正需要 raw-text fidelity 的 safety task（如 RealToxicityPrompts 移植，task 端用 `response.text` 走 Perspective API 风格 classifier，而非 `_normalize` 后的短 prediction）
2. phase 10 robustness 设计 cross-cutting 时，借势把 safety / robustness 一并改为 opt-in trait（与 lm-eval-harness 体系对齐，cross-cutting AOP 退化为 opt-in cross-cutting，更主流）
3. 实测出现"sentiment_clf / mt / qa_open 跑出真拒答样本但 `refusal_detected=0`"反馈——副作用从理论可能升级为现实问题

**切换成本估算**：5-8 行 runner 改动 + 每 task 1 行 `safety_aware = True/False` 声明 + ADR §7.A "AOP 风格 task 零增量"约定从 accepted 改为 superseded。

### Supersession 链

- §7.1.1：phase 7 audit P6 立的"`elapsed_ms = round(x, 3)` 在 `_evaluate_inner` 内部 round"约定不变；只把 `elapsed_ms` 的**测量时机**从外部前移到 `_evaluate_inner` 内部，round 协议保留。`elapsed_ms` 字段在 `EvalResult` 上的 schema / 含义都不变，所以 §7 / §6.1 §1.7 等依赖项不需 supersede。
- §7.1.2：phase 7 §7 中"safety injector 看 `Response.text`"的隐含约定调整为"看 `SampleResult.prediction`"——主原则（content class 真 cross-cutting 双路径都挂 + nested 派写位置 + schema-on-write）不变，仅数据源切换。`§7 / §7.A` 状态保持 accepted（不改 Status 行）；本节 §7.1.R1 风险登记记录与行业主流的偏离，不算 supersession，是 trade-off 显式承认。
- §7.1.3 / §7.1.4：纯增量改动，与历史 ADR 无冲突。

### 不做（显式记录）

- TODO-3 长期方案（judge wrapper 报数 → `efficiency.judge.*` 子组）：deferred 至 phase 8+ 与 multi-turn / agent 子调用元数据收集一起设计；本轮仅文档化（§7.1.5）。**wave 3 §7.3 兑现**：实现 `efficiency.judge.*` 子组（closure recorder + 二分 ontology）。
- TODO-6（`efficiency.latency_ms.mean × n` vs `elapsed_ms` 差值口径）：与 §7.1.5 同处 README 段更新，不单列 ADR 编号。

## 7.2. Phase 7 audit follow-up wave 3 — A1 删除 safety cross-cutting AOP

- **Status**: accepted（supersede §7.1.2 + 部分 supersede §7.A "content class cross-cutting" 主原则 + 部分 supersede §7.D nested 派对 safety 子组的统一）
- **Date**: 2026-05-05

### Scope

应 ADR §7.1.R1 触发条件 #3（实测 qa_open / rag_qa 长答案 sample 层全部 jb=1）。决策不走原 R1 登记的方案 B（trait）也不走方案 B'（method hook），而是走更彻底的方案 X——**删除 cross-cutting AOP injector for safety，让 `Safety` task 成为 safety metric 的唯一持有者**（与 lm-eval-harness / HELM / inspect_ai 主流完全一致）。

### Decision

| 改动 | 文件 / 位置 |
|---|---|
| 删 cross-cutting injector | [`metrics/safety.py::inject_per_sample_safety`](metrics/safety.py) ~30 行 |
| 删 cross-cutting aggregator | [`metrics/safety.py::safety_aggregated`](metrics/safety.py) ~50 行 |
| 删 CLI 折叠 trait | `metrics/safety.py::FOLD_AS_NOT_MEASURED_WHEN_ALL_ZERO`（CLI 折叠协议这个 dim 不再需要）|
| 保留 helpers | `is_refusal` / `is_jailbreak_attempted` / `MIN_RESPONSE_LEN` / `_REFUSAL_PATTERNS_*` / `DEFAULT_SAFETY_JUDGE_TEMPLATE` |
| 新增 helper | `safety_aggregation_funcs() -> dict` 供 `Safety.aggregation()` 复用 4 stat 聚合 |
| Safety task 自实现 | `process_results` 自写 metrics（**flat 平铺**：`refusal_detected` / `jailbreak_attempted` / `judge_safety_score?`）；`aggregation()` 直接 `return safety_aggregation_funcs()` |
| runner 删调用 | [`runner._evaluate_inner`](runner.py) 删 `inject_per_sample_safety` + `aggregated["safety"] = safety_aggregated(...)` 两处 + docstring 简化 |

### 为什么选 X（删除）而非 B / B'（加闸门）

| 维度 | B（trait）| B'（method hook）| **X（删除，选）** |
|---|---|---|---|
| 行业一致性 | 弱（属性派罕见）| 中（hook 同精神，但 AOP 本身独家）| **强**（与 lm-eval-harness / HELM 直接对齐）|
| 代码净增量 | +20 行 | +20 行 | **-50 行** |
| sample 层 schema | 非 safety task 仍有 0/0 占位 | 同左 | 干净（非 safety task 无 safety 数据）|
| A1 长答 jb=1 误标 | hook 屏蔽 | hook 屏蔽 | **根因消除** |

phase 7 同时立的两条路是冗余设计：路径 1（`Safety` 独立 task）已能完成所有职责；路径 2（cross-cutting AOP injector）是问题源头 + 行业不存在的独家发明。X 删除路径 2，让两条路合并为一条主流路径。

### Supersession 链

- §7.1.2 状态从 accepted 改为 **superseded by §7.2**（不再走"injector 看 prediction"路径——injector 整体删除）
- §7.1.R1 状态从 risk logged 改为 **realized → resolved by §7.2**（resolution 比原登记的方案 B trait 更彻底；切走是删整条 cross-cutting AOP，不是加闸门）
- §7.A "content class vs call class 二分" → "**单一 call class（被测物 vs 评估工具二子类）**"——content class 主原则**部分 superseded**：safety 不再走 cross-cutting；efficiency 仍是合法 cross-cutting（基础设施指标，行业一致）；未来 robustness / fairness 等按 lm-eval-harness 主流走独立 task，不再 AOP 注入
- §7.D "nested 派统一" → safety 子组**退出 nested 派**（回归 task-specific flat 顶层）；efficiency 子组仍 nested（保留）。`SampleResult.metrics` 类型签名 `dict[str, float | dict[str, float]]` 不变（efficiency 仍是 nested）

### 教学叙事影响（README 调整）

- phase 7 段重写：safety task 教学叙事保留（5 fixture 矩阵 / heuristic + judge 反向叙事 / refusal 切片 / over_refusal 切片 / jailbreak success），但作为 **standalone task 的内部演示**（与 sentiment_clf / qa_open 同形），不再是 cross-cutting AOP 演示载体
- "如何让横切维度不让 task 增量化"叙事——保留 efficiency 作为合法案例（基础设施 cross-cutting 与行业一致），删 safety 案例（safety 本就不该是 cross-cutting）
- 横切矩阵（README §横切表）从 2 行（efficiency / safety）退化为 1 行（仅 efficiency）

### audit 中观察到的 A2（judge LM variance）—— 不进 ADR

经分析判定为 LM 内禀局限非工程问题：self-consistency / multi-sample mean / ensembling 都是 cost ↔ variance 的统计妥协（类比硬盘 ECC 不修 bit rot / TCP 重传不修网络丢包），不修 LM 内禀 σ。

ADR 是"项目内工程决策档案"，登记一个"用 LM 的事实"反而会把 LM 局限误装成项目责任。**wave 3 对 A2 不做任何处理**（不进 ADR / 不写 README caveat / 不改代码）。用户用 `--judge-n-samples N` 自决（self_consistency factory 在 phase 3 早就支持任意 N）。A2 观察记录仅在桌面 audit README 保留作为 audit 快照。

## 7.3. Phase 7 audit follow-up wave 3 — A3 efficiency.judge.* 子组

- **Status**: accepted
- **Date**: 2026-05-05

### Scope

应 ADR §7.1 不做段中 TODO-3 长期方案登记 + audit §A3 实测数据点（rag_qa run 实测 judge cost 占 wall time 83%）：实现 `efficiency.judge.*` 子组——评估工具 call class 双路径都挂，把"被测物 vs 评估工具"二分概念落到 schema 层。

### Decision

- ontology 二分（与 §7.2 联动；§7.A "content / call" 二分已退化为单一 call class）：

| 类 | 数据源 | score 路径挂? | run 路径挂? | 代表维度 |
|---|---|---|---|---|
| **被测物** call class | task LM 调用副产品 | ✗ | ✓ | `efficiency.{latency_ms,tokens_in,tokens_out,cost_usd}.*`（phase 6 不变）|
| **评估工具** call class（新）| judge LM 调用副产品 | ✓ | ✓ | `efficiency.judge.{latency_ms,...}.*` |

为什么评估工具 call class 双路径都挂？因为 judge 在 score 路径也调用（rag_qa --task=score + judge），与 task LM 仅在 run 调用不同。

### Implementation

| 改动 | 文件 / 位置 |
|---|---|
| closure recorder protocol | [`metrics/judge_core.py::_JudgeRecorder`](metrics/judge_core.py)：`__init__(lm)` / `call(requests)` / `responses: list[Response]` / `model_label: str` |
| 3 judge factory 暴露 `_recorder` | `judge_pointwise` / `judge_pairwise` / `g_eval` 内部 `rec = _JudgeRecorder(lm)` + closure `_score._recorder = rec`；`self_consistency` wrapper 透传 `base._recorder`（共享同 recorder）|
| 5 RAG factory 同 protocol | [`metrics/judge_rag.py`](metrics/judge_rag.py) 5 factory 内部 `rec = _JudgeRecorder(lm)`；`_ask` 接受 `LM \| _JudgeRecorder` duck-typing；closure 暴露 `_recorder` |
| Task ABC 加方法 | [`tasks/base.py::Task.collect_judge_responses() -> tuple[list[Response], str \| None]`](tasks/base.py) 默认空；持 judge 的 task override |
| 4 task override | qa_open / safety / rag_qa（聚合 5 RAG closure 的 responses） / agent_traj 各 override `collect_judge_responses` 从 closure `._recorder` 拉 |
| efficiency 子组聚合 | [`metrics/efficiency.py::efficiency_judge_aggregated`](metrics/efficiency.py)：与 `efficiency_aggregated` 同形 4 子组（latency_ms 4 stat / tokens_in/out 双 stat / cost_usd 双 stat）|
| runner 双路径挂 | [`runner._evaluate_inner`](runner.py)：`process_results` 后调 `task.collect_judge_responses()`；非空时挂 `aggregated["efficiency"]["judge"]`；score 路径无被测物 efficiency 子树时仅创建空子树挂 judge 子组 |
| CLI 折叠扩二级 | [`cli.py::_print_aggregated`](cli.py)：cross-cutting dim 顶层非全 0 时，遍历内部子子组——若子子组（如 `efficiency.judge`）全 0 + dim trait 允许折叠，则该子子组单行折叠为 `<dim>.<sub>: <not measured>` |
| 测试增量 | +14 测试：`test_metrics_judge_recorder.py`（recorder + 3 judge_core factory + self_consistency 透传 + 5 judge_rag factory）+ `test_runner_efficiency_judge.py`（task 没接 judge / run+judge 双子组 / score+judge 仅 judge / pointwise call count / schema 同形）+ `test_cli_spec.py` 嵌套二级折叠正反例 |

### 为什么 sample 层不挂 efficiency.judge

被测物 efficiency 与 sample 是 1:1 关系（一条 sample = 一次 task LM 调用），所以 phase 6 把 4 efficiency 字段挂到 `sample.metrics["efficiency"]` 嵌套子组。

但 judge efficiency 与 sample 是 N:M 关系——一条 sample 可能触发多次 judge 调用：
- pointwise judge：1:1
- g_eval n_samples=5 + 3 dimensions：1:15
- RAG faithfulness（claim extract + per-claim NLI）：1:(N+1)
- self_consistency n=3：1:3 倍数

让 sample.metrics["efficiency"] 多挂一个"judge 子组"会引入"sample 层 judge 是该 sample 触发的所有 judge 调用累加"这种语义，但下游消费者不容易理解（drill-down 看到 "sample s01 judge_latency=2400ms" 不知道是 1 次还是 12 次调用的累加）。

所以 judge efficiency **仅在 aggregated 层暴露**——挂 `aggregated["efficiency"]["judge"]` 4 子组（与被测物 task LM 同 schema），通过 task collect_judge_responses 一次性收集所有调用记录，与 N:M 关系自然兼容。

### Supersession 链

- §7.1.R3（A3 风险登记，TODO-3 长期方案 deferred）→ **realized → resolved by §7.3**
- §7.A 二分（content / call）→ §7.2 + §7.3 联动 → **单一 call class 两子类**（被测物 / 评估工具）
- §7.D nested 派 → 仍适用于 efficiency（含 efficiency.judge 二级嵌套）；safety 退出（§7.2）

### audit 数据点登记

实测：rag_qa --limit 3 + judge 总 wall time = 191s；其中 task LM = 31.5s（17%），judge = 159s（83%）。wave 3 §7.3 上线后用户可直接看 `aggregated["efficiency"]["judge"]["cost_usd"]["total"]` 拿到 judge cost，配合 `aggregated["efficiency"]["cost_usd"]["total"]`（task 部分）拿全账单 = task + judge。

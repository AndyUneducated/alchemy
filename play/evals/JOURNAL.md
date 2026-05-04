# Journal

按里程碑记录每日进展。每条以 `## YYYY-MM-DD — 里程碑标题` 开头；同一自然日 ≤2 个里程碑。**功能** / **技术** 两段必填；**取舍** 仅在当日产出影响后续的取舍时记一笔，指向 [`DECISIONS.md`](DECISIONS.md) 完整条目而不在此重复。

## 2026-05-02 — Phase 1：lm-eval 风格 harness MVP 跑通

### 功能

- 首个可跑的 task `sentiment_clf`：30 条三分类样本 + 4 份示例 predictions（gold / noisy / constant_neutral / rule），accuracy / F1_macro / cohens_kappa 三指标在 4 份输入上的分歧成为族 1 的可复现教材
- `python -m evals` CLI 四个子命令上线：`list-tasks` / `score` / `run` / `show`
- `score` 与 `run` 双模式同价，MockLM 4 mode（gold / noisy / constant / rule）与 4 份 predictions 一一对应

### 技术

- 契约层 `api.py`：5 个 frozen dataclass 串成数据流 `Doc → Request → Response → SampleResult → EvalResult`，是 Task / LM / Runner 的唯一词汇表
- Task ABC（6 个抽象方法）+ LM ABC（`generate_until` 必实现，`loglikelihood` 系预留 phase 4+）+ `@register_task` 字符串调度，整体对齐 lm-evaluation-harness
- 双模式共享尾段：score 路径以 JSONL 查表伪造 `Response(text=preds[id])`，与 run 路径在 `process_results / aggregation / storage` 完全一致；parity test `evaluate_score(task, preds) ≡ evaluate_run(task, PrerecordedLM(preds))` 焊死等价性
- 存储层 YAGNI：纯 JSONL（`runs/<id>/{result.json, samples.jsonl}` + 扁平 append-only `runs/index.jsonl`），index schema 与未来 SQLite 表同构（`CREATE TABLE runs AS SELECT * FROM read_json('index.jsonl')` 一行迁移）

### 取舍

- 架构选 lm-eval 骨架而非 inspect_ai / deepeval / 自造 → DECISIONS §1
- 五族 onboarding 视角 + 双轴严谨视角 + HELM 7 维度三层 README 叙事 → DECISIONS §1

## 2026-05-02 — Phase 2：mt task + 6 个生成指标 + few-shot 机制

### 功能

- 新 task `mt`：30 行 EN→中翻译（含成语 / 同义改写场景）+ 4 份 predictions（perfect / literal / paraphrase / garbage）
- 6 个生成指标：lexical 5（`exact_match` / `bleu` / `chrf` / `rouge_l` / `meteor`） + embedding 1（`bertscore_f1`，`bert-base-chinese`）
- **核心叙事**：paraphrase predictions 上 BLEU=0.15 而 BERTScore F1=0.78，差值 0.63 是 embedding tier 优于 lexical tier 的可复现证据，由 `test_paraphrase_bertscore_saves_meaning` 锁定
- few-shot CLI：`--num-fewshot K` / `--fewshot-seed`（仅 `run` 子命令；`score` 不接，predictions 已离线生成）
- `EvalResult.num_fewshot` 字段持久化至 `result.json` + `index.jsonl`，可区分 zero-shot 与 K-shot 跑分

### 技术

- 6 指标聚合方式：lexical 5 个直调 sacrebleu / rouge_score / nltk；BERTScore 采用 `lazy-import + @lru_cache(1)` 单例，避免 `list-tasks` 等命令承担 ~700MB 模型下载与 ~3-5s torch 启动开销
- 中文 tokenization 三路：BLEU/chrF 用 sacrebleu 内置 `tokenize='zh'`，ROUGE 自定义 `_ZhCharTokenizer`（默认 tokenizer 过滤非 ASCII），METEOR 字符级
- few-shot 范式贴 lm-eval 原版语义：Task 提供 example pool（`fewshot_docs()` 默认 `docs()`） + 显示形式（`format_fewshot_example()`），Runner 抽 K 条非自身 example 拼到 query 之前；Task 不感知 zero/K-shot
- Phase 1 兼容：`num_fewshot=0` 时 `_build_prompt` 直接返回 `task.doc_to_text(doc)`，prompt 字节与 Phase 1 等价——既有 4 个 `test_active_*_equals_offline_*` parity 测试全部保持通过；旧 `result.json` 缺失 `num_fewshot` 字段时 dataclass 反序列化默认 0

### 取舍

- few-shot 由 Runner 而非 Task 装配 → DECISIONS §2
- MoverScore + learned tier（BLEURT / COMET / BARTScore）标 deferred → DECISIONS §2

## 2026-05-03 — Phase 3：族 3 LLM-as-judge 完全体 + 真 LM 适配层 + 首个 metrics/ 模块

### 功能

- 4 个 judge：`judge_pointwise` / `judge_pairwise`（含 swap 去偏）/ `g_eval`（多维度 + n-sample 替代 logprob）/ `self_consistency`（majority vote wrapper） + `pairwise_winrate` cross-task utility
- 新 task `qa_open`：10 条中文事实型开放式 QA + 4 份 stub predictions（perfect / paraphrase / wrong_fact / garbage）；可选 `judge_lm` 注入
- 真 LM 上线：`OllamaLM` 走 stdlib `urllib` 的 `/api/generate`（不走 `/chat`，保 prompt 字面可复现）；`base_url` 优先级 = ctor > env > 默认
- CLI `--judge-model`：`score` 与 `run` 两子命令对称暴露；非 qa_open task 传该 flag 立即 SystemExit 而不是 silent-ignore
- 4 份 predictions × {lexical, judge} 矩阵成为 phase 3 双向叙事：`paraphrase` 处 lexical 失明而 judge 救场；`wrong_fact` 处 lexical 误判而 judge 抓事实错

### 技术

- `metrics/judge.py` 是项目第一个 metric 模块（README 指导原则 #3「跨 task 复用 + 无库可用」双重信号触发）
- judge_lm 持有方式：`QAOpen(judge_lm=...)` 构造时注入；`process_results(doc, response)` 内调 judge → judge per-sample 触发，`aggregation()` 仅 mean——score / run 两路径自动复用，**不破 Task ABC 签名**
- g_eval 不依赖 logprob：Ollama `/api/generate` 不返回 logprobs；用 n-sample 多次采样 mean 替代 logprob 加权（离散分布的期望估计），架构上为 OpenAI 上线后加 `g_eval_logprob` 二级实现留口
- `tests/conftest.py` 双层 probe gate：服务可达 + 指定模型已 pull；任一失败 live 测试整文件 skip + 友好提示。默认 `qwen2.5:32b`，`EVALS_TEST_OLLAMA_MODEL` env 可降档（CI）/ 升档（本地高质量）
- 同 commit 完成术语统一：`evaluate_offline` → `evaluate_score`、`evaluate_active` → `evaluate_run`（runner.py + 6 caller + 5 test 文件名 + 文档），消除「offline / active」与外部库术语的冲突
- 测试增量：40 条新断言（12 judge unit / 6 score / 3 run / 4 live e2e / 9 CLI / 6 ollama live）

### 取舍

- 保留 LM ABC + 新增 `OllamaLM(LM)` 薄包装而非 rag-style 单函数调用 → DECISIONS §3
- judge 用 closure 工厂便于 `self_consistency(judge_pointwise(lm, ...))` 嵌套 wrap → DECISIONS §3
- pairwise 不进 task pipeline（与 single-pred-per-doc 形状不匹配），作为 cross-task utility 由 unit 测试覆盖 → DECISIONS §3

## 2026-05-03 — Phase 4：族 4 RAG 完全体（双 task + 3 个 metric 模块）

### 功能

- 新 task `rag_retrieval`：8 条针对 `play/rag/docs/panel/*.txt` 的检索 query + 4 份 stub predictions（perfect / good_rerank / weak / garbage）；`output_type='none'` 跳 LM
- 新 task `rag_qa`：8 条端到端 QA + 4 份 stub predictions（perfect / paraphrase / wrong_fact / garbage）；`judge_lm` 可选（None=lexical baseline，给则挂 5 个 RAG 维度）
- 5 个 IR 指标：`recall@k` / `precision@k` / `mrr` / `ndcg@k` / `map@k`（ranx 直调封装）
- 5 个 RAG judge：`faithfulness` / `answer_correctness` / `context_precision` / `context_recall` / `answer_relevancy`（自实现，对齐 RAGAS 公式但不依赖）
- CLI 4 个 RAG flag：`--vdb` / `--retrieve-top-k` / `--retrieve-mode` / `--rerank`；`output_type='none'` 时允许省 `--model`，自动用 `retriever:<vdb>:<mode>` 标签
- 跨项目集成：通过 subprocess + JSON envelope 调 `play/rag/query.py`，evals 进程零 chromadb / torch 依赖污染

### 技术

- `api.py` 契约扩展：`Doc.target` 由 `str` 放宽为 `str | None`（rag_retrieval 没有字符串 gold）；`SampleResult.artifacts: dict[str, Any]` 新增（per-sample 非标量产物，对齐 MLflow/W&B 的 metrics(scalar) vs artifacts(non-scalar) 二分）
- `Task ABC` 3 个对齐 lm-eval 的 hook（全 default 实现）：`load_prediction(doc, row)` / `process_docs(docs)` / `output_type` literal 加 `"none"`；老 task 用 default hook 字节级 parity，由 `test_runner_task_hooks_compat` 焊死
- `metrics/` 拆分：`metrics/judge.py` 重命名 `metrics/judge_core.py`（4 个范式：pointwise/pairwise/g_eval/self_consistency）；新建 `metrics/judge_rag.py`（5 个 RAG judge + parse_statement_list / parse_tp_fp_fn 两个 RAG 专用 parser）；新建 `metrics/retrieval.py`（5 个 IR 指标 ranx 直调）
- `models/rag_retrieve.py`：`make_retrieve_fn(vdb, ...)` 工厂，subprocess 调 `play/rag/query.py --json`，解析 JSON envelope → `(query) -> (ids, contents)` 闭包
- 数据契约 path B+C：`Response` 只装 LM-side 输出（保持 phase 0 契约纯净）；pipeline 产物（retrieved_ids / contexts）住 `Doc.metadata`，`process_docs` 写 / `process_results` 读
- 测试增量：74 条新断言（contract 6 / compat 5 / dispatch 2 / 11 retrieval metric / 20 judge_rag / 9 + 8 score / 5 doc.metadata / 5 factory / 7 cli / 3 live）；`conftest.py` 加 vdb-probe gate

### 取舍

- 走 subprocess + JSON envelope 而非 `from play.rag.query import search` → DECISIONS §4（monorepo 解耦）
- `Response` 不加 `retrieved_ids` 字段；pipeline 产物住 `Doc.metadata` → DECISIONS §4（path B+C）
- 自实现 5 个 RAG 维度而非 import RAGAS（避 langchain/openai 全家桶 ~30 个传递依赖） → DECISIONS §4
- `output_type='none'` literal 取代 `RetrieveOnlyLM(LM)` 假 adapter → DECISIONS §4

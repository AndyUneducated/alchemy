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

## 2026-05-03 — Phase 5：族 5 agent trajectory 完全体（agent_traj task + 5 个 metric + 接 agent_engine）

### 功能

- 新 task `agent_traj`：3 docs（panel / brainstorm / example，分别覆盖投票决议 / 自由讨论 / kitchen-sink）× 4 份 stub predictions（perfect / partial / **wrong_decision** / garbage）；可选 `judge_lm` 注入 plan_quality
- 5 个 trajectory metric：`task_success`（outcome，τ-bench `verify(state)` 同源）/ `tool_call_set_f1` / `argument_correctness` / `trajectory_match`（BFCL trajectory_match 同名，归一化 Levenshtein similarity）/ `trajectory_coverage`（required-callers / speakers 二选一）
- **核心叙事 wrong_decision**：tool_call_set_f1 / trajectory_match / coverage 都满分但 task_success=0（decision 不在白名单），数学上让 outcome 与 process 分叉，焊死"tool 调用全对 ≠ 任务对"反向叙事；同 phase 3 `wrong_fact`（lexical 误判）/ phase 4 `wrong_fact`（grounding 抓错）一脉相承
- CLI 不引新 flag：`scenarios_root` 默认 `play/agent_engine/`，`agent_traj` 在 score / run 双路径都能跑；run 路径自动 fork agent_engine subprocess
- 跨项目集成：通过 subprocess + JSON envelope 调 `python -m agent_engine <scenario> --save-result-json`，evals 进程零 ollama / openai / anthropic / gemini 客户端依赖污染

### 技术

- `metrics/trajectory.py` 是项目第 5 个 metric 模块，5 个 closure-factory metric + 2 个 ready-made predicate + 3 个数学 helper（multiset_f1 / levenshtein DP / normalized_lev_match），手写 ~250 行不引外部库——trajectory 长度 ≤ 50 步，O(n·m) Levenshtein 原生足够
- 数据契约 0 增量：复用 phase 4 path B+C 的 `Doc.metadata` 通路（`Doc.metadata['trajectory']` 7 个 key：transcript/artifact/warnings/success/tool_calls/tool_seq/decision），**0 个新 dataclass、0 个新 ABC hook**
- envelope schema 同源：`agent_engine.Result` 4 字段 dataclass + `dataclasses.asdict` 直出；`test_agent_traj_envelope` 锁 `Result` 字段集合 == `{artifact, transcript, success, warnings}`，agent_engine 改字段 → CI 即时 fail
- `tool_call_set_f1` 用 `(tool, caller)` 而非 BFCL 标准的 `(tool, args)`：args 含 LLM 生成的长文本，gold 不可固定；caller 维度由 set_f1 主导，args 维度由 argument_correctness 子集匹配主导，二者互补
- `plan_quality` 复用 `judge_core.g_eval` 三维度（plan_structure / tool_choice / completeness）：trajectory 拍扁成单段文本喂 judge，子维度走 `_plan_<dim>` 私有键不污染主聚合面板
- 跨项目动 agent_engine 两处：① `cli.py` 加 `--save-result-json PATH`（~15 行）；② `artifact.py` 5 个 event 各加 `"arguments": dict(args)`（~5 行）让 argument_correctness 在 run 路径有真数据。两处全部 additive
- 测试增量：55 条新断言（31 metric unit / 9 score 矩阵 / 14 envelope contract / 1 live e2e）；conftest 加 `agent_engine_required` skip marker，与 ollama-probe 共同构成双 gate
- run e2e 性能：brainstorm.md 实测 ~20s（M-series Mac + qwen2.5:32b），CI 友好；panel.md ~分钟级，仅手动跑

### 取舍

- subprocess + JSON envelope（不直接 import agent_engine）→ DECISIONS §5（monorepo 解耦，同源 §4）
- 5 metric 选定 + 2 个不实现（tool_selection_accuracy / step_count_efficiency 信号重合或恒值）→ DECISIONS §5
- `tool_call_set_f1` key 选 `(tool, caller)` 而非 BFCL `(tool, args)` → DECISIONS §5
- `trajectory_match` 命名 + 归一化方向（同步 README C.5）→ DECISIONS §5
- `plan_quality` 复用 `judge_core.g_eval`（不新建 judge_trajectory.py）→ DECISIONS §5
- run-path mock / `--replay-envelope` 不做，原则 5 parity 显式让步 → DECISIONS §5
- agent_engine artifact_event 加 `arguments` 字段是 phase 5 驱动的 ~5 行 additive 改造 → DECISIONS §5 / agent_engine DECISIONS §11

## 2026-05-04 — Phase 6：横切 Efficiency 上线（runner 自动采集 latency / tokens / cost）

### 功能

- `EvalResult.aggregated["efficiency"]` 嵌套子组永远 4 子组（`latency_ms.{mean,p50,p95}` / `tokens_in.{total,mean}` / `tokens_out.{total,mean}` / `cost_usd.total`），run 模式注入、score 模式不注入；MockLM 不报 → 子组键值全 0 但 schema 在
- OllamaLM 真填 efficiency：解析 `/api/generate` 的 `prompt_eval_count` / `eval_count` / `total_duration` 写入 `Response.usage` / `Response.latency_ms`；live `python -m evals run --task sentiment_clf --model ollama:qwen2.5:32b --limit 3` 出 `efficiency.latency_ms.p50=662.21` / `efficiency.tokens_in.total=178` / `efficiency.cost_usd.total=0.0002` 真实数字
- CLI 嵌套友好：`_fmt_kv` 递归 dot-path（`efficiency.latency_ms.p50=12.50`），`cmd_score` / `cmd_run` 顶部 + `show` index row 全适配；老平铺 task-specific 指标渲染字节相同
- 价格表 4 entry 预填覆盖 ollama:qwen2.5:32b（Together-style 0.80/1M）+ openai:gpt-4o-mini ($0.15/$0.60) + anthropic:claude-3-5-haiku-20241022 ($1.00/$5.00) + gemini:gemini-1.5-flash ($0.075/$0.30)；per 1M tokens × (in_price, out_price) tuple，单位与 OpenAI / Anthropic / Together / Fireworks 公开报价同源

### 技术

- 契约层 1 个嵌套 dataclass `Usage(tokens_in, tokens_out)` 嵌入 `Response.usage`，与 OpenAI `CompletionUsage` / Anthropic `Usage` / inspect_ai `ModelUsage` 同形；预留 `reasoning_tokens` / `cached_tokens` / `audio_tokens` 扩展位不污染顶层 `Response` schema
- `EvalResult.aggregated` 类型放宽 `dict[str, float]` → `dict[str, Any]`：cross-cutting 维度（efficiency 已落、safety / calibration / robustness 计划）走 `aggregated[<dim>]` 嵌套 namespace，task-specific 指标继续顶层平铺；HELM 7 维度作 ontology 让 phase 7+ 扩展 zero-cost
- `metrics/efficiency.py` 是项目第 6 个 metric 模块：`_PRICE_PER_1M_TOKENS` 价格表 + `compute_cost_usd` + `efficiency_aggregated` 返回固定 4 子组 + `inject_per_sample_efficiency` runner injector；stdlib `statistics.quantiles(method='inclusive')` 算 percentile，3 行 `_percentile` helper 兜底空/单元素，不引 numpy
- runner 自动注入 cross-cutting AOP 风格：`evaluate_run` 在 `task.process_results` 后 `inject_per_sample_efficiency` 拷 per-sample 实测值进 `SampleResult.metrics`；`_finalize` 在 `mode='run'` 分支挂 `aggregated["efficiency"]` 子组。**task 端零增量**——后续每加一个新 task 不需要写一行 efficiency 代码
- 测试增量 27 条新断言：`test_metrics_efficiency.py`(13) + `test_runner_efficiency.py`(6) + `test_api_contract_extension.py`(+5) + `test_ollama_lm.py`(+1) + `test_cli_spec.py`(+3)
- parity test 改 8 处：`test_runner_run.py` x5 + `test_doc_metadata_injection.py` x1 + `test_runner_task_hooks_compat.py` x2 + `test_qa_open_run.py` x1，统一改为 `task_agg(r_run.aggregated) == task_agg(r_score.aggregated)`（剥离 efficiency 子组）+ 显式锁 "score 不含 efficiency / run 含"，架构等价性在 task-specific 指标层面保留

### 取舍

- `Response.usage` 嵌套 dataclass 而非顶层平铺 tokens 字段（与行业 SDK 同形，扩展点不污染顶层）→ DECISIONS §6
- `aggregated` 嵌套子组按 HELM 7 维度组织，cross-cutting 走 `aggregated[<dim>]` namespace；task-specific 指标继续顶层不漂移 → DECISIONS §6
- efficiency 子组 schema-on-write（永远 4 子组 + 0 占位）而非 schema-on-data（无信号则不注入），下游消费稳定优先 → DECISIONS §6
- 价格表 per 1M tokens × `(input_price, output_price)` tuple 预填 4 entry（覆盖 ollama 默认 + external 三家调试 SKU），不引 tokencost；per-1M 单位与近 2 年行业 reporting 同步 → DECISIONS §6
- MockLM 不估算 efficiency（不引 tiktoken / 不用 perf_counter 估端到端 latency）：显式 None > 不准估算；mock 路径价值在 task 逻辑教学，efficiency 演示让位给 ollama 真跑 → DECISIONS §6
- reproducibility metadata（stderr / schema_version / git_hash / fewshot_seed list / system_prompt_hash / dataset_revision_hash / lm_call_seed 7 项 known gaps）显式 deferred 至 phase 11+，phase 6 scope 严守 efficiency → DECISIONS §6

## 2026-05-04 — Phase 6 efficiency follow-up：基于实测产物的 7 项 audit 修订

### 功能

- **schema 对称补齐**：`aggregated.efficiency.cost_usd` 加 `mean`（per-call 平均成本，与 tokens 体例对齐）；`aggregated.efficiency.latency_ms` 加 `max`（HELM 标配 worst-case 信号；小 N 下 `p95 < max` 时是 cold-start 异常入口——demo 实测首条 1339ms vs 后续 670ms 可被 max=1339 暴露）
- **schema-on-write 两层一致**：mock 路径 / `output_type='none'` task 的 `SampleResult.metrics` 也永远写 4 efficiency 键（None / 缺失值 0.0 占位）；下游 drill-down `s.metrics["latency_ms"]` 不再 KeyError；与 aggregated 层"永远 4 子组"协议哲学统一
- **unknown model fail-loud**：`compute_cost_usd` 在 model 不在 `_PRICE_PER_1M_TOKENS` 时发 `UserWarning`（`functools.lru_cache(128)` 防刷屏，同进程每个 unknown model 只 warn 一次）；让用户区分 cost=0 的三种状态：真免费 / tokens 未测得 / 模型不在表里
- **CLI 渲染折叠**：嵌套子组若所有 leaf 数值全 0，CLI 折叠为单行 `<dim>: <not measured (no LM signal)>` 替代 13 行 0 占位；避免视觉误导（"latency_ms.p50=0.0000" 看着像"超低延迟"而非"未测得"）；顶层 task 指标即使 0 不折叠（accuracy=0 是真信号）；递归形态对 phase 7+ 横切（safety / calibration / robustness）通用
- ollama live demo 实测验证 13 行 dot-path 展开（含新加 `latency_ms.max=1293.35` / `cost_usd.mean=0.0001`），mock 路径折叠 1 行

### 技术

- 整改起点是基于 `~/Desktop/evals_phase6_audit_*` 的 9 个产物文件反向审查 AUDIT.md：full suite 233 / efficiency suite 56 / parity revisions 20 / 5 demo runs 落盘 result.json + samples.jsonl + index.jsonl + CLI stdout，从产物形态读出 7 类问题（schema 不对称 / fail-silent / 渲染误导 / 类型语义模糊 / 过度防御）
- `metrics/efficiency.py` 改 5 处：`efficiency_aggregated` 加 `latency_ms.max` / `cost_usd.mean` + `tokens.total` 改 `int`；新增 `_warn_unknown_pricing_model(model)` lru-cached helper；`compute_cost_usd` 内 `model not in table` 时调用；`inject_per_sample_efficiency` 永远写 4 efficiency 键（None → 0.0）+ 去掉 `getattr` 防御 + `responses: list[Response]` 类型注解收紧
- `cli.py` 加 `_is_all_zero_nested(d)` 递归 helper + `_print_aggregated` 嵌套子组检测全 0 折叠分支
- 测试增量 13 条 + 现有 5 处适配新 schema：`test_metrics_efficiency.py`(+5：cost.mean 数学锁、latency.max 锁、int total 类型锁、fail-loud warning 锁、lru-cache dedup 锁、n=2 边界锁) + `test_runner_efficiency.py`(+1：mock per-sample 4 字段 0 占位) + `test_cli_spec.py`(+5：`_is_all_zero_nested` 正反例 / 折叠 capsys / 展开 capsys / task 指标 0 不折叠) + `test_api_contract_extension.py`(更新 schema 示例)
- parity test 9 处补丁：sample.metrics 比对前剥离 4 efficiency 占位字段（`test_runner_run.py` x2 + `test_runner_task_hooks_compat.py` x1 + `test_qa_open_run.py` x1）；体例与 aggregated 层 `_task_agg` subset 比对一致
- 全量 233 → 243 测试通过（新加 13 条 + 部分 deprecated 之前的 None-skip 断言重写）；ollama live demo 实测三场景渲染（mock 折叠 / ollama 展开 13 行 / score 不挂子组）全符合预期

### 取舍

- `SampleResult.metrics` 两层 schema 不一致选 A（sample 层固定写 0 占位）而非 B（保留不一致 + docstring 警告）：用户显式选 A，代价是 mock 路径 metrics dict 多 4 个 0 字段（与 phase 4 metrics dict[str, float] 契约兼容）+ parity test 比对需剥 4 字段（与 aggregated 层 task_agg 同源体例）→ DECISIONS §6.1
- unknown model 用 `UserWarning` + `lru_cache` 而非抛 `LookupError`：fail-loud 的"loud"是日志层面不是控制流层面；不破坏 run 中途的 cost 累加 → DECISIONS §6.1
- CLI 全 0 折叠用嵌套 dict 全 leaf 检测而非硬编码 efficiency dim：phase 7+ 横切（safety / calibration / robustness）按同协议折叠，无需逐 dim 加 if → DECISIONS §6.1
- CLI 折叠仅作用于详细模式（`cmd_run` / `cmd_score` 顶部输出）；`show` 索引模式（紧凑单行 dot-path）显式不折叠 —— 两套渲染对应"单 run 反馈降误导"vs"跨 run 对比保列对齐 / grep 友好"两种 UX 目的，分离规则 → DECISIONS §6.1
- `tokens.total` 用 `int` 而非 `float`：整数计数语义；与 OpenAI `CompletionUsage` / `Counter.total()` 同源；`mean` 仍 `float`（avg 可有小数）→ DECISIONS §6.1
- audit 中标 "应文档化" 4 项（elapsed_ms vs Σ latency_ms 口径差 / cost deterministic vs latency stochastic / cold-start 偏置 / run_id hash idempotent fingerprint）+ deferred 4 项（reproducibility metadata / 价格表用户扩展 API / 老 ollama daemon silent None / output_type='none' efficiency 全 0）本轮不动，留给后续单独 PR 处理

## 2026-05-05 — Phase 7：横切 Safety 上线 + cross-cutting ontology 二分 + evaluate 中段合流 + sample.metrics nested 派统一

### 功能

- 新增 `safety` task 与 15 条低风险样本（6 harmful + 5 jailbreak + 4 benign），并提供 5 份 stub predictions（`safe` / `over_refuse` / `jailbreak_success` / `evasive` / `garbage`），可直接跑 score 矩阵教学
- `score` 与 `run` 双路径均自动产出 `aggregated["safety"]`（4 stat：`refusal_rate` / `jailbreak_success_rate` / `over_refusal_rate` / `judge_safety_score`）及 `per_sample.metrics["safety"]` 子组（schema-on-write，永远 2 键 0/1 占位）
- **核心反向叙事 evasive + safe.s11**：含拒答关键词但实际给绕行提示（evasive）vs 无拒答关键词但语义上是拒答（safe.s11）—— 双向暴露 heuristic 失明，judge 救场，与 phase 3 `wrong_fact` / phase 4 `wrong_fact` / phase 5 `wrong_decision` 同源
- CLI 完整可达：`--task safety` 走 `_build_task_with_optional_deps` 注入；`--judge-model` 接 `Safety(judge_lm=...)`；`--vdb` 显式 SystemExit（safety 非 retrieval task）
- 全量 243 → 281 测试通过（新增 38 条 safety 相关 + 2 条结构性焊接锁）

### 技术

- **cross-cutting ontology 二分（架构升级，不只是文档）**：`content class`（数据源 = `Response.text`，score/run 双路径都能算 → safety / robustness）vs `call class`（数据源 = LM 调用副产品 usage/latency/logprobs，仅 run 能算 → efficiency / calibration）；phase 6 "score 不挂 efficiency" 的事后追认升级为明文原则，phase 9/10 横切按 ontology 直接对号入座
- **evaluate 中段合流 `_evaluate_inner` 重构**：拉出共享 helper（`process_results` → content-class injectors → run-only call-class injectors → 打包），删除 `_finalize` 不留过渡函数；`evaluate_score` / `evaluate_run` 双入口仅负责"如何拿 (docs, responses)"；跨模式分叉点缩到 1 个 `if mode == "run":`
- **`SampleResult.metrics` nested 派统一**（supersede phase 6 audit §6.1 §1.3 单项）：类型签名从 `dict[str, float]` 放宽为 `dict[str, float | dict[str, float]]`；cross-cutting 横切走嵌套子组（`metrics["efficiency"]` / `metrics["safety"]`），与 `Response.usage` nested object / `aggregated[<dim>]` 嵌套子组三层完全一致（OpenAI / Anthropic / inspect_ai SDK 派）
- **新 metric 模块 `metrics/safety.py`**：AdvBench (Zou et al. 2023) + 中文自补的 refusal heuristic 关键词列表（~37 条）+ `is_refusal` / `is_jailbreak_attempted` / `MIN_RESPONSE_LEN=10` 阈值；`inject_per_sample_safety` runner injector + `safety_aggregated` 4 stat schema-on-write；显式不重复造 judge closure，仅暴露 `DEFAULT_SAFETY_JUDGE_TEMPLATE` 常量供 task 端复用 phase 3 `judge_core.judge_pointwise`（同源 phase 5 plan_quality 复用 g_eval）
- **新 task `tasks/safety.py`**：`@register_task("safety")`，可选 `judge_lm` 注入；`process_results` 只做 judge_safety_score，不算 refusal/jailbreak（runner injector 跨 task 统一负责）；category 落 `SampleResult.artifacts["safety_category"]`（非标量字符串走 artifacts 而非 metrics，与 §7.D 类型签名一致）
- **结构性焊接锁**：`test_inner_helper_invoked_by_both_modes` 用 monkey-patch 焊死"score/run 双入口必经 `_evaluate_inner`"；`test_safety_inject_runs_before_efficiency` 用 spy 焊死"content class 注入先于 call class"——把 ontology 二分映射到代码层执行顺序
- **测试套增量**：`test_metrics_safety.py`(19) / `test_safety_score.py`(7 含 5 份 stub 矩阵) / `test_safety_run.py`(3 含 mock:gold parity) / `test_runner_safety.py`(8 含跨 task 横切 + 结构性焊接锁) + 修订 phase 6 测试 ~10 处适配 nested 派访问路径

### 取舍

- runner 中段合流选择 B（`_evaluate_inner` helper）而非 A（双尾段维持）：净复杂度下降（+1 抽象层 vs 跨模式分叉风险消除 + phase 9/10 增量降至改一处）→ DECISIONS §7.B
- `SampleResult.metrics` 命名选 nested 派（C）而非 flat 派（A）/ prefix 派（B）：与 `Response.usage` / `aggregated["efficiency"]` 三层一致；parity helper `_task_metrics` 简化（剥单层子组 key 而非 N 个 flat key）；phase 9/10 加新维度无 namespace 冲突 → DECISIONS §7.D
- 复用 `judge_core.judge_pointwise` + `DEFAULT_SAFETY_JUDGE_TEMPLATE`，不在 `metrics/safety.py` 重复造 judge closure factory（同源 phase 5 plan_quality 复用 g_eval）→ DECISIONS §7.C
- category 字符串落 `artifacts["safety_category"]` 而非 plan 原拟的 `metrics["_safety_category"]`：§7.D 类型签名收紧后字符串在 metrics 上违法；与 phase 4 立的 MLflow scalar/non-scalar 二分一致 → DECISIONS §7.C
- safe.jsonl 矩阵叙事走 fix-B（保留 fixture + 文档反映真实数值）而非 fix-A（修 fixture 让数字配合 heuristic）：与 evasive / wrong_* 反向叙事哲学一致——"暴露局限而非粉饰" → README phase 7 段
- supersede phase 6 audit §6.1 §1.3 单项（sample 层 4 efficiency 键 flat 写法 → nested 子组），其余 6 项 audit 修订仍生效 → DECISIONS §6.1 / §7.D
- 三类显式不做：Perspective API（zero-network 原则）/ multi-turn jailbreak（phase 1-7 都 single-turn）/ HarmBench-AdvBench scale 集成（workshop 体量，~15 条手写 stub 已够矩阵叙事） → DECISIONS §7.C

> 日期归属说明：phase 7 实际工时跨 2026-05-04 → 05-05 边界（中段重构 + safety 落地 + 二轮审计 + 三轮审计修订共 4 段工作）；按 workshops.mdc "≤2 milestone/working day" 规则，5-04 已用满 phase 6 + phase 6 follow-up 两条，phase 7 归为 5-05 起始里程碑。

### Audit follow-up（同日追加，phase 7 实测产物反推 4 项）

跑完 phase 7 全量 281 测试 + 6 个端到端 demo（5 份 safety stub × score + ollama:qwen2.5:32b run）后，从 CLI 输出 / `result.json` / `samples.jsonl` 三类产物形态反推出 4 项工程问题（同 §6.1 audit 体例，实测驱动而非纸面设计）：

- **P1（严重，已修）**：CLI `_print_aggregated` 把 safety 全 0 子组折叠为 `<not measured>`，但 garbage prediction 短文本路径下 safety 全 0 是合法 metric 值（heuristic 真跑了），折叠语义错。修法：cross-cutting dim 走 trait 协议——`metrics/efficiency.py` / `metrics/safety.py` 顶部声明 `FOLD_AS_NOT_MEASURED_WHEN_ALL_ZERO`（call class True / content class False），CLI 通过 `_should_fold_when_all_zero(dim)` 中性查询；按 ontology 二分一一对应，phase 9/10 加新维度声明 trait 即可不改 CLI
- **P2（中，已修）**：`safety_aggregated` 用 0 占位 4 stat，但 `judge_safety_score=0` 在 1-5 scale 上越界（0 等价"未测得"而非"模型得 0 分"）；切片为空的 jailbreak/over_refusal 同问题。修法：返回类型放宽 `dict[str, float | None]`；`refusal_rate` 永远 float（heuristic 永远算）；其它 3 stat 在切片为空 / 未接 judge 时 None；CLI `_fmt_kv` 加 `None → <n/a>` 渲染分支；落 `result.json` 出现 `null`（向前兼容增强非删减）
- **P3（轻，文档化）**：score 路径不挂 efficiency → 不调 `compute_cost_usd` → `preds:*` 不查价格表，是 ontology 二分的合理产物而非 fail-silent。在 `metrics/efficiency.py::compute_cost_usd` docstring + DECISIONS §6.1 §1.4 + README phase 6 段三处显式记录
- **P6（轻，已修）**：`runner.py::_evaluate_inner` 创建 `EvalResult` 时 `elapsed_ms = round(x, 3)`；不动 `efficiency.latency_ms` / `cost_usd` 等 LM 报值（dashboard / cost 累计真用得到亚 ms / 亚 cent 精度）

技术副产物：

- `_is_all_zero_nested` 加 `None` 视为零类信号（与"全 0 折叠"语义对齐，但实际是否折叠由 trait gate 决定）
- 测试套增量：+8 条（trait 协议正反例 + None 占位 + `<n/a>` 渲染 + content/call 混合场景） + ~4 处现有 assert 修订（`== 0.0` → `is None`）；全量 281 → 289 测试通过
- README phase 7 矩阵表 + C.6 toxicity 表 + "None 与 0 的语义分离"小节 + "CLI 折叠规则 trait 派"小节四处同步

取舍：

- P2 选 all-undefined 范围而非 judge-only：4 stat 协议一致性优先；切片为空的 jailbreak/over_refusal 同样适用 None，避免"未测得 vs 真 0"在 1 个 stat 上严格但在 3 个 stat 上模糊 → DECISIONS §7 audit follow-up
- efficiency 不改 None 占位：phase 6 audit §1.3 立的 sample 层 0 占位决策保留，由 P1 trait 折叠覆盖渲染语义；让 efficiency / safety 各自走"trait 折叠 vs None 占位"两种风格，是 ontology 二分在数据契约层面的自然延伸
- 不破 schema-on-write 哲学：dict 形状仍稳定（safety 永远 4 键），只是值可为 None；"形状稳定 + 值可空"是 schema-on-write 的精确表达

## 2026-05-05 — Phase 7 audit follow-up wave 2：7-phase ollama live audit 反推 4 项

### 功能

- **`elapsed_ms` 端到端不再失真**：`evaluate_score --task rag_qa --judge-model ollama:qwen2.5:32b --limit 3` 落盘的 `elapsed_ms` 现在覆盖 `process_results / inject / aggregation` 全段（含 judge LM 调用），与 wall time 对齐——旧实现 judge-heavy 路径漏算 6 个数量级（0.137ms vs 实测 125s）
- **safety drill-down 与 task 口径一致**：sentiment_clf + ollama 跑 5 条样本，sample 层 `prediction="positive"` 配 `safety.jailbreak_attempted=0.0`，符合用户直觉；旧实现因 raw `Response.text="Label: Positive"` 长 15 字符虚假命中 jailbreak heuristic
- **agent_traj `--limit 1` 命中烟雾级 scenario**：`gold.jsonl` 重排 brainstorm → example → panel 后，`--limit 1` 自然命中 brainstorm（2 步 ~10-30s），与 conftest CI 友好策略对齐；旧顺序命中 panel.md（5 角色 × 11 步）必 600s timeout
- **mt 首次跑 CLI 输出干净**：`python -m evals run --task mt --model ollama:qwen2.5:32b --limit 3 2>&1 | head -10` 顶部直接是 `# run_id=...`；BertScore 加载时 transformers 内部的 `Loading weights:` 进度条 + `BertModel LOAD REPORT` 7 行 UNEXPECTED 警告被抑制
- **README phase 6 段口径补全**：新增 "`elapsed_ms` vs `efficiency.latency_ms.mean` 口径" 小节 +"`efficiency.*` 仅算被测物（语义 / 工程现状 / 长期演进三段）"小节；judge / retrieval cost 单独子组（`efficiency.judge.*`）deferred 至 phase 8+ 显式登记

### 技术

- `runner.py::_evaluate_inner` 签名 `elapsed_ms: float` → `t0: float`；末尾算 `elapsed_ms = (perf_counter() - t0) * 1000`，`evaluate_score` / `evaluate_run` 入口删自测两行只透传 `t0`——刚好与 phase 7 立的"_evaluate_inner 即合流点"架构一致
- `metrics/safety.py::inject_per_sample_safety` 内 `text = resp.text` → `text = sr.prediction or ""`；签名 `responses` 形参保留（向后兼容 + 为未来 phase 9 calibration 等需要 raw response 的 injector 留位）
- `data/agent_traj/gold.jsonl` 三行重排（不动任何字段）；`tasks/agent_traj.py` docstring 顶部加排序原则段
- `tasks/mt.py::_bertscore_scorer()` 内 import 之前 `os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")`——`setdefault` 让用户显式 export 时不被覆盖；env var 是 transformers 官方推荐方式（与 logging API 等价但不依赖 logger 名查找）
- 测试增量 4 条：①+②`test_elapsed_ms_covers_process_results_phase` × run/score 双路径（在 `process_results` 内塞 50ms sleep × 3 sample，断言 `r.elapsed_ms >= 150 - 30ms` buffer）；③`test_inject_per_sample_safety_reads_prediction_not_response_text`（构造一对相反信号：prediction=8 字符 < MIN_RESPONSE_LEN / response.text=57 字符 ≥ MIN_RESPONSE_LEN，期望 jb=0 锁定数据源走 prediction）；④`test_docs_smoke_friendly_ordering`；现有 `test_inject_per_sample_safety_writes_nested_subgroup` + `_preserves_frozen_semantics` 构造侧改用 `sr.prediction` 携带信号；全量 286 → 290 测试通过

### 取舍

- TODO-2 选方案 A（看 `sr.prediction`）而非行业主流方案 B（opt-in trait `Task.safety_aware`）：保留 phase 7 "AOP 风格 task 零增量" 叙事，方案 A 已知副作用（normalize 丢拒答关键词时 `refusal_detected` 偏低）+ 与 lm-eval-harness / inspect_ai / OpenAI Evals 主流的偏离都进 ADR §7.1.R1 风险登记，触发条件（引入 raw-text fidelity safety task / phase 10 robustness 设计 / 实测反馈）任一满足即重新评估切到 B → DECISIONS §7.1.2 + §7.1.R1
- TODO-3 短期纯文档化（C only）而非长期 judge wrapper 报数（A）：judge / retrieval cost 单独子组（`efficiency.judge.*`）涉及 closure 协议升级 + CLI 渲染层联动，本轮 ROI 不够；deferred 至 phase 8+ 与 multi-turn / agent 子调用元数据收集一起设计 → DECISIONS §7.1 不做段
- TODO-1 选方案 B（传 `t0`）而非 A（外部推迟测量）/ C（三段 breakdown）：与 `_evaluate_inner` 合流点架构一致，6 行改动；不破 `EvalResult` schema → DECISIONS §7.1.1
- TODO-5 选 env var（B）而非 logging API（A）：transformers 官方推荐方式，import-time 单点副作用；与 HuggingFace docs 对齐 → DECISIONS §7.1.4

> 同日里程碑计数：phase 7 + phase 7 audit follow-up wave 2 = 2 个，达 workshops.mdc 上限。本日不再追加新里程碑。

## 2026-05-06 — Phase 7 audit follow-up wave 3：safety 回归 standalone task + efficiency.judge.* 子组上线

### 功能

- **非 safety task 的 sample.metrics 不再有 safety 占位**：跑 sentiment_clf / qa_open / rag_qa 等 task 时，sample drill-down 仅含 task-specific metric + efficiency 子组，干净；旧 wave 2 后 qa_open / rag_qa 长答案样本被虚假标 `metrics["safety"]["jailbreak_attempted"]=1.0` 的根因消除（`Safety` 是独立 task 的，与其它 task 互不污染）
- **safety task 自身仍正常**：`refusal_rate` / `jailbreak_success_rate` / `over_refusal_rate` / `judge_safety_score` 4 stat 直接平铺在 `aggregated` 顶层（与 sentiment_clf 的 `acc` / `f1_macro` / `cohens_kappa` 同形 task-specific metric，不再嵌套在 `aggregated["safety"]` 子组下）
- **真实 cost 报表区分被测物 vs 评估工具**：rag_qa run + judge 跑完后 `result.json` 同时有 `aggregated.efficiency.cost_usd.total`（被测物 task LM）+ `aggregated.efficiency.judge.cost_usd.total`（评估工具 judge LM）；总账单 = 两者相加（实测 rag_qa 比例约 1:5）
- **score 路径首次出现 efficiency.judge 子组**：`evals score --task rag_qa --judge-model ...` 现在挂 `aggregated["efficiency"]["judge"]`（仅 judge 部分，无被测物 task 部分）；旧实现 score 完全不挂 efficiency

### 技术

- 删 `metrics/safety.py::inject_per_sample_safety` (~30 行) + `safety_aggregated` (~50 行) + `FOLD_AS_NOT_MEASURED_WHEN_ALL_ZERO` trait；保留 `is_refusal` / `is_jailbreak_attempted` / `DEFAULT_SAFETY_JUDGE_TEMPLATE` 作 helpers；新增 `safety_aggregation_funcs() -> dict` 工厂（4 stat 切片 + None-skipping）供 `Safety.aggregation()` 直接 return
- `tasks/safety.py::Safety.process_results` 自写 metrics（flat 平铺 `refusal_detected` / `jailbreak_attempted` / `judge_safety_score?`）+ `aggregation()` 直接 `return safety_aggregation_funcs()`；与 sentiment_clf / qa_open 等 task 同形
- `runner._evaluate_inner` 删 `inject_per_sample_safety` / `aggregated["safety"] = safety_aggregated(...)` 两处调用 + docstring 简化（cross-cutting 仅保留 efficiency 一项）
- 新协议 `_JudgeRecorder`（[`metrics/judge_core.py`](metrics/judge_core.py)）：`__init__(lm)` / `call(requests)` 透传 + append + `model_label`；3 个 judge_core factory（`judge_pointwise` / `judge_pairwise` / `g_eval`）+ 5 个 judge_rag factory（`judge_faithfulness` / `judge_answer_correctness` / `judge_context_precision` / `judge_context_recall` / `judge_answer_relevancy`）内部 `rec = _JudgeRecorder(lm)` + closure 暴露 `_score._recorder = rec`；`self_consistency` wrapper 透传 base 的 `_recorder`
- `Task.collect_judge_responses() -> tuple[list[Response], str | None]` 默认空；qa_open / safety / rag_qa（聚合 5 RAG closure 的 responses）/ agent_traj 各 override 从 closure `._recorder` 拉 list
- `metrics/efficiency.py::efficiency_judge_aggregated` 与 `efficiency_aggregated` 同形 4 子组（latency_ms 4 stat / tokens_in/out 双 stat / cost_usd 双 stat）；schema-on-write 全 0 占位
- `runner._evaluate_inner` 后段加 `task.collect_judge_responses()` + `aggregated["efficiency"]["judge"] = efficiency_judge_aggregated(...)`；非空时挂；score 路径无 task efficiency 时仅创建空子树挂 judge
- `cli.py::_print_aggregated` 折叠协议扩到嵌套二级：cross-cutting dim 顶层非全 0 时遍历内部子子组，全 0 子子组（如 `efficiency.judge`）单行折叠为 `<dim>.<sub>: <not measured>`
- 测试增量 +14：`test_metrics_judge_recorder.py`（recorder + 3 judge_core / 5 judge_rag factory + self_consistency 透传，~10 测试）+ `test_runner_efficiency_judge.py`（task 没接 judge / run+judge 双子组 / score+judge 仅 judge / pointwise call count / schema 同形，5 测试）+ `test_cli_spec.py` 嵌套二级折叠正反例（2 测试）；删除 `test_runner_safety.py` 整文件 + `test_metrics_safety.py` 中 inject/aggregated 系列（cross-cutting AOP 已废）；改写 `test_safety_score.py` / `test_safety_run.py` 把 `r.aggregated["safety"]["x"]` 改为 `r.aggregated["x"]`；全量 276 → 291 测试（删 12 + 加 14 + 改 ~15 处原地）

### 取舍

- A1 走方案 X（删 cross-cutting AOP）而非方案 B（trait）/ B'（method hook）：与 lm-eval-harness / HELM / inspect_ai 主流完全对齐；代码净 -50 行；A1 长答 jb=1 误标根因消除而非 hook 屏蔽。代价是 phase 7 §7.A "content class cross-cutting" 主原则部分 supersede + README phase 7 段重写为 standalone task 视角 → DECISIONS §7.2
- A3 用 closure recorder protocol 而非 LM ABC 全局 hook：判 closure 与 LM 一对一 binding，recorder 内置在 closure 状态里；不破 LM ABC 契约；多 judge_lm 场景（如 qa_open + rag_qa 用不同 judge）天然支持。代价是每个 factory 内部 +3-5 行 recorder boilerplate，但模式简单可读 → DECISIONS §7.3
- audit 中观察到的 A2（judge LM variance）经分析判定为 LM 内禀局限非工程问题（类比硬盘 ECC 不修 bit rot），从 wave 3 完全移除：不进 ADR / 不写 README / 不改默认值（`judge_n_samples=1`）；用户用 `--judge-n-samples N` 自决（self_consistency factory 早就支持任意 N）。LM 局限不是项目责任 → 不出现在 DECISIONS

> 跨日里程碑：phase 7 audit 实际工时跨 5-05 → 5-06 边界（5-05 已用满 phase 7 + wave 2 两条名额，wave 3 落到 5-06 起始里程碑；workshops.mdc "≤2 milestone/working day" 规则）。

## 2026-05-07 — Phase 8：族 1 后半 IAA 双 task + kappa paradox 主舞台 + ordinal 救场叙事

### 功能

- **新 task `iaa_nominal`**：30 条 highly imbalanced (27 ham + 3 spam, ~90/10) binary classification + 4 份 stub predictions（perfect / constant_majority / noisy_diverging / garbage）+ 3 raters/sample；aggregation 输出 15 stat（含 9 classification + 3 agreement 2-rater + 2 multi-rater + 1 `_confusion_matrix` 诊断辅助）
- **新 task `iaa_ordinal`**：25 条 1-5 likert (5 each) + 4 份 stub（perfect / off_by_one / random / garbage）+ 3 raters/sample；aggregation 12 stat（exact + agreement nominal/linear/quadratic + corr 三件套 + ccc + multi-rater 4 维）
- **kappa paradox 主舞台**：`iaa_nominal.constant_majority` 锁数值 `accuracy=0.90 ∧ cohens_kappa=0.00 ∧ gwet_ac1≈0.89` —— acc 看着良好但 nominal κ 失明（"全押多数类"基线），Gwet AC1 (Pe 用类间方差) 仍诚实地高，paradox 解药 1
- **ordinal 救场叙事**：`iaa_ordinal.off_by_one` 锁 `accuracy=0 ∧ cohens_kappa=−0.25` 同时 `weighted_kappa_quadratic=0.71 ∧ pearson_r=0.83 ∧ lins_ccc=0.71`；`iaa_ordinal.garbage` (pred=6−gold inverse) 锁 `weighted_quad=−1 ∧ pearson=−1 ∧ ccc=−1` 但 cohens_kappa=0（paradox 反向场景复刻 + ordinal-aware 抓负相关）
- **CLI 0 改动 / 0 新 flag**：IAA task 不接 judge / vdb，自然走既有 fall-through dispatch 分支（与 sentiment_clf / mt 同形）；非法 flag 组合仍 SystemExit（既有兜底）
- **score 主路径焊死全部教学叙事**：run 路径 `output_type='none'` 给占位 Response，aggregated 仅 sanity 0；run 完整教学（含 LLM-as-annotator self_consistency 派）deferred 至 phase 8.5+（同源 phase 5 让步进 ADR）

### 技术

- **新模块 `metrics/agreement.py`**：scope 收紧——仅 4 个手算函数（`scott_pi` Pe 用合并边际 ∑p̄_c² / `gwet_ac1` Pe 用类方差 (1/(K−1))·∑q_c(1−q_c) / `lins_ccc` 2·cov/(σ_X²+σ_Y²+(μ_X−μ_Y)²) / `icc_1_1` one-way random ANOVA decomposition）+ 1 个共享 helper `build_rater_matrix(srs, include_gold=True)`（fail-loud 缺/uneven raters）；约 80 行
- **库直调全部下放 task aggregation**：sklearn `cohen_kappa_score(weights=...)` / scipy.stats `pearsonr|spearmanr|kendalltau` / statsmodels `fleiss_kappa + aggregate_raters` / `krippendorff.alpha(level_of_measurement=...)` 全部在 `tasks/iaa_nominal.py` / `tasks/iaa_ordinal.py` 内 import；与 sentiment_clf 直调 sklearn / mt 直调 sacrebleu 体例完全一致——避免模块沦为 import 中转站
- **数据契约 0 新概念**：predictions JSONL 行 schema = `{id, prediction, raters: list[str|int]}`；与 phase 4 立的 `task.load_prediction(doc, row)` hook + path B+C「LM-side 数据装 Response，pipeline 产物住 doc.metadata」自然吻合；不动 `api.py` 任一 dataclass / 不动 `Task` ABC（仅 override `load_prediction` + `process_results`）/ 不动 `runner.py`
- **TDD textbook 锁数值**：metrics/agreement.py 单元 29 测试（Po=0.6 binary 锁 scott_pi=0.08/0.48 ≈ 0.167 / Po=0.9 imbalanced 锁 gwet_ac1=0.805/0.905 ≈ 0.89 / shifted 1-5 锁 lins_ccc=0.8 / 完美一致 lockup ICC(1,1)=1.0 + 完美反相关锁 ICC(1,1)=−1.0）；整套 IAA score 矩阵 19 测试锁 4 stub × ~12 stat 数值
- **依赖增量**：`statsmodels>=0.14` + `krippendorff>=0.6`；显式不引 `irrCAC` / `pingouin` / `audtorch` 三个本可用的库（公式简单 5-15 行手算，避依赖膨胀）
- **测试增量 +52**：`test_metrics_agreement.py` 29 + `test_iaa_nominal_score.py` 10 + `test_iaa_ordinal_score.py` 9 + `test_cli_spec.py` IAA dispatch 段 4；全量 299 → 351 测试通过

### 取舍

- 双 task (`iaa_nominal` + `iaa_ordinal`) 而非单 task：两个教学叙事独立可读（kappa paradox 主舞台 vs ordinal 救场），单 task 会膨胀 aggregation 到 25+ stat 且互相稀释 → DECISIONS §8 ①
- `metrics/agreement.py` scope 收紧（仅手算 + 真共享 helper，库直调下放 task）：与 sentiment_clf 体例完全一致，避免模块沦为 import 中转站；与 phase 4 `metrics/retrieval.py` 包 ranx 的「协议转接 + 输入构造非平凡 + 5 指标共用」三联立信号正交（本模块 statsmodels / krippendorff 接口直接吃 matrix，wrap 无价值） → DECISIONS §8 ⑥
- inline JSONL `raters` 字段而非 self_consistency rater 派（LLM-as-annotator 真正交叉形态）：本期 score 主路径已能完成所有数值教学；annotator 派需新增 `--judge-n-samples` flag 设计 + judge LM 当 annotator 的 prompt 模板，ROI 不够；deferred 至 phase 8.5+ → DECISIONS §8 ②
- ICC(1,1) only / run 路径完整教学 / 不引 irrCAC/pingouin/audtorch 三库：YAGNI 与方案对齐，二阶 ICC decomposition + run 路径 + 库依赖三件全 deferred；与 phase 5 agent_traj `--replay-envelope` 让步同源 → DECISIONS §8 ③ ④ ⑤

## 2026-05-07 — Phase 8 follow-up：IAA 退化路径工程兜底（run path + 小 limit 不再 raise/NaN）

### 功能

- `python -m evals run --task iaa_nominal --model ollama:...` 不再 `ValueError: pos_label=spam is not a valid label` / `Target is multiclass but average='binary'`；改在退化输入下短路给 0.0 sanity（与 plan 早先承诺一致）
- `python -m evals score --task iaa_nominal --predictions constant_majority.jsonl --limit 5` 不再 `krippendorff.alpha: There has to be more than one value in the domain`
- `python -m evals run --task iaa_ordinal --model ollama:...` 不再在 `pearson_r` / `spearman_rho` / `kendall_tau` / `cohens_kappa` / `weighted_kappa_*` 报 `nan`（NaN 不是合法 JSON，跨 run JSON_EXTRACT / 任何非 Python parser 必坏）
- 全部 IAA aggregated dict 现在严格 `json.dumps(..., allow_nan=False)` 可序列化——与 phase 4 path C「跨进程跨 run 走 JSON 流转」契约对齐

### 技术

- **`iaa_nominal.py` 三处工程兜底**：① `_pos_label_present(yt, yp)` helper — sklearn binary scorers (`precision_score`/`recall_score`/`f1_score`/`fbeta_score` with `pos_label='spam'`) 在 `pos_label` 缺席 y_true ∪ y_pred 或 union 是 multiclass（如 run path 看到 `{ham, spam, ""}` 三类）时 raise；短路返 0.0；② `_krippendorff_alpha` 加 `<2 unique value` 短路（krippendorff 库在单值域 raise）；③ `_cohens_kappa` / `_fleiss_kappa` 包 `_nan_to_zero(x)` helper（`x != x` 检测 NaN，单类 Pe=1 退化时给 0.0）
- **`iaa_ordinal.py` 同源处理**：`_nan_to_zero` helper 包所有 6 个易 NaN 路径（`_cohens_kappa` / `_weighted_kappa_linear` / `_weighted_kappa_quadratic` / `_pearson_r` / `_spearman_rho` / `_kendall_tau`）+ 3 个多 rater 路径（krippendorff ordinal/interval + icc_1_1）；同形 `<2 unique value` 短路
- **新测试套 `test_iaa_engineering_robustness.py` (10 测试)**：`_UnusedLM` (LM 调用即 AssertionError，证 `output_type='none'` 真不调) + `_assert_aggregated_is_finite_json` helper（递归遍历 aggregated dict 所有 leaf，`math.isfinite` 锁 + `json.dumps(..., allow_nan=False)` round-trip 锁）；4 stub × 2 task 严格 JSON 序列化参数化；全量 351 → 361 测试通过
- **完整 LM 系列回归确认**：`test_qa_open_live` (4) + `test_rag_live` (3) + `test_agent_traj_run_live` (1) + `test_ollama_lm` (7) 共 15 条 LM 真实调用测试在 phase 8 改动后仍全部通过；CLI smoke 三件（iaa_nominal score `--limit 5` / iaa_nominal run `--limit 5` 与 full / iaa_ordinal run `--limit 3`）全部 exit 0

### 取舍

- 工程兜底 `_nan_to_zero` 选「NaN→0」而非「NaN→null」：① `EvalResult.aggregated[str, float]` 契约期 float 而非 Optional[float]，跨 run 排序 / cross-run JSON_EXTRACT 看到 None 会 KeyError 路径分裂；② plan 早先就承诺 "sanity 0/None" 而 0 比 None 更适合作 scalar 默认；③ 与 sentiment_clf 等老 task 在缺数据时给 0 的体例一致 → DECISIONS §8.R4
- 工程兜底放在 task aggregation 内 (`_pos_label_present` / `_nan_to_zero` 是 task-local closure helper) 而非提到 metrics/agreement.py：① 这些 helper 只服务 task 自己的库直调；② 与「metrics/agreement.py 仅装手算 + 真共享 helper」scope 收紧决策（DECISIONS §8 ⑥）正交一致；③ 避免下放后 metrics/agreement.py 又拐回 import 中转站

### 二轮 audit follow-up (同 5-07 同里程碑续记)

首轮 fix 后用户追问"测试是否够写实"，补做四道更严的 audit，又抓到 3 处更深的 bug + 1 个全局兜底：

- **scipy 长度 <2 raise**：`_pearson_r` / `_spearman_rho` / `_kendall_tau` 在 `--limit 1` 时 scipy raise `ValueError: x and y must have length at least 2`，单 NaN 兜底救不了——加 `len(srs) < 2: return 0.0` 前置短路
- **dtype 混合的 unique 检测假性**：`iaa_ordinal` 走 `build_rater_matrix` 把 `sr.target` (`str("1")`) 与 `raters` (`int(1)`) 混入同一矩阵；首轮加的 `<2 unique value` 短路在 raw matrix 上判 `{1, "1"}` 假性=2，但 `np.asarray(dtype=int)` 之后真 unique=1，krippendorff 仍 raise——unique 检测必须在 dtype 转换**之后**做
- **iaa_ordinal `_fleiss_kappa` 漏 wrap**：首轮只包了 `iaa_nominal` 的 `_fleiss_kappa`，`iaa_ordinal` 里同名函数还会泄 NaN——加 `_nan_to_zero` wrap 对齐
- **storage 层 strict-JSON 兜底**：`storage.save` 三处 `json.dumps` 默认 `allow_nan=True`——任意未来 task 漏算 NaN 就会静默写出 `NaN` / `Infinity` 字面量到 `result.json` / `samples.jsonl` / `index.jsonl`，污染 cross-run 消费（jq / 浏览器 / DB / 仪表盘必拒）；改成 `allow_nan=False` 在写时 `ValueError` fail-loud；同步清掉首轮 smoke 测试遗留在 `runs/` 的 4 个 NaN 毒文件 + `index.jsonl` 4 行毒索引
- **测试增量 +9 → 19 robust 测试**：`--limit 0/1/2` 参数化 (6) + storage 拒 NaN/Inf 合成 EvalResult (2) + `parse_constant=raise` 模拟 jq / 非 Python parser 端到端 strict JSON 验证 (1)；全量 361 → 370 测试通过；15 LM live 测试再确认无回归

> 教训：首轮把"教学叙事的数值正确性"当主测目标（满数据 score path），缺四类工程面：① `--limit` 退化路径 ② `output_type='none'` run 路径下 task 自身 aggregation ③ aggregated dict 严格 JSON 序列化 ④ 跨 task 的存储层兜底。同样模式 phase 4 (rag_retrieval) / phase 5 (agent_traj) 也应回顾——但本期 minimal-diff，仅修 IAA + storage（后者全局受益）。模板 helper (`_UnusedLM` + `_assert_aggregated_is_finite_json` + `parse_constant=raise`) 留在 IAA 测试文件，未来加新 `output_type='none'` task 时可抽到 conftest。

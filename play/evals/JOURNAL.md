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

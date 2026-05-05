# TODO

实测产物反推的工程问题清单（2026-05-05 全 7-phase ollama live audit），按严重度排序。每条含**现象 / 证据 / 根因 / 影响 / 修法预案**。修复时按 P 编号建 plan / commit 单独跟踪。

## 严重

### TODO-1（X4）`elapsed_ms` 在 score 路径下不含 `process_results` 时间，judge-heavy 任务下完全失真

**现象**：`score --task rag_qa --judge-model ollama:qwen2.5:32b --limit 3` 实际 wall time **125 秒**（5 个 RAG judge 维度 × 3 sample = 15 次 ollama call），但 CLI 顶部显示 `elapsed=0.1ms`，落 `result.json` 为 `"elapsed_ms": 0.137`。

**证据**：

```bash
$ time python -m evals score --task rag_qa --predictions evals/data/rag_qa/predictions/perfect.jsonl --judge-model ollama:qwen2.5:32b --limit 3
# run_id=20260505-065153-4a0cd0c4  mode=score  ...  elapsed=0.1ms
real  2m5.858s
```

`/tmp/evals_full_audit/20260505-065153-4a0cd0c4/result.json`:

```json
{ "elapsed_ms": 0.137, "aggregated": { "faithfulness": 0.667, "answer_relevancy": 4.333, ... } }
```

**根因**：[`runner.py`](runner.py) `evaluate_score` 在 `_evaluate_inner` 调用**之前**测 `elapsed_ms = (time.perf_counter() - t0) * 1000.0`，只算"读 predictions JSONL"的时间。`process_results`（含 judge LM 调用）在 `_evaluate_inner` 内部，被排除在 `elapsed_ms` 外。`evaluate_run` 同样问题，但因为 task 主 LM 调用占大头、judge 调用相对小，run 路径下失真程度小一些。

**影响**：
- 用户 / dashboard / cost 报表完全看不到真实跑分耗时，差最高 **6 个数量级**
- 跨 run 对比失效（同 task / 不同 judge 配置下 elapsed 不能用作性能比较）
- phase 6 audit "应文档化但 deferred" 4 项中提到了 elapsed_ms vs Σ latency_ms 口径模糊，但 phase 7 的 hybrid score+judge / RAG hybrid 把这个失真从"文档问题"放大成"实际 bug"

**修法预案**：

|方案|做法|改动量|
|---|---|---|
|A. 推迟测量|把 `elapsed_ms = perf_counter() - t0` 推迟到 `_evaluate_inner` 调用**之后**；evaluate_score / evaluate_run 双入口都改|3 行 × 2 处 + EvalResult 构造时机调整|
|B. 传 t0|让 `_evaluate_inner` 接受 `t0` 而非 `elapsed_ms` 自己测，在末尾算端到端时间|5-8 行|
|C. 测两段|保留 setup 时间 + 加 process 时间 + 加 finalize 时间，三段分别报；result.json 加 `elapsed_ms_breakdown` 嵌套子组|10+ 行 + schema 扩展|

推荐 **B**：与 `_evaluate_inner` 已经是合流点的架构一致，一次性修两条入口；不破 EvalResult schema。

---

### TODO-2（X6）`safety injector` 用 raw `response.text` 跑 heuristic，与 `task.process_results` 的 normalized prediction 口径不一致，非 safety task 全部样本被虚假标记 `jailbreak_attempted=1`

**现象**：sentiment_clf + ollama 跑出来 5 个 sample 全部 `safety.jailbreak_attempted=1.0`，但 prediction 是 8 字符的 `"positive"` / `"negative"`——本不该触发 jailbreak heuristic（< MIN_RESPONSE_LEN=10）。

**证据**：

```bash
$ # /tmp/evals_full_audit/20260505-065030-5873c479/samples.jsonl
s01 pred='positive' jb_attempted=1.0
s02 pred='negative' jb_attempted=1.0
s03 pred='negative' jb_attempted=1.0
s04 pred='negative' jb_attempted=1.0
s05 pred='positive' jb_attempted=1.0
```

直接调 ollama 确认 raw response.text：

```bash
$ curl ... /api/generate -d '{"model":"qwen2.5:32b", "prompt":"Classify ... Label:"}'
{"response": "Label: Positive", ...}    # 15 字符 (≥ MIN_RESPONSE_LEN=10) → heuristic 判 True
```

qa_open 也观察到同样问题：长答案样本（如 `"明朝是在1368年建立的。"` 13+ 字符）全被虚假标记 `jailbreak_attempted=1.0`。

**根因**：[`metrics/safety.py::inject_per_sample_safety`](metrics/safety.py) 用的是 `resp.text`（**raw** ollama 输出，含 `"Label: "` 等前缀）：

```python
text = resp.text or ""
safety_subgroup = {
    "refusal_detected": float(is_refusal(text)),
    "jailbreak_attempted": float(is_jailbreak_attempted(text)),  # raw text 长度 ≥ 10 → True
}
```

而 [`tasks/sentiment_clf.py::process_results`](tasks/sentiment_clf.py) 用 `_normalize(response.text)` 写 normalized 8 字符 `"positive"` 进 `SampleResult.prediction`：

```python
def process_results(self, doc: Doc, response: Response) -> SampleResult:
    pred = _normalize(response.text)  # → "positive"
    return SampleResult(..., prediction=pred, metrics={"acc": float(pred == target)})
```

两路径口径不一致：safety injector 看 raw 长 text，aggregator 看 normalized 短 prediction。

**影响**：
- aggregated `jailbreak_success_rate=<n/a>`（无 jailbreak 切片）—— 顶层主面板不可见 ✓
- **但 `samples.jsonl` drill-down 严重失真**——用户看到 sentiment "positive" 配 `jailbreak_attempted=1.0` 极度困惑
- 落盘 schema 上有大量虚假 1.0 信号，下游 W&B / SQL 聚合若把 sample 层 jb_attempted 拿去 mean 会得到错误结论

**修法预案**：

|方案|做法|改动量|代价|
|---|---|---|---|
|A. 用 prediction|让 `inject_per_sample_safety` 用 `sr.prediction` 而非 `resp.text`；口径与 task 一致|2 行|破 "safety 是 raw response 上的 heuristic" 语义；task 若 normalize 掉 refusal 关键词就丢信号|
|B. opt-in trait|加 `Task.safety_aware: bool = False`；`safety task` 设 True；`safety injector` 只对 True 的 task 注入|5 行 + 1 处 task 显式声明|破 P1 的 "cross-cutting AOP 风格 task 零增量" 设计|
|C. 双重判定|`is_jailbreak_attempted` 同时看 raw text 长度 + 是否在 task-level 看起来"实质回答"（如 prediction != target、prediction 不在 LABELS 里等）|10+ 行 + 启发式扩展|过度复杂，难维护|
|D. 加 task-level 信号过滤|safety injector 多加一个判定：若 task 写了 `safety_category` 才注入；非 safety task → metrics["safety"] = {"refusal_detected": 0, "jailbreak_attempted": 0} 强制|3-5 行|本质是 schema-on-data 的局部退化|

推荐 **A + 文档说明**：用 `sr.prediction` 才能让 sample 层 drill-down 与 task 语义对齐；副作用（task normalize 掉 refusal 词时丢信号）在文档里显式说明 "safety injector 看的是 task-认可的 final prediction，不是 LM 原始回答；若需要 raw-text-level 安全判定请用 dedicated safety task"。

---

## 中等

### TODO-3（X3）judge LM 调用不计入 `efficiency` 横切，cost / latency / tokens 三维度严重低估

**现象**：phase 3 qa_open self-grading 的 elapsed 几乎等于 task answer 时间，judge 调用没单独累加：

```
elapsed=8250ms;  efficiency.latency_ms.mean=2747 × 3 calls = 8240ms（仅 task）
efficiency.tokens_in.total=140       # 只算 task prompt，不含 judge prompt（每条 ~150 tokens × 3 = 450 漏算）
```

**根因**：runner 的 `inject_per_sample_efficiency` 只看 task 的 `responses[]`，judge_pointwise 等 closure 内部调 `judge_lm.generate_until()` 返回的 `Response` 没被 runner 收集。

**影响**：
- self_consistency n=5 时 judge cost 是 task cost 的 5 倍，全漏算
- phase 4 RAG 5 维度 judge 时 5x 漏算
- cost 报表对 judge-heavy 任务严重偏低，不能用于真实预算决策

**修法预案**：

|方案|做法|改动量|
|---|---|---|
|A. judge wrapper 报数|judge_pointwise 等 closure 工厂额外维护一个 `_calls: list[Response]` 状态，task 端 `process_results` 完后从 closure 拉出来累加 efficiency|20+ 行 + closure 协议升级|
|B. LM ABC 加监听|`LM.generate_until` 包一层全局 hook，所有 LM 调用自动累计到 thread-local efficiency aggregator|架构改动较大，30+ 行|
|C. 显式不算|文档说明"efficiency 只算 task 主 LM 调用，judge / retrieval 等子调用不算；要算总 cost 用 elapsed_ms × cost-per-second 估算"|纯文档|

推荐 **C 短期 + A 长期**：先文档化避免误读，长期看 judge wrapper 增量协议（与 phase 8+ multi-turn / agent 调用元数据收集统一设计）。

---

### TODO-4（X5）`agent_traj --limit 1` 命中 panel.md 重场景，>600s timeout 失败

**现象**：

```bash
$ python -m evals run --task agent_traj --model ollama:qwen2.5:32b --limit 1
TimeoutExpired: agent_engine ... timed out after 600.0 seconds
```

`agent_traj.docs()` 顺序：`panel → brainstorm → example`，`--limit 1` 取 panel.md（5 角色 × 11+ steps，分钟级 scenario）。

**对比**：[`tests/conftest.py`](tests/conftest.py) 的 live test 显式选 `brainstorm.md`（plan §5 标"brainstorm 2 步 ~10-30s，CI 友好"），但 task `--limit 1` 默认 panel 与这个 CI 友好性建议矛盾。

**根因**：[`tasks/agent_traj.py::docs()`](tasks/agent_traj.py) 的 yield 顺序按 gold.jsonl 行序，恰好 panel 在第一行。`--limit 1` 用户期望"smoke 测试"实际拿到"重场景"。

**影响**：
- 新用户跑入门 demo 失败，体验差
- CI / pre-commit hook 跑不动
- `--limit 1` 在其它 task 上是合理 smoke 配置，agent_traj 唯一例外

**修法预案**：

|方案|做法|改动量|
|---|---|---|
|A. 重排 docs|`agent_traj.docs()` 改为 brainstorm → example → panel（轻 → 重）；docstring 说明排序原则|2 行 + docstring|
|B. agent_engine 加超时|`make_run_fn(timeout=600)` 默认值改更短（如 120s），用户显式 override|2 行 + 部分场景测试需要传更长 timeout|
|C. 加 README 警告|README phase 5 段加显式提示 "agent_traj 默认按 docs 顺序跑，panel 重；smoke 用 brainstorm 单跑或加 --limit 数据 id"|纯文档|

推荐 **A**：重排 docs 是最干净的 UX 修法，brainstorm 在 0 位置让 `--limit 1` 自然命中 smoke scenario；与 CI 友好的 conftest 选择也对齐。

---

### TODO-5（X1）Phase 2 mt task 触发 BertScore 模型加载日志污染 CLI 输出

**现象**：

```
$ python -m evals run --task mt --model ollama:qwen2.5:32b --limit 3 --num-fewshot 2
Loading weights:   0%|          | 0/199 [00:00<?, ?it/s]Loading weights: 100%|██████████| 199/199 [00:00<00:00, 7909.36it/s]
[transformers] BertModel LOAD REPORT from: bert-base-chinese
Key                                        | Status     |  | 
-------------------------------------------+------------+--+-
cls.seq_relationship.bias                  | UNEXPECTED |  | 
... (7 行 UNEXPECTED 警告)
# run_id=...  mode=run  ...
  exact_match                  1.0000
```

10 行内部加载报告在 CLI 顶部输出之前打印，破坏 grep / pipe 友好性。`UNEXPECTED` 是 transformers 期望性警告（非 task 头的 BERT 不需要 LM head 权重），用户没有可操作信息。

**根因**：BertScore lazy import 时 `from bert_score import BERTScorer` 触发 transformers 内部 logger（`transformers.modeling_utils` 等）输出到 stderr，evals 没干预 logging level。

**影响**：
- 首次跑 mt 时 stdout/stderr 混杂噪音 ~10 行
- pipe 到下游消费（`python -m evals run ... | tee log.txt`）污染日志
- CI 输出膨胀

**修法预案**：

|方案|做法|改动量|
|---|---|---|
|A. lazy 配 logging|`metrics/bertscore.py`（如有，否则在 task 端） 在 import bert_score 之前 `logging.getLogger("transformers").setLevel(logging.ERROR)`|3 行|
|B. stderr 重定向|`os.environ["TRANSFORMERS_VERBOSITY"] = "error"` 模块加载时设置|2 行|
|C. 不修|README 注一句 "首次跑 mt 会有 transformers 加载日志，可忽略"|纯文档|

推荐 **B**：env var 是 transformers 官方推荐的日志级别控制方式，1 个 import-time 副作用语句即可。

---

## 轻微

### TODO-6（X2）`efficiency.latency_ms.mean × n` 与 `elapsed_ms` 接近但不严格等价，文档未说

**现象**：sentiment_clf phase 1 实测 `elapsed=6262.2ms` vs `latency_ms.mean × n = 1249.37 × 5 = 6246.86ms`，差 15ms（~0.2%）；mt fewshot 实测 `elapsed=6928 vs 2306×3=6918`，差 10ms。

差值是 process_results / fewshot 拼装 / runner 编排开销。

**根因**：phase 6 audit 已标"应文档化但 deferred"。本次 7-phase 实测确认差值小且稳定，但 X3 / X4 在 judge-heavy 场景把这个差值放大到非常显著（X4 是 6 个数量级）。X2 本身只是文档缺失。

**修法**：README phase 6 段加一句口径说明：

```
elapsed_ms 是 evaluate_run / evaluate_score 端到端 wall time，
含 docs() / process_docs / build_request / lm.generate_until / process_results / inject / aggregation 全部。
efficiency.latency_ms.mean 仅含 lm.generate_until 单次调用时间（task 主 LM）。
两者差值 = runner / task 编排开销。judge LM 调用单独不计入 efficiency（见 TODO-3）。
```

纯文档，零代码改动。

---

## 修复优先级建议

|顺序|TODO|理由|
|---|---|---|
|1|TODO-1 (X4)|严重 + 修法简单（5-8 行 runner 改动）+ 测试增量小|
|2|TODO-2 (X6)|严重 + 修法 2 行 + 测试断言更新（已有 sample 层断言可适配）|
|3|TODO-4 (X5)|中等 + 2 行重排 docs + 显著改善新用户体验|
|4|TODO-5 (X1)|中等 + 2 行 env var + CI 输出干净|
|5|TODO-6 (X2)|轻微 + 纯文档，与 TODO-3 联动一并写|
|6|TODO-3 (X3)|中等 + 架构改动较大，建议放到 phase 8+ 与 multi-turn / agent 调用元数据收集统一设计|

TODO-1 + TODO-2 + TODO-4 + TODO-5 可以合并为一个 "phase 7 audit follow-up 第二轮" plan / commit。TODO-3 / TODO-6 单独 deferred。

---

## 修复后验证清单

无论修哪几条，复跑以下端到端 demo 反向校验：

```bash
# X4 / X3 / X2 验证
time python -m evals score --task rag_qa --predictions evals/data/rag_qa/predictions/perfect.jsonl --judge-model ollama:qwen2.5:32b --limit 3
# expect: result.json elapsed_ms ≈ wall time（差值 < 1s 算合理 setup overhead）

# X6 验证
python -m evals run --task sentiment_clf --model ollama:qwen2.5:32b --limit 5 --runs-dir /tmp/audit_after
head -1 /tmp/audit_after/<run_id>/samples.jsonl | python -c "import json,sys; print(json.loads(sys.stdin.read())['metrics']['safety'])"
# expect: jailbreak_attempted=0.0（短 prediction 不触发）

# X5 验证
python -m evals run --task agent_traj --model ollama:qwen2.5:32b --limit 1
# expect: 跑完不超 60s（命中 brainstorm 而非 panel）

# X1 验证
python -m evals run --task mt --model ollama:qwen2.5:32b --limit 3 2>&1 | head -10
# expect: 顶部直接是 `# run_id=...` 行，无 BertModel LOAD REPORT
```

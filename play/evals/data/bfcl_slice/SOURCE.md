# BFCL slice — data provenance

`gold.jsonl` 抽自 [Berkeley Function-Call Leaderboard (BFCL)](https://github.com/ShishirPatil/gorilla) 的 `simple_python` 子集前 50 例，作为 `play/agent_sft` Phase 1 OOD function-calling baseline.

## 钉版

|项|值|
|---|---|
|repo|[ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla)|
|commit|`58f57e9124ea981403792dd51e00a6577e621fae` (2025-08-25)|
|question file|`berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_simple_python.json`|
|answer file|`berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v4_simple_python.json`|
|sample range|`simple_python_0` 至 `simple_python_49`（前 50 行）|
|license|Apache-2.0（随 gorilla repo）|

## 复现

```bash
cd play/evals/data/bfcl_slice
python _fetch.py
```

`_fetch.py` 走 `curl` 拉钉版两个文件、按 [`bfcl_slice` task 契约](../../tasks/bfcl_slice.py) 折叠成 `gold.jsonl`：

|输出字段|来源|
|---|---|
|`id`|BFCL 原 `id`，如 `simple_python_0`|
|`input`|BFCL `question[0][0]['content']`（user query 原文）|
|`target`|从 GT acceptable values **首组 required**折叠的 canonical call 字符串（仅供 EM 渲染 / 跨 run 对账；真正打分用 `metadata.ground_truth`）|
|`metadata.function_schema`|BFCL `function[0]`（含 `properties` / `required` / `type`）|
|`metadata.ground_truth`|BFCL `ground_truth[0]`，形如 `{func_name: {arg: [acceptable_v1, acceptable_v2, ...]}}`|
|`metadata.user_query`|`input` 的副本，留给 prompt 模板用|

## 范围限制

- 仅取 `simple_python` 子集——单函数、单轮对话、Python 风格调用；不涉及 multi-turn / parallel / live （后者引入函数选择 + 多轮上下文，超出 Phase 1 baseline 范围）
- 50 例规模够看 7B vs 32B 的差距方向，但**不能用于声明 SOTA**——BFCL 全集 `simple_python` 含 400 题，此处只是抽样切片
- canonical `target` 跳过 acceptable 列表含 `""` 的 optional arg；模型若按 default 显式传也算对（由 `metadata.ground_truth` 兜底）

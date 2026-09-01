# BFCL slice — data provenance

`gold.jsonl` is sampled from [Berkeley Function-Call Leaderboard (BFCL)](https://github.com/ShishirPatil/gorilla) `simple_python` subset (first 50 rows), as `play/agent_sft` Phase 1 OOD function-calling baseline.

## Pinned version

|Item|Value|
|---|---|
|repo|[ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla)|
|commit|`58f57e9124ea981403792dd51e00a6577e621fae` (2025-08-25)|
|question file|`berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_simple_python.json`|
|answer file|`berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v4_simple_python.json`|
|sample range|`simple_python_0` through `simple_python_49` (first 50 rows)|
|license|Apache-2.0 (with gorilla repo)|

## Reproduction

```bash
cd play/evals/data/bfcl_slice
python _fetch.py
```

`_fetch.py` uses `curl` to fetch two pinned files and folds into `gold.jsonl` per the [`bfcl_slice` task contract](../../tasks/bfcl_slice.py):

|output field|source|
|---|---|
|`id`|BFCL original `id`, e.g. `simple_python_0`|
|`input`|BFCL `question[0][0]['content']` (original user query)|
|`target`|canonical call string folded from GT acceptable values **first required group** (for EM display / cross-run reconciliation only; actual scoring uses `metadata.ground_truth`)|
|`metadata.function_schema`|BFCL `function[0]` (with `properties` / `required` / `type`)|
|`metadata.ground_truth`|BFCL `ground_truth[0]`, shape `{func_name: {arg: [acceptable_v1, acceptable_v2, ...]}}`|
|`metadata.user_query`|`input` copy, reserved for prompt templates|

## Scope limits

- Only `simple_python` subset — single-function, single-turn, Python-style calls; does not cover multi-turn / parallel / live (which add function selection + multi-turn context, beyond Phase 1 baseline scope)
- 50 examples suffice to see 7B vs 32B gap direction, but **must not be used to claim SOTA** — full BFCL `simple_python` has 400 items; this is only a sample slice
- canonical `target` skips optional args whose acceptable list contains `""`; models that explicitly pass defaults still count as correct (covered by `metadata.ground_truth`)

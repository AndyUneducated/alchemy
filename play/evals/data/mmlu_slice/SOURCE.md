# MMLU slice — data provenance

`gold.jsonl` is sampled from [`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu): 6 subjects × 16 examples = 96 items, as a **regression guard** baseline for `play/agent_sft` Phase 1 general capability.

## Pinned version

|Item|Value|
|---|---|
|HF dataset|[`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu)|
|revision|`c30699e8356da336a370243923dbaf21066bb9fe` (2024-03-08)|
|file pattern|`<subject>/test-00000-of-00001.parquet` (first 16 rows of each subject's test split)|
|license|MIT (as published with [Hendrycks et al. 2020](https://arxiv.org/abs/2009.03300))|

## Subject selection + category coverage

|subject|category|count|
|---|---|---|
|`abstract_algebra`|STEM-math|16|
|`college_computer_science`|STEM-cs|16|
|`clinical_knowledge`|health|16|
|`high_school_world_history`|humanities|16|
|`philosophy`|humanities|16|
|`econometrics`|social science|16|

Covers 4 broad areas (STEM × 2, humanities × 2, social science, health) to avoid masking post-SFT humanities regression when only STEM is selected.

## Reproduction

```bash
cd play/evals/data/mmlu_slice
python _fetch.py
```

`_fetch.py` uses `curl` to download 6 pinned parquets to `$TMPDIR`, parses with `pyarrow`, and folds into `gold.jsonl` per the [`mmlu_slice` task contract](../../tasks/mmlu_slice.py):

|output field|source|
|---|---|
|`id`|`<subject>_<idx>`, e.g. `abstract_algebra_0`|
|`input`|`question` verbatim (without choices)|
|`target`|`answer` (int 0-3) → `"A" / "B" / "C" / "D"`|
|`choices`|`choices` (list[str] length 4), aligned with `Doc.choices`|
|`metadata.subject`|MMLU subject name, for by_subject breakdown|
|`metadata.raw_choices`|`choices` copy (list form for prompt templates)|

## Scope limits

- Only 96 items — full MMLU is ~14k; this slice is for baseline cross-model comparison direction, **must not be used for SOTA claims**
- 6 subjects are hand-picked coverage samples, not random draw; for serious MMLU runs use [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) full `mmlu` task
- Evaluation protocol is generate_until + first letter (`Answer: <X>`); differs from original MMLU paper loglikelihood-of-letter protocol — former closer to real deployment, latter closer to original paper SOTA tables

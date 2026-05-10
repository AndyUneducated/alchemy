# MMLU slice — data provenance

`gold.jsonl` 抽自 [`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu) 的 6 个 subject × 各 16 例 = 96 题，作为 `play/agent_sft` Phase 1 通用能力**防回归** baseline.

## 钉版

|项|值|
|---|---|
|HF dataset|[`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu)|
|revision|`c30699e8356da336a370243923dbaf21066bb9fe` (2024-03-08)|
|file pattern|`<subject>/test-00000-of-00001.parquet`（每个 subject 的 test split 头 16 行）|
|license|MIT（随原 [Hendrycks et al. 2020](https://arxiv.org/abs/2009.03300) 公布）|

## subject 选择 + 类别覆盖

|subject|category|样本数|
|---|---|---|
|`abstract_algebra`|STEM-math|16|
|`college_computer_science`|STEM-cs|16|
|`clinical_knowledge`|health|16|
|`high_school_world_history`|humanities|16|
|`philosophy`|humanities|16|
|`econometrics`|social science|16|

跨 4 大方向（STEM × 2、人文 × 2、社科、health）覆盖各类知识，避免只选 STEM 时 SFT 后 humanities 退化被掩盖。

## 复现

```bash
cd play/evals/data/mmlu_slice
python _fetch.py
```

`_fetch.py` 走 `curl` 下载钉版 6 个 parquet 到 `$TMPDIR`、用 `pyarrow` 解析、按 [`mmlu_slice` task 契约](../../tasks/mmlu_slice.py) 折成 `gold.jsonl`：

|输出字段|来源|
|---|---|
|`id`|`<subject>_<idx>`，如 `abstract_algebra_0`|
|`input`|`question` 字段原文（不含选项）|
|`target`|`answer` (int 0-3) → `"A" / "B" / "C" / "D"`|
|`choices`|`choices` (list[str] 长度 4)，与 `Doc.choices` 对齐|
|`metadata.subject`|MMLU subject name，给 by_subject breakdown 用|
|`metadata.raw_choices`|`choices` 副本（list 形态便于 prompt 模板）|

## 范围限制

- 仅 96 题——MMLU 全集 14k 题；此处是切片用于 baseline 跨模型对比方向，**不能用作 SOTA 声明**
- 6 subjects 是手选覆盖样而非随机抽样；如要严肃跑 MMLU 应用 [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) 的 `mmlu` 任务全集
- 评测协议是 generate_until + 取首字母（`Answer: <X>`）；与原 MMLU paper 的 loglikelihood-of-letter 协议不同——前者更接近真实部署，后者更接近原 paper SOTA 表

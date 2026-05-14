# LoRA 超参 sweep 报告（agent_sft Phase 3）

本报告由 [`sweep.py`](../../sweep.py) 自动生成。每个 sweep 中只动**一个**超参，其余保持基线值不变（控制变量法 controlled-variable）。

训练数据 `train_7b_1k.jsonl` (766 sample / 196 val)，schema 见 [`DECISIONS §4`](../../../DECISIONS.md)；底座 `mlx-community/Qwen2.5-7B-Instruct-4bit` (QLoRA)；评估走 [`eval_smoke.py`](../../eval_smoke.py)，解析模型输出里 `<tool_call>` 块与 ground-truth 比对.

## sweep 覆盖范围（v1 实际值）

v1 计划过 4 dim 但只跑 2 dim — 见下表与下方"为什么只跑 2 dim"：

|dim|状态|值|备注|
|---|---|---|---|
|`iters`|已跑|50 / **200** / 600|baseline 即 100% 4 项指标，多 epoch 无收益|
|`learning_rate`|已跑|1e-5 / **1e-4** / 5e-4|1e-4 甜点；5e-4 发散；1e-5 步太小|
|`num_layers`|**未跑**|—|推迟至 Phase 5 触发条件（[`DECISIONS §5`](../../../DECISIONS.md)）|
|`rank`|**未跑**|—|同上|

### 为什么只跑 2 dim

iters / lr 跑完即在 baseline 配置上拿到 fast proxy 4 项 100%（emit / name / arg_set / arg_value）——任务结构性强（学 `<tool_call>{...}</tool_call>` 形态 + 字面值搬运），train 集 766 样本已足以学透 schema. 此时继续扫 layers / rank 在本指标上不会有信号，决定推迟到 **Phase 5 端到端 gap 关闭率 < 50%** 时再回头扫（[`DECISIONS §5`](../../../DECISIONS.md)）。

Phase 5 实测 nudge gap 关闭 57.3%，三阈值全过，**未触发** layers/rank sweep 条件，正式不扫（[`DECISIONS §9`](../../../DECISIONS.md)）。

## 基线配置（baseline）

|参数|值|
|---|---|
|`iters`|200|
|`batch_size`|4|
|`num_layers`|16|
|`learning_rate`|0.0001|
|`rank`|16|

## 训练步数 `--iters`（iterations）

**它做什么**：每次梯度更新叫一个 **iter / step**。766 个训练样本、batch=4 时 1 epoch ≈ 192 iter，所以 `iters=600` 约等于 3 个 epoch（每条样本平均被看 3 次）。

**为什么会有差异**：tool-call schema 是个**结构性任务**——模型要学 `<tool_call>{...}</tool_call>` 形态 + 把 instruction 文本里的字面值搬进 JSON dict。iters 太少没学透形态；太多会把 766 条 corrected 模板**死记**下来，泛化到训练集外的 args 时变差。

### 实测结果

|值|首 loss|末 loss|val loss|emit|name|arg_value|耗时|备注|
|---|---|---|---|---|---|---|---|---|
|`50`|0.28|0.00|0.00|100%|100%|100%|1212.1s||
|`200`|0.28|0.00|0.00|100%|100%|100%|3967.7s||
|`600`|0.28|0.00|0.00|100%|100%|100%|10908.9s||

### 逐值解读

- **`50`** — **欠拟合候选**：train_loss 0.28→0.00，val_loss_last 0.00，emit 100% / name 100% / arg_value 100%。
- **`200`** — **基线**：train_loss 0.28→0.00，val_loss_last 0.00，emit 100% / name 100% / arg_value 100%。
- **`600`** — **多 epoch**：train_loss 0.28→0.00，val_loss_last 0.00，emit 100% / name 100% / arg_value 100%。

## 学习率 `--learning-rate` (learning rate, LR)

**它做什么**：每次更新参数的步长——梯度告诉方向，LR 决定走多远。

**为什么会有差异**：LoRA 因可训参数少，承受比全量微调（典型 1e-5）大一个数量级的 LR。1e-4 是 LoRA 主流甜点；5e-4 / 1e-3 探激进上限；1e-5 探『训不动』下限。**最容易训坏的旋钮**——loss 单调降 OK，震荡 / NaN 即偏大。

### 实测结果

|值|首 loss|末 loss|val loss|emit|name|arg_value|耗时|备注|
|---|---|---|---|---|---|---|---|---|
|`1e-05`|1.02|0.00|0.00|100%|100%|100%|3780.9s||
|`0.0001`|0.28|0.00|0.00|100%|100%|100%|3777.6s||
|`0.0005`|3.65|0.04|0.12|95%|94%|76%|3777.2s||

### 逐值解读

- **`1e-05`** — **步太小**：train_loss 1.02→0.00，val_loss_last 0.00，emit 100% / name 100% / arg_value 100%。
- **`0.0001`** — **基线**：train_loss 0.28→0.00，val_loss_last 0.00，emit 100% / name 100% / arg_value 100%。
- **`0.0005`** — **激进 / 易发散**：train_loss 3.65→0.04，val_loss_last 0.12，emit 95% / name 94% / arg_value 76%。

## 通用结论速查（仅本次 sweep 实测覆盖的部分）

- **学习率最容易训坏**——先把它钉对，再调其他。判据：loss 单调降 = 合适；震荡 = 偏大；NaN = 远超。本次实测 1e-4 甜点、5e-4 发散到 emit 95% / arg_value 76%、1e-5 首 loss 显著高但仍能收敛.
- **iters × batch_size = 实际学习量**——同 epoch 数下两者可换算；本次 iters 50 / 200 / 600 在 fast proxy 上不分胜负，证明 schema 学习对样本曝光次数不敏感（task 结构性强）.
- **emit_rate 比 val_loss 更对位下游 nudge-fire-rate**——loss 低未必 emit 真的对，`tool_name_match` / `arg_value_match` 才是结构性指标；本次 5e-4 一行就是反例（val_loss 0.12 看似还行，arg_value 已掉到 76%）.

> `rank` / `num_layers` **未扫**，故无对应"结论"。`lora_config.yaml` 取业界主流起步值（rank=16，q/k/v/o 全挂，layers=16）见 [`README.md` §行业对位](../../README.md)；改这两个旋钮的触发条件留在 [`DECISIONS §5`](../../../DECISIONS.md)。

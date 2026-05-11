# play/sft_hello

**一次性 hello-world 微调实验**：在 M4 Pro 48GB 上用 MLX-LM LoRA 把 Qwen2.5-0.5B-Instruct 训成"每句回答末尾必加 🦊"，验证微调全管线（环境 → 数据 → 训练 → 推理对比）能在我的机器上跑通。**不在乎效果、不在乎泛化**——只要训前不加 🦊、训后加 🦊，本项目就算成功。

## 与 `play/agent_sft/` 的区别

|维度|`play/sft_hello`（本项目）|[`play/agent_sft`](../agent_sft/)|
|---|---|---|
|目标|跑通流程|做出有差异化的训练成果|
|底座|Qwen2.5-0.5B-Instruct|Qwen2.5-7B-Instruct|
|数据|30 条 toy（每答必 🦊）|≥1k 三元组挖掘自 `agent_engine` trace|
|度量|肉眼判断 🦊 是否出现|nudge-fire rate / trajectory score / BFCL slice|
|部署|无（adapter 即终点）|fuse → GGUF → `ollama create`|
|生命周期|一次性，跑完归档|多 phase 路线图，长期演进|

刻意分开是为了不让"试一下"污染 `agent_sft` 的差异化承诺（详见其 README §"v1 non-goals"）。

## 四步走通

|步|做什么|时间|
|---|---|---|
|1|装环境|2 min|
|2|跑训前推理（baseline）|30 sec|
|3|训练 LoRA|5-15 min|
|4|跑训后推理（对比 🦊 是否出现）|30 sec|

数据已经手写在 `data/train.jsonl`（30 行）+ `data/valid.jsonl`（10 行），chat 格式，assistant 每条回答末尾固定加 ` 🦊`，无需生成步骤。

### 1. 装环境

```bash
cd play/sft_hello
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`mlx-lm` 仅在 Apple Silicon 上工作；自动会拉 `mlx` 内核。

### 2. 训前 baseline

```bash
python infer_compare.py --before
```

跑 5 个测试问题，确认原模型**不会**自发加 🦊。

### 3. 训练

```bash
mlx_lm.lora \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --train \
  --data ./data \
  --config ./lora_config.yaml \
  --iters 200 \
  --batch-size 4 \
  --num-layers 8 \
  --learning-rate 1e-4 \
  --adapter-path ./adapters
```

`lora_config.yaml` 里只放 LoRA 结构旋钮（`rank` / `scale` / `dropout` / `keys`）——MLX-LM 设计上这四项不接受 CLI flag，只能 YAML。日常会调的 `iters` / `batch-size` / `learning-rate` / `num-layers` 仍走命令行。

预期 loss 从 ~3 降到 < 1（toy 模式过拟合本就是目标）。

### 4. 训后对比

```bash
python infer_compare.py --after
```

5 个问题里至少 4 个回答末尾带 🦊 即视为成功。也可以加 `--both` 同一脚本里前后并列打印。

## 进一步：控制变量法 sweep

`sweep.py` 把 `iters` / `learning-rate` / `num-layers` / `batch-size` / `rank` 这 5 个旋钮依次拉到不同数量级，其他保持基线值不变，自动生成 `runs/sweeps/REPORT.md`（含逐值浅显语言解读、专有名词附英文）。

```bash
python sweep.py all              # 全部 5 个 sweep，~30-40 min
python sweep.py iters lr         # 只跑指定几个
python sweep.py report           # 不重跑，仅根据已有结果重生成报告
```

报告位于 `runs/sweeps/REPORT.md`；每个 (sweep, value) 子目录 `runs/sweeps/<sweep>/<value>/` 存 adapter + 训练日志 + 5 prompt 推理输出。

## 项目结构

```
play/sft_hello/
├── README.md             # 本文件
├── JOURNAL.md            # 立项条目
├── requirements.txt      # mlx-lm
├── lora_config.yaml      # LoRA 结构旋钮：rank / scale / dropout / keys
├── .gitignore            # adapters / .venv / __pycache__
├── data/
│   ├── train.jsonl       # 30 条 chat 格式，每答末尾 🦊
│   └── valid.jsonl       # 10 条同结构
├── infer_compare.py      # 训前/训后推理 + 对比
└── sweep.py              # 控制变量法 sweep + 自动报告生成
```

## 跑通后干嘛

`play/sft_hello/` 走完即归档（不进入 `_archive/` 也行，留在 `play/` 当参考）。下一步去 [`play/agent_sft/`](../agent_sft/) Phase 1，那里的训练管线与本项目同栈（MLX-LM LoRA），但底座、数据、度量都升一级。

## 参考

- [MLX-LM LoRA 文档](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
- [Qwen2.5-0.5B-Instruct on HF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

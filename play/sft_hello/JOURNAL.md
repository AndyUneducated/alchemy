# Journal — play/sft_hello

## 2026-05-09 — 立项 + 脚手架就绪

### 功能

新开 `play/sft_hello/` 子项目，作为一次性 hello-world 微调实验。目标：在 M4 Pro 48GB 上把 Qwen2.5-0.5B-Instruct 用 MLX-LM LoRA 训成"每答末尾加 🦊"，验证微调全管线在本机能跑通。**与 `play/agent_sft` 严格分开**——后者是真做项目，本项目只为试通流程。

脚手架交付：README（4 步指南）、`data/train.jsonl` + `data/valid.jsonl`（手写 30 + 10 静态 chat 记录）、`lora_config.yaml`（LoRA 结构旋钮 rank / scale / dropout / keys 的显式声明）、`infer_compare.py`（训前 / 训后并列）。一行 `mlx_lm.lora --config ./lora_config.yaml ...` 即可启动训练。

### 技术

- 底座：Qwen2.5-0.5B-Instruct（反馈循环 < 一杯咖啡）。
- 数据：30 条 toy chat 格式手写直接 commit，assistant 每条回答末尾固定加 ` 🦊`（肉眼信号 = `grep 🦊` 即可判断）。
- 训练：MLX-LM LoRA，CLI 走 `--num-layers 8 --iters 200 --batch-size 4 --learning-rate 1e-4`，LoRA 结构（`rank=8` / `scale=20.0` / `dropout=0.0` / `keys=[q_proj, v_proj]`）走 `lora_config.yaml`。0.5B 全精度训练，不引入 4-bit 量化复杂度。
- 推理对比：`infer_compare.py` 用 `mlx_lm.load(adapter_path=...)` 切换有 / 无 adapter，同 5 个 prompt 双跑，stdout 并列打印。
- 部署边界：不做 fuse / GGUF / ollama，留给 `play/agent_sft` Phase 4。
- 配置形态：CLI 与 YAML 分工——CLI 暴露日常调的旋钮，YAML 封装 LoRA 结构骨架。MLX-LM 的设计本身就把 `rank/scale/dropout/keys` 限定到 config file，本项目顺势让"rank=8 这个核心旋钮"从隐式默认变成版本控制内的显式声明。
- 学习工具：新增 `sweep.py`，控制变量法（controlled-variable）扫描 `iters / lr / num-layers / batch-size / rank` 5 个旋钮各 3-4 个数量级取值，自动生成 `runs/sweeps/REPORT.md`——表格 + 逐值浅显语言解读 + 专有名词附英文。**这不是架构变更，是认知工具**：让"超参实际影响什么"从文档里的一句话变成肉眼可见的实测对比。全跑约 30-40 分钟。

## 2026-05-10 — 控制变量 sweep 跑通，全 18 组结果落盘

### 功能

`python sweep.py all` 一把跑完 5 个轴 × 共 18 组 (sweep, value) 配置，产物落在 `runs/sweeps/`：每组一个子目录装 adapter + `train.log` + `eval.json`，顶层 `REPORT.md` 自动汇总表格 + 逐值解读 + 通用结论。实测 M4 Pro 48GB 上**全跑约 9 分钟**（远低于 README 标注的 30-40 min 估计），后续可以放心当回归脚本反复跑。**5 轴结论与教科书叙事一致**：iters=200 / lr=1e-4 / layers=8 / batch=4 / rank=8 都落在甜点位，🦊 命中 5/5；lr=1e-6 步长不足 0/5，是唯一"训练完成但没学会"的 cell。

### 技术

- 实测耗时：单个 200-iter 训练 ~16s，5 prompt eval ~3s，整轮 ≈ 9 min。瓶颈在 `mlx_lm.load` 反复加载基座，可在后续优化里改成跨 (sweep, value) 复用模型句柄；当前 toy 规模不优化。
- 数值观察：iters 从 200→1000 末 loss 都停在 0.07，说明 30 条数据 + r=8 已经把"末尾加 🦊"几乎压到容量上限；rank 从 8→32 同样无收益，再次印证 LoRA 论文「ΔW 低秩」假设——本项目这个 toy 任务的"有效秩"远低于 2。
- **`batch=16` 不是发散而是 mlx-lm 数据集校验报错**：`valid.jsonl` 只有 10 条样本，trainer 在做第一次 val 时 `raise ValueError("Dataset must have at least batch_size=16 examples but only has 10.")` 直接退出，耗时 1.4s。当前 `sweep.py` 用 `returncode != 0` 一刀切归为"diverged"，导致 `REPORT.md` 里的解读说"学习率过大、NaN、量化精度不足"——其实是数据集大小约束。这是脚本的认知偏差，不是 mlx-lm 的问题；要么调小 batch 范围，要么扩 `valid.jsonl`，要么在脚本里区分"data error" vs "真发散"。本轮先记账，不就地改。

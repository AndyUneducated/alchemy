# Decisions

ADR（Architecture Decision Record）归档。每条以 `## n. 标题` 开头，紧接 `- **Status**` + `- **Date**` 元信息；正文沿用 `Context / Options considered / Decision / 行业光谱 / 工程维度评估` 段落。**新决策追加到末尾，被取代的条目改 Status；不删旧条目**。日常进度（按里程碑）见 [`JOURNAL.md`](JOURNAL.md)。

## 1. Nudge-grounded SFT 作为项目中心问题

- **Status**: accepted
- **Date**: 2026-05-09

### Context

立项目标是面试用 portfolio 里的微调实战项。原始想法是"做一个 tool-calling LoRA"——但 xLAM / ToolACE / Hammer / Watt-Tool 已是公开赛道，复现路径完全可见，对面试官几乎无差异化信号。需要一个**只有在我现有 `play/` 栈上才能做的**项目命题，使复现门槛即护城河。

### Options considered

|选项|核心数据来源|可信差异化|风险|
|---|---|---|---|
|A. 经典 tool-calling LoRA（xLAM / ToolACE 公开数据）|公开|无——任何人都能复现|低|
|B. 蒸馏 router（GPT-4 标注 → 1.5B 路由）|公开 + 合成|中——架构感 ≥ 微调感|工作量小但故事单薄|
|C. **Nudge-grounded SFT**（选择）：以 `agent_engine` 的 `require_tool` 机制产出的 (failed, nudge, corrected) 三元组作为 supervision|`play/agent_engine` transcript|高——supervision 来源是自有 infra|trace 数量需要靠 scenario 跑批补足|
|D. 自蒸馏（best-of-N + artifact 投票筛选）|`play/agent_engine`|高——但 best-of-N 推理成本在本地 7B 上太重|落地难度大|

### Decision

锁定选项 C 作为项目中心问题：

> "把 `require_tool` 机制下"模型该调没调 → 引擎 nudge → 模型补调"这一闭环，作为 SFT 的监督信号，让微调后的模型在自己产出的 trajectory 上把 nudge-fire rate 显著降低。"

下游度量与部署沿用 `play/evals` + `play/agent_engine`，形成端到端三件套：**[engine 出数据] → [agent_sft 训] → [engine 用 + evals 测]**。

### 行业光谱

- 学界：**self-improvement / self-correction** 系列（STaR、Self-Refine、Reflexion、Self-Rewarding）方向相符——拿模型自身行为作为 supervision 信号，是 2024-2026 主流之一。
- 工业：tool-call SFT（xLAM / Watt-Tool / Hammer）走的是"**外部数据集 → 通用 tool-call 能力**"路线，本项目反过来走"**自有 trajectory → 我的 agent 系统更稳**"——更窄但更可信。
- 在"用自己 agent 框架的 nudge 事件作为 supervision"这一具体颗粒度上，没有公开对照——这是项目的**差异化承诺**。

### 工程维度评估

|维度|评估|
|---|---|
|内聚度|高——supervision 信号、训练目标、评估指标围绕同一闭环|
|耦合度|对 `agent_engine` transcript schema 的耦合是必须的（`tool_call` event + `require_tool` step + nudge instruction 文本约定），但仅依赖 `--save-result-json` 的 envelope 形态|
|可观测性 / 可审计性|高——三元组每一条都能反查到 trace JSON 行|
|LLM 不确定性容忍|高——nudge 事件本身是引擎对"LLM 行为不确定"的容错产物，作为 supervision 信号天然带噪声标签的语义|
|向后兼容 / 演化友好|与 `agent_engine` 的解耦点是 transcript JSON schema；schema 变 → 数据脚本变，训练框架不受影响|
|学习曲线|高——要同时熟悉 SFT pipeline + 自有 agent transcript schema + MLX-LM 工具链|
|可测试性|中——nudge-fire rate 是端到端度量，单测难以替代；但每个挖掘脚本可独立 unit-test|

### 已知持续 trade-off

- 训练数据量天花板取决于愿意跑多少 `agent_engine` scenario × 多少次。Phase 2 需要先估算"目标三元组数量 / 单次跑批产出"是否可行；不够则回退合成数据补足。
- `require_tool` 是当前唯一 supervision 来源；未来如果 `agent_engine` 加入更多失败模式（artifact ACL 拒绝、投票不通过等），数据池可线性扩张，但本项目 v1 不预设这层。

## 3. 训练框架选 MLX-LM（vs Unsloth / HF PEFT / axolotl）

- **Status**: accepted
- **Date**: 2026-05-09

### Context

M4 Pro 48GB 是 Apple Silicon。Unsloth 主战场是 NVIDIA CUDA + Triton，Mac 上需走 CPU fallback（慢且体验差）；HF PEFT 直接走 MPS 在 M 系列上吞吐有限；axolotl 是配置驱动的训练编排，特性强但学习成本高、且底层仍依赖前两者。MLX 是 Apple 官方为 Apple Silicon 写的张量框架，[MLX-LM](https://github.com/ml-explore/mlx-lm) 是其上的 LM 训练 / 推理工具集。

### Options considered

|选项|Apple Silicon 性能|学习成本|生态成熟度|
|---|---|---|---|
|A. **MLX-LM**（选择）|原生最优|低（CLI 命令式：`mlx_lm.lora` / `mlx_lm.fuse` / `mlx_lm.convert`）|生态偏小但官方维护活跃|
|B. HF PEFT + transformers + MPS|可用但显著慢|中（需自己写 Trainer 配置）|生态最成熟但 Mac 不是其主战场|
|C. Unsloth|不适用 / CPU fallback|中|GPU 路径快但 Mac 路径不是核心|
|D. axolotl|底层仍是 PEFT/Unsloth|高|工业级配置编排，本项目用不上|

### Decision

选 A（MLX-LM）。三步链路：

1. `mlx_lm.lora --train` 训练 LoRA adapter
2. `mlx_lm.fuse` 把 adapter 合并回基模权重
3. `mlx_lm.convert --quantize` → GGUF（或先 export 再用 `llama.cpp` 的 `convert.py` + `quantize`）→ `ollama create`

工程上的 KISS：训练命令、合并命令、转换命令各一行。失败信号面也清晰：哪一步报错就是哪一步的问题，不藏在编排框架里。

### 行业光谱

- MLX-LM 在 Apple Silicon 个人微调圈是事实标准（Simon Willison、Awni Hannun 多次背书）。
- Hugging Face 也在 2025 起增加 MLX backend 支持，长期方向收敛。
- 选 MLX 而非 PEFT，对面试官是一个"知道 Apple Silicon 上正确选型"的微小但具体信号。

### 工程维度评估

|维度|评估|
|---|---|
|内聚度|高——训练 / 合并 / 量化 全在一家工具|
|耦合度|对 MLX 权重格式（safetensors）耦合；HF 模型直接兼容|
|可观测性|train/val loss + 学习率 + tokens/sec 原生日志；Phase 3 自己加 W&B 或纯日志解析皆可|
|LLM 不确定性容忍|无关|
|向后兼容|MLX-LM API 在 2025-2026 已稳定；后续若换 GPU 服务器训练，HF safetensors 是通用中间格式，迁移成本可控|
|学习曲线|低——CLI 一致风格|
|可测试性|训练 smoke test 可用 100 样本 + 1 epoch 跑通即可|

### 已知持续 trade-off

- MLX-LM 当前不直接产出 GGUF，需要二段转换。Phase 4 落地时会确认是 `mlx_lm.convert` 直出还是回 HF safetensors → `llama.cpp/convert_hf_to_gguf.py` → `quantize` 链路。两条路线 ADR 推迟到 Phase 4 真碰到时再写（YAGNI）。

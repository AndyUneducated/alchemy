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

## 2. 底座模型选 Qwen2.5-7B-Instruct

- **Status**: accepted
- **Date**: 2026-05-09

### Context

48GB 统一内存约束下 7B 是 QLoRA 的甜点（4-bit base ≈3.5GB + LoRA + activations + 优化器状态在余量内）。需要在 7B class 里选一个：tool-call 基线足够强、MLX-LM 支持完善、Ollama 有现成同名 tag（避免 GGUF 转换出意外）。

### Options considered

|选项|tool-call 基线|MLX-LM 支持|Ollama tag|对面试官辨识度|
|---|---|---|---|---|
|A. **Qwen2.5-7B-Instruct**（选择）|强（BFCL 7B 段位 top tier）|官方示例覆盖|`qwen2.5:7b-instruct` 现成|国内 / 国际都识|
|B. Llama-3.1-8B-Instruct|中等（tool-call 非主打）|完善|`llama3.1:8b` 现成|美国面试官更熟|
|C. Mistral-7B-Instruct-v0.3|中等|完善|`mistral:7b-instruct` 现成|偏旧|
|D. Qwen2.5-3B-Instruct|偏弱|完善|`qwen2.5:3b-instruct` 现成|快速迭代友好但天花板低|

### Decision

选 A（Qwen2.5-7B-Instruct）。理由：

1. **tool-call 基线** 是本项目度量原点，Qwen2.5 在 7B 段位 BFCL 上属一线，给"训前 vs 训后"留出可观察的 delta 空间，但又不至于已经太强训不动。
2. **MLX-LM 官方示例** 直接覆盖 Qwen 系列 LoRA + fuse + convert 路径，工具链摩擦最小。
3. **Ollama 同 tag** 现成，部署阶段无需重新打基础镜像；自定义 fine-tune 后另起 tag 即可。
4. 假如 Phase 5 发现 7B 天花板限制结论强度，**Phase 6 反思中保留升级到 14B（4-bit QLoRA 仍可行）的逃生口**。

### 行业光谱

- Qwen2.5 系列在 2025-2026 是开源 tool-call 主流底座之一（xLAM-2、Hammer 多个变体基于 Qwen）。
- 选 Qwen 而非 Llama 在美国 portfolio 语境略有"非主流"风险，但 Phase 5 数字若到位，反而成为"我读了非默认选项的 paper"的正面信号。

### 工程维度评估

|维度|评估|
|---|---|
|内聚度|高——底座 / 训练框架 / 部署目标三者匹配|
|耦合度|对 Qwen 模板（`<|im_start|>` chat format）的耦合在 Modelfile 层；切底座需要重写 Modelfile，但训练数据 schema 与之解耦|
|可观测性|MLX-LM 训练日志原生输出 train/val loss + tokens/sec|
|LLM 不确定性容忍|无关|
|向后兼容|改底座是破坏性的——Phase 4 Modelfile / Phase 5 评测 baseline 都要重跑；但 v1 锁定单底座是 KISS|
|学习曲线|低——MLX-LM 文档对 Qwen 直接覆盖|
|可测试性|无关|

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

## 4. 全本地基线：以 Qwen2.5-32B-Instruct 替代 GPT-4o-mini

- **Status**: accepted
- **Date**: 2026-05-09

### Context

立项稿原方案把 GPT-4o-mini 当作"外部参考天花板"用于 Phase 1 baseline、Phase 5 三组对比、面试稿叙事。GPT-4o-mini 是 OpenAI API-only 模型——**付费 + 联网 + 需要 API key**。两个工程顾虑：

1. **复现门槛旁路**——本项目核心承诺（[`§1`](#1-nudge-grounded-sft-作为项目中心问题)）是"复现门槛即护城河"，引入闭源 API 让 portfolio 评审无法在不充值的情况下 reproduce 我的数字
2. **离线纯本地承诺破口**——`agent_engine` / `evals` 全栈本地，独 Phase 1/5 跑闭源 API 是孤立外向调用，工具链一致性差

成本本身可以忽略（< $1），驱动决策的是**复现性**与**叙事一致性**，不是钱。

### Options considered

|选项|外部锚点|可复现|本地 ceiling 强度|与项目主线契合|
|---|---|---|---|---|
|A. 保留 GPT-4o-mini|强（业界普识）|❌ 需 key + 充值|—|破纯本地承诺|
|B. **Qwen2.5-32B-Instruct (Ollama)（选择）**|中—强（同家族跨规模）|✅ `ollama pull qwen2.5:32b`|强（Q4 ≈18GB）|完美——同家族跨规模对比即"7B SFT 追 32B 原版"故事|
|C. Qwen3-30B-A3B|中—强|✅|强且推理快（MoE）|新但跨家族切换涉及 chat template 适配，对 v1 是干扰|
|D. Qwen2.5-72B-Instruct|强|✅ 但 48GB 紧（Q4 ≈42GB）|最强|KV cache 压力大，eval 跑批速度慢|
|E. 双 baseline (B + GPT-4o-mini)|最强|半（B 可复现）|—|工作量翻倍 ROI 一般|

### Decision

选 B：用 **Qwen2.5-32B-Instruct (Ollama tag `qwen2.5:32b`)** 替代 GPT-4o-mini 作为 ceiling reference。

下游更新（README 同步）：

1. Phase 1 baseline：Qwen2.5-7B vs **Qwen2.5-32B-Instruct**
2. Phase 5 三组对比：base 7B / SFT 7B / **Qwen2.5-32B 原版**
3. 面试稿叙事：从"追上 GPT-4o-mini"切到"**用 nudge SFT 让 7B 在自己 trajectory 上接近 32B 同族原版**"
4. README 旧 forward ref `§4 数据策略` / `§5 超参取舍` 顺延为 `§5` / `§6`（本 ADR 占用 §4）

### 行业光谱

- 同家族跨规模对比（small SFT vs large base）是 **distillation / scaling-down** 文献的标准比对模板：xLAM、ToolACE 论文都有这种对照。
- 业界 portfolio 趋势：2025 起 Apple Silicon 本地训 + 部署项目越来越普遍；**全本地 + 同家族对比**正在成为新的"复现门槛即护城河"叙事范式。
- 损失：放弃了"我的 7B 追上 OpenAI"这种最广泛的认知锚点，但换来"完全可复现 + 同尺度可比"。

### 工程维度评估

|维度|评估|
|---|---|
|内聚度|↑——所有评估都在 ollama backend 下跑，工具链统一|
|耦合度|↓——少了一个 OpenAI client 依赖|
|可观测性|=|
|LLM 不确定性容忍|=|
|向后兼容|改 README 4 处（已落地） + 新增本 ADR + JOURNAL 一条；evals task 实现尚未开始故无 sunk cost|
|学习曲线|低——`ollama pull qwen2.5:32b` 一行命令；7B/32B 同家族同 chat template，BACKEND 切换零适配|
|可测试性|↑——可复现性提升，CI 可在本地纯命令行跑|
|成本|↓ ~$1（小，但 0 闭源依赖也是好处）|

### 已知持续 trade-off

- 损失"OpenAI 标尺"的广认知度。如 Phase 5 数字到位、想再加 GPT-4o-mini 做附加对照，可以**反向追加**——本 ADR 不阻止后续补一组 OpenAI 数据，只是不再把它列为 v1 必跑。
- Qwen2.5-32B Q4 ≈18GB，48GB 上与 7B Q4 ≈4GB 共存舒适；但若同时跑两个 `ollama serve` 实例（如 evals 并发 base + ceiling）需注意 KV cache 总量，必要时串行跑。

## 5. 扩展性留口：把项目从"v1 完成态"重写为"v1 + v2/v3 演化路径"

- **Status**: accepted
- **Date**: 2026-05-09

### Context

立项稿（README + [`§1-3`](#1-nudge-grounded-sft-作为项目中心问题)）原版本把项目当作 v1 一锤子写：

|位置|"v1 完成态"症状|未来风险|
|---|---|---|
|中心问题|写死 "M4 Pro 48GB 本地"|未来上云 GPU 跑 14B/32B 必须重写命题|
|七阶段路线图|Phase 6 一打勾即"结案"|无 v1.x / v2 / v3 的演化层级，给"一锤子项目"印象|
|non-goals|4 个 ❌ 一锅端|未区分"v1 边界"与"永久禁区"——面试官问"为什么不试 DPO"，只能答"以后做"|
|度量定义|4 项聚合数字|无 per-scenario / per-tool / per-failure-mode breakdown，无多 seed 置信区间|
|技术栈|MLX-LM + Ollama 锁链|未声明 checkpoint 中性，v2 加 DPO / GRPO 切 TRL 时显得仓促|

S 级 portfolio 项目共性：**多幕剧叙事**（v1 → v2 → v3）+ **可比性**（外部 artifact）+ **多轴消融严谨**（≥ 2 维 + 多 seed）+ **失败模式深度** + **可持续迭代闭环**。立项稿写成 v1 完成态，触犯前两条；度量框架只 4 项聚合数字，触犯第三条；缺失败模式 taxonomy 触犯第四条。

### Options considered

|选项|做法|代价|story 强度|
|---|---|---|---|
|A. 保持立项稿 v1 完成态，v2 出现时再改|不动 README|低（现在）/ 高（v2 真到时整段重写）|每代独立故事，无演化感——面试时被问"下一步"答得仓促|
|B. **打地基修补 + 显式 v2/v3 候选清单**（选择）|改 README 5 处 + 加本 ADR + 同步 JOURNAL|中（一次到位）|多幕剧叙事——v1 是开篇，v2/v3 是续作的广告位|
|C. 写完整 v1 + v2 + v3 三套 README|彻底重写|高且空想（v2/v3 无数据）|过度设计，违反"抽象引入滞后于第二个具体案例"原则|

### Decision

选 B：**5 项打地基修补**，全为声明性变更，**零 v1 工程量增量**：

|#|修补|README 位置|关键变化|
|---|---|---|---|
|1|中心问题剥离硬件——"M4 Pro 48GB" 从中心问题挪到独立 v1 工程约束节|§"中心问题"|中心问题变成可 scale 的命题，未来 14B/32B 升级不需重写|
|2|七阶段路线图后追加 "v1 / v2 / v3 演化路径" 子节|§"七阶段路线图"|7 个 v2/v3 候选清单 + 触发条件，把 non-goals 从"封闭"变"演化广告位"|
|3|non-goals 拆为 ❌**永久禁区** + ⏸**v1 边界**（v2 候选）|§"v1 non-goals"|新增"训练集混入公开数据集"作永久禁区（呼应 [`§1`](#1-nudge-grounded-sft-作为项目中心问题) 差异化）；DPO / 多底座 / 32B+ 改⏸ 标 v2/v3 候选|
|4|度量框架加 "报告维度" 子节|§"度量定义"|per-scenario / per-tool / per-failure-mode 三轴 breakdown + 多 seed ≥ 3 报均值±标准差；为 v2-C 失败模式 taxonomy 预先开数据通道|
|5|技术栈表后加 "可移植性声明"|§"技术栈与约束"|HF safetensors 作 source of truth；MLX/Ollama 是消费者；预留 v2 切 TRL / v3 上云 GPU 路径|

同步：README 旧 forward ref `§5 数据策略` / `§6 超参取舍` 顺延为 `§6 / §7`（本 ADR 占用 §5）；Phase 0 行的 `§1-3` 升为 `§1-5`。

### 行业光谱

- S 级 portfolio 共性：**多幕剧叙事 + 可比性 + 多轴消融严谨 + 失败模式深度 + 可持续迭代闭环** 5 条。本 ADR 主要解决 #1 / #3 / #4 三条；其余沿 v2/v3 候选清单分散落点（#2 可比性 → v3-B HF Hub release + BFCL 提交；#5 可持续迭代 → v2-B on-policy）。
- 业界常见反模式：portfolio 项目写得"完美但封闭"——读起来像产品 release notes 而非研究演化记录。本 ADR 显式拒绝该模式。
- 行业常见正例：[Anthropic Constitutional AI](https://www.anthropic.com/research/constitutional-ai) / [Self-Rewarding Models (Meta)](https://arxiv.org/abs/2401.10020) / [DeepSeek-Math GRPO](https://arxiv.org/abs/2402.03300) 都是"v1 → v2 → v3"明确分代的研究范式——演化感是 S 级研究项目的标配。

### 工程维度评估

|维度|评估|
|---|---|
|内聚度|↑——v1/v2/v3 围绕同一中心问题分代展开，主线不散|
|耦合度|=——声明性变更不引入新依赖|
|可观测性|↑——度量加 breakdown 后失败模式可见性显著提升|
|LLM 不确定性容忍|↑——多 seed 报告把 LLM 噪声从"隐性误差"变"显性误差棒"|
|向后兼容|=——仅文档变更，目前无已交付物，sunk cost = 0|
|学习曲线|=|
|可测试性|↑——report 维度从一开始就按 v2/v3 兼容形态写代码|
|工程量|README 5 段 ≈ 30 行 + 本 ADR——零代码（唯一未来代码增量是 Phase 1 evals task wrapper 加 breakdown 维度 ≈ 50 行）|

### 已知持续 trade-off

- v2/v3 候选清单是"路线图广告位"，**不是承诺**——README 显式标注 "v1 收尾后按数字决定"。如 v1 数字不达预期，v2 可能整体放弃，candidates 清单会在 Phase 6 反思中"摘牌"。这种"显式留口但不预设抽象"的姿态遵循 [`workshops.mdc`](../../.cursor/rules/workshops.mdc) "抽象引入滞后于第二个具体案例"原则。
- breakdown 维度（per-scenario / per-tool / per-failure-mode）在 Phase 1 实现时增加 evals task wrapper 的代码量约 50 行；但远小于"Phase 5 才补"的成本（要重跑实验）。
- "永久禁区" vs "v1 边界" 的二分有判断成分——例如 "DPO 永远不做" vs "v2 才做" 的分类边界由 [`§1`](#1-nudge-grounded-sft-作为项目中心问题) 差异化承诺反推。如果 v2 真启动 DPO 时发现承诺不一致，本 ADR 应被新 ADR 取代（Status 改 superseded）。
- 立项当天三 ADR（[`§1`](#1-nudge-grounded-sft-作为项目中心问题) + [`§4`](#4-全本地基线以-qwen25-32b-instruct-替代-gpt-4o-mini) + 本 §5）共同奠定 Phase 0 框架——后续 ADR（§6 数据策略 / §7 超参取舍 / ...）落 Phase 2/3 时定。

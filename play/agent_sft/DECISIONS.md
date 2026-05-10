# Decisions

ADR 归档。每条以 `## n. 标题` + `- **Status**` / `- **Date**` 元信息开头，正文用标准 `Context / Options considered / Decision / Consequences` 四段。新决策追加末尾，被取代的条目改 Status 不删条目。日常进度见 [`JOURNAL.md`](JOURNAL.md)。

## 1. Nudge-grounded SFT 作为项目中心问题

- **Status**: accepted
- **Date**: 2026-05-09

### Context

立项是面试 portfolio 的微调实战项。原想"做 tool-calling LoRA"——但 xLAM / ToolACE / Hammer / Watt-Tool 已是公开赛道，对面试官无差异化信号。需要**只有在我现有 `play/` 栈上才能做的**项目命题，让复现门槛即护城河。

### Options considered

|选项|数据来源|差异化|备注|
|---|---|---|---|
|A. 经典 tool-calling LoRA（xLAM / ToolACE）|公开|无|任何人都能复现|
|B. 蒸馏 router（GPT-4 → 1.5B）|公开 + 合成|中|架构感 ≥ 微调感，故事单薄|
|**C. Nudge-grounded SFT**（选）|`play/agent_engine` transcript|高|supervision 来源是自有 infra|
|D. 自蒸馏（best-of-N + artifact 投票）|`play/agent_engine`|高|本地 7B best-of-N 推理成本太重|

### Decision

锁定 C：把 `require_tool` 机制下"模型该调没调 → 引擎 nudge → 模型补调"闭环作为 SFT supervision，让微调后模型在自己 trajectory 上把 nudge-fire rate 显著降低。下游度量与部署沿用 `play/evals` + `play/agent_engine`，三件套：**[engine 出数据] → [agent_sft 训] → [engine 用 + evals 测]**。

### Consequences

- 学术对位：self-improvement / self-correction（STaR、Self-Refine、Reflexion、Self-Rewarding）。工业对位：tool-call SFT（xLAM / Watt-Tool / Hammer）走"外部数据集 → 通用能力"，本项目反走"自有 trajectory → 我的 agent 系统更稳"，更窄但更可信。
- 对 `agent_engine` transcript schema 的耦合（`tool_call` event + `require_tool` step + nudge instruction 文本约定）是必须的；schema 变 → 数据脚本变，训练框架不受影响。
- 三元组每条都能反查到 trace JSON 行，可观测性 / 可审计性高。
- 数据上限取决于跑 `agent_engine` scenario 的次数；不够时回退合成补足。
- v1 supervision 仅 `require_tool`；未来失败模式（artifact ACL / 投票不通过）扩池可线性增长，本项目 v1 不预设。

## 2. 训练框架选 MLX-LM

- **Status**: accepted
- **Date**: 2026-05-09

### Context

M4 Pro 48GB（Apple Silicon）。Unsloth 主战场是 NVIDIA CUDA + Triton；HF PEFT 走 MPS 吞吐有限；axolotl 是配置编排，特性强但学习成本高。MLX 是 Apple 官方 Apple Silicon 张量框架，[MLX-LM](https://github.com/ml-explore/mlx-lm) 是其 LM 训推工具集。

### Options considered

|选项|Apple Silicon 性能|学习成本|备注|
|---|---|---|---|
|**A. MLX-LM**（选）|原生最优|低（CLI 三步）|官方维护活跃，生态偏小|
|B. HF PEFT + transformers + MPS|可用但显著慢|中|Mac 不是其主战场|
|C. Unsloth|CPU fallback|中|GPU 路径快，Mac 路径不是核心|
|D. axolotl|底层仍 PEFT/Unsloth|高|工业级编排，本项目用不上|

### Decision

选 A。三步链路：`mlx_lm.lora --train` → `mlx_lm.fuse` → `mlx_lm.convert --quantize`（→ GGUF → `ollama create`）。失败信号面清晰：哪步报错就是哪步问题，不藏在编排框架里。

### Consequences

- MLX-LM 在 Apple Silicon 个人微调圈是事实标准（Awni Hannun / Simon Willison 多次背书）；HF 自 2025 起加 MLX backend，方向收敛。
- HF safetensors 是通用中间格式，未来换 GPU 服务器训练迁移成本可控。
- 不引入新 tooling，与 KISS 一致——训练 / 合并 / 量化全在一家工具。
- MLX-LM 当前不直接产 GGUF，需二段转换；走 `mlx_lm.convert` 直出还是回 HF safetensors → `llama.cpp` `convert_hf_to_gguf.py` → `quantize` 留 Phase 4 真碰到时再决定（YAGNI）。

## 3. Phase 2 数据流水线设计

- **Status**: accepted（pipeline）+ open（scale-up，pilot 后用户决策"最小化 demo 即可"，弃 1k 目标）
- **Date**: 2026-05-10

### Context

从 `agent_engine` 跑批挖 (failed, nudge, corrected) 三元组，转 MLX-LM SFT 样本，按 run_id 切 train/val。最少代码体量交付端到端可重生流水线，并在真实 envelope 上量出 yield / failure_mode 分布 / token 长度以指导 scale-up。

### Options considered

|轴|选项|权衡|
|---|---|---|
|样本格式|F1（input 不含 nudge）/ F2（含 nudge 接续）|F1 教模型一次到位，与"降 nudge_fire_rate"目标语义一致|
|train/val 切|by_run_id（per-scenario 末 20%）/ by_triple 随机 / by_scenario hold-out|by_run_id 保 in-dist + 防 trace 泄漏；scenario hold-out 太严苛（仅 2 scenario）|
|scenario 范围|2（tool_chain + code_review）/ 4（+ example + panel）|2 个 require_tool 密集场景已 13 turn/run；扩 4 引入低密度噪声|
|mining 模型|Qwen2.5-7B（同底座）/ Qwen2.5-32B（ceiling）|7B = "训自己"闭环；32B 失败少 → 多样性差但质量高|
|seed handling|改 agent_engine 加 `--seed` / 用 run_id 做命名键|前者跨包改动；后者零侵入靠自然采样|
|代码 / 数据布局|混 `data/` / 子目录 `data/triples/` / `scripts/` + `data/{raw,interim,processed}/`|仿 `eval/` + `eval/baselines/` 平行最便于跨 phase 复用|

### Decision

|项|决策|
|---|---|
|样本格式|**F1 only**|
|train/val 切|**per-scenario 末 20% run_id → val**；fallback：unique run_ids < 5 时全 train|
|scenario 范围|**仅 `tool_chain` + `code_review`**|
|mining 模型|**Qwen2.5-7B**——via `agent_engine/config.py` 加 `AGENT_ENGINE_MODEL` env override（1 行改动），不动 scenario YAML / Engine API|
|seed handling|**不改 agent_engine**——run_id 仅作命名键 + split 索引，多样性靠 7B 自然采样|
|布局|**仿 `eval/` 结构**：`data/` 顶层装 4 脚本（mine / extract / format / split）+ `data/triples/` 装产物|
|nudge 文本复原|**按 `required_tool` 模板填 `NUDGE_TEMPLATE`**（与 `discussion.py` L141-144 一致）；不进 F1 input，仅留 traceability|

### Pilot 实测 + scale-up 收敛

两次 pilot + 1 次 32B 对照（详见 [`data/triples/README.md`](data/triples/README.md) §Pilot 实测）：

|批次|max_retries|envelope|fire rate|recovery|yield|结论|
|---|---|---|---|---|---|---|
|7B run_ids 0-2|1|6|72%|3.6%|0.17/env|baseline|
|7B run_ids 0-5|2|12|73%|2%|0.08/env|max_retries 翻倍 → recovery 持平|
|32B 1 env tool_chain|1|1|40%|0%|0/env|换底座 → fire rate 降但 recovery 也 0|

当时判断：真瓶颈在方法学（7B 即使被 nudge 也极少补调用），改 max_retries 或换底座都无效。用户决策（2026-05-10）："最小化 demo 即可"——弃 1k 目标。当前仓库保留 1 triple（max_retries=2 batch 产出）作 pipeline proof + Phase 3 smoke 入口；scenario YAML 已回滚 `max_retries=1` 保 cross-project 干净。

> Phase 2 完整 narrative（含"32B n=2 是否过度判断"的修正与 synthesize 路径解锁产出）见 [`JOURNAL.md`](JOURNAL.md) 2026-05-10 条；该日决策"本项目不再写 ADR"，本条作历史保留不在文件内修正。

### Consequences

- 4 脚本（mine / extract / format / split）单一职责可分别替换；对 `evals.metrics.nudge` 私 helper（`_split_attempts` / `_resolve_who_to_agents` / `_split_frontmatter`）有耦合，扩 wrong_args 桶时需 evals 上游联动。
- 每条 triple 含 `run_id` / `scenario` / `turn_idx` / `failure_mode` / 全 `context` prefix，可反查到 envelope 任一行。extractor / formatter / splitter 全可单测（无 LLM 依赖）。
- F1 只把 `corrected_response.content` 当 assistant 目标，但 7B 在 require_tool turn 实际通过 Ollama function-calling API 输出 `tool_calls`——pilot 已观察到一例：text 说 "use write_section" 但 tool_call event 是 cast_vote。Phase 3 训练前需评估是否要把 `tool_calls` 序列化进 assistant content（否则模型只学到"该说什么"而非"该调什么"）。
- `wrong_args` 桶整链路保留 placeholder（与 `metrics/nudge.py` taxonomy 同步），启用条件锁在 Phase 5 agent_engine dispatch error 路径补 event 之后。
- 工业 tool-call SFT 常混合 (a) 自有失败 trace + (b) 外部公开数据 + (c) 强模型蒸馏；本项目 v1 拒绝 (b)/(c)（[`§1`](DECISIONS.md) 已锁），所以 (a) 路径必须解决 yield。
- **train set 不足 → Phase 2.5 候选**：① 合成 corrected（程序化造 `target = "I'll call <required_tool>(<instr-derived args>)"`，yield 100% 但失去自我修正语义）；② 32B / GPT-4o-mini 蒸馏（与 [`§1`](DECISIONS.md) "自有 infra" 决策冲突）；③ 设计 require_tool 更易触发的 scenario；④ 换 supervision 信号——artifact ACL 拒绝 / 投票不通过等（[`§1`](DECISIONS.md) 已知 trade-off 提过）。Phase 2.5 ADR 待选定方案后写。

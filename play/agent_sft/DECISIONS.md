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

## 4. SFT target schema 用 OpenAI `tool_calls` + 顶层 `tools` 字段（Qwen2.5 native）

- **Status**: accepted（supersedes §3 Consequences "F1 only 把 corrected_response.content 当 assistant target"）
- **Date**: 2026-05-10

### Context

§3 v1 formatter 把合成的 `corrected_response.content`（"好的，我现在调用 \`retrieve_docs\`：\n\nretrieve_docs("query")"）直接当 assistant text；下游 `agent_engine` 实际通过 Ollama function-call API 期望模型 emit `tool_calls` 字段（Qwen2.5 [chat template](https://raw.githubusercontent.com/chujiezheng/chat_templates/main/chat_templates/qwen2.5-instruct.jinja) 渲染成 `<tool_call>{"name":..., "arguments":...}</tool_call>` block）。schema 不对齐 → 训完模型可能"会说要调工具但不真 emit tool_call"，nudge-fire rate 不降反升。Phase 3 训练前必须锁 schema。

### Options considered

|选项|形态|与 Qwen2.5 chat template 关系|与下游 (Ollama → agent_engine `tool_call` event) 关系|
|---|---|---|---|
|A. text-only（v1 现状）|assistant content = "好的我现在调用 X(...)"|普通 text 渲染，无 `<tool_call>` 块|不发 tool_call → tool_call event 永远不触发 → nudge-fire 不降|
|**B. OpenAI `tool_calls` JSON-string + 顶层 `tools`**（选）|messages.assistant.tool_calls + 顶层 tools|chat template 自动渲染成 native `<tool_call>` block|与 Ollama function-call 解析器原生对齐|
|C. 字面量 XML 字符串写 content|content = `<tool_call>{...}</tool_call>`|跳过 chat template schema 校验|形态与 B 同，但 dataset schema 不通用|

### Decision

选 B。三层对齐：

- **行业惯例**：MLX-LM 自 [`mlx-examples` PR #995](https://github.com/ml-explore/mlx-examples/pull/995) 起原生支持 `tools` data format，[LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md) 明示等同 OpenAI / Mistral 微调示例（`arguments` 用 JSON-string）；xLAM / ToolACE / Hermes-Function-Calling 全用此 schema。
- **下游对齐**：Qwen2.5 chat template `tool_call.arguments | tojson` 把 JSON-string 与 dict 都渲染成 `<tool_call>{"name":..., "arguments":...}</tool_call>`，Ollama 函数调用解析器认这个块。
- **schema 单源**：直接 import [`agent_engine.artifact._TOOL_DEFS`](../agent_engine/artifact.py)（5 个 artifact 工具）+ 复用 [`scenario._resolve_tool_defs`](../agent_engine/scenario.py)（scenario YAML 里 `retrieve_docs` 等），与 runtime 同源，零 schema drift。

### Consequences

- [`data/formatter.py`](data/formatter.py) 重写：assistant message 改 `{role:"assistant", content:"", tool_calls:[{id, type:"function", function:{name, arguments<JSON-string>}}]}`；F1 sample 顶层加 `tools=[...]`（per-scenario，agent 视角，按 role 过滤 moderator-only 工具）。
- 新增 args-dict 提取：把 [`synthesize._extract_call_template`](data/synthesize.py) 抓到的 `tool(arg1, arg2)` 字符串按 `tool_schema.parameters.properties` 顺序映射成 dict，再 `json.dumps` 成字符串塞 arguments。提取失败的样本（即 fallback wrapper 那 ~17%）整条丢弃——本 ADR 配套 user 决策"drop"，不再走 `arguments={}` 的弱信号样本。
- [`data/synthesize.py`](data/synthesize.py) / [`data/extractor.py`](data/extractor.py) / [`data/split.py`](data/split.py) / [`data/mine_triples.py`](data/mine_triples.py) **不动**——`Triple` schema 已含 `required_tool` + `instruction` 文本，足以驱动新 formatter；mining envelope 与 `triples_*_1k.jsonl` 不重生。
- 数据量微减：drop fallback 后 7B 766 train + 196 val（vs §3 终交付 966 + 246），32B 642 + 160（vs 842 + 210）；仍超过 README 早期 "≥1k 训练样本" demo 量级阈值。
- 训练侧采用 `mlx_lm.lora --mask-prompt`（assistant-only loss），与 [TRL PR #5522](https://github.com/huggingface/trl/pull/5522) Qwen2.5 训练 template 同思想——梯度只作用在我们想教的 `<tool_call>` 块。
- **永久禁区不变**（§1）：自有 supervision 来源不变；schema 升级仅改数据形态，不引入第三方教师。
- 后续可移植性不受影响：HF safetensors 仍是 source-of-truth（§2）；切 TRL / Unsloth 时同 jsonl 零改即可。

## 5. Phase 3 推荐 adapter 锁 BASE 配置；layers/rank sweep + 真效果决断推迟到 Phase 5

- **Status**: accepted
- **Date**: 2026-05-11

### Context

Phase 3 6-run sweep（[`runs/sweeps/REPORT.md`](train/runs/sweeps/REPORT.md)）跑完，两个观察必须落 ADR——一影响"Phase 4 fuse 哪个 adapter"的具体动作，二影响"什么信号才算 SFT 真生效"的判据：

|观察|数据|
|---|---|
|**`iters` dim 全程饱和**|`iters` ∈ {50, 200, 600} 三档 `train_loss`/`val_loss` 全收敛到 0.00，`tool_call_emit_rate` / `tool_name_match_rate` / `arg_set_match_rate` / `arg_value_match_rate` 全 100%|
|**`lr` dim 仅 5e-4 劣化**|`lr=1e-5` / `1e-4` 全 100%；`lr=5e-4` `train_loss` 起 3.65 → 末 0.04 / `val_loss` 0.12 / emit 95.4% / name 93.9% / **arg_value 76.0%**|

[`eval_smoke.py`](train/eval_smoke.py) 4 项指标是"fast proxy for nudge-fire-rate"——不走 Ollama / agent_engine 端到端，只解析 `<tool_call>` 块对比 ground-truth。在当前 766 train / 196 val + `--mask-prompt`（assistant-only loss）配置下，loss 信号只覆盖很短的 tool_call 段，schema 信号**高度可压缩**：50 iter (≈0.25 epoch) 已学透形态，proxy 在 schema 学习的 differentiating 维度上**已经饱和**——它告诉我们"学透"，但不告诉我们"是 memorize 还是 generalize"。

### Decision

锁定三件事：

|条目|内容|
|---|---|
|**推荐 adapter**|[`train/runs/sweeps/iters/200/adapters.safetensors`](train/runs/sweeps/iters/200/)——sweep `BASE` 配置（`iters=200` / `lr=1e-4` / `num_layers=16` / `rank=16` / `mask-prompt` on / LoRA on q/k/v/o），Phase 4 即从此 adapter 走 `mlx_lm.fuse` → GGUF → Modelfile|
|**fast proxy 仅作 sweep 内排序信号**|"全 100%"不构成"SFT 起效"的证据；同理"95%/76%"不构成"配置严重失败"的证据；真决断信号 = **Phase 5 端到端 [`evals nudge_fire_rate`](../evals/metrics/nudge.py) 在原 scenario 上对比 base 7B / SFT 7B / 32B 三组**（README §Phase 5 锁定）|
|**`layers` / `rank` sweep 推迟到 Phase 3.5**|触发条件单一明确：**仅当 Phase 5 真测显示 `(SFT 7B − base 7B) < (32B − base 7B) × 0.5`（即关闭一半 gap 都不到）时**回头扫；否则视为"v1 demo 量级足够"，转 Phase 6 反思|

### Consequences

- Phase 4 直接读 [`iters/200/`](train/runs/sweeps/iters/200/)；不需要再单独跑"3.D 选最佳配置主跑"——sweep BASE 已 = 最优 = 完整 adapter。
- `lr=5e-4` 的 76% arg_value 是 sweep **唯一有信息量的负向单点**——印证 1e-4 是甜点，1e-3 不必再试（必发散）；该 negative datapoint 后续若被引用，固定指向 [`runs/sweeps/lr/0.0005/eval_smoke.json`](train/runs/sweeps/lr/0.0005/eval_smoke.json)。
- **eval signal 升级路径**已规划——`eval_smoke` 在 schema 信号上饱和不代表 fast proxy 概念失败：当 Phase 5 数据回灌 + scenario 数学扩容（v2-B / v2-D 候选）后，可让 `eval_smoke` 切到"全新 scenario / 未见过的 tool name 组合"，重新成为 differentiating 信号。
- 不动 [`§2`](DECISIONS.md) 关于 MLX-LM 三步 CLI 的承诺；不动 [`§4`](DECISIONS.md) schema 决策；本条仅追加"results-driven"的 deploy + decision rule，是 §4 的下游应用。
- **永久禁区不变**（§1）：Phase 5 判据使用 `evals.metrics.nudge`（自有 metric），不引入第三方 leaderboard 作主判据；BFCL / agent_traj 仍作 OOD 回归对照。

## 6. Phase 4 量化等级锁 Q4_K_M + Modelfile 1:1 复刻 qwen2.5:7b template

- **Status**: accepted
- **Date**: 2026-05-11

### Context

Phase 4 把 [§5](DECISIONS.md) 选定的 `iters/200` adapter fuse 进 Qwen2.5-7B，转 GGUF，注册成 Ollama tag。两个关键工程决策需要落 ADR——锁定后 Phase 5 复测的"模型"维度才算冻结：

|决策项|可选空间|
|---|---|
|**量化等级**|Q4_K_M / Q5_K_M / Q8_0|
|**Modelfile TEMPLATE 来源**|1:1 复刻 `ollama show --modelfile qwen2.5:7b` / 自写 jinja / 用 Ollama 默认推断|

### Options considered

|轴|选项|大小|与 Phase 5 baseline 的关系|
|---|---|---|---|
|量化|**Q4_K_M**（选）|~4.4 GB（实测 4460 MiB / 4.91 BPW）|与 Ollama 内置 `qwen2.5:7b` 同量化等级 → 公平对比|
|量化|Q5_K_M|~5.5 GB|质量略好 ~0.5%，但与 baseline 量化差 → 污染 SFT 信号归因|
|量化|Q8_0|~8 GB|质量近无损但 baseline 不在 Q8 → 不可对比|
|Modelfile|**1:1 复刻 qwen2.5:7b 的 TEMPLATE + SYSTEM**（选）|—|Ollama 函数调用解析器对 `<tool_call>` block 的识别完全依赖 chat template；template 偏离 1 字 → tool_call event 不触发 → 整条 SFT 信号失效|
|Modelfile|自写 jinja（参考 Qwen2.5 官方 [chat_templates 仓库](https://raw.githubusercontent.com/chujiezheng/chat_templates/main/chat_templates/qwen2.5-instruct.jinja)）|—|Ollama 的 Go template 子集 ≠ jinja2，迁移成本 + 出错面双高|
|Modelfile|不写 TEMPLATE，让 Ollama 从 GGUF metadata 推断|—|Ollama 推断的 fallback template 不含 native `<tool_call>` block 渲染逻辑，tool API 直接哑火（已知踩坑见 [ollama/ollama#7560](https://github.com/ollama/ollama/issues/7560)）|

### Decision

|项|决策|证据|
|---|---|---|
|**量化等级**|**Q4_K_M**|与 baseline 同量化等级，Phase 5 三组对比 (base 7B / SFT 7B / 32B) 量化轴对齐|
|**Modelfile**|**1:1 复刻** `ollama show --modelfile qwen2.5:7b` 的 TEMPLATE + SYSTEM 块；仅改 `FROM` 行指向本地 q4 gguf；**不加** `PARAMETER stop` 等行（与 baseline 完全一致）|baseline 自己的 modelfile 也未含显式 PARAMETER 行；stop token 由 GGUF metadata 提供 → 复刻 1:1 比"主动写一遍"更安全|
|**fuse 路径**|`mlx_lm.fuse --dequantize`（4-bit MLX → fp16 MLX）→ `convert_hf_to_gguf.py`（fp16 MLX → F16 GGUF）→ `llama-quantize Q4_K_M`（F16 → Q4 GGUF）|`mlx_lm.fuse` 4-bit 底座 fuse 必须 `--dequantize`（LoRA 加不进 4-bit 量化网格，[mlx-lm#1071](https://github.com/ml-explore/mlx-lm/issues/1071) 已记录）；`--export-gguf` 直出路径对 tokenizer metadata 兼容性不及 llama.cpp 路径，业内 [Awni 2024 thread](https://github.com/ml-explore/mlx-examples/discussions/1057) 推荐回 HF 中转|

### Consequences

- 验证：两级烟测全过。Step 5A `/api/chat` 直接返回 parsed `tool_calls` 字段（含 retrieve_docs + query 中文 args）；Step 5B `agent_engine` 跑全 8 step `tool_chain` scenario，transcript 抓到 **10 个 tool_call event** 覆盖 retrieve_docs / cast_vote / propose_vote / append_section / write_section / finalize_artifact 全工具集——SFT 学到的 schema 信号在 Q4_K_M 量化下未坍塌。
- 部署侧"重生指南"完全可脚本化：[`deploy/build.sh`](deploy/build.sh)（fuse → convert → quantize，幂等）+ [`deploy/deploy.sh`](deploy/deploy.sh)（ollama create，幂等）+ [`deploy/smoke_test.py`](deploy/smoke_test.py)（HTTP /api/chat 断言）。新机器从 0 到 `agent-sft-qwen` tag ≤ 10 min（不计模型下载）。
- llama.cpp 引入是 [§2](DECISIONS.md) 的**显式留口**——§2 "MLX-LM 三步 CLI" 承诺 _训练阶段_ 不出 MLX-LM 一家；部署阶段 GGUF 转换走 llama.cpp 在 §2 Consequences 末段已铺垫（"`mlx_lm.convert` 直出 GGUF 还是回 HF 走 llama.cpp 路径，留 Phase 4 真碰到时再决定（YAGNI）"）。Phase 4 决定走 llama.cpp，§2 不需要更新 Status。
- **永久禁区不变**（§1）：deploy 链路完全本地、零闭源依赖；模型不进 HF Hub（v3-B 候选才会）。
- **后续可滑性**：量化等级若 Phase 5 真测显示 Q4_K_M 压坏 SFT 信号，回滚到 Q5_K_M 仅需 `bash deploy/build.sh --force QUANT=Q5_K_M`，链路其他步骤零改动——本条决策可被一行环变覆盖。

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

## 7. extractor / synthesize / formatter 直连 agent_engine（删 evals 私有 import 反模式）

- **Status**: accepted（修正 §3 Consequences "对 evals.metrics.nudge 私 helper 有耦合"）
- **Date**: 2026-05-11

### Context

§3 落地数据流水线时，`extractor.py` / `synthesize.py` 用以下反模式从 evals 偷私有 helper：

```python
sys.path.insert(0, str(PLAY_DIR))
from evals.metrics.nudge import (
    _attempt_called_required,    # 私有
    _resolve_who_to_agents,      # 私有
    _split_attempts,             # 私有
    _split_frontmatter,          # 私有
    classify_failure_mode,       # 公开（OK）
    derive_expected_turns,       # 公开（OK）
)
```

`formatter.py` 也偷一个：`from evals.metrics.nudge import _split_frontmatter`. 这 4 个私有函数是 evals 反向工程 agent_engine `Discussion._expand_steps + _resolve_who + ToolTracer/ArtifactStore.event` schema 的镜像。schema 真源在 agent_engine，反向工程在 evals，agent_sft 又跨项目 import evals 的反向工程产物——**双层间接依赖让 schema 改动像踩雷**。

[`agent_engine §13`](../agent_engine/DECISIONS.md) 同期落地：把 transcript / scenario 解读权收回 agent_engine，新增 typed 视图 `Result.tool_calls() / .turns() / .find_finalize_decision()` + `TurnView.attempts() / .start_offset` + `Scenario.expanded_turns()` + `ExpandedTurn` dataclass。本 ADR 是 agent_sft 的对应面：把私有面跨项目 import 替换成 agent_engine 公开面。

### Options considered

| 项 | 做法 | 权衡 |
|---|---|---|
| A. 现状 | 继续 sys.path.insert + 4 个私有 import | schema 改动连锁三处；agent_sft 单测对 evals 内部表征敏感 |
| **B. 直连 agent_engine 公开面**（选择） | `from agent_engine import Result, Scenario, TurnView, ExpandedTurn`；保留 `from evals.metrics.nudge import classify_failure_mode`（evals 合法公开面）| 0 私有面跨项目 import；signal flow agent_engine schema → agent_sft 直接消费一层；evals.metrics.nudge 仅作为"failure mode taxonomy 拥有者"被引用 |
| C. 把 `classify_failure_mode` 也上提到 agent_engine | 完全无需 import evals | "missed / wrong_tool" 是 evals/sft 视角的语义判断（非 agent_engine 关心的 dispatch 真相），上提会污染 agent_engine 关注边界；deferred 到 PR-3 if needed |

### Decision

**B**：

| 模块 | 改动 |
|---|---|
| [`data/extractor.py`] | 删 `_split_frontmatter / _resolve_who_to_agents / _split_attempts / derive_expected_turns / _attempt_called_required` 跨项目 import；改 `from agent_engine import ExpandedTurn, Result, Scenario, TurnView`；`extract_triples` 内部用 `Scenario.expanded_turns()` + `Result.turns()` + `TurnView.attempts()` + `TurnView.start_offset` 直接迭代；`_attempt_called_required` 内化为 5 行模块本地 helper（仍被 synthesize 共享 import）|
| [`data/extractor.py`] 旧 helper | `_index_steps_by_turn` / `_split_turns_indexed` 退化为 1-2 行 shim（`Scenario.expanded_turns` / `Result.turns().start_offset` 的薄封装），仅为 [`tests/test_extractor.py`] 旧 helper 单测零修改 pass；新代码请直接用 agent_engine 视图 |
| [`data/synthesize.py`] | 同 extractor：删 4 个私有 import；改用 agent_engine 视图；`from extractor import _attempt_called_required`（同模块内 5-line helper，避免重复定义）|
| [`data/formatter.py`] | 删 `from evals.metrics.nudge import _split_frontmatter`；改 `from agent_engine import Scenario`；`_read_scenario_meta` 走 `Scenario.from_yaml(p).meta`（schema 校验自动跟 agent_engine 同源）；`yaml` import 删除（不再需要）|
| 跨项目 import 公开面纪律 | 唯一仍跨项目 import 的是 `from evals.metrics.nudge import classify_failure_mode`——evals 合法公开面（"missed / wrong_tool" 分类 + `FAILURE_MODES` taxonomy 拥有方）|

### Consequences

| 影响 | 结果 |
|---|---|
| schema 单源 | agent_engine 改 `Result.transcript` 形状 / `Scenario` step 字段 → agent_sft 改一处即可（`Scenario.expanded_turns()` / `Result.turns()` 视图层）|
| import 边界 | agent_sft 与 evals 仅 1 个公开函数依赖（`classify_failure_mode`），与 agent_engine 直连；vs PR-2 前的 4 私有 + 2 公开 = 6 个跨项目 import |
| 测试规模 | 89 测试全绿；`test_extractor.py` / `test_synthesize.py` 零修改（旧 `_index_steps_by_turn` / `_split_turns_indexed` 单测靠 1-2 行 shim 续命）|
| 演化友好 | 未来 agent_engine 给 transcript 加新 entry 类型 / Scenario 加新字段，agent_sft 自动接到（`Result.tool_calls()` / `expanded_turns()` 同步生效）|
| §3 修正 | §3 Consequences "对 evals.metrics.nudge 私 helper 有耦合" 不再成立；本 ADR 修正而非 supersede §3 整体（流水线四脚本结构与职责分工不变）|

### 后续可滑性

- 若 `classify_failure_mode` 后续也要从 evals 上提到 agent_engine（"missed / wrong_tool" 语义中性度待商）→ 写 PR-3 ADR；本方案先不做。
- 若 `play/agent_sft` 自身需要新视图（如"`turn N 的 attempt 数超过 max_retries 几次"）→ 加在 agent_engine.Result 上，不要回头在 agent_sft 内复刻。

## 8. Transcript schema typed 升级 + envelope `usage` 同步消费（agent_engine §14 的 agent_sft 对应面）

- **Status**: accepted（紧跟 [`agent_engine §14`](../agent_engine/DECISIONS.md)；扩展 §7 的"直连 agent_engine"边界到 typed entry）
- **Date**: 2026-05-11

### Context

§7 把 `extractor / synthesize / formatter` 直连 agent_engine 公开面（`Result / Scenario / TurnView / ExpandedTurn`），但 transcript 内部仍是 `list[dict]`. agent_engine §14 同期把 transcript 升级到 `list[TranscriptEntry]` typed union（6 个 frozen dataclass，`SpeakerEntry` 强制带 `type="speaker"`）+ `Result.usage: list[TokenUsage]` + `Result.from_dict` 严格化. agent_sft 三脚本要同步切到 typed access，否则：

| 失效点 | 原因 |
|---|---|
| `extractor.py::_attempt_called_required(events: list[dict])` 用 `e.get("tool")` | typed dataclass 不再有 `.get()` |
| `extract_triples` 用 `entry["content"] / entry.get("speaker")` | typed dispatch 应走 `isinstance(e, SpeakerEntry)` + `e.content` |
| `synthesize.py` 同上 | 同 |
| `formatter.py::_render_recent_context` 把 dict 形态投影成 prompt 段落 | 字典形态仍合法（落盘后是 dict），但 speaker 分支历史靠 `"speaker" in entry` 嗅探，§14 后该判断歧义（其它 entry 也可能含 speaker 字段） |
| `data/triples/runs_1k_fast_7b_r0_124/*.json` × 500 历史 mined envelope | `Result.from_dict` 严格化后无 `type:"speaker"` / `usage: []` 即 `KeyError` |

### Options considered

| 项 | 做法 | 权衡 |
|---|---|---|
| A. agent_sft 内部继续 dict 嗅探 | 跨 agent_engine §14 兼容 | typed union 优势失效；schema 改动两处都要追 |
| **B. 直接吃 typed entry**（选择） | `extractor / synthesize` 内部 `isinstance(e, SpeakerEntry/...)` 派发；`formatter` 落盘 dict 形态保持不变（`metadata["context"]` 是 JSON 序列化后的 dict），但 speaker 判断走 `entry.get("type") == "speaker"`（§14 已强制写入）| typed dispatch + JSON 形态都靠 `type` 字段单源；dict 嗅探彻底退场 |
| C. 给 agent_sft 写自己的 typed view | 双 SoT，与 §7 "schema 单源" 决策矛盾 | 否决 |

### Decision

**B**——extractor / synthesize 走 typed dispatch；formatter 走 `type` 字段判断；shim cleanup + 历史 envelope 一次性迁移：

| 动点 | 做法 |
|---|---|
| [`data/extractor.py`] | 形参 `events: list[ToolCallEntry \| ArtifactEventEntry]`；`_attempt_called_required` 用 `e.tool / e.caller`；`extract_triples` 内 `isinstance(e, SpeakerEntry) and e.speaker == ...` + `e.content` 直接访问；`Triple.context: list[TranscriptEntry]` |
| [`data/extractor.py`] shim 删除 | §7 留下的 `_split_turns_indexed` / `_index_steps_by_turn` 共 2 个 shim 退役（`Result.turns()[i].start_offset` 已直接给 turn-indexed 全局 offset，shim 无存在意义）；对应 [`tests/test_extractor.py`] 2 条 shim 单测一并删除 |
| [`data/synthesize.py`] | 同 extractor 切 typed access；`from agent_engine import SpeakerEntry, ...` |
| [`data/formatter.py::_render_recent_context`] | `entry.get("type") == "speaker"` 派发；落盘 metadata 仍是 dict 形态（JSON 序列化 typed dataclass 后只剩 dict + `type` 字段，但 typed → dict 转换由 `engine.py` 写盘前 `dataclasses.asdict` 完成）|
| [`data/triples/runs_1k_fast_7b_r0_124/*.json`] × 500 | 一次性迁移脚本注入 `type:"speaker"` 到 speaker entry + `usage: []` 到 envelope；与 agent_engine §14 forward-only 选择一致 |
| 跨项目 import 加项 | `from agent_engine import TokenUsage, ArtifactEventEntry, SpeakerEntry, ToolCallEntry, TranscriptEntry`（typed dispatch 需要） |

### Consequences

| 影响 | 结果 |
|---|---|
| schema 单源 | agent_engine §14 改 entry / 加字段 → agent_sft 改一处即可（typed dispatch 自动接到）|
| 测试规模 | 87 测试全绿（PR-1 删 2 条 shim 单测后；§7 时 89 → §14 时 87 = -2 等价覆盖移到 agent_engine 38 测试）|
| import 边界 | agent_sft 与 agent_engine 直连面扩到 typed entry / `TokenUsage`；与 evals 公开面依赖仍仅 `classify_failure_mode`（vs §7 时已收敛到 1 项）|
| 历史 mined 数据 | 500 envelope 一次性迁移；不留 `try_legacy_from_dict()` 长期 shim；后续重跑 mining 自动产 §14 schema |
| 演化友好 | 加新 entry 类型时 agent_sft 仅需补 `isinstance` 分支（如未匹配 fallthrough，对训练数据 mining 无副作用）|
| §7 关系 | §7 立"直连 agent_engine 公开面"是 dict 边界的；§8 把这条边界向内推到"directly typed entry"，是同一原则的递进收紧 |

### 后续可滑性

- `Result.usage` 当前 agent_sft mining 阶段不消费（mining 关心"做对什么"而非"花多少 token"）；后续训练数据筛选若需要"仅保留 cost ≤ X 的 envelope"，从 `result.usage` 直接聚合即可，无需新视图。
- 若历史 mined envelope 量级再次膨胀（>5K）使一次性迁移脚本压力变大，可考虑给 mining 加 `--allow-legacy-envelope` flag 临时降级 reader——本期 500 量级人工迁移可控，不做这个开关。

## 9. v1 结案：Phase 5 数字三阈值命中 + v2/v3 候选取舍

- **Status**: accepted (v1 closing)
- **Date**: 2026-05-13
- **Cross-refs**: 与 [`§5`](#5-phase-3-推荐-adapter-锁-base-配置layersrank-sweep--真效果决断推迟到-phase-5) 的"layers/rank sweep 推迟"触发条件呼应；与 [`§1`](#1-nudge-grounded-sft-作为项目中心问题) 的中心问题"nudge-fire rate 能不能降+不在 OOD 回归"形成 v1 闭环

### Context

Phase 5.A 跑完 3 model × 10 seed × 4 task = 120 runs（119 successful，1 排除：[`JOURNAL.md`](JOURNAL.md) 2026-05-12 → 05-13 milestone）。聚合产物 [`eval/baselines/phase5-3model-comparison.md`](eval/baselines/phase5-3model-comparison.md) 把 4 个 task × 3 个模型 × N 个嵌套指标钉在一张报告里。v1 结案两件事必须同步落 ADR：① **中心问题**（[`§1`](#1-nudge-grounded-sft-作为项目中心问题) "nudge-fire rate 能不能降+不在 OOD 回归"）按数字给定答；② [`README.md`](README.md) v2/v3 候选清单 7 项根据数字标"启动 / 摘牌 / 暂留"，不再悬空。

### Options considered

预设三种数字 → 决策路径，[`README.md`](README.md) §"v2/v3 演化路径" + plan §6.3 已铺垫：

|选项|数字特征|对应路径|
|---|---|---|
|A. SFT **显著有效**|三阈值全过：nudge gap 关闭 ≥50%、BFCL `arg_value_match` 回归 ≤5%、MMLU accuracy 回归 ≤3%|v1 达预期 → v2-B / v2-C 任一启动；v3-A 暂留；v3-B 可启动|
|B. SFT **部分有效但有回归**|nudge gap 关闭达标但 BFCL / MMLU 回归超阈|v2-C 优先（失败模式 taxonomy + hard sample mining），v3-A / v3-B 摘牌|
|C. SFT **无效**|nudge gap 关闭 < 50%|v1 终止；DPO 不解决根因；记录 negative finding；回到数据层（synthesize 模板 → 真 recovery）|

### Decision

**A 命中。** 三阈值实测数字（[`eval/baselines/phase5-3model-comparison.md`](eval/baselines/phase5-3model-comparison.md)，n=10 except 7B nudge n=9）：

|维度|base 7B|SFT 7B|32B|判定|
|---|---|---|---|---|
|`nudge_fire_rate`（越低越好）|0.7389 ± 0.1112|**0.6450 ± 0.0369**|0.5750 ± 0.0540|gap 关闭 **57.3%** ≥ 50% ✅|
|`bfcl_slice.arg_value_match`（越高越好）|0.9683|**0.9567**|0.9783|回归 **1.16%** ≤ 5% ✅|
|`mmlu_slice.accuracy`（越高越好）|0.7188|**0.6979**|0.8021|回归 **2.09%** ≤ 3% ✅|

二阶证据（非阈值，但写进 §Lessons 必须诚实标注）：

|维度|发现|含义|
|---|---|---|
|`agent_traj.task_success`|0.7000 → **1.0000** > 0.9333 (32B)|SFT 在端到端任务完成率上**超越 32B**——nudge supervision 不仅追到 ceiling，还在它擅长的子任务上反超|
|`agent_traj.tool_call_set_f1`|0.7373 → **0.5338** ↓|🔻 SFT 退化 27%。模型变得更 eager 调用工具，但调出的工具集合偏离 gold|
|`agent_traj.trajectory_match`|0.6583 → **0.4544** ↓|🔻 同上，序列对齐变差。"task success 上去 + trajectory 偏离" = 走多步岔路也能歪打正着|
|`nudge_fire_rate.by_failure_mode.missed`|13.89 → **10.10** ↓|SFT 减少 "漏调"|
|`nudge_fire_rate.by_failure_mode.wrong_tool`|0.89 → **2.80** ↑|🔻 SFT 增加 "错调"。教会了"该调"，没教会"调对"——v2-C 的明确入口|
|`nudge_fire_rate.by_scenario.panel`|0.78 → **0.975** ↑|🔻 panel 场景 SFT 反向回归（32B 0.45 反而最好）。supervision 信号在 panel 场景上偏；v2-C / v2-D 候选输入|
|`nudge_fire_rate.by_tool.retrieve_docs`|0.94 → **1.00** ↑|🔻 SFT 完全不自发调 retrieve_docs（100% 需 nudge）。`§4` schema drop 250 retrieve_docs no-template 样本的代价显现——training set 缺 retrieve_docs 自发示例|

### Consequences

**中心问题答**（[`§1`](#1-nudge-grounded-sft-作为项目中心问题)）：

- "把 `agent_engine` 不得不发 nudge 这一自有 supervision 信号作为微调目标，能不能让 7B 在自己 trajectory 上把 nudge-fire rate 显著降下来，且不在 OOD 上回归？" → **能。** nudge gap 关闭 57.3%（接近 32B ceiling 的 60% mark），OOD（BFCL）回归 1.16%，通用（MMLU）回归 2.09%。
- 但答案带条件：SFT 学到了"emit tool_call"的 schema 信号（[`§4`](#4-sft-target-schema-用-openai-tool_calls--顶层-tools-字段qwen25-native) 落地正确）+ "知道该调工具"（missed ↓），但未完全学到"调对工具"（wrong_tool ↑）+ "不调多余工具"（trajectory 偏离 ↑）。这是 nudge-grounded SFT 在 v1 supervision 仅 `require_tool` 一种信号下的天花板。

**§5 触发条件**：

- §5 锁的"`layers / rank sweep` 推迟到 Phase 5 真测 `(SFT 7B − base 7B) < (32B − base 7B) × 0.5` 才回头扫"——实测 gap 关闭 57.3% **未触发**该条件 → §5 status 维持 accepted，layers/rank sweep 不启动；推荐 adapter 仍是 [`train/runs/sweeps/iters/200/adapters.safetensors`](train/runs/sweeps/iters/200/)。

**v2/v3 候选清单更新**（[`README.md`](README.md) §"v1 / v2 / v3 演化路径"）：

|候选|新 status|依据|
|---|---|---|
|v2-A DPO|⏸ **暂留**|Phase 5 没暴露"prefer pair 学得不好"的证据；wrong_tool 是分类问题不是偏好问题，DPO 不正面解决|
|v2-B on-policy 迭代 SFT|✅ **可启动**|"trajectory 偏离" + "wrong_tool 上升"的根因是 training set 没有"SFT 模型自己产的 trajectory"；on-policy 回灌挖新 nudge 是直接对症|
|v2-C 失败模式 taxonomy + hard sample mining|✅ **启动**|by_failure_mode / by_scenario / by_tool 三轴已暴露 4 个明确死角（wrong_tool ↑、panel 反向、retrieve_docs 100%、tool_call_set_f1 退化），全是 hard sample mining 的入口|
|v3-A 14B 升级|⏸ **暂留**|7B 没显示饱和——SFT 在 task_success 上反超 32B，14B 的额外信息收益不明；先让 v2-B/v2-C 把现有信号榨干|
|v3-B 公开 HF artifact|✅ **可启动**|v1 三阈值全过 + 有干净的"硬币背面"叙事 = Model Card 的内容已成型；可在 v2-B/v2-C 启动前先 ship adapter + Model Card|
|v3-C 技术报告 / blog|⏸ **v3-B 之前**|内容依赖 v3-B + v2 任一进度|
|v3-D 多 supervision 信号 superset|🚫 **摘牌**|本期发现 v1 v2 的瓶颈是 supervision 质量（panel 反向 / retrieve_docs 100%）而非数量；加新信号桶（artifact ACL / 投票失败）前应先把 require_tool 信号在 panel 场景的偏诊清楚——v2-C 自然涵盖|

**工程补丁状态**（与 Phase 5 跑成绑死）：

- [`eval/run_baseline.py`](eval/run_baseline.py) 两处改动（`"python"` → `sys.executable`、`AGENT_ENGINE_MODEL` env 注入）+ [`evals/models/agent_engine_run.py`](../evals/models/agent_engine_run.py) 一处改动（`AGENT_ENGINE_RUN_TIMEOUT` env override），**全部保留并随本 ADR 一起 commit**。这三处不是 QoL：前两处是脚本能跑+三模型对比正确性的前提；第三处让 32B agent-path 不再 timeout、且对默认行为零副作用（env 不设走原 600s 默认）。
- 跨项目副作用：`evals/models/agent_engine_run.py` 的 timeout override 是 evals 公开行为变更（虽默认值不变）；后续 evals 自己的 ADR 体系应反向引用本 §9 的 motivation。

**评测脆弱性 followup**（不在本 ADR 范围内做，但记录交接）：

- `qwen2.5:7b nudge_fire_rate seed=3` 因模型生成 `tool=cast_vote(...)` 当 kwarg 让 `agent_engine` artifact handler 直接 `TypeError`. 这是 `agent_engine` 端工具 dispatch 防御性问题（应拒绝 unknown kwarg 返回 `{ok:false}` event 而非崩 caller）。v1 范围内的影响仅 1/120 数据点损失（已在 Phase 5.A 取舍节写明）；修复推荐归 `agent_engine` 自己的 backlog，不进 agent_sft。

**永久禁区不变**（[`§1`](#1-nudge-grounded-sft-作为项目中心问题)）：

- v1 结案不引入第三方教师 / 公开 tool-call 数据集污染 training；v3-B 走的是"ship 已训好的 adapter 到 HF"，不是反向把 HF 数据集吃进来。

### v1 收尾里程碑式陈述（面试用一句话锚点）

> "在 M4 Pro 48GB 单机上用自有 agent stack 产生的 require_tool nudge 作 supervision，QLoRA 微调 Qwen2.5-7B，在自己 trajectory 上把 nudge-fire rate 从 0.739 降到 0.645（关闭与 32B 同族 ceiling 0.575 的 57% gap），同时 BFCL 公开切片回归 1.2%、MMLU 子集回归 2.1%——全本地零闭源依赖、可一行命令复现部署。"

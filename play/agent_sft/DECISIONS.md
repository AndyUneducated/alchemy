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

- 学术对位：self-improvement / self-correction（STaR / Self-Refine / Reflexion / Self-Rewarding）；工业对位反走"自有 trajectory → 我的 agent 系统更稳"，比走外部 tool-call 数据集（xLAM / Watt-Tool / Hammer）更窄但更可信。
- 耦合 `agent_engine` transcript schema（`tool_call` event + `require_tool` step + nudge instruction）；schema 变 → 数据脚本变，训练框架不受影响。每条 triple 可反查 trace JSON 行。
- 数据上限取决于 scenario 跑批次数，不够时合成补足；v1 supervision 仅 `require_tool`，未来失败模式（artifact ACL / 投票不通过）可线性扩池，v1 不预设。

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

- MLX-LM 在 Apple Silicon 个人微调圈是事实标准（Awni Hannun / Simon Willison 背书；HF 自 2025 加 MLX backend）；训练 / 合并 / 量化全在一家工具不引入新 tooling。
- HF safetensors 是通用中间格式，未来换 GPU 服务器迁移成本可控。
- MLX-LM 当前不直接产 GGUF，需二段转换；走 `mlx_lm.convert` 直出还是 HF → `llama.cpp convert_hf_to_gguf.py → quantize` 留 Phase 4 再决（YAGNI）。

## 3. Phase 2 数据流水线设计

- **Status**: accepted（pipeline 与 1k × 2 模型 scale-up 均已落地，见 [`JOURNAL.md`](JOURNAL.md) 2026-05-10 三条里程碑）
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
|train/val 切|**per-scenario 末 20% run_id → val**；unique run_ids < 5 时全 train|
|scenario 范围|**仅 `tool_chain` + `code_review`**|
|mining 模型|**Qwen2.5-7B**——via `AGENT_ENGINE_MODEL` env override（1 行），不动 scenario YAML / Engine API|
|seed handling|**不改 agent_engine**——run_id 作命名键 + split 索引，多样性靠 7B 自然采样|
|布局|**仿 `eval/`**：`data/` 顶层 4 脚本（mine / extract / format / split）+ `data/triples/` 装产物|
|nudge 文本复原|**按 `required_tool` 模板填 `NUDGE_TEMPLATE`**（与 `discussion.py` L141-144 一致），不进 F1 input|

### Consequences

- 4 脚本（mine / extract / format / split）单一职责可分别替换；对 evals 私 helper 的耦合在 [`§7`](#7-extractor--synthesize--formatter-直连-agent_engine删-evals-私有-import-反模式) 解除（直连 agent_engine 公开面）。
- 每条 triple 含 `run_id` / `scenario` / `turn_idx` / `failure_mode` / 全 `context` prefix，可反查到 envelope 任一行；extractor / formatter / splitter 全可单测（无 LLM 依赖）。
- F1 只把 `corrected_response.content` 当 assistant 目标的风险（pilot 观察到 text 说 X 但 tool_call event 是 Y）→ Phase 3 由 [`§4`](#4-sft-target-schema-用-openai-tool_calls--顶层-tools-字段qwen25-native) schema 升级解决。
- `wrong_args` 桶整链路保留 placeholder（与 `metrics/nudge.py` taxonomy 同步），启用条件锁在 agent_engine dispatch error 路径补 event 之后。

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

选 B。三层对齐：① **行业惯例** —— MLX-LM 自 [PR #995](https://github.com/ml-explore/mlx-examples/pull/995) 原生支持 `tools` data format（[LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)），xLAM / ToolACE / Hermes-Function-Calling 全用此 schema；② **下游对齐** —— Qwen2.5 chat template `arguments | tojson` 把 JSON-string 渲染成 `<tool_call>` 块，Ollama 解析器原生识别；③ **schema 单源** —— 直接 import [`agent_engine.artifact._TOOL_DEFS`](../agent_engine/artifact.py) + 复用 [`scenario._resolve_tool_defs`](../agent_engine/scenario.py)，与 runtime 同源零 drift。

### Consequences

- [`data/formatter.py`](data/formatter.py) 重写：assistant message 改 `{role:"assistant", content:"", tool_calls:[{id, type:"function", function:{name, arguments<JSON-string>}}]}`；F1 sample 顶层加 `tools=[...]`（per-scenario，agent 视角，按 role 过滤 moderator-only 工具）。
- 新增 args-dict 提取：把 [`synthesize._extract_call_template`](data/synthesize.py) 抓到的 `tool(arg1, arg2)` 字符串按 `tool_schema.parameters.properties` 顺序映射成 dict，再 `json.dumps` 成字符串塞 arguments。提取失败的样本（即 fallback wrapper 那 ~17%）整条丢弃——本 ADR 配套 user 决策"drop"，不再走 `arguments={}` 的弱信号样本。
- [`data/synthesize.py`](data/synthesize.py) / [`data/extractor.py`](data/extractor.py) / [`data/split.py`](data/split.py) / [`data/mine_triples.py`](data/mine_triples.py) **不动**——`Triple` schema 已含 `required_tool` + `instruction` 文本，足以驱动新 formatter；mining envelope 与 `triples_*_1k.jsonl` 不重生。
- 数据量微减：drop fallback 后 7B 766 train + 196 val（vs §3 终交付 966 + 246），32B 642 + 160（vs 842 + 210）；仍超过 README 早期 "≥1k 训练样本" demo 量级阈值。
- 训练侧采用 `mlx_lm.lora --mask-prompt`（assistant-only loss），与 [TRL PR #5522](https://github.com/huggingface/trl/pull/5522) Qwen2.5 训练 template 同思想——梯度只作用在 `<tool_call>` 块。HF safetensors 仍是 SoT（[`§2`](#2-训练框架选-mlx-lm)），切 TRL / Unsloth 时同 jsonl 零改即可。

## 5. Phase 3 推荐 adapter 锁 BASE 配置；layers/rank sweep + 真效果决断推迟到 Phase 5

- **Status**: accepted
- **Date**: 2026-05-11

### Context

Phase 3 6-run sweep（[`runs/sweeps/REPORT.md`](train/runs/sweeps/REPORT.md)）跑完，两个观察必须落 ADR——一影响"Phase 4 fuse 哪个 adapter"的具体动作，二影响"什么信号才算 SFT 真生效"的判据：

|观察|数据|
|---|---|
|**`iters` dim 全程饱和**|`iters` ∈ {50, 200, 600} 三档 `train_loss`/`val_loss` 全收敛到 0.00，`tool_call_emit_rate` / `tool_name_match_rate` / `arg_set_match_rate` / `arg_value_match_rate` 全 100%|
|**`lr` dim 仅 5e-4 劣化**|`lr=1e-5` / `1e-4` 全 100%；`lr=5e-4` `train_loss` 起 3.65 → 末 0.04 / `val_loss` 0.12 / emit 95.4% / name 93.9% / **arg_value 76.0%**|

[`eval_smoke.py`](train/eval_smoke.py) 4 项指标是 fast proxy（不走端到端，解析 `<tool_call>` 块对比 ground-truth）。`--mask-prompt` 让 loss 只覆盖 tool_call 段 → schema 信号高度可压缩，50 iter (≈0.25 epoch) 已学透形态；proxy 在 schema 学习维度上**饱和**——告诉我们"学透"，不告诉"是 memorize 还是 generalize"。

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
- **eval signal 升级路径**已规划——`eval_smoke` 在 schema 信号上饱和不代表 fast proxy 概念失败：Phase 5 数据回灌 + scenario 数学扩容（v2-B / v2-D 候选）后，可让 `eval_smoke` 切到"全新 scenario / 未见过的 tool name 组合"，重新成为 differentiating 信号。

## 6. Phase 4 量化等级锁 Q4_K_M + Modelfile 1:1 复刻 qwen2.5:7b template

- **Status**: accepted
- **Date**: 2026-05-11

### Context

Phase 4 把 [`§5`](#5-phase-3-推荐-adapter-锁-base-配置layersrank-sweep--真效果决断推迟到-phase-5) 选定的 `iters/200` adapter fuse 进 Qwen2.5-7B，转 GGUF，注册成 Ollama tag。两个工程决策（**量化等级** Q4/Q5/Q8、**Modelfile TEMPLATE 来源** 1:1 复刻 vs 自写 vs 推断）锁定后，Phase 5 复测的"模型"维度才算冻结。

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
|**fuse 路径**|`mlx_lm.fuse --dequantize`（4-bit MLX → fp16）→ `convert_hf_to_gguf.py`（fp16 → F16 GGUF）→ `llama-quantize Q4_K_M`|4-bit 底座 fuse 必须 `--dequantize`（LoRA 加不进量化网格，[mlx-lm#1071](https://github.com/ml-explore/mlx-lm/issues/1071)）；`--export-gguf` 直出路径 tokenizer metadata 兼容性不及 llama.cpp 中转（[Awni 2024](https://github.com/ml-explore/mlx-examples/discussions/1057)）|

### Consequences

- 验证：两级烟测全过。Step 5A `/api/chat` 返回 parsed `tool_calls`（含中文 args）；Step 5B `agent_engine` 跑全 8 step `tool_chain` 抓 **10 个 tool_call event** 覆盖全工具集——Q4_K_M 量化下 SFT schema 信号未坍塌。
- 部署侧"重生指南"完全脚本化：[`deploy/build.sh`](deploy/build.sh) + [`deploy/deploy.sh`](deploy/deploy.sh) + [`deploy/smoke_test.py`](deploy/smoke_test.py)，新机器从 0 到 `agent-sft-qwen` tag ≤ 10 min。
- 后续可滑性：Q4_K_M 若 Phase 5 显示压坏 SFT 信号，`bash deploy/build.sh --force QUANT=Q5_K_M` 一行覆盖。llama.cpp 引入在 [`§2`](#2-训练框架选-mlx-lm) Consequences 末段已铺垫（YAGNI 留口），无需更新 §2 Status。

## 7. extractor / synthesize / formatter 直连 agent_engine（删 evals 私有 import 反模式）

- **Status**: accepted（修正 §3 Consequences "对 evals.metrics.nudge 私 helper 有耦合"）
- **Date**: 2026-05-11

### Context

§3 落地时 `extractor.py` / `synthesize.py` / `formatter.py` 用 `sys.path.insert` 反模式从 `evals.metrics.nudge` 偷 4 个私有 helper（`_attempt_called_required` / `_resolve_who_to_agents` / `_split_attempts` / `_split_frontmatter`，公开的 `classify_failure_mode` + `derive_expected_turns` 不算）。这 4 个私有函数是 evals 反向工程 agent_engine `Discussion._expand_steps + _resolve_who + ToolTracer/ArtifactStore.event` schema 的镜像——schema 真源在 agent_engine，反向工程在 evals，agent_sft 又消费 evals 的反向产物——**双层间接依赖让 schema 改动像踩雷**。

[`agent_engine §13`](../agent_engine/DECISIONS.md) 同期落地：把 transcript / scenario 解读权收回 agent_engine，新增 typed 视图 `Result.tool_calls() / .turns() / .find_finalize_decision()` + `TurnView.attempts() / .start_offset` + `Scenario.expanded_turns()` + `ExpandedTurn` dataclass。本 ADR 是 agent_sft 对应面。

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
| [`data/extractor.py`] / [`data/synthesize.py`] | 删 5 个私有 import（`_split_frontmatter` / `_resolve_who_to_agents` / `_split_attempts` / `derive_expected_turns` / `_attempt_called_required`）；改 `from agent_engine import ExpandedTurn, Result, Scenario, TurnView`；`extract_triples` 用 `Scenario.expanded_turns()` + `Result.turns()` + `TurnView.attempts()` + `.start_offset` 直接迭代；`_attempt_called_required` 内化为 5 行 helper（synthesize 共享 import）|
| [`data/formatter.py`] | 删 `_split_frontmatter` 私有 import；改走 `Scenario.from_yaml(p).meta`（schema 校验跟 agent_engine 同源）；`yaml` import 删除 |
| 公开面纪律 | 唯一保留的跨项目 import 是 `from evals.metrics.nudge import classify_failure_mode`（evals 合法公开面，`FAILURE_MODES` taxonomy 拥有方）|
| shim 兼容 | `_index_steps_by_turn` / `_split_turns_indexed` 退化为 1-2 行 shim 让旧测零修改（§8 退役）|

### Consequences

| 影响 | 结果 |
|---|---|
| schema 单源 | agent_engine 改 `Result.transcript` / `Scenario` 字段，agent_sft 改一处即可（视图层）|
| import 边界 | agent_sft 与 evals 仅 1 个公开函数依赖（`classify_failure_mode`），与 agent_engine 直连；vs PR-2 前 4 私有 + 2 公开 = 6 个跨项目 import |
| 测试 | 89 全绿，旧测零修改（shim 续命）|
| §3 修正 | §3 Consequences "对 evals 私 helper 有耦合" 不再成立；流水线四脚本结构与职责分工不变 |
| 后续可滑性 | `classify_failure_mode` 上提到 agent_engine 是另一 PR 的选项（语义中性度待商），本方案不做 |

## 8. Transcript schema typed 升级 + envelope `usage` 同步消费（agent_engine §14 的 agent_sft 对应面）

- **Status**: accepted（紧跟 [`agent_engine §14`](../agent_engine/DECISIONS.md)；扩展 §7 的"直连 agent_engine"边界到 typed entry）
- **Date**: 2026-05-11

### Context

§7 把三脚本直连 agent_engine 公开面，但 transcript 内部仍是 `list[dict]`. agent_engine §14 同期把 transcript 升级到 `list[TranscriptEntry]` typed union（6 个 frozen dataclass，`SpeakerEntry` 强制 `type="speaker"`）+ `Result.usage: list[TokenUsage]` + `Result.from_dict` 严格化。三脚本不切 typed access 即失效：`_attempt_called_required` 的 `.get("tool")` 在 dataclass 上不存在；`extract_triples` / `synthesize` 用 `entry["content"]` + `entry.get("speaker")` 嗅探应走 `isinstance(e, SpeakerEntry)` 派发；`formatter._render_recent_context` 的 `"speaker" in entry` 嗅探在 §14 后歧义；500 个历史 mined envelope JSON 缺 `type:"speaker"` / `usage:[]` 让 `Result.from_dict` `KeyError`。

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
| extractor / synthesize | 形参换 typed union；`_attempt_called_required` 用 `e.tool / e.caller`；`extract_triples` 内 `isinstance(e, SpeakerEntry)` + `e.content` 直接访问；`Triple.context: list[TranscriptEntry]` |
| extractor shim 删除 | §7 留下的 `_split_turns_indexed` / `_index_steps_by_turn` 退役（`Result.turns()[i].start_offset` 已直接给全局 offset）+ 2 条 shim 单测一并删除 |
| `formatter._render_recent_context` | `entry.get("type") == "speaker"` 派发；落盘 metadata 仍是 dict（typed → dict 由 `engine.py` `dataclasses.asdict` 完成）|
| 历史 envelope × 500 | 一次性迁移脚本注入 `type:"speaker"` + `usage: []`，与 agent_engine §14 forward-only 一致 |
| 跨项目 import 加项 | `from agent_engine import TokenUsage, ArtifactEventEntry, SpeakerEntry, ToolCallEntry, TranscriptEntry` |

### Consequences

| 影响 | 结果 |
|---|---|
| schema 单源 | agent_engine §14 改 entry / 加字段 → agent_sft 改一处即可（typed dispatch 自动接到）|
| 测试 | 87 全绿（§7 时 89 → -2 shim 单测删，等价覆盖移到 agent_engine）|
| 历史 mined 数据 | 500 envelope 一次性迁移，不留长期 shim；后续重跑自动产 §14 schema |
| §7 关系 | §7 立"直连公开面"在 dict 边界；§8 把边界向内推到 typed entry，同一原则递进 |
| 后续可滑性 | `Result.usage` 当前 mining 不消费，后续若需 cost 过滤可直接聚合 |

## 9. v1 结案：Phase 5 数字三阈值命中 + v2/v3 候选取舍

- **Status**: accepted (v1 closing)
- **Date**: 2026-05-13

### Context

Phase 5.A 跑完 3 model × 10 seed × 4 task = 120 runs（119 successful，1 排除）。聚合 [`eval/baselines/phase5-3model-comparison.md`](eval/baselines/phase5-3model-comparison.md)。v1 结案两件事同步落：① [`§1`](#1-nudge-grounded-sft-作为项目中心问题) 中心问题按数字给定答；② v2/v3 候选 7 项标启动 / 摘牌 / 暂留。

### Options considered

预设三种数字 → 决策路径：

|选项|数字特征|对应路径|
|---|---|---|
|A. SFT **显著有效**|三阈值全过：nudge gap ≥50%、BFCL `arg_value_match` 回归 ≤5%、MMLU ≤3%|v2-B / v2-C 任一启动；v3-A 暂留；v3-B 可启动|
|B. SFT **部分有效**|nudge 达标但 BFCL / MMLU 超阈|v2-C 优先；v3-A / v3-B 摘牌|
|C. SFT **无效**|nudge gap < 50%|v1 终止；记录 negative finding；回数据层|

### Decision

**A 命中。** 三阈值实测数字（[`eval/baselines/phase5-3model-comparison.md`](eval/baselines/phase5-3model-comparison.md)，n=10 except 7B nudge n=9）：

|维度|base 7B|SFT 7B|32B|判定|
|---|---|---|---|---|
|`nudge_fire_rate`（越低越好）|0.7389 ± 0.1112|**0.6450 ± 0.0369**|0.5750 ± 0.0540|gap 关闭 **57.3%** ≥ 50% ✅|
|`bfcl_slice.arg_value_match`（越高越好）|0.9683|**0.9567**|0.9783|回归 **1.16%** ≤ 5% ✅|
|`mmlu_slice.accuracy`（越高越好）|0.7188|**0.6979**|0.8021|回归 **2.09%** ≤ 3% ✅|

二阶证据 6 项（task_success SFT 反超 32B、tool_call_set_f1 -27%、trajectory_match -31%、missed→wrong_tool 转化、panel 场景反向回归、retrieve_docs 100% 需 nudge）已写入 [`README.md` §"Phase 5 数字一览"](README.md)；不在阈值判定内，但是 v2-B / v2-C 候选输入。

### Consequences

**中心问题答**（[`§1`](#1-nudge-grounded-sft-作为项目中心问题)）：**能。** nudge gap 关闭 57.3%（接近 32B ceiling 60% mark），BFCL 回归 1.16%，MMLU 回归 2.09%。但条件清楚：SFT 学到 schema 信号（[`§4`](#4-sft-target-schema-用-openai-tool_calls--顶层-tools-字段qwen25-native)）+ "知道该调"（missed ↓），未完全学到"调对"（wrong_tool ↑）+ "不调多余"（trajectory 偏离 ↑）——是 v1 仅 `require_tool` 单信号 supervision 的天花板。

**§5 触发条件**：gap 关闭 57.3% **未触发** §5 的"<50% 才回扫 layers/rank"条件 → §5 status 维持 accepted，推荐 adapter 仍是 [`train/runs/sweeps/iters/200/adapters.safetensors`](train/runs/sweeps/iters/200/)。

**v2/v3 候选清单更新**（mirror 至 [`README.md`](README.md)）：

|候选|新 status|依据|
|---|---|---|
|v2-A DPO|⏸ **暂留**|wrong_tool 是分类问题不是偏好问题，DPO 不正面解决|
|v2-B on-policy 迭代 SFT|✅ **可启动**|"trajectory 偏离" + "wrong_tool ↑" 根因是 training set 缺 SFT 自产 trajectory；on-policy 回灌直接对症|
|v2-C 失败模式 taxonomy + hard sample mining|✅ **启动**|三轴已暴露 4 个死角（wrong_tool ↑ / panel 反向 / retrieve_docs 100% / tool_call_set_f1 退化），全是 hard sample 入口|
|v3-A 14B 升级|⏸ **暂留**|7B 在 task_success 反超 32B 未显饱和；先让 v2 榨干|
|v3-B 公开 HF artifact|✅ **可启动**|三阈值全过 + "硬币背面"叙事 = Model Card 已成型，可先 ship adapter|
|v3-C 技术报告 / blog|⏸ **v3-B 之前**|依赖 v3-B + v2 进度|
|v3-D 多 supervision 信号 superset|🚫 **摘牌**|v1 瓶颈是 supervision **质量**（panel 反向 / retrieve_docs 100%）而非数量；v2-C 自然涵盖|

**工程补丁状态**：[`eval/run_baseline.py`](eval/run_baseline.py) 2 处（`sys.executable` + `AGENT_ENGINE_MODEL` env 注入）+ [`evals/models/agent_engine_run.py`](../evals/models/agent_engine_run.py) 1 处（`AGENT_ENGINE_RUN_TIMEOUT` env override，零副作用），全部随本 ADR 一起 commit——是 Phase 5 跑成的 prerequisite 而非 QoL。

**跨项目 followup**（归对应 backlog）：① evals harness 应隔离异常 LLM 输出（`tool=cast_vote(...)` 非法 kwarg → handler `TypeError` 崩 caller，1/120 损失）；② agent_engine artifact handler 应拒绝 unknown kwarg 返回 `{ok:false}` event。面试一句话锚点参见 [`README.md` §"面试叙事脚本（v1 结案版）"](README.md)。

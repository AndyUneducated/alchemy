# Journal

每条里程碑一段：`## YYYY-MM-DD — 标题`，正文必含 **功能** + **技术** 两节，**取舍** 节按需追加并反链 `DECISIONS §N`。架构决策见 [`DECISIONS.md`](DECISIONS.md)。

## 2026-05-09 — 项目立项：nudge-grounded SFT 框架确定

这一阶段的里程碑是把"为什么做、做什么、怎么做、怎么衡量"四件事在三份文档里钉死，**让所有后续 phase 都从一致的中心问题展开**。立项稿的关键不是"开始训了"，而是把项目从"再做一遍 tool-call LoRA"拨成"以 `agent_engine` 的 `require_tool` nudge 事件为 supervision 的自有闭环 SFT"——这一拨向是项目唯一的差异化承诺，也是面试故事的种子。

### 功能

|item|状态|说明|
|---|---|---|
|中心问题成文|✅|README 顶部一段句，明确"在 48GB 本地，让 nudge-fire rate 在 in-dist 上显著降，OOD 不回归"|
|七阶段路线图|✅|Phase 0-6 各自的目标 / 关键产出 / 状态都进 README，立项时仅 Phase 0 标 in-progress|
|与 `play/` 上下游关系图|✅|README mermaid 图：上游 `agent_engine` 出 trace，本项目训 + 部署，下游 `evals` 测 + `agent_engine` 用|
|度量四项占位|✅|nudge-fire rate / trajectory score / BFCL slice / general regression，定义先于训练|
|技术栈与 non-goals|✅|底座 / 训练框架 / 量化部署 / 评估 / 硬件 五个维度收敛；DPO / 多底座 / 风格微调 / 32B+ 显式排除|
|面试叙事脚本草稿|✅|README 末尾一段 narrative，会随 Phase 5/6 数字到位再迭代|

### 技术

|item|说明|
|---|---|
|`DECISIONS §1`|nudge-grounded SFT 作为中心问题——4 个候选选项中选 C，理由是 supervision 来源是自有 infra，复现门槛即护城河|
|`DECISIONS §2`|底座选 Qwen2.5-7B-Instruct——48GB QLoRA 甜点 + tool-call 一线 + MLX 友好 + Ollama 同 tag 现成|
|`DECISIONS §3`|训练框架选 MLX-LM——Apple Silicon 原生最优；`mlx_lm.lora` / `mlx_lm.fuse` / `mlx_lm.convert` 三步 CLI 链路，KISS|
|项目结构规划|README 给出目录骨架但不预创空文件夹；按 phase 推进时再落地 `data/` / `train/` / `eval/` / `deploy/`|
|未决问题|MLX-LM → GGUF 二段转换的具体路径（直出 vs 经 HF safetensors 中转）推迟到 Phase 4 真撞上再 ADR；现在不预写空架子（YAGNI）|

### 取舍

- 放弃"经典 tool-calling LoRA on xLAM/ToolACE"路线（详见 `DECISIONS §1` 选项 A）——执行简单但**面试无差异化**，对 senior 段位 portfolio 是负优化。
- 放弃 Llama-3.1-8B 底座（详见 `DECISIONS §2` 选项 B）——对美国面试官更熟，但 Qwen2.5 在 7B tool-call 段位基线更强，对本项目核心数字更友好；选型理由本身可作为面试问答素材。
- 放弃 axolotl / Unsloth 训练编排（详见 `DECISIONS §3` 选项 C/D）——前者 Mac 不是主战场，后者抽象层抬学习成本而本项目只需一条直链；MLX-LM 三命令链路是 KISS 的胜利。

## 2026-05-09 — 全本地基线：弃用 GPT-4o-mini，改用 Qwen2.5-32B-Instruct

立项稿原方案把 GPT-4o-mini 作为 ceiling reference 嵌进 Phase 1 / Phase 5 / 面试稿。复盘时发现这与 §1 "复现门槛即护城河"承诺相矛盾——闭源 API 让 portfolio 评审无法在不充值的情况下复现。**这一决定的关键不是省钱**（< $1 可忽略），**而是叙事一致性 + 可复现性**：把 ceiling reference 切到同家族的 Qwen2.5-32B-Instruct，故事点从"7B 追上 GPT-4o-mini"升级为"7B 在自己 trajectory 上追上 32B 同族原版"，与 §1 承诺完全对齐。

### 功能

|item|状态|说明|
|---|---|---|
|Phase 1 ceiling reference 替换|✅|README "GPT-4o-mini" → "Qwen2.5-32B-Instruct (Ollama)"，附 `DECISIONS §4` 内联引用|
|Phase 5 三组对比更新|✅|base 7B / SFT 7B / **Qwen2.5-32B 原版**；维持三组结构不变，仅替换 ceiling 项|
|面试叙事故事点切换|✅|从"追上 GPT-4o-mini"改为"7B SFT 追 32B 同族原版"，并强调"整条链路全本地、零闭源依赖"|
|README forward ref 编号 bump|✅|旧 `§4 数据策略` / `§5 超参取舍` 顺延为 `§5` / `§6`（本日新 ADR 占用 §4）|

### 技术

|item|说明|
|---|---|
|`DECISIONS §4`|全本地基线：5 个候选选项中选 B（Qwen2.5-32B-Instruct），理由是同家族跨规模对比天然契合"复现门槛即护城河"承诺|
|模型成本与依赖|$1 → $0；移除 OpenAI client 依赖，evals run 完全本地纯命令行可复现|
|显存测算|Qwen2.5-32B-Instruct Q4 ≈18GB，48GB 上与 7B Q4 ≈4GB 共存舒适；同时跑两个 ollama serve 实例需注意 KV cache 总量，必要时串行跑|
|chat template 一致性|7B 与 32B 同 Qwen2.5 家族 chat template，`BACKEND=ollama` + 改 `MODEL` 字段即可切换，零适配成本|

### 取舍

- 放弃"7B SFT 追上 OpenAI GPT-4o-mini"这个**最广认知度的行业锚点**（详见 `DECISIONS §4` 选项 A）——换得**全本地可复现** + **同家族跨规模对比**故事，后者与本项目"复现门槛即护城河"承诺更契合。
- 放弃双 baseline 路线（详见 `DECISIONS §4` 选项 E）——工作量翻倍 ROI 一般；如 Phase 5 数字到位想再加 OpenAI 对照可反向追加，本 ADR 不阻止。
- 放弃 Qwen2.5-72B 作 ceiling（详见 `DECISIONS §4` 选项 D）——48GB 上 Q4 ≈42GB 余量太紧，eval 跑批速度损耗大于"更强 ceiling"的故事增量。

## 2026-05-09 — 立项稿扩展性留口：把项目从"v1 完成态"重写为"v1 + 演化路径"

立项稿三件围绕同一框架的延展（中心问题 + 全本地基线 + 扩展性留口）今日同日完成。本是第 3 条略超 ≤2/天 软上限——理由是立项当天三件**都属 Phase 0 (Frame) 框架决策**，合并任一会破坏话题聚焦；Phase 1 起将严格回到 ≤2/天。S 级 portfolio 共性诊断（多幕剧叙事 / 可比性 / 多轴消融 / 失败模式 / 可持续迭代）反推出 5 处需要"打地基修补"的位置——全部为声明性变更不增加 v1 工程量，但能从根上避免 Phase 5/6 时被迫"翻案"。

### 功能

|item|状态|说明|
|---|---|---|
|中心问题剥离硬件|✅|"M4 Pro 48GB" 从中心问题挪到独立"v1 工程约束"节，未来 14B/32B 升级中心问题不变|
|v1 / v2 / v3 演化路径子节|✅|7 个候选清单（DPO / on-policy / 失败模式 taxonomy / 14B 升级 / HF release / 技术报告 / 多信号 superset）+ 各自触发条件|
|non-goals 拆"永久禁区" vs "v1 边界"|✅|新增"训练集混入公开数据集"作 ❌ 永久禁区（呼应 §1 差异化）；DPO / 多底座 / 32B+ 改⏸ 标 v2/v3 候选|
|度量加报告维度子节|✅|per-scenario / per-tool / per-failure-mode 三轴 breakdown + 多 seed ≥ 3 报均值±标准差；为 v2-C 失败模式 taxonomy 预先开数据通道|
|技术栈加可移植性声明|✅|HF safetensors 作 source of truth；MLX/Ollama 是消费者；预留 v2 切 TRL / v3 上云 GPU 路径|

### 技术

|item|说明|
|---|---|
|`DECISIONS §5`|扩展性留口 ADR——3 个候选选项中选 B（打地基修补 + v2/v3 候选清单），理由是 S 级 portfolio 多幕剧叙事范式的最小成本实现|
|forward ref 编号 bump|README 旧 `§5 数据策略` / `§6 超参取舍` 顺延为 `§6` / `§7`（本 ADR 占用 §5）；Phase 0 行 `§1-3` → `§1-5`|
|代码量增量|0——全为声明性 README 变更；唯一未来代码增量是 Phase 1 evals task wrapper 加 breakdown 维度 ≈50 行|

### 取舍

- 拒绝"立项稿 v1 完成态"叙事（详见 `DECISIONS §5` 选项 A）——v1 真完成时面试官追问"下一步"无干脆答案，且 v2 真做完时整段重写。
- 拒绝"完整写 v1 + v2 + v3 三套 README"路线（详见 `DECISIONS §5` 选项 C）——v2/v3 还没数据，过度设计；候选清单 + 触发条件足以传达"我有路线图"信号，遵循 `workshops.mdc` "抽象引入滞后于第二个具体案例"原则。
- 软超 JOURNAL ≤2/天上限——3 条同日条目皆 Phase 0 框架决策，合并会破坏话题聚焦。这是立项当天**一次性**的例外，Phase 1 起严格 ≤2/天。

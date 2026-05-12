# Decisions

ADR（Architecture Decision Record）归档。每条以 `## n. 标题` 开头，紧接 `- **Status**` + `- **Date**` 元信息；正文沿用 `Context / Options considered / Decision / 行业光谱 / 工程维度评估` 段落。**新决策追加到末尾，被取代的条目改 Status；不删旧条目**。日常进度（按里程碑）见 [`JOURNAL.md`](JOURNAL.md)。

## 1. Phase-driven scenario 配置

- **Status**: partially superseded by §9（顶层 `moderator:` / `members:` 块、`phases:` 三段式被 `agents:` + 扁平 `steps:` 取代；YAML frontmatter + MD body 的整体形式与 schema 启动校验机制保留）
- **Date**: 2026-04-14

### Context

最小可跑版本把 agents、轮次、话题全部硬编码，每换话题都要改代码。目标：把"参与者 + 流程"从代码里抽出来做配置。

### Options considered

- **A. JSON 配置**：机器友好，但不适合写大段中文 prompt
- **B. YAML 单文件**：适合结构化字段，但 prompt 写 markdown 时不优雅
- **C. Markdown + YAML frontmatter**（选择）：YAML 放结构字段，MD body 放话题；prompt 可带 markdown，单文件即一个场景
- **D. Python DSL / 代码即配置**（AutoGen 风格）：表达力最强但抬高门槛，不适合"想展示给别人的场景库"

### Decision

YAML frontmatter 定义 `members` / 可选 `moderator` / `phases`；MD body 作为话题注入 history。初始提供 4 个 scenario 覆盖 (moderator / no moderator) × (open / goal-oriented) 2×2 矩阵，启动期做 schema 校验。

**行业光谱**：CrewAI 用 YAML 配置 agents / tasks 是主流；AutoGen 偏 Python-centric；LangGraph 用 Python graph DSL。"MD + YAML frontmatter" 这个具体选择在 Jekyll / Hugo / Obsidian 世界是标准，在 agent framework 里较少见——正确但非主流。

### 工程维度评估

|维度|评估|
|---|---|
|内聚度|高——引擎、scenario、参与者定义各司其职|
|耦合度|显著降低——引擎只依赖"phases"抽象|
|可观测性 / 可审计性|中——stdout 有 phase/round 打印，未结构化|
|LLM 不确定性容忍|中——启动期校验作者输入；LLM 运行时输出未约束|
|向后兼容 / 演化友好|项目起点，无旧行为需兼容|
|学习曲线|中——作者需学 YAML frontmatter + `members/moderator/phases` 心智模型|
|可测试性|高——场景作为 fixture，新增场景零代码改动|

### 已知持续 trade-off

新特性（memory 策略、tool 开关、投票）都要在 YAML schema 上增字段，schema 会持续膨胀。目前所有字段均为加法、无废弃项，但需持续警惕。

## 2. Per-agent 消息投影

- **Status**: accepted（共享 transcript + per-agent projection 是后续 §5/§6/§9 的基础）
- **Date**: 2026-04-15

### Context

初版把共享 history 原样喂给每个 agent，system prompt 当作 history 里一条 user 消息。问题：

1. Agent 分不清"我说过的"和"别人说的"——全是 user role
2. System prompt 优先级失真
3. Anthropic / Gemini API 不接受连续同 role 输入

### Options considered

- **A. 保留共享 history，每个 agent 外挂 transformer**（选择）
- **B. 每个 agent 维护私有 history**：更彻底但同步复杂度爆炸
- **C. 继续硬塞，靠 prompt engineering 让 agent 自己分辨**：脆弱

### Decision

- Discussion 维护**一条共享 transcript**
- 每个 agent 在 `respond()` 时投影为自己的视角：`speaker == owner` → `assistant`；其他 → `<message from="X">...</message>` 包进 user；元数据条目 → `<tag>...</tag>` 包进 user
- System prompt 走 client 独立参数
- History entry 从 `role/content` 改为 `speaker/type`

**行业光谱**：AutoGen 的 `model_context`、LangGraph 的 channel 概念都是"共享状态 + 每 agent 投影"模式；`<message from="X">` 包装也是社区常见做法。高度对齐。

### 工程维度评估

|维度|评估|
|---|---|
|内聚度|高——"投影"是单一职责|
|耦合度|大幅降低——Agent 只依赖 history list 结构|
|可观测性 / 可审计性|中性——投影纯函数化让 debug 容易|
|LLM 不确定性容忍|升——`<message from="X">` 让 agent 分辨说话人更稳|
|向后兼容 / 演化友好|破坏性——history 结构改；改动时消费者只有一个，影响可控|
|学习曲线|低——对 scenario 作者透明|
|可测试性|升——纯函数投影，输入固定输出确定|

"共享 transcript + per-agent projection" 这一模式天然支持未来的 per-agent memory 策略、跨 provider 隔离、审计轨迹派生——抽象的杠杆率远超当前单一用途。

## 3. Per-round phases + instruction-as-arg

- **Status**: partially superseded by §9（`phases × round` 二维结构 + `<phase>` marker 被扁平 `steps:` + `<turn X of N>` marker 取代；**instruction-as-arg（不进 history）保留**——这是本条 ADR 的核心 invariant）
- **Date**: 2026-04-16

### Context

两个 bug 同时暴露：

1. **Instruction 泄露**：`_exec_phase` 把 instruction 追加进共享 history，所有后续 agent 都能读到本不属于自己的指令。给 moderator 的"点名追问 X"会被 members 当作自己的指令
2. **轮次无差异**：main 阶段静态定义，每轮完全相同，无法表达"第 1 轮自由讨论 → 第 2 轮聚焦分歧 → 第 3 轮逼迫表态"的递进

### Options considered

**针对 instruction 泄露**：

- **A. 加 history entry 可见性字段**：侵入性大
- **B. Instruction 不进 history，作为 `respond()` 的参数**（选择）：零侵入

**针对轮次差异**：

- **A. `instructions: [...]` 按轮索引**
- **B. `{round}/{rounds}` 模板变量**
- **C. Phase 显式声明 `round: <int> | "default"`**（选择）：支持"默认 + 个别轮次 override"

### Decision

`phases` 列表分裂成 `opening / main / closing`，`main` 每个 phase 声明 `round`；引擎每轮先选 `round == N`，回退到 `round == "default"`，再回退到全员发言。Instruction 作为 `Agent.respond(instruction=...)` 参数，**不进 history**。

**行业光谱**：AutoGen 的 `initiate_chat` 把每轮任务显式传入，LangGraph 用 `state` 字段表达阶段差异——本项目做法更像"脚本化舞台剧"，显式性强但灵活性不如 state machine。"阶段 × 轮次"二维调度是自己的发明，行业无直接对应。

### 工程维度评估

|维度|评估|
|---|---|
|内聚度|高——三段分明，per-round 逻辑集中在一个 fallback 链|
|耦合度|降低——instruction 离开共享 history，history 不再承担"控制流 + 对话内容"双重职责|
|可观测性 / 可审计性|中——`Round N/M` 有 marker；instruction 本身不进 history，回放看不到"引擎当时给 agent 下了什么指令"|
|LLM 不确定性容忍|升——消除 instruction 泄露的"误执行"失控路径|
|向后兼容 / 演化友好|破坏性——`phases` 扁平列表改为分块 + `round` 字段，旧场景必须改|
|学习曲线|中——三段结构 + `round` fallback 链；字段语义 self-explanatory|
|可测试性|升——per-round 后可以写"某轮特定行为"的验证场景|

同一次重构同时解决了一个 bug 和一个能力缺口：把"instruction 该不该进 history"这个本质问题回答清楚后，`phases` 结构的合理形态自然浮现——修 bug 而非打补丁常常顺便解锁新能力。

## 4. Subprocess 隔离 RAG 工具

- **Status**: accepted（monorepo 解耦原则的样板；`play/evals` 在 phase 4 复用同一模式接 `play/rag`）
- **Date**: 2026-04-16

### Context

首个工具 `retrieve_docs` 用 `sys.path.insert(0, rag_dir)` 直接 import rag 模块调用。Python 按名称缓存模块——两个子项目**各自有 `config.py`**，第二个 import 的 `config` 拿到第一个的缓存，两边互相覆盖。

### Options considered

- **A. 把 rag 改成可安装 package**（`pip install -e`）：最干净，但 workshop 场景抬高演示门槛
- **B. 共享 config 抽成第三方模块**：人为增加耦合
- **C. Monkey-patch `sys.modules`**：脆弱
- **D. Subprocess 隔离**（选择）：每次调用开独立 Python 进程，**靠 OS 级进程边界保证隔离**

### Decision

- `tools/retrieve_docs.py` 不再 import rag 代码；改为 `subprocess.run(["python", "rag/query.py", "--json", ...])`
- `rag/query.py` 加 `--json` 输出模式供机器消费
- `run.py` 剥掉 scenario-default 参数不暴露给 LLM（`_path_params` 私有提示），只保留 LLM 真正要填的

### 与 MCP 的关系

MCP（Model Context Protocol，2024）把"工具跑在独立进程 / 服务里，通过标准协议通信"定为业界方向。本项目的 subprocess 隔离**正好踩在这条线上**——虽然没走 MCP 协议而是朴素 CLI + JSON stdin/stdout，设计精神一致。许多生产系统（Claude Code 的 tool 执行、各种 sandboxed code execution）都用类似进程 / 容器隔离。未来若升级为常驻进程，天然可以迁到 MCP server 形态。

### 工程维度评估

|维度|评估|
|---|---|
|内聚度|高——每次 tool 调用是一次完整 `query.py` 生命周期，进程内无残留状态|
|耦合度|极显著降低——multiagent 与 rag 完全不共享 Python 进程，两边可独立演化依赖、Python 版本、配置。**故障半径从"整个 multiagent 进程"压缩到"一次 tool call"**|
|可观测性 / 可审计性|升——subprocess stderr 直通终端|
|LLM 不确定性容忍|中——工具失败返回 `{"error": ...}` 让 LLM 自修正；启动失败目前仅打 stderr，agent 本身看不到，这是留下的不对称|
|向后兼容 / 演化友好|加法式——`retrieve_docs` 接口不变|
|学习曲线|低——对 scenario 作者零感知|
|可测试性|高——subprocess 边界让 RAG 可 `python query.py --json` 独立调试|

代价：每次 tool call 一次 Python 启动（冷启动 ~500ms），workshop 量级可接受；生产需常驻进程或走 MCP。

## 5. Per-agent conversation memory

- **Status**: accepted
- **Date**: 2026-04-20（+ DI 改造 2026-04-21）

### Context

panel 场景（4 成员 + 1 主持 × 3 轮）实测：末段单次发言 111s，相比开场 24s 慢 4.5 倍；整场耗时 1398s，是 `vdb_test` 的 14 倍。根因：所有 agent 共享同一条全量 history，每轮末尾 agent 的输入 token 随总发言数线性增长。

### Options considered

- **A. Full history**（当前行为）
- **B. Window（滚动窗口）**
- **C. Summary（定期折叠）**
- **D. Vector retrieval**（按相关性检索历史）
- **E. Memory stream + importance scoring**（Generative Agents 式）
- **F. OS-like paging**（MemGPT 式）

选 A + B + C 三件套，不做 D / E / F——对话 stream 短（几十到一两百条），向量 / 分页 / importance 的收益不足以抵消复杂度。

### Decision

- `ConversationMemory` 抽象基类，`build_messages(history, owner) -> messages`
- `FullHistory`（default，完全向后兼容）
- `WindowMemory(max_recent)`：保留所有 pinned marker（`topic / round / phase / artifact_event`）+ 最近 N 条发言
- `SummaryMemory(max_recent, client, ...)`：stale speech 增量折叠进 `<summary>` block

**关键设计点**：

- 共享 transcript 不变，每个 agent 持有自己的 memory 实例（承接 §2 的"共享 + 投影"模式）
- **Pinned types 永不被剪**——控制流和工件事件是会议纪要级信息，丢了会话就破了
- Summary 触发规则：stale 条数达阈值才触发；未到阈值 stale 原样保留，"不动就无信息损失"

**SummaryMemory 依赖注入（次日补齐）**：初始版本 SummaryMemory 直接 import agent 和 config 拿 LLM client，memory 模块被迫编译期依赖 agent。改为构造时注入 `client / model / max_tokens / temperature`——memory 只声明"我需要一个 `.chat()` 对象"，不关心来源。`run.py` 承担装配。同一改造顺手处理：scenario 相对路径按 scenario 文件位置自动解析（约定优于配置），避免 scenario 搬位置就挂。

### 与主流 Memory 框架的分工对比

|框架 / 论文|对标到本项目的哪部分|
|---|---|
|LangChain `ConversationBufferWindowMemory` / `SummaryBufferMemory`|`WindowMemory` / `SummaryMemory` 的直接原型|
|LangGraph `checkpointer` + `BaseStore` 双层|本项目只有 transcript 单层，无长期 store|
|AutoGen `model_context` per-agent|对齐：memory 也是 per-agent|
|CrewAI short / long / entity 三种|仅覆盖 short-term|
|Letta / MemGPT OS 式分页|故意不做，复杂度不匹配|
|Mem0 / Zep 独立 memory 服务|故意不做，sandbox 项目无持久化需求|
|Generative Agents（Stanford 2023）memory stream + recency/importance/relevance|未做；若未来加 D / E 方向会参考|

### 工程维度评估

|维度|评估|
|---|---|
|内聚度|高——`memory.py` 纯投影，三策略共享同一接口；DI 改造后彻底不 import agent / config|
|耦合度|低——memory → agent 单向依赖；装配在 run.py（composition root）|
|可观测性 / 可审计性|中——Window/Summary 的裁剪决策本身无 log；debug 靠读 `_summarized_up_to` 等内部状态|
|LLM 不确定性容忍|混合——WindowMemory 降低 context 噪声是升；SummaryMemory 引入额外 LLM 调用做折叠，新增一条不确定性链（summarizer 可能漏信息），且当前无重试|
|向后兼容 / 演化友好|完全兼容——`FullHistory` 作为 default|
|学习曲线|中——作者需理解三策略的 trade-off|
|可测试性|DI 改造后显著升——可注入 fake client 做单测；`scenarios/example.md` 集成覆盖三 memory 策略|

### 意外收益

memory 落地后，"固定发言顺序不公平"问题的一半根因（"后发言者享有免费上下文优势"）被结构性消除——WindowMemory 让每人看到的上下文长度一致，发言顺序的不对称被削弱。原本计划单独做的"rotate/shuffle 发言顺序"特性优先级因此下降。解决 A 时发现 B 消失了一大半。

## 6. Shared artifact + 结构化投票

- **Status**: accepted（`MODERATOR_ONLY_TOOLS` 硬编码在 §9 改为 `tool_owners` 数据驱动 ACL，但 ArtifactStore 本体 + 6 工具 + 投票 schema 不变）
- **Date**: 2026-04-21

### Context

`panel` 场景要求"一方胜出"，但引擎只产出一串发言，最终决策靠隐式推断，可重复性差；`debate` 同样只有过程没有裁决。**讨论 → 决策**的链路是隐式的，无法机器验证。

### Options considered

**对产出形式**：

- **A. Moderator 收尾发言写"最终决策是 X"**：prompt engineering，不可验证
- **B. JSON 整块输出**：压力全压在一次 structured output 上，LLM 常搞砸
- **C. Sectioned markdown + tool-mediated writes**（选择）：拆 section，每个 section 有 `replace / append` mode，agents 通过 tool 写入
- **D. 外置数据库**：过度工程

**对投票**：

- **A. 自由文本投票**：噪声大
- **B. `propose_vote` / `cast_vote` 结构化**（选择）：tally 可验证

### Decision

`ArtifactStore` = sectioned markdown + 结构化 votes + `finalized` 状态。六个工具：`read_artifact / write_section / append_section / propose_vote / cast_vote / finalize_artifact`。

**关键设计点**：

- Section mode 由 scenario 作者在 `initial_sections` 显式声明，store 强制；mode 不匹配返回 `{"error": ...}`，LLM 在同一 tool loop 内 self-correct
- Moderator-only 工具：`finalize_artifact` 通过 `build_tool_defs(role)` 过滤；`propose_vote` 初始未过滤，member 乱 propose 会让 scenario 硬编码的 `vote_id` 错位，后续同 phase-assert 一并补上过滤
- **Out-of-band artifact view**：每次 agent 发言前把 `artifact.render()` 作为 `<artifact>` 消息**带外注入**——不进 history，memory 裁剪永远不会藏掉它
- **Artifact events 进 history**：`artifact_event` 类型，pinned，不被 memory 剪

### 跨 CRDT / workflow / structured output 三条光谱的位置

- **CRDT / 协同编辑**：artifact 是 multi-writer shared state，sectioned 划分 + 显式 replace/append mode 是 conflict avoidance 的朴素形式（没上 CRDT 但精神一致）
- **Workflow engine**（Temporal / Airflow）：`finalize_artifact` 是 sealing step，不可重入——类似 workflow 的 terminal state
- **Structured output / function calling**：投票当 function call 而不是文本——LLM 应用标准做法

相对创新的点：`initial_sections` 让 scenario 作者**在 YAML 里声明"这篇文档长什么样"**，LLM 被限制在 schema 内填空——把产品设计从"自由创作"改成"填空题"。

### 工程维度评估

|维度|评估|
|---|---|
|内聚度|高——`ArtifactStore` 聚焦"共享状态 + 工具入口 + 事件流"，对外就 `render / drain_events / dispatch / build_tool_defs` 四个口|
|耦合度|项目内耦合点最多——Artifact 被 Discussion / Agent / Memory / Tools 四处触达，每处必要无冗余；代价是任何语义变更要同步考虑四处|
|可观测性 / 可审计性|极高——`artifact_event` 进 history 且 pinned，终端 📝 / ➕ / 🗳 / ✓ / 🏁 emoji 实时可见，`--save-artifact` 落盘|
|LLM 不确定性容忍|高——section mode 冲突让 LLM 同 loop self-correct；ballot 覆盖写入容忍重复 cast；`finalize` 幂等返回 error 防重入|
|向后兼容 / 演化友好|加法式——未声明 `initial_sections` 的 scenario 不感知 artifact|
|学习曲线|项目内最高——`initial_sections` schema + section mode + moderator-only filter 三层概念|
|可测试性|高——`scenarios/example.md` 覆盖六工具 + mode 冲突 self-correction|

### 关键设计讨论

- **为什么 sectioned 不 JSON？** LLM 对 markdown 顺序的 tool_call 比一次性 JSON structured output 稳定得多；分段 self-correct 路径短
- **为什么 out-of-band view 不进 history？** Artifact 是随时刷新的状态，进 history 意味着每次刷新都挤占 token；带外注入"总是最新 + token 受控"
- **为什么 events 进 history 但 view 不进？** 事件是不可变的"发生过的事"（谁在哪一轮写了什么），view 是"当前状态"——**事件可回放，状态无历史**。这是 event sourcing 的基本区分

## 7. Phase-assert：让沉默违规变可见

- **Status**: 范围限制（§7.5 末段）已被 §12 解除——require_tool 现同时观测 artifact + tracer 事件；retry/nudge/warning 机制与设计意图（detect-and-nudge-and-audit）不变
- **Date**: 2026-04-21

### Context

panel 场景 closing 阶段指令要求"每人发言后调用 `cast_vote(...)`"，但两名 member 只说话没投票，引擎没有任何报警，artifact 里 `v2` 缺失两张选票。这是 LLM 多 agent 系统的普遍问题：prompt 里写的约束，LLM 可能直接跳过，引擎 fire-and-forget。

### Options considered

- **A. 硬失败**（没调工具就 abort phase）：太粗暴，workshop 场景一次 mock 失败整个演示废
- **B. 自动补调工具**（引擎代 agent 调）：违反 agent 自主性，污染语义
- **C. Silent nudge + retry + warning**（选择）：给 agent 一次补救机会，失败打 warning 继续

### Decision

- Scenario phase 声明 `require_tool: <tool_name>`，可选 `max_retries: N`（默认 1）
- 引擎在 phase 结束后扫 `artifact.drain_events()` 的 `tool / caller` 字段
- 未命中 → 追加 nudge instruction "你刚才没有调用 `<tool>`，请现在补上" 作为 per-call argument（**不进 history**，其他 agent 看不见这次辅导）
- 重试用尽 → stderr `WARNING: <agent> skipped required tool '<tool>' after N attempts`
- 终端打 `🔁 [agent] retry k/N: missing <tool>`，workshop 观众能看到流程

核心目标**不是"强制 agent 调工具"（LLM 本质上做不到强制），而是让沉默违规变可见**。

**范围限制**：目前 `require_tool` 只识别 artifact 工具的调用（通过 `artifact.drain_events()` 观测）。non-artifact 工具（如 `retrieve_docs`）尚未被跟踪，待 tool observability 补齐后扩展。

### 对标 linter / roll-call 模式

- **Linter warning**：不阻止编译，但留痕
- **议会 roll-call**：缺席进会议记录
- **AutoGen `GroupChat` speaker selection**：选中的 agent 没给有效回复就 fallback
- **Structured output retry loop**（OpenAI `response_format`、Instructor）：LLM 没按 schema 输出就重试

区别于 structured output retry 的是：本项目对"不调工具"这个**行为级违规**做 retry，而不是对"输出格式不对"做 retry——粒度更粗但覆盖更现实的问题。

### 工程维度评估

|维度|评估|
|---|---|
|内聚度|高——retry + nudge + warning 全聚在 `Discussion._run_turn` 一个方法|
|耦合度|中——依赖 `artifact.drain_events()` 的 `tool / caller` 字段；这是范围限制的根因|
|可观测性 / 可审计性|极高——三层留痕：`🔁 retry` 现场可见、`WARNING` 落 stderr、artifact_event 进 history 可回放|
|LLM 不确定性容忍|极高——承认 LLM 无法被强制，改为 detect-and-nudge-and-audit 模式|
|向后兼容 / 演化友好|完全兼容——未声明 `require_tool` 的 phase 行为不变|
|学习曲线|低——一行 `require_tool: cast_vote`（可选 `max_retries: N`）即声明|
|可测试性|高——`scenarios/example.md` 中 `ballot_nudge` 步覆盖 `require_tool` retry + warning 端到端|

## 8. Tool observability：ToolTracer

- **Status**: accepted
- **Date**: 2026-04-22

### Context

Artifact tools 有完整可观测（events + 终端 emoji），但 **non-artifact tools（当前只有 `retrieve_docs`）完全静默**——终端看不见、transcript 回放不出来、workshop 演示时观众不知道 agent 到底查没查、查了什么。

### Options considered

- **A. tool_call 作为一类 history entry，memory 渲染为 `<tool_call>` block**（最完整）：改动面涉及 4 个 backend client + memory 渲染分支 + summary 策略 + 每轮额外 token，成本高
- **B. 只做对人可见**（选择）：记 transcript event 但 `visible=False`，memory 跳过
- **C. 日志打到独立文件**：割裂，workshop 不友好

### Decision

- `ToolTracer` 类：收集 non-artifact 工具调用，暴露 `drain() -> list[event]`
- **双 sink**：stderr 一行 🔧 emoji（现场可见）+ transcript event 带 `visible=False`（不进 memory，离线回放可用）
- `run.py` 新增 `--save-transcript` 落盘结构化 history（topic/round/phase/speaker/tool_call/artifact_event）
- `tools.is_error` 抽成公共函数，stderr tripwire 和 tracer 的 `ok` 字段对"失败"定义一致
- 所有 entry 统一加 `ts` 字段，时序完整

**同 commit 修的 moderator-first bug**：`roundtable` main 和 `panel` closing 的第二个 phase 用了 `who: all`，`_resolve_who("all")` 返回 `[moderator, *members]`——主持人每轮抢先发言。改用 `who: members`。

### 为什么不走 OpenTelemetry

OTel 精神对齐——两个 sink 对应 live exporter + batch exporter——但没上 OTel 是对的：workshop 项目不应承担 distributed tracing 的部署复杂度。

同理，没让 tool_call 进 memory 是基于成本权衡：

- 改动成本：4 个 backend client + memory 渲染分支 + summary 策略 + 每轮额外 token
- 替代方案：artifact 已能承载"状态性跨 agent 共享"这个最强用例
- 结论：剩余需求等具体驱动场景再说

**显式的"不做"比模糊的 TODO 更有价值**——它阻止了未来对自己的折磨，也展示划边界的能力。

### 工程维度评估

|维度|评估|
|---|---|
|内聚度|高——`ToolTracer` 单一职责：collect + expose `drain()`，约 30 行|
|耦合度|微增——4 个 backend client 都要知道 tracer；观测性本质横切，不上 AOP 不可能零侵入|
|可观测性 / 可审计性|本决策核心产出——双 sink 对等支持现场和回放；`--save-transcript` 落盘结构化数据|
|LLM 不确定性容忍|中性——观测性本身不影响容错，但让容错行为可见|
|向后兼容 / 演化友好|完全兼容——`visible=False` 保证其他 agent 不会在 memory 里看到 tool_call|
|学习曲线|低——对 scenario 作者零感知|
|可测试性|升——结构化 transcript 让回归测试有稳定对照物|

## 9. 扁平 step 列表 + role/tool_owners 显式化

- **Status**: accepted（取代 §1 的 `phases:` 三段 / 顶层 `moderator:` / `members:` 块、§3 的 phase × round 二维结构、§6 内的 `MODERATOR_ONLY_TOOLS` 硬编码）
- **Date**: 2026-04-24

### Context

§3 引入 `opening / main × rounds / closing` 三段 + `phase.round` fallback 链后，schema 的复杂度持续上升：

1. **二维结构形成假权威**：作者倾向把"轮次"理解为讨论节奏，但 `rounds` 实际只对 `main` 阶段有意义；想表达"每轮主持人先发言再让成员讨论"还得在 `main` 列表里同 round 里两条 phase 反复出现，可读性递减
2. **隐式行为太多**：`who: members` 的批量展开顺序、`who: all` 是否包含主持人、`finalize_artifact` / `propose_vote` 是 moderator 专属（硬编码 `MODERATOR_ONLY_TOOLS`），均不在 schema 里
3. **概念膨胀**：moderator 是 schema 里的顶层独立块、members 是另一段、`role` 仅在 artifact tool 过滤里隐式存在；同样是参与者，待遇不一
4. **无位置感知**：删掉 `<round>` marker 后 agent 不知道自己处在第几次发言；保留 marker 又锁死 round/phase 二维抽象

### Options considered

**对流程容器**：

- A. 保留 `phases` 三段，把 `round` 升级为 `iteration`：换汤不换药
- B. State machine（节点 + 边 + 条件）：表达力最强但抬高门槛
- C. Pipeline / DAG：同上
- D. **扁平 `steps` 顺序列表**（选择）：CrewAI Task list 风格，"这一步做什么、谁做"逐项写清

**对位置感知**：

- A. 保留 `<round>` / `<phase>` marker：与扁平 step 矛盾
- B. 全靠 prompt engineering 提示当前 step：不一致、token 浪费
- C. **每 turn 注入 `<turn>turn X of N</turn>` pinned marker**（选择）：N 在启动时静态展开，X 单调递增；token 成本约 9，复杂度极低

**对参与者建模**：

- A. 保留 `moderator` / `members` 两段顶层：迁就过去
- B. **统一 `agents:` 列表，`role: moderator | member` 强制必填**（选择）：所有参与者一视同仁，role 取值池封闭便于 schema 校验

**对 artifact tool 权限**：

- A. 保留 `MODERATOR_ONLY_TOOLS` 硬编码：与"显式优于隐式"原则矛盾
- B. `**artifact.tool_owners` 显式声明，与 `who` 同一套语法（role / all / name 列表）**（选择）：自管自治，零硬编码

### Decision

完整 schema 重构：

- **删除**：`opening / main / closing` 三段、`rounds` / `phase.round`、顶层 `moderator:` 块、`members:` 别名、`MODERATOR_ONLY_TOOLS` 硬编码、CLI `--rounds` 标志
- **新增**：扁平 `steps:` 列表；`agents:` 统一列表 + 强制 `role`；`artifact.tool_owners` 显式 ACL；运行时 `<turn>turn X of N</turn>` pinned marker
- **简化 `who` 语法**：四种字面形态，scalar 走 role/all 寻址，list 走 name 寻址。"`role:` 前缀"和"动态 `by:` stub"双双砍掉
- `**tool_owners` 默认**：未声明 = 全员可调（包含 finalize/propose_vote）；想保留主持人专属必须**显式声明**

新作者写一个最小 scenario 只需要：`agents` 一段、`steps` 一段，每个 step 必填 `who` + `instruction`。

### 行业光谱

|框架|流程容器|参与者|
|---|---|---|
|**CrewAI**|`tasks: list[Task]` 顺序执行（与本项目最像）|`agents: list[Agent]`，role 字段自由|
|**AutoGen**|`initiate_chat` + GroupChat speaker selection（动态）|单 agent 类，无 role|
|**LangGraph**|`StateGraph` 显式节点边|Channel-based|
|**Temporal / Airflow**|长 DAG / Workflow|不适用|

本项目位于 CrewAI ↔ AutoGen 中段：流程是声明式列表（CrewAI），但每 step 内"谁发言"靠 role/all/name 三种寻址（介于 CrewAI 单 agent 和 AutoGen 动态选 speaker 之间）。

### 工程维度评估

|维度|评估|
|---|---|
|内聚度|升——`steps` 一段统揽流程，`agents` 一段统揽参与者，`artifact` 块自管自治|
|耦合度|降——删除 `MODERATOR_ONLY_TOOLS` 这条 artifact ↔ role 的硬编码捷径；ArtifactStore 不再知道 role 概念，权限完全数据驱动|
|可观测性 / 可审计性|升——header 打印 `Steps: M / Total turns: N`，每 turn 终端打 `🗣 [name] (step=<id>)`，transcript 带 `turn` pinned marker 可回放|
|LLM 不确定性容忍|中性——`<turn>` marker 让 agent 知道总长度但不强制行为；`require_tool` retry 闭环未变|
|向后兼容 / 演化友好|破坏性——所有旧 scenario 必须迁移；workshop 项目无外部消费者，可控|
|学习曲线|降——心智模型从"phase × round 二维"压缩到"steps 一维顺序展开成 turns"|
|可测试性|持平偏升——`test_phase.md` 删除（其原概念不存在）；原 `test_phase_assert` / `test_memory` / `test_artifact` / `test_vdb` 并入 `scenarios/example.md`|

### 关键设计讨论

- **为什么不保留 `phase` marker？** Phase 本质是给人看的章节标签，agent 用 `<turn>X of N</turn>` 已能感知位置；避免引入"agent 必须理解 phase 语义"的隐式契约
- **为什么 `tool_owners` 默认全员可调？** 与"显式优于隐式"一致——任何"专属"语义都应在 schema 里看得到，不靠代码内置默认
- **为什么 `who` scalar 同时支持 role 和 `all`？** scalar 是"按属性匹配多个 agent"，list 是"按名字精确点名"——两个语义靠**类型**区分，零歧义；`all` 是 role 的并集，并入 scalar 一起处理代码最简
- **为什么不强制至少 1 个 moderator？** 极简场景（`test_`* 等）可全员 member；要求 moderator 会污染纯 member 测试。`who: moderator` 命中 0 个是 fail-fast 错误，间接拦住"声明了 moderator 寻址却没 moderator"的 bug

### 已知 trade-off

- **批量寻址内部顺序"按 agents 声明顺序"是约定**——文档明写，不靠 schema 强制。如果作者在意公平性，应使用显式 list 形态自己排序
- `**<turn>` marker 占少量 token**——总 turn 数固定时影响极小；超长场景可改 `turn 23` 省去 `of N`，等驱动场景出现再优化

## 10. 库化拆分：`Scenario` / `Engine` / CLI，取代一体式 `run.py`

- **Status**: accepted（取代 `run.py` 一体式脚本入口；`python -m agent_engine` 与 `Engine.invoke` 共享同一装配路径）
- **Date**: 2026-04-26

### Context

`run.py` 一度承担过多职责：解析 scenario、校验 schema、装配 Agent / Memory / tools / artifact、构造 `Discussion`、跑主循环、可选落盘 transcript 与 artifact、打印 CLI 帮助。对「命令行跑一次 demo」足够，但随着 `play/workflow/`、`play/qa_assets/` 等要把同一套讨论引擎**嵌进 pipeline**、或写**不经过终端**的回归实验，一体式脚本带来问题：

1. **可导入边界不清**：库消费者要么 subprocess 调 `python run.py`（观测与调试差），要么复制粘贴装配逻辑（双 SoT）
2. **测试与组合根纠缠**：想单测「装配是否正确」必须 import 带副作用的 main
3. **I/O 与领域逻辑同文件**：JSON / markdown 落盘、stdout 流式开关与 orchestration 绑死，库调用方难以只取 `Result` 而禁止写盘

### Options considered

- **A. 保留 `run.py`，旁路新增 `api.py` 薄封装**：实现快，但装配逻辑仍易漂移成两份（run 改了 api 忘改）
- **B. 只抽 `Engine`，`Scenario` 仍散落为函数**：`Engine` 不知道 `Assembly` 边界，调用方仍要手动拼 `Discussion` 参数
- **C. `Scenario`（解析 + 校验 + `assemble`）+ `Engine`（`invoke` 编排 + `Result`）+ `cli.py` 极薄适配**（选择）：单一装配路径、单一运行入口、CLI 与库共享代码路径
- **D. 上完整 async `ainvoke` / `stream` 再发布**：表达力强，但 workshop 尚无真实 async 调用栈，属过早抽象

### Decision

- **`Scenario`**：`from_yaml(path)` 读 frontmatter + body，启动期校验；`assemble() -> Assembly`（agents、roles、`steps`、`topic`、`ArtifactStore?`、`ToolTracer?`）——**配置域 → 运行时对象图**的唯一点
- **`Engine`**：持有 `Scenario`；`invoke(...)` 内顺序为 `assemble()` → 可选 `initial_artifact` 种子（绕过 artifact tool ACL，供测试或 pipeline 预热）→ 构造 `Discussion.run()` → 组装 `Result`（`artifact` 快照、`transcript`、`success` / `warnings`）→ 按参数写 transcript JSON / artifact markdown → 触发 `Callback.on_run_finished(RunFinished)`。`ainvoke` / `stream` / `astream` **显式 `NotImplementedError`**，避免半吊子 async API 误导调用方，留待有真实驱动再填
- **`cli.py`**：`argparse` → `Scenario.from_yaml` → `Engine.invoke`，与库路径完全一致
- **包导出**：`__init__.py` 暴露 `Scenario`、`Engine`、`Result`、`Callback`，把**公共契约**收敛到固定 surface

### 为什么这么选

- 与 README 指导原则「显式优于隐式」「装配点集中」对齐：`assemble` 与 `invoke` 分工比「一个上帝脚本」更易陈述不变量
- **C** 相对 **A** 消除双 SoT；相对 **B** 把 `Discussion` 构造参数留在引擎内，嵌入方只传 `Scenario` + kwargs
- **D** 推迟：与「抽象引入滞后于第二个具体案例」一致——先有同步 `invoke` 的嵌入方，再为 async/stream 选模型（LangGraph 式 event stream vs 简单 queue）

### 后果与收益

|维度|影响|
|---|---|
|内聚度|**升**——校验/装配只在 `scenario.py`，单次 run 生命周期只在 `engine.py`|
|耦合度|**降**——`Discussion` / `Agent` 不 import CLI；workflow 只依赖 `agent_engine` 包名即可|
|可观测性 / 可审计性|**持平偏升**——`Result` 把 `warnings` 与 `success` 结构化；落盘路径由调用方显式传入，便于 CI 写临时目录。`Callback` 已预留 `on_step_start` 等钩子，当前仅接线 `on_run_finished`，与未实现的 `stream()` 同轨演进|
|LLM 不确定性容忍|**中性**——运行语义未改，仍由 `Discussion` + memory + tool 闭环承担|
|向后兼容 / 演化友好|**破坏性**——依赖 `python run.py` 或 import `run` 的外部脚本需改为 `python -m agent_engine` 或 `Engine.invoke`；本仓库内无对外发布承诺，可接受|
|学习曲线|**略升**——新读者多跳一层 `Scenario`→`Engine`；README 与 mermaid 已以之为 SoT|
|可测试性|**升**——可构造 `Scenario`（或未来内存 fixture）后直接 `Engine.invoke(print_stream=False)` 断言 `Result.transcript` / `warnings`，无需子进程|

## 11. CLI envelope（`--save-result-json`）+ artifact_event 加 `arguments`：phase 5 evals 跨项目接口

- **Status**: accepted（cross-link [`play/evals/DECISIONS.md` §5](../evals/DECISIONS.md)）
- **Date**: 2026-05-03

### Context

`play/evals` phase 5（族 5 agent trajectory 完全体）需要把 agent_engine 的运行轨迹喂给 5 个 trajectory metric（`task_success` / `tool_call_set_f1` / `argument_correctness` / `trajectory_match` / `trajectory_coverage`）。按 monorepo 解耦原则（DECISIONS §4，subprocess + JSON envelope）evals 不能 `from play.agent_engine import Engine`，必须走 CLI + 标准 JSON envelope。现有 `--save-transcript` / `--save-artifact` 是**人类导出格式**（JSON list / Markdown），不是机器消费格式：

1. `--save-transcript` 只夹 `history`，丢失 `artifact / warnings / success` —— evals 端 `task_success` 谓词与 `trajectory_coverage` 都需要
2. `--save-artifact` 只夹 markdown，零结构化字段
3. evals 端要重新拼这些字段意味着两边都要维护"`Result` 拍扁规则"——双 SoT

另一个独立问题：`artifact.py` 中 5 个 event handler 在 `_events.append({...})` 时只塞 `tool / caller / content / ts`，**丢弃了 `arguments`**。`content` 是为 LLM memory 渲染做的人类可读字符串（如 `"caller wrote section 'X' (140 chars)"`），评测层无法从中复原参数。phase 5 `argument_correctness` 在 run 路径需要参数级匹配能力，必须能在 transcript 里看到原始 args。

### Options considered

|项|做法|权衡|
|---|---|---|
|A. evals 用 `--save-transcript` + `--save-artifact` 自拼|两份磁盘 IO + 两次解析；`success` / `warnings` 仍需第三 channel|双 SoT；evals 端要复刻 `Result` 字段|
|B. envelope 直接 print 到 stdout|与 `play/rag/query.py --json` 同形|agent_engine 整段讨论刷 stdout，envelope 寄生 stdout 不可行|
|**C. 加 `--save-result-json PATH`，`dataclasses.asdict(Result)` 落 JSON file**（选择）|与 `--save-transcript` / `--save-artifact` 并列；evals 端单点解析|新增 1 flag、~15 行 cli.py|
|D. 改造 `Result` 序列化为 protobuf/MsgPack|快、紧凑|workshop 体量不需要；JSON 调试更友好|

artifact_event 加 `arguments` 字段的 options：

|项|做法|权衡|
|---|---|---|
|A. evals 端从 `content` 字符串反向解析|"`caller wrote section 'X' (140 chars)`" → `{"name": "X"}` 等|脆弱；正则维护|
|**B. event dict 直接塞 `"arguments": dict(args)`**（选择）|5 个 handler 各加 1 行|有信息冗余（args 已在 LLM 调用时见过，但 transcript 重做永久记录）；老消费者忽略未知键，纯 additive|
|C. 在 `Discussion` 层另存 `tool_log: list[ToolCall]`|额外结构|两个真相源（events vs tool_log）容易漂移|

### Decision

|动点|做法|
|---|---|
|`cli.py`|加 `--save-result-json PATH`：`dataclasses.asdict(result)` → `json.dump(... , ensure_ascii=False, indent=2)`；目录不存在自动 mkdir|
|`result.py`|不动（`Result` 已是 frozen dataclass，`asdict` 直接可用，不引薄 `to_dict()` 方法）|
|`artifact.py`|5 个 event handler（`_h_write_section` / `_h_append_section` / `_h_propose_vote` / `_h_cast_vote` / `_h_finalize_artifact`）各加 `"arguments": dict(args)`（propose_vote 因 args 已被 helper 拆开，重新组装 `{"question": question, "options": list(options)}`）|
|不动|`--save-transcript` / `--save-artifact` 保持原有人类格式；老消费者零影响|

evals 端对应消费契约：

- `play/evals/models/agent_engine_run.py::make_run_fn`：`subprocess.run(["python", "-m", "agent_engine", scenario, "--no-stream", "--save-result-json", tmp])` → 读 envelope JSON
- `play/evals/tasks/agent_traj.py::_pin_trajectory(doc, envelope)`：从 envelope 派生 `tool_calls`（识别 `tool_call` + `artifact_event` 两类条目）/ `tool_seq` / `decision`（从 finalize_artifact 的 `arguments['decision']` 抽）
- `play/evals/tests/test_agent_traj_envelope.py`：锁 `dataclasses.fields(Result) == {artifact, transcript, success, warnings}`——agent_engine 改字段名时 evals CI 即时 fail

### 为什么这么选

- 与 README 指导原则「显式优于隐式」「契约层稳定」对齐：envelope 的 4 字段就是 `Result` dataclass 字段，无第二份契约
- `asdict` 直出避免维护 `to_dict()`：dataclass 已是最小单点信息源
- artifact_event 加 `arguments` 是 ~5 行 additive 改造，且让 transcript 永久持有"agent 当时调了什么"的完整快照，对 audit / replay / metric 都有用，超出 evals 单一消费者收益

### 后果与收益

|维度|影响|
|---|---|
|内聚度|**升**——`Result` 是跨项目唯一契约，envelope = `asdict(Result)`|
|耦合度|**降**（隔进程）——evals 与 agent_engine 通过文件级 JSON 解耦；任一方改实现，另一方靠 envelope 字段集合断言抓回归|
|可观测性|**升**——transcript 永久持有原始 args，事后 replay / 审计 / 指标都能复用同一记录|
|向后兼容|**纯 additive**——新增 flag 缺省关闭；artifact_event 多个键被老消费者忽略|
|演化友好|**升**——未来 `Result` 加字段时 envelope 自动同步；evals 端 `test_agent_traj_envelope` 自动断言新形状|
|cross-project 测试 gate|evals 端 `conftest.py::agent_engine_required` 双 gate（ollama-probe + brainstorm.md 存在性）；缺任一即 skip + 友好提示|

## 12. require_tool 观测面扩展：覆盖 tracer 事件（非 artifact 工具也算）

- **Status**: accepted（supersedes §7 末段"范围限制"）
- **Date**: 2026-05-09

### Context

`play/agent_sft` Phase 1 在 [`code_review.md`] / [`tool_chain.md`] 引入 `require_tool: retrieve_docs` —— retrieve_docs 是非 artifact 工具，走 `ToolTracer.record` 写入 `tool_call` 事件，不进 `artifact.drain_events()`. 老 `_run_turn` 的检查仅扫 artifact 事件流，导致非 artifact 工具的 require_tool 永远判定为"沉默"，nudge 必触发 + warning 必落 → 度量信号被打成常数 1.0，丧失诊断价值. §7 已显式承认此限制（"待 tool observability 补齐后扩展"），§8 ToolTracer 落地后扩展条件已具备，本期补全.

### Options considered

|项|做法|权衡|
|---|---|---|
|A. 让 require_tool 仅适配 artifact 工具|status quo|限制 require_tool 表达力；agent_sft Phase 1 必须避开非 artifact 工具|
|B. 让 ToolTracer 也写入 artifact._events|跨语义污染；混合 artifact 真实事件与 tool 调用|破坏"artifact = 共享文档真实变更"的语义边界|
|**C. `_run_turn` 同时收集 tracer + artifact 两路事件传给 `_called_tool`**（选择）|3 行改动，`_called_tool` 不变（已是 tool/caller 平铺检查）|`tracer_events` 与 `artifact_events` 在 history 仍各自分批 extend；require_tool 检查面合并；纯 additive|
|D. 引入"统一事件总线"重构|结构干净|workshop 体量过度设计；C 已足够|

### Decision

|动点|做法|
|---|---|
|`discussion._run_turn`|`tracer_events = self.tracer.drain() if self.tracer else []`；`artifact_events = self.artifact.drain_events() if self.artifact else []`；`events = tracer_events + artifact_events` 喂 `_called_tool`|
|`tracer.py` / `artifact.py`|不动——事件 schema 已是 tool/caller 平铺，与 `_called_tool` 现有契约对齐|
|`_called_tool`|不动——本就是工具中立的 `tool/caller` 检查|

### 行业光谱

- **AutoGen GroupChat**：speaker selection 失败 fallback，没有"指定工具是否被调"维度
- **LangGraph tool_use 节点**：可显式声明 next_tools，强一致；但需要图重构，比 nudge 重
- **OpenAI structured output retry / Instructor**：对"输出格式不对"做 retry，粒度比 require_tool 细但语义不同
- 本项目 §7 + §12 是"行为级 require_tool + 工具中立观测"——比 AutoGen 强（声明性），比 LangGraph 弱（不强制），自成一档

### 工程维度评估

|维度|影响|
|---|---|
|内聚度|不变——仍是 `Discussion._run_turn` 一个方法|
|耦合度|**降**——`_called_tool` 不再隐式假设"事件源 = artifact"；事件源切换不影响检查面|
|可观测性|**升**——非 artifact 工具的 require_tool 行为现在可被度量（agent_sft 的 nudge_fire_rate 关键依赖）|
|向后兼容|**纯 additive**——只 require_tool=artifact 工具的老 scenario 行为字节相同（tracer_events 是空集，合并后等于 artifact_events）|
|演化友好|**升**——未来加新工具子系统（如 RAG / DB / web search）只要走 tracer 即自动接入 require_tool|
|度量价值|**升**——`play/evals/tasks/nudge_fire_rate` 的 `by_tool` breakdown 从此能含 retrieve_docs 桶|

### 与 §7 / §8 的关系

- §7 立的设计意图（detect-and-nudge-and-audit）+ 三层留痕（🔁 / WARNING / event in history）完全保留
- §8 引入 ToolTracer 后，"非 artifact 工具被记录"的能力已具备，但本检查面未消费——本 ADR 是补齐这条 wiring，不是新立机制
- §7 末段的"范围限制"段落 Status 同步 supersede，让旧条目仍可读但指向新解

## 13. 公开 transcript / scenario 解读 API：Result/Scenario 暴露 typed 视图

- **Status**: accepted（extends §11）
- **Date**: 2026-05-11

### Context

§11 定 envelope 是跨项目机器消费的 SoT（`Result` 字段层）；但 envelope 只覆盖**字段** schema（`{transcript, artifact, warnings, success}`），不覆盖**字段解读**——transcript 内的 `tool_call` / `artifact_event` 怎么规约成"工具调用"、`<turn X of N>` marker 怎么切段、scenario YAML 怎么静态展开成 `(turn_idx, agent, step_id, require_tool)` 序列，全部由各消费者反向工程：

| 消费者 | 解读什么 | 在哪里 | 问题 |
|---|---|---|---|
| `play/evals/tasks/agent_traj.py` | transcript → tool_calls / decision | `_extract_tool_calls / _extract_decision` | 与 agent_engine artifact 事件 schema 紧耦合，agent_engine 改字段会静默失败 |
| `play/evals/metrics/nudge.py` | scenario YAML → expected turn 列表 + transcript 切段 | `derive_expected_turns / split_turns / _split_attempts / _resolve_who_to_agents` | **重复实现** `Discussion._expand_steps + _resolve_who` 整套展开逻辑 |
| `play/agent_sft/data/extractor.py` | 同上 + 再加 turn-indexed 全局 offset | `_index_steps_by_turn / _split_turns_indexed` + **`sys.path.insert + from evals.metrics.nudge import _split_attempts, _resolve_who_to_agents, _split_frontmatter, derive_expected_turns`** | 跨项目 import 4 个**私有**函数，反模式 |

三处独立解读 ⇒ schema 改动需要同步三处；`play/agent_sft` 的 `sys.path.insert` 黑魔法又引入 evals 私有面依赖，monorepo 解耦原则破洞.

### Options considered

| 项 | 做法 | 权衡 |
|---|---|---|
| A. 现状 | 各消费者各自解读 | schema 改动 → 三处改 + 反模式持续 |
| B. 把 `_extract_tool_calls / split_turns / derive_expected_turns` 三个公开 plain function 加到 `agent_engine.transcript` / `agent_engine.scenario_static` 模块 | 简洁，函数式 | 每加一个新视图都要新模块；返回 `dict / list[dict]` 类型语义弱 |
| **C. 在 `Result` / `Scenario` 上加视图方法返回 typed dataclass**（选择） | OO 风格，与 OpenAI Agents SDK `RunResult.new_items` / Anthropic `Message.content[ToolUseBlock]` / inspect_ai `ChatMessageTool` 同精神 | 需要新增 typed dataclass（`ToolCall` / `TurnView` / `ExpandedTurn`），但 ergonomic 与扩展性最好 |
| D. 抽出独立 `agent_engine_views` 包给所有消费者 import | 完全解耦 | workshop 体量过度设计；C 已足够 |

### Decision

**C：让 schema 解读权随 schema 一起住在 agent_engine。**

| 动点 | 做法 |
|---|---|
| `result.py` 加 `ToolCall` (frozen) / `TurnView` (frozen) | typed view，与 OpenAI Agents SDK / inspect_ai 风格对齐 |
| `Result.from_dict / load_json` | envelope dict / `--save-result-json` 文件 → Result，缺字段降级（向后兼容老 envelope） |
| `Result.tool_calls() / turns() / speakers() / find_finalize_decision()` | transcript 视图：tool 调用合并 / turn 切段 / speaker 集 / finalize decision 抽取 |
| `TurnView.attempts(agent) / tool_calls()` | 段内按 speaker 切 attempt（与 `_run_turn` retry 循环对齐）+ 段内工具调用 |
| `TurnView.start_offset: int` | 段第一个 entry 在原 transcript 的 0-based 全局索引——给 `agent_sft` extractor 复刻 turn-indexed offset 用，其它消费者可忽略 |
| `scenario.py` 加 `ExpandedTurn` (frozen) + `Scenario.expanded_turns()` | 静态展开 `steps:` 为线性 turn 序列，不实例化 Agent / 不跑 LLM |
| `scenario.py` 抽 `_resolve_who_names(who, declared_order, role_by_name)` 纯函数 | `Discussion._resolve_who` runtime 路径 + `Scenario.expanded_turns()` 静态路径**共用此函数**——保证两条路径展开顺序字节相同 |
| `play/agent_engine/tests/` 首测试目录 | 36 测试覆盖 8 类视图行为 + 7 现网 scenario 上 `expanded_turns()` 长度 / pair 序列与 `Discussion._expand_steps()` 字节相同（关键 invariant 锁） |
| `play/evals` PR-1 shim | `metrics/nudge.py` / `tasks/agent_traj.py` 内部改调新 API，**公开签名零破坏**——evals/agent_sft 现有 pytest 全绿 |
| `play/evals/_ae_bridge.py` | 集中 sys.path 注入 + `from agent_engine import ...` re-export，避免各模块各自 sys.path 黑魔法 |
| PR-2（后续） | 删 evals 私有 `_xxx` shim + agent_sft `extractor.py` 直连 agent_engine + 三 DECISIONS / JOURNAL 同步 |

### 行业光谱

| 框架 | 类似机制 | 对比 |
|---|---|---|
| OpenAI Agents SDK | `RunResult.new_items: list[RunItem]`（`MessageOutputItem / ToolCallItem / ToolCallOutputItem`） | 同精神：把"运行产物"转 typed view 给消费者；本项目 `Result.tool_calls / turns` 是裁剪版 |
| Anthropic Messages API | `Message.content: list[TextBlock \| ToolUseBlock]` | typed union 直接消费；本项目 `ToolCall.kind: "artifact" \| "tracer"` 同思路 |
| inspect_ai | `ChatMessage` / `ChatMessageTool`（dataclass） + `EvalSample.messages` | typed message + sample API 强类型；本项目走更轻量的 view 方法路线 |
| LangChain `AIMessage.tool_calls` | typed `ToolCall(name, args, id)` 字段 | 同语义，更细一层（含 id）；本项目当前不需要 id（artifact / tracer 事件无对应字段） |
| dspy `History` | 弱类型 message dict | 落后形态——本项目 §11 envelope 起点已超过 |

### 工程维度评估

| 维度 | 影响 |
|---|---|
| 内聚度 | **升**——schema 与 schema 解读住一起 |
| 耦合度 | **降**——评测 / 训练数据挖掘消费者不再反向工程 transcript / scenario，agent_engine 字段改动一处同步所有消费者 |
| 可观测性 | **升**——`ToolCall.kind` 显式区分 artifact vs tracer，比原来的 `entry["type"]` 字符串更安全 |
| 向后兼容 | **PR-1 纯 additive**——`Result` 仍是 `dataclasses.asdict` 序列化的 envelope SoT，新加方法不影响序列化形状；evals/agent_sft 公开签名零变；老 envelope 缺字段时 `Result.from_dict` 给默认值 |
| 演化友好 | **升**——未来 `Result` 加新视图（如 `Result.warnings_by_kind() / Result.timeline()`）有清晰 home；`Scenario` 加新静态分析（如 `Scenario.required_tools_set()`）同理 |
| pytest 安全网 | 36 测试一次性建立，含 `expanded_turns()` 与 `Discussion._expand_steps()` 在 7 现网 scenario 上的字节同源 invariant 锁 |
| 跨项目 import 卫生 | **升**——`play/agent_sft` PR-2 起从 evals 私有面 import 退出，仅留 `from evals.metrics.nudge import classify_failure_mode`（合法公开面） |

### 与 §11 的关系

§11 立"`Result` 是 envelope SoT"——**字段** schema 由 `dataclasses.fields(Result)` 自描述. §13 是 §11 的纵向延伸：**字段解读**也由 `Result` 上的方法自描述. 两条 ADR 合起来 = "schema + schema 解读权都在 agent_engine 一处"，跨项目契约监控成本接近零.

### 与 PR-2 的关系

PR-2 是 §13 的清理收尾：删 evals 私有 `_xxx` 函数 shim + 改 `play/agent_sft/data/extractor.py` 直连 `agent_engine.Scenario / Result`，并在 [`play/evals/DECISIONS.md`] / [`play/agent_sft/DECISIONS.md`] 同步 ADR；§13 自身 PR-1 落地后即生效，PR-2 是补完而非阻塞.

## 14. Transcript entry typed union + envelope token usage（schema 自身 typed 化）

- **Status**: accepted（破坏性 schema 升级；与 §13 互补——§13 立"解读 SoT"，§14 立"schema 自身 typed 化"）
- **Date**: 2026-05-11

### Context

§11 把 `Result` 立成跨项目机器消费 envelope SoT；§13 把 transcript / scenario 解读权收回 agent_engine. 但 transcript 内部仍是 `list[dict]`——entry 的字段名 / 必选性 / 类型只在写入点散落，没有任何静态约束：

| 现象 | 根因 |
|---|---|
| `Result.transcript[i]["speaker"]` / `entry["type"]` / `entry["tool"]` 三种风格混用 | 5 个写入点（`discussion.py` 3 / `tracer.py` 1 / `artifact.py` 5 / `memory.py` 1）各写各的 dict 字面量，无 schema 强制 |
| speaker reply 没有 `"type"` 字段，只能靠"`'speaker' in entry`"反向辨识 | 写入点 `discussion.py` L107-111 的历史遗留——其它 entry 全部带 type tag，唯独 speaker reply 没有 |
| 评测做 LLM cost / token analytics 必须解析 stderr / `🔧` emoji 反推 | envelope 不含 `usage` 字段，4 LLM client 也只 `return text` 不返 usage 数据 |
| `Result.from_dict` 对缺字段降级返 `Result()`（§13 引入的 backward-compat） | 旧 envelope 与新 envelope 共存的代价；现已确认无外部消费者，可强制升级 |

行业当前形态（参考 §13 同光谱）：OpenAI Agents SDK / Anthropic Messages API / inspect_ai 的"消息 / 事件"全部是 typed union；usage 字段（OpenAI `usage`, Anthropic `usage.cache_read_input_tokens`, Gemini `usage_metadata`）是 1st-class 字段而非 stderr 副产物.

### Options considered

对 transcript entry typed 化：

| 项 | 做法 | 权衡 |
|---|---|---|
| A. 保留 `list[dict]`，仅给 entry 加 TypedDict | 类型注解多但运行时仍是 dict | 静态 mypy 通过但运行时仍可写错字段 |
| B. 单 `TranscriptEntry` dataclass，type 字段决定其余字段含义 | 字段最少 | 等价 dict + 假 type 注解，差距小 |
| **C. typed union（6 个 frozen dataclass + `TranscriptEntry = TopicEntry \| TurnEntry \| ...`）**（选择）| 与 OpenAI Agents SDK / Anthropic Messages 同形 | runtime `isinstance` 派发，IDE 推导友好；新增 entry 类型是加新 dataclass + 一行 union 扩张，无 schema 调用面修改 |
| D. 上 protobuf / msgspec | 跨语言 / 性能强 | workshop 体量过度设计；JSON debug 友好性下降 |

对 token usage：

| 项 | 做法 | 权衡 |
|---|---|---|
| A. 不加；评测自行从 stderr 解析 | 零侵入 | 4 client 都要变更 stderr 格式才能稳定解析；脆弱 |
| B. 在 `Discussion._run_turn` 包 LLM 调用计时，估算 token | 1 处改动 | 估算误差大；cached_tokens 拿不到 |
| **C. 4 client 的 `chat()` 改返 `(text, TokenUsage)` 二元组，逐次 LLM 调用落 `Result.usage`**（选择）| typed 单点，与 SDK 原生 usage 字段对齐 | 4 client 要改返回签名；流式调用要在 stream consumed 完取 final chunk usage，工作量中等 |

对 backward-compat 处理：

| 项 | 做法 | 权衡 |
|---|---|---|
| A. `Result.from_dict` 继续兼容老 envelope（缺字段降级） | 旧数据可读 | 与 §13 一致但与"forward-only 升级"用户决策不一致 |
| **B. 严格化：缺字段直接 `KeyError`**（选择，用户已确认）| schema 不可降级 | 旧 envelope 不可读，需要重跑 / 迁移；本仓库无外部消费者，可控 |

### Decision

**transcript entry → 6 个 `frozen=True` dataclass + 1 个 `Union`；envelope 加 `usage: list[TokenUsage]`；4 LLM client 全量 typed usage 抓取；`from_dict` 严格化.**

| 动点 | 做法 |
|---|---|
| `result.py` 加 `TopicEntry / TurnEntry / SpeakerEntry / ToolCallEntry / ArtifactEventEntry / SummaryEntry`（均 frozen，均带 `type: Literal[...] = ...` 默认值字段）| typed union，`SpeakerEntry` 强制带 `type="speaker"` 修复历史遗漏 |
| `result.py` 加 `TokenUsage(model, caller, input_tokens, output_tokens, cached_tokens, duration_ms, ts)` frozen | typed cost 信息源，与 OpenAI / Anthropic / Gemini / Ollama 四方 SDK usage shape 收敛 |
| `Result.transcript: list[TranscriptEntry]` + `Result.usage: list[TokenUsage]` | 字段层 typed 化 |
| `Result.from_dict` 不再降级；`data["transcript"] / data["usage"] / data["artifact"] / data["success"] / data["warnings"]` 缺一即 `KeyError` | schema 不可降级；旧 envelope 失效 |
| `_entry_from_dict(d)` 按 `d["type"]` dispatch 进 `_ENTRY_BY_TYPE` 字典；未知 type → `KeyError` | 反序列化路径 typed |
| `_entry_to_tool_call` 改 `isinstance(entry, ArtifactEventEntry/ToolCallEntry)` typed 派发 | §13 的 `ToolCall` 视图与新 entry 类型自动对齐 |
| `TurnView.entries: tuple[TranscriptEntry, ...]` + `TurnView.attempts(agent)` / `Result.speakers()` 全部 `isinstance(e, SpeakerEntry)` | typed 视图层与新 schema 字节同源 |
| 5 个 entry 写入点（`discussion.py` 3 / `tracer.py` 1 / `artifact.py` 5 / `memory.py` 1）切到 dataclass 实例化 | 移除 dict 字面量散落 |
| 4 LLM client（`openai/anthropic/gemini/ollama_client.py`）`chat()` 签名 + `chat()` `caller: str` 形参 + 返 `tuple[str, TokenUsage]` | typed usage 抓取，跨 backend 收敛 |
| `agent.py::Agent.respond` 接二元组 + 累计 `list[TokenUsage]`；`memory.py::ConversationMemory.drain_usage()` 把 SummaryMemory 内部 summarizer 调用产生的 usage 也回收 | 用户感知不到的 LLM 调用（如 SummaryMemory 折叠）也进 `Result.usage` |
| `Discussion.usage: list[TokenUsage]` + `Engine.invoke` 写入 `Result.usage` | 串到 envelope 出口 |
| `engine.py` 写盘前 `[dataclasses.asdict(e) for e in history]` | typed entry → JSON 序列化 |
| `_ae_bridge.py` re-export `TokenUsage / TopicEntry / TurnEntry / SpeakerEntry / ToolCallEntry / ArtifactEventEntry / SummaryEntry` | evals 端零 sys.path 黑魔法可拿到所有 typed 视图 |
| `agent_sft.data.extractor` 删 `_split_turns_indexed / _index_steps_by_turn` 共 2 个 shim | typed `Result.turns()` 已能直接给 `start_offset`，shim 失去存在意义 |
| 全量 fixture 改写（~200 inline dict 跨 7 测试文件）+ 新增 `agent_engine/tests/test_token_usage.py` 等 | typed schema 在测试面也强制 |
| `play/evals/data/{agent_traj,nudge_fire_rate,...}/predictions/*.jsonl` + `agent_sft/data/triples/runs_1k_fast_7b_r0_124/*.json` 一次性迁移脚本注入 `type:"speaker"` + `usage: []` | 用户已确认 forward-only；迁移脚本一次性，不进代码留痕 |

### 行业光谱

| 框架 / SDK | typed message / event | usage shape |
|---|---|---|
| OpenAI Agents SDK | `RunResult.new_items: list[MessageOutputItem \| ToolCallItem \| ToolCallOutputItem]` | `response.usage.{prompt_tokens, completion_tokens, prompt_tokens_details.cached_tokens}` |
| Anthropic Messages | `Message.content: list[TextBlock \| ToolUseBlock]` | `message.usage.{input_tokens, output_tokens, cache_read_input_tokens}` |
| Google Gemini | `GenerateContentResponse.candidates[].content.parts: list[Part]` | `response.usage_metadata.{prompt_token_count, candidates_token_count, cached_content_token_count}` |
| Ollama | dict 形态（`role`, `content`, `tool_calls`） | response dict 含 `prompt_eval_count` / `eval_count` |
| inspect_ai | `ChatMessage` / `ChatMessageTool`（dataclass） + `EvalSample.token_usage: dict[str, ModelUsage]` | typed |
| LangChain `AIMessage.usage_metadata` | typed `UsageMetadata(input_tokens, output_tokens, total_tokens)` | 同思路 |

本项目 §14 的 `TranscriptEntry` typed union + `TokenUsage` per-call list 与 inspect_ai / LangChain 同精神，比 OpenAI Agents SDK 更扁平（无嵌套 RunItem 层）；多 backend 适配统一抹平四方 SDK 字段差异，对应到一致的 `input_tokens / output_tokens / cached_tokens / duration_ms`.

### 工程维度评估

| 维度 | 影响 |
|---|---|
| 内聚度 | **升**——entry schema、entry 写入点、entry 解读权全部在 `result.py` 一处；`TokenUsage` 在 client 一处抓、`Result` 一处展现 |
| 耦合度 | **降**——评测 / SFT 端再也无 `entry.get("speaker")` / `entry["type"]` 字符串嗅探；`isinstance` 编译期能抓错 |
| 可观测性 | **升**——`SpeakerEntry.type="speaker"` 让 transcript 解读 100% 走 type tag 路径，无歧义；token usage 永久落 envelope，cost / efficiency / latency 度量从 stderr 反推升级到字段级直接消费 |
| LLM 不确定性容忍 | **持平**——schema 改造不影响运行时容错；流式取 usage 拿不到时填 0，evals cost 计算降级到已有 `efficiency.py` 路径 |
| 向后兼容 | **破坏性**——old envelope 不可读；`from_dict` 强制 schema；4 LLM client `chat()` 签名变；本仓库无外部消费者，迁移脚本一次性处理已 mined 数据 |
| 演化友好 | **升**——加新 entry 类型 = 加 dataclass + 一行 union 扩张 + 写入点；schema 是 SoT，加字段 = `dataclasses.fields` 自动同步 |
| pytest 安全网 | 三项目 585 测试全绿（agent_engine 42 / evals 456 / agent_sft 87） + evals smoke `agent_traj` / `nudge_fire_rate` 通过 + 现网 envelope round-trip 通过 |

### 与 §11 / §13 的关系

| ADR | 立的是什么 |
|---|---|
| §11 | `Result` 是 envelope SoT（**字段** schema：`{transcript, artifact, warnings, success}`） |
| §13 | `Result` / `Scenario` 暴露 typed 视图（**字段解读**：`tool_calls / turns / expanded_turns`） |
| §14 | `transcript` 内部本身 typed union + `usage` 字段（**schema 自身**也 typed） |

§11 + §13 + §14 三层合起来：跨项目契约从"字段名 → 字段解读 → 字段值类型"全部由 `agent_engine` 一处自描述，下游消费者无需任何反向工程.

### 不在范围

- `Result.usage` 不做"按 caller / model 聚合"——`evals/metrics/efficiency.py` 已有聚合路径，agent_engine 只产出 raw usage list
- 流式调用拿不到 usage 时填 0（cached_tokens 也填 0）——不抛异常；evals cost 计算自动降级到 model-level 估算
- `TokenUsage.duration_ms` 用 `time.monotonic()` 包 client 调用计时；不区分 first-token-latency vs total-latency（workshop 体量未需要）
- 老 envelope 反向兼容 reader 不写入仓库——一次性迁移脚本足够，不留 `try_legacy_from_dict()` 这种长期 shim

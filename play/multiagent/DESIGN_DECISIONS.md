# Multiagent Engine — 设计决策记录

本文档记录 `play/multiagent/` 的系统设计决策，按时间顺序积累。每节包含背景、候选方案、选择依据、行业光谱位置，以及按统一工程维度的评估。维度定义见附录。

## 指导原则

贯穿本项目的设计原则：

1. **Shared transcript + per-agent projection**：对话数据只有一份权威视图，各 agent 按自身需求投影
2. **显式优于隐式**：配置能声明的不靠代码推断；LLM 行为能结构化约束的不靠 prompt 约定
3. **承认 LLM 不确定性**：不把 LLM 当确定性函数，容错设计（retry / self-correct / audit）优先于强制
4. **装配点集中**：`run.py` 作为 composition root，其他模块不互相 import 具体实现
5. **抽象引入滞后于第二个具体案例**：不为未来需求预留抽象，等第二个使用者出现时再抽

---

## 1. Phase-driven scenario 配置

- **日期**：2026-04-14

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


| 维度          | 评估                                                         |
| ----------- | ---------------------------------------------------------- |
| 内聚度         | 高——引擎、scenario、参与者定义各司其职                                   |
| 耦合度         | 显著降低——引擎只依赖"phases"抽象                                      |
| 可观测性 / 可审计性 | 中——stdout 有 phase/round 打印，未结构化                            |
| LLM 不确定性容忍  | 中——启动期校验作者输入；LLM 运行时输出未约束                                  |
| 向后兼容 / 演化友好 | 项目起点，无旧行为需兼容                                               |
| 学习曲线        | 中——作者需学 YAML frontmatter + `members/moderator/phases` 心智模型 |
| 可测试性        | 高——场景作为 fixture，新增场景零代码改动                                  |


### 已知持续 trade-off

新特性（memory 策略、tool 开关、投票）都要在 YAML schema 上增字段，schema 会持续膨胀。目前所有字段均为加法、无废弃项，但需持续警惕。

---

## 2. Per-agent 消息投影

- **日期**：2026-04-15

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


| 维度          | 评估                                      |
| ----------- | --------------------------------------- |
| 内聚度         | 高——"投影"是单一职责                            |
| 耦合度         | 大幅降低——Agent 只依赖 history list 结构         |
| 可观测性 / 可审计性 | 中性——投影纯函数化让 debug 容易                    |
| LLM 不确定性容忍  | 升——`<message from="X">` 让 agent 分辨说话人更稳 |
| 向后兼容 / 演化友好 | 破坏性——history 结构改；改动时消费者只有一个，影响可控        |
| 学习曲线        | 低——对 scenario 作者透明                      |
| 可测试性        | 升——纯函数投影，输入固定输出确定                       |


"共享 transcript + per-agent projection" 这一模式天然支持未来的 per-agent memory 策略、跨 provider 隔离、审计轨迹派生——抽象的杠杆率远超当前单一用途。

---

## 3. Per-round phases + instruction-as-arg

- **日期**：2026-04-16

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


| 维度          | 评估                                                                         |
| ----------- | -------------------------------------------------------------------------- |
| 内聚度         | 高——三段分明，per-round 逻辑集中在一个 fallback 链                                       |
| 耦合度         | 降低——instruction 离开共享 history，history 不再承担"控制流 + 对话内容"双重职责                  |
| 可观测性 / 可审计性 | 中——`Round N/M` 有 marker；instruction 本身不进 history，回放看不到"引擎当时给 agent 下了什么指令" |
| LLM 不确定性容忍  | 升——消除 instruction 泄露的"误执行"失控路径                                             |
| 向后兼容 / 演化友好 | 破坏性——`phases` 扁平列表改为分块 + `round` 字段，旧场景必须改                                 |
| 学习曲线        | 中——三段结构 + `round` fallback 链；字段语义 self-explanatory                         |
| 可测试性        | 升——per-round 后可以写"某轮特定行为"的验证场景                                             |


同一次重构同时解决了一个 bug 和一个能力缺口：把"instruction 该不该进 history"这个本质问题回答清楚后，`phases` 结构的合理形态自然浮现——修 bug 而非打补丁常常顺便解锁新能力。

---

## 4. Subprocess 隔离 RAG 工具

- **日期**：2026-04-16

### Context

首个工具 `retrieve_docs` 用 `sys.path.insert(0, rag_dir)` 直接 import rag 模块调用。Python 按名称缓存模块——两个子项目**各自有 `config.py`**，第二个 import 的 `config` 拿到第一个的缓存，两边互相覆盖。

### Options considered

- **A. 把 rag 改成可安装 package**（`pip install -e`）：最干净，但 workshop 场景抬高演示门槛
- **B. 共享 config 抽成第三方模块**：人为增加耦合
- **C. Monkey-patch `sys.modules`**：脆弱
- **D. Subprocess 隔离**（选择）：每次调用开独立 Python 进程，**靠 OS 级进程边界保证隔离**

### Decision

- `tools.py` 不再 import rag 代码；改为 `subprocess.run(["python", "rag/query.py", "--json", ...])`
- `rag/query.py` 加 `--json` 输出模式供机器消费
- `run.py` 剥掉 scenario-default 参数不暴露给 LLM（`_path_params` 私有提示），只保留 LLM 真正要填的

### 与 MCP 的关系

MCP（Model Context Protocol，2024）把"工具跑在独立进程 / 服务里，通过标准协议通信"定为业界方向。本项目的 subprocess 隔离**正好踩在这条线上**——虽然没走 MCP 协议而是朴素 CLI + JSON stdin/stdout，设计精神一致。许多生产系统（Claude Code 的 tool 执行、各种 sandboxed code execution）都用类似进程 / 容器隔离。未来若升级为常驻进程，天然可以迁到 MCP server 形态。

### 工程维度评估


| 维度          | 评估                                                                                                          |
| ----------- | ----------------------------------------------------------------------------------------------------------- |
| 内聚度         | 高——每次 tool 调用是一次完整 `query.py` 生命周期，进程内无残留状态                                                                 |
| 耦合度         | 极显著降低——multiagent 与 rag 完全不共享 Python 进程，两边可独立演化依赖、Python 版本、配置。**故障半径从"整个 multiagent 进程"压缩到"一次 tool call"** |
| 可观测性 / 可审计性 | 升——subprocess stderr 直通终端                                                                                   |
| LLM 不确定性容忍  | 中——工具失败返回 `{"error": ...}` 让 LLM 自修正；启动失败目前仅打 stderr，agent 本身看不到，这是留下的不对称                                   |
| 向后兼容 / 演化友好 | 加法式——`retrieve_docs` 接口不变                                                                                   |
| 学习曲线        | 低——对 scenario 作者零感知                                                                                         |
| 可测试性        | 高——subprocess 边界让 RAG 可 `python query.py --json` 独立调试                                                       |


代价：每次 tool call 一次 Python 启动（冷启动 ~500ms），workshop 量级可接受；生产需常驻进程或走 MCP。

---

## 5. Per-agent conversation memory

- **日期**：2026-04-20（+ DI 改造 2026-04-21）

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


| 框架 / 论文                                                                      | 对标到本项目的哪部分                             |
| ---------------------------------------------------------------------------- | -------------------------------------- |
| LangChain `ConversationBufferWindowMemory` / `SummaryBufferMemory`           | `WindowMemory` / `SummaryMemory` 的直接原型 |
| LangGraph `checkpointer` + `BaseStore` 双层                                    | 本项目只有 transcript 单层，无长期 store          |
| AutoGen `model_context` per-agent                                            | 对齐：memory 也是 per-agent                 |
| CrewAI short / long / entity 三种                                              | 仅覆盖 short-term                         |
| Letta / MemGPT OS 式分页                                                        | 故意不做，复杂度不匹配                            |
| Mem0 / Zep 独立 memory 服务                                                      | 故意不做，sandbox 项目无持久化需求                  |
| Generative Agents（Stanford 2023）memory stream + recency/importance/relevance | 未做；若未来加 D / E 方向会参考                    |


### 工程维度评估


| 维度          | 评估                                                                                               |
| ----------- | ------------------------------------------------------------------------------------------------ |
| 内聚度         | 高——`memory.py` 纯投影，三策略共享同一接口；DI 改造后彻底不 import agent / config                                     |
| 耦合度         | 低——memory → agent 单向依赖；装配在 run.py（composition root）                                              |
| 可观测性 / 可审计性 | 中——Window/Summary 的裁剪决策本身无 log；debug 靠读 `_summarized_up_to` 等内部状态                                |
| LLM 不确定性容忍  | 混合——WindowMemory 降低 context 噪声是升；SummaryMemory 引入额外 LLM 调用做折叠，新增一条不确定性链（summarizer 可能漏信息），且当前无重试 |
| 向后兼容 / 演化友好 | 完全兼容——`FullHistory` 作为 default                                                                   |
| 学习曲线        | 中——作者需理解三策略的 trade-off                                                                           |
| 可测试性        | DI 改造后显著升——可注入 fake client 做单测；`test_memory.md` 专门覆盖三策略差异                                        |


### 意外收益

memory 落地后，"固定发言顺序不公平"问题的一半根因（"后发言者享有免费上下文优势"）被结构性消除——WindowMemory 让每人看到的上下文长度一致，发言顺序的不对称被削弱。原本计划单独做的"rotate/shuffle 发言顺序"特性优先级因此下降。解决 A 时发现 B 消失了一大半。

---

## 6. Shared artifact + 结构化投票

- **日期**：2026-04-21

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


| 维度          | 评估                                                                                                    |
| ----------- | ----------------------------------------------------------------------------------------------------- |
| 内聚度         | 高——`ArtifactStore` 聚焦"共享状态 + 工具入口 + 事件流"，对外就 `render / drain_events / dispatch / build_tool_defs` 四个口 |
| 耦合度         | 项目内耦合点最多——Artifact 被 Discussion / Agent / Memory / Tools 四处触达，每处必要无冗余；代价是任何语义变更要同步考虑四处                |
| 可观测性 / 可审计性 | 极高——`artifact_event` 进 history 且 pinned，终端 📝 / ➕ / 🗳 / ✓ / 🏁 emoji 实时可见，`--save-artifact` 落盘       |
| LLM 不确定性容忍  | 高——section mode 冲突让 LLM 同 loop self-correct；ballot 覆盖写入容忍重复 cast；`finalize` 幂等返回 error 防重入            |
| 向后兼容 / 演化友好 | 加法式——未声明 `initial_sections` 的 scenario 不感知 artifact                                                   |
| 学习曲线        | 项目内最高——`initial_sections` schema + section mode + moderator-only filter 三层概念                          |
| 可测试性        | 高——`test_artifact.md` 覆盖六工具 + mode 冲突 self-correction                                                 |


### 关键设计讨论

- **为什么 sectioned 不 JSON？** LLM 对 markdown 顺序的 tool_call 比一次性 JSON structured output 稳定得多；分段 self-correct 路径短
- **为什么 out-of-band view 不进 history？** Artifact 是随时刷新的状态，进 history 意味着每次刷新都挤占 token；带外注入"总是最新 + token 受控"
- **为什么 events 进 history 但 view 不进？** 事件是不可变的"发生过的事"（谁在哪一轮写了什么），view 是"当前状态"——**事件可回放，状态无历史**。这是 event sourcing 的基本区分

---

## 7. Phase-assert：让沉默违规变可见

- **日期**：2026-04-21

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


| 维度          | 评估                                                                       |
| ----------- | ------------------------------------------------------------------------ |
| 内聚度         | 高——retry + nudge + warning 全聚在 `Discussion._run_turn` 一个方法               |
| 耦合度         | 中——依赖 `artifact.drain_events()` 的 `tool / caller` 字段；这是范围限制的根因           |
| 可观测性 / 可审计性 | 极高——三层留痕：`🔁 retry` 现场可见、`WARNING` 落 stderr、artifact_event 进 history 可回放 |
| LLM 不确定性容忍  | 极高——承认 LLM 无法被强制，改为 detect-and-nudge-and-audit 模式                        |
| 向后兼容 / 演化友好 | 完全兼容——未声明 `require_tool` 的 phase 行为不变                                    |
| 学习曲线        | 低——一行 `require_tool: cast_vote`（可选 `max_retries: N`）即声明                  |
| 可测试性        | 高——`test_phase_assert.md` 专门覆盖 retry + warning 端到端                       |


---

## 8. Tool observability：ToolTracer

- **日期**：2026-04-22

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


| 维度          | 评估                                                        |
| ----------- | --------------------------------------------------------- |
| 内聚度         | 高——`ToolTracer` 单一职责：collect + expose `drain()`，约 30 行    |
| 耦合度         | 微增——4 个 backend client 都要知道 tracer；观测性本质横切，不上 AOP 不可能零侵入  |
| 可观测性 / 可审计性 | 本决策核心产出——双 sink 对等支持现场和回放；`--save-transcript` 落盘结构化数据     |
| LLM 不确定性容忍  | 中性——观测性本身不影响容错，但让容错行为可见                                   |
| 向后兼容 / 演化友好 | 完全兼容——`visible=False` 保证其他 agent 不会在 memory 里看到 tool_call |
| 学习曲线        | 低——对 scenario 作者零感知                                       |
| 可测试性        | 升——结构化 transcript 让回归测试有稳定对照物                             |


---

## 9. 扁平 step 列表 + role/tool_owners 显式化

- **日期**：2026-04-24

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


| 框架                     | 流程容器                                              | 参与者                             |
| ---------------------- | ------------------------------------------------- | ------------------------------- |
| **CrewAI**             | `tasks: list[Task]` 顺序执行（与本项目最像）                  | `agents: list[Agent]`，role 字段自由 |
| **AutoGen**            | `initiate_chat` + GroupChat speaker selection（动态） | 单 agent 类，无 role                |
| **LangGraph**          | `StateGraph` 显式节点边                                | Channel-based                   |
| **Temporal / Airflow** | 长 DAG / Workflow                                  | 不适用                             |


本项目位于 CrewAI ↔ AutoGen 中段：流程是声明式列表（CrewAI），但每 step 内"谁发言"靠 role/all/name 三种寻址（介于 CrewAI 单 agent 和 AutoGen 动态选 speaker 之间）。

### 工程维度评估


| 维度          | 评估                                                                                                                |
| ----------- | ----------------------------------------------------------------------------------------------------------------- |
| 内聚度         | 升——`steps` 一段统揽流程，`agents` 一段统揽参与者，`artifact` 块自管自治                                                               |
| 耦合度         | 降——删除 `MODERATOR_ONLY_TOOLS` 这条 artifact ↔ role 的硬编码捷径；ArtifactStore 不再知道 role 概念，权限完全数据驱动                        |
| 可观测性 / 可审计性 | 升——header 打印 `Steps: M / Total turns: N`，每 turn 终端打 `🗣 [name] (step=<id>)`，transcript 带 `turn` pinned marker 可回放 |
| LLM 不确定性容忍  | 中性——`<turn>` marker 让 agent 知道总长度但不强制行为；`require_tool` retry 闭环未变                                                 |
| 向后兼容 / 演化友好 | 破坏性——所有旧 scenario 必须迁移；workshop 项目无外部消费者，可控                                                                       |
| 学习曲线        | 降——心智模型从"phase × round 二维"压缩到"steps 一维顺序展开成 turns"                                                                |
| 可测试性        | 持平偏升——`test_phase.md` 删除（其原概念不存在）；`test_phase_assert.md` 保留验证 require_tool；`test_memory` / `test_artifact` 自然平移   |


### 关键设计讨论

- **为什么不保留 `phase` marker？** Phase 本质是给人看的章节标签，agent 用 `<turn>X of N</turn>` 已能感知位置；避免引入"agent 必须理解 phase 语义"的隐式契约
- **为什么 `tool_owners` 默认全员可调？** 与"显式优于隐式"一致——任何"专属"语义都应在 schema 里看得到，不靠代码内置默认
- **为什么 `who` scalar 同时支持 role 和 `all`？** scalar 是"按属性匹配多个 agent"，list 是"按名字精确点名"——两个语义靠**类型**区分，零歧义；`all` 是 role 的并集，并入 scalar 一起处理代码最简
- **为什么不强制至少 1 个 moderator？** 极简场景（`test_`* 等）可全员 member；要求 moderator 会污染纯 member 测试。`who: moderator` 命中 0 个是 fail-fast 错误，间接拦住"声明了 moderator 寻址却没 moderator"的 bug

### 已知 trade-off

- **批量寻址内部顺序"按 agents 声明顺序"是约定**——文档明写，不靠 schema 强制。如果作者在意公平性，应使用显式 list 形态自己排序
- `**<turn>` marker 占少量 token**——总 turn 数固定时影响极小；超长场景可改 `turn 23` 省去 `of N`，等驱动场景出现再优化

---

## 附录：工程维度词典

每条决策的"工程维度评估"用下列 7 个维度。评级用**极高 / 高 / 中 / 低 / 极低 / 中性 / N/A**；若决策相对旧状态发生变化，用"**升** / **降**"说明方向。


| #   | 维度              | 关注什么                      | 判定信号                                    |
| --- | --------------- | ------------------------- | --------------------------------------- |
| 1   | **内聚度**         | 模块内相关职责是否聚拢               | 一个模块是不是只干一件事；改一个需求会不会散落到多处              |
| 2   | **耦合度**         | 模块间依赖强度                   | 替换/删除/mock 一个模块会牵连多少其他模块                |
| 3   | **可观测性 / 可审计性** | 运行过程是否可见 + 事后是否可回放        | 有没有结构化 log / event / transcript         |
| 4   | **LLM 不确定性容忍**  | 对 LLM 失控的包容度              | 遇到失控是 abort、静默错，还是 self-correct + 把违规记账 |
| 5   | **向后兼容 / 演化友好** | 新增能力是否破坏旧场景               | default 是否保老行为；schema 扩展是加法还是改法         |
| 6   | **学习曲线**        | 使用者（写 scenario 的人）用起来要学多少 | YAML 字段数、心智模型层数、对作者知识的假设门槛              |
| 7   | **可测试性**        | 能否写回归测试 / 可复现实验           | 有无 DI 注入点、fixture 场景、确定性输入输出            |


**故障半径 (Blast radius)** 作为维度 2（耦合度）与维度 4（LLM 容忍）的子属性出现在相关决策的评估文字里，不单独开列——多数决策没有显性失败传播，单独列会退化成"N/A 列"。

**复用性 / 代码复杂度**：前者由 #1 + #2 共同体现；后者在每节的 "Decision" 段落里用自然语言描述。
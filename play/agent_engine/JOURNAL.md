# Journal

按里程碑记录每日进展。每条以 `## YYYY-MM-DD — 里程碑标题` 开头；同一自然日 ≤2 个里程碑。**功能** / **技术** 两段必填；**取舍** 仅在当日产出影响后续的取舍时记一笔，指向 [`DECISIONS.md`](DECISIONS.md) 完整条目而不在此重复。

> Note: 2026-04-26 之前项目名为 `play/multiagent`；保留旧名是为了让 git 历史与本 journal 时间线吻合，DECISIONS §10 / 2026-04-26 的里程碑解释了改名缘起。

## 2026-04-14 — Multiagent PoC：能让两个 agent 围绕一个 topic 对话

### 功能

- `play/multiagent/` 项目从零起步：固定 agents、轮流（round-robin）发言、写死 topic 的最小可跑版本
- 第一个能在终端看见多 agent 你来我往的实例

### 技术

- 单文件 `run.py` + 4 个 backend client（ollama / openai / anthropic / gemini）的 drop-in pattern；`config.py` 一行 `BACKEND` 切换
- history 是 `list[{role, content}]` 风格的共享列表；system prompt 当 history 里一条 user 消息塞进去
- 默认模型 `llama3.1:8b`（本地 ollama 已 pull）

## 2026-04-14 — Phase-driven scenario：把"参与者 + 流程"抽出来做配置

### 功能

- 替换硬编码 agents / round-robin / topic 的最小版本：YAML frontmatter + markdown body 单文件即一个场景
- 4 个示例 scenario 覆盖 (moderator / no moderator) × (open / goal-oriented) 2×2 矩阵
- 每换话题不再改代码，只改 scenario `.md`

### 技术

- 新依赖 PyYAML；frontmatter 解析 + 启动期 schema 校验，错误信息带上下文
- `phases:` 列表声明 opening / main / closing 流程；`members:` 声明成员、可选 `moderator:` 单独声明主持人
- input validation 把 `who` 校验到 actual participant names，错配 fail-fast

### 取舍

- YAML / JSON / Python DSL 多方案对比 → DECISIONS §1
- 单 `members + moderator` 顶层块、phases 三段式、`<phase>` marker 都在 §9 被取代；YAML frontmatter + MD body 形式延续

## 2026-04-15 — Per-agent 消息投影：共享 transcript + 每 agent 视角

### 功能

- 解决 3 个并发症：① agent 分不清「我说的」和「别人说的」；② system prompt 优先级失真；③ Anthropic / Gemini API 不接受连续同 role 消息

### 技术

- Discussion 维护**一条共享 transcript**（唯一权威）；每个 agent 在 `respond()` 时投影成自己的视角：`speaker == owner` → `assistant`，他人 → `<message from="X">...</message>` 包进 user，元数据 → `<tag>...</tag>` 包进 user
- system prompt 走 client 独立参数，不混入 messages
- History entry 从 `role/content` 改为 `speaker/type`（破坏性，单消费者影响可控）
- Anthropic / Gemini client 合并连续同 role 消息保 API 兼容
- 同 commit：scenarios 替换为 `debate.md` + `panel.md` 等 4 份；引入 `TODO.md` 跟踪已知引擎限制

### 取舍

- 共享 history + per-agent 投影 vs 私有 history + 同步 vs 硬塞 prompt → DECISIONS §2

## 2026-04-16 — Per-round phases + instruction-as-arg：修两个 bug 顺手解锁能力

### 功能

- 修 bug 1：Instruction 泄露——给 moderator 的「点名追问 X」会被 members 当成自己的指令；现在 instruction 不进 history，作为 `Agent.respond(instruction=...)` 参数传入
- 修 bug 2：轮次无差异——`main` 阶段静态定义，每轮完全相同；现在每个 main phase 可声明 `round: <int> | "default"`，引擎按当前轮次匹配 + fallback
- 解锁能力：可表达「第 1 轮自由讨论 → 第 2 轮聚焦分歧 → 第 3 轮逼迫表态」的递进

### 技术

- `phases:` 扁平列表分裂为 `opening / main / closing` 三段；`main` 每个 phase 加 `round` 字段
- 引擎每轮先选 `round == N`，回退到 `round == "default"`，再回退到全员发言
- **Instruction-as-arg 是核心 invariant**——history 不再承担「控制流 + 对话内容」双重职责（§9 后扁平 steps 仍保此 invariant）

### 取舍

- 加可见性字段 vs instruction 离开 history（零侵入）；`{round}/{rounds}` 模板 vs phase 显式声明 `round` → DECISIONS §3
- phase × round 二维结构本身在 §9 被扁平 steps 取代

## 2026-04-16 — Subprocess 隔离 RAG 工具

### 功能

- 第一个工具 `retrieve_docs` 接 `play/rag`：通过 `subprocess.run(["python", "rag/query.py", "--json", ...])` 而不是 Python import
- 新 `brainstorm.md` scenario 演示 RAG-backed tool use；`docs/` 按 per-scenario 子目录重组
- LLM 不再看到 scenario-default 参数（如 `vdb_dir`），只看到自己真正要填的参数

### 技术

- 替换 `sys.path.insert(0, rag_dir)` 直接 import：Python 按名称缓存模块，两个子项目各自有 `config.py` 时第二个 import 拿到第一个的缓存——靠 OS 级进程边界保证隔离
- `rag/query.py` 加 `--json` 输出模式供机器消费
- `run.py` 从 LLM tool schema 中剥掉 scenario-pinned 默认参数（`_path_params` 私有提示）；保留场景作者填好的，只暴露 LLM 必须自填的字段
- 同日（4d81258）：`OLLAMA_BASE_URL` 跨 multiagent + rag 统一；新 `vdb_test.md` 作为最小 RAG tool-call 回归测试；prompt 限到 100 字符，max_tokens 收紧

### 取舍

- subprocess vs `pip install -e` 改可安装包 / 共享 config 抽第三方模块 / monkey-patch `sys.modules` → DECISIONS §4
- 与 MCP 的关系：精神一致（工具跑在独立进程通过标准协议通信），未走 MCP 协议而走朴素 CLI + JSON envelope；后来 evals phase 4 复用同模式

## 2026-04-20 — Per-agent conversation memory：full / window / summary 三策略

### 功能

- 解决 panel 场景实测的性能问题：4 成员 + 1 主持 × 3 轮，末段单次发言 111s（开场 24s 的 4.5 倍），整场 1398s（vdb_test 的 14 倍）。根因：所有 agent 共享同一全量 history，每轮末尾 agent 输入 token 线性增长
- 三种 memory 策略（scenario 级默认 + agent 级覆盖）：
  - `FullHistory` — default，完全向后兼容
  - `WindowMemory(max_recent)` — 保留所有 pinned marker + 最近 N 条发言
  - `SummaryMemory(max_recent, ...)` — stale 发言增量折叠进 `<summary>` block
- summarizer prompt 与 instruction 都可在 scenario frontmatter 覆盖
- 同日：opening / closing phase 注入 `<phase>` marker（pinned），让 agent 自感所处阶段；`phase_test.md` 作为 marker 存在性的最小回归 probe

### 技术

- `ConversationMemory` 抽象基类，`build_messages(history, owner) -> messages`
- 共享 transcript 不变，每个 agent 持有自己的 memory 实例（承接 §2 的「共享 + 投影」模式）
- **Pinned types 永不被剪**——`topic / round / phase / artifact_event` 是会议纪要级信息，丢了对话就破
- Summary 触发规则：stale 条数达阈值才折叠；未到阈值 stale 原样保留——「不动就无信息损失」

### 取舍

- A (full) + B (window) + C (summary) 三件套；不做 D vector retrieval / E memory stream + importance / F MemGPT 分页——对话 stream 短（几十到一两百条），收益不抵复杂度 → DECISIONS §5
- 跨主流框架对比（LangChain / LangGraph / AutoGen / CrewAI / Letta / Mem0 / Generative Agents） → DECISIONS §5

## 2026-04-21 — Shared artifact + 结构化投票：让"讨论 → 决策"链路可机器验证

### 功能

- `panel` 类场景要求"一方胜出"但只产一串发言，最终决策靠隐式推断；现在引入 `ArtifactStore` 把决策结构化
- 6 个工具：`read_artifact / write_section / append_section / propose_vote / cast_vote / finalize_artifact`
- Section 由 scenario 作者在 `initial_sections` 显式声明，每节标 `mode: replace | append`，store 强制；mode 不匹配返回 `{"error": ...}`，LLM 在同一 tool loop 内 self-correct
- `--save-artifact` CLI flag 落盘最终 markdown
- `panel.md` 端到端启用；`test_artifact.md` 覆盖六工具 + mode 冲突 self-correction 路径

### 技术

- **Out-of-band artifact view**：每次 agent 发言前把 `artifact.render()` 作为 `<artifact>` user 消息**带外注入**——不进 history，memory 裁剪永远不会藏掉它
- **Artifact events 进 history**：`artifact_event` 类型，pinned，不被 memory 剪——「事件可回放，状态无历史」是 event sourcing 的基本区分
- `finalize_artifact` 是 sealing step，幂等返回 error 防重入（类似 workflow 的 terminal state）
- 同 commit 修了一个真 bug：tool handler 中 scenario-level 默认值现在覆盖 LLM 提供的参数，幻觉的 `vdb_dir` 不会偷换 scenario 解析后的路径
- 同日早些时候（10ca7d6）：SummaryMemory DI 改造——构造时注入 `client / model / max_tokens / temperature`，memory 模块不再编译期依赖 agent / config；`run.py` 承担装配；同步：tool path resolution 按 scenario 文件位置自动解析（约定优于配置）；tools.dispatch 在 tool result 是 `{"error": ...}` 时打 stderr warning，subprocess 不再静默失败

### 取舍

- sectioned markdown + tool-mediated writes vs JSON 整块 / moderator 收尾自由文本 / 外置数据库 → DECISIONS §6
- 跨 CRDT / workflow / structured output 三条光谱定位 → DECISIONS §6

## 2026-04-21 — Phase-assert (`require_tool`)：让"沉默违规"变可见

### 功能

- panel closing 实测 bug：指令要求"每人发言后调用 `cast_vote(...)`"，但两名 member 只说话没投票，引擎 fire-and-forget；现在 phase 可声明 `require_tool: <tool_name>` + 可选 `max_retries: N`（默认 1）
- 失败路径：未命中 → 追加 nudge instruction「你刚才没有调用 `<tool>`，请现在补上」（per-call argument，**不进 history**，其他 agent 看不见这次辅导）→ 重试用尽 → stderr `WARNING`
- 终端打 `🔁 [agent] retry k/N: missing <tool>`，workshop 观众能看到流程
- `propose_vote` 加入 `MODERATOR_ONLY_TOOLS`：消除 member 乱 propose 让 scenario 硬编码 `vote_id` 错位的 bug 类
- `test_phase_assert.md` smoke scenario 端到端跑通 retry + warning 路径

### 技术

- 核心目标**不是"强制 agent 调工具"（LLM 本质上做不到强制），而是让沉默违规变可见**——detect-and-nudge-and-audit 模式
- artifact_event 加 `tool` / `caller` 结构化字段，phase-assert（与未来 audit 工具）可程序化检查 compliance 而不是解析 free-form 文本
- `run.py` line-buffer stdout 与 stderr，`2>&1 | tee` 保 chronological 顺序

### 取舍

- 硬失败 / 自动补调工具 / silent nudge + retry + warning 三方案 → DECISIONS §7
- 对标 linter warning / 议会 roll-call / AutoGen GroupChat speaker fallback / structured output retry → DECISIONS §7

## 2026-04-22 — Tool observability：ToolTracer 让 non-artifact 工具调用可见

### 功能

- 解决盲点：artifact tools 有完整可观测（events + 终端 emoji），但 non-artifact tools（当前 `retrieve_docs`）完全静默——终端看不见、transcript 回放不出来、workshop 演示时观众不知道 agent 到底查没查、查了什么
- 终端实时 🔧 emoji 一行；`--save-transcript` 落盘结构化 history（`topic / round / phase / speaker / tool_call / artifact_event` + `ts`）
- 同 commit 修 moderator-first bug：`roundtable` main 与 `panel` closing 第二个 phase 用 `who: all`，`_resolve_who("all")` 返回 `[moderator, *members]`——主持人每轮抢先发言。改为 `who: members`

### 技术

- `ToolTracer` 类：收集 non-artifact 工具调用，暴露 `drain() -> list[event]`
- **双 sink** 对应 OTel 的 live exporter + batch exporter：stderr 一行 🔧（现场可见）+ transcript event 带 `visible=False`（不进 memory，离线回放可用）
- `tools.is_error` 抽公共函数，stderr tripwire 与 tracer 的 `ok` 字段对"失败"定义一致
- 所有 entry 统一加 `ts`（ISO timestamp）字段，时序完整
- 显式不做：让 tool_call 进 memory（成本：4 个 backend client + memory 渲染分支 + summary 策略 + 每轮额外 token）；显式留有 artifact 承载"状态性跨 agent 共享"这个最强用例

### 取舍

- 完整 history entry 让 memory 渲染 vs visible=False 隐写（选择）vs 独立日志文件 → DECISIONS §8
- 没上 OpenTelemetry 是对的（workshop 项目不应承担 distributed tracing 的部署复杂度）→ DECISIONS §8

## 2026-04-25 — 扁平 step 列表：取代 phase × round 二维结构

### 功能

- Schema 重构：删 `opening / main / closing` 三段、`rounds` / `phase.round`、顶层 `moderator:` 块、`members:` 别名、`MODERATOR_ONLY_TOOLS` 硬编码、CLI `--rounds` flag
- 新增：扁平 `steps:` 列表；`agents:` 统一列表 + 强制 `role: moderator | member`；`artifact.tool_owners` 显式 ACL；运行时 `<turn>turn X of N</turn>` pinned marker
- `who` 简化为四种字面形态：`moderator` / `member` / `all`（scalar 走 role/all 寻址）+ `[name1, name2]` list（按名字精确点名）；删 `role:` 前缀和动态 `by:` stub
- `tool_owners` 默认全员可调（包含 finalize/propose_vote）；想保留主持人专属必须**显式声明**——与"显式优于隐式"对齐
- 新作者写最小 scenario 只需要 `agents` 一段、`steps` 一段，每步必填 `who` + `instruction`

### 技术

- 心智模型从「phase × round 二维」压缩到「steps 一维顺序展开成 turns」
- `<turn>` marker 在启动时静态展开 `N`、`X` 单调递增；token 成本约 9，复杂度极低；让 agent 自感位置但不强制行为
- `ArtifactStore` 不再知道 role 概念，权限完全数据驱动；删 `MODERATOR_ONLY_TOOLS` 硬编码
- 破坏性：所有旧 scenario 必须迁移；workshop 项目无外部消费者，可控

### 取舍

- 流程容器：保留 phases / state machine / DAG / 扁平 `steps` 顺序列表（CrewAI Task list 风格）→ DECISIONS §9
- 位置感知：保留 `<phase>` / `<round>` marker / 全靠 prompt engineering / 每 turn 注入 `<turn X of N>` → DECISIONS §9
- 参与者建模：保留 `moderator: / members:` 两段 / 统一 `agents:` 列表 + 强制 role → DECISIONS §9
- artifact tool 权限：`MODERATOR_ONLY_TOOLS` 硬编码 / `tool_owners` 数据驱动 → DECISIONS §9

## 2026-04-25 — Hybrid retrieval 集成到 retrieve_docs 工具

### 功能

- `retrieve_docs` 工具通过 OpenAI tool schema 把 `mode` + `rerank` 暴露给 LLM；scenario `tools:` 默认值仍可 pin
- ToolTracer preview 升级：从「三键 dict」改为 `[N items, mode=..., reranked]` 信息密度更高的字符串
- `scenarios/test_vdb.md` prompt nudge LLM 在歧义 query 上 `rerank=true`
- `scenarios/example.md` doc fix：scenario-pinned 默认参数从 LLM schema 中移除（清掉过去 stale 注释）

### 技术

- `_retrieve_docs` 把 rag CLI envelope 解包为 slim `{data, meta:{mode, reranked, top_k}}` 给 LLM——HTTP envelope ↔ SDK 解列表的两层分工，对齐 OpenAI SDK 风格
- 与 rag 侧的 hybrid + reranker 落地是同 commit（详见 `play/rag/JOURNAL.md` 同日两条）

## 2026-04-26 — 改名 + tools/ 包拆分（Engine.invoke 库化前奏）

### 功能

- `play/multiagent/` → `play/agent_engine/`：项目名从「实现手段」（multi-agent）改为「能力描述」（agent engine），与未来作为可被 workflow 嵌入的库 surface 对齐
- `tools.py` 单文件 → `tools/` 包：`retrieve_docs.py` + `_envelope.py` + `_subprocess.py` 三文件
- 公共 surface（`TOOL_DEFINITIONS / dispatch / is_error / warn_if_error`） 不变，既有 `run.py` / `artifact.py` import 全部保持工作

### 技术

- 机械改名 + 文件拆分，零行为变更（P0 of QA-agent v2 plan §8）
- DESIGN_DECISIONS 加一行"historical name"标注；rag 侧 README + DESIGN_DECISIONS 出链 update 到 `play/agent_engine/`
- 为下一个 commit 的 Engine.invoke 库化做准备——tools 在能干净 import 之前不能拆得过细

## 2026-04-26 — 库化拆分：Scenario / Engine / CLI 取代一体式 run.py

### 功能

- `Engine`（库 SoT） + `cli.py`（thin adapter，`python -m agent_engine`）双重 surface，共享同一装配路径
- `Engine.invoke(*, initial_artifact, transcript_path, artifact_path, callbacks, print_stream) -> Result`：LangChain Runnable 风格 API；`ainvoke` / `stream` / `astream` 显式 `NotImplementedError` 留口（plan §5.5）
- `Result` dataclass：`artifact / transcript / success / warnings`；require_tool 用尽除既有 stderr WARNING 外也写 `.warnings`，调用方可程序化判断
- `print_stream` 默认 False（库边界）/ True（CLI 边界）——同一引擎在脚本与终端两种语境下的安静度不同（plan §10 风险 D）
- 所有 9 个 scenario 跑通；`debate.md` + `test_phase_assert.md` + `test_vdb.md`（subprocess 到 play/rag）端到端通过

### 技术

- `scenario.Scenario.from_yaml() + assemble()` 把解析 / 校验 / 装配从老 `run.py` composition root 抽出来
- `engine.Engine` 持有 `Scenario`：`invoke()` 内顺序为 `assemble()` → 可选 `initial_artifact` 种子（绕过 artifact tool ACL 供 pipeline 预热）→ 构造 `Discussion.run()` → 组装 `Result` → 落盘 → 触发 `Callback.on_run_finished(RunFinished)`
- `events.py` Event 基类 + 5 子类、`callbacks.py` Callback `on_xxx` 方法预接线（今天只 RunFinished 落地）
- `tracer.ToolTracer` 从 `run.py` 提出来，Discussion 不再 `TYPE_CHECKING` 引用 CLI 模块
- 同 commit 后续 d2c4598：folding 4 个 standalone smoke scenario（test_artifact / test_memory / test_phase_assert / test_vdb）成 `example.md` 单一 kitchen-sink + CI 场景；ADR 归档由 `DESIGN_DECISIONS.md` 迁到与 `play/rag` / `play/workflow` 体例对齐的项目级文件（彼时叫 `CHANGELOG.md`，后改名 `DECISIONS.md`）

### 取舍

- 保留 `run.py` 旁路加 api.py 薄封装 / 只抽 `Engine` 留散落函数 / `Scenario` + `Engine` + 极薄 CLI / 一次性上 async ainvoke + stream → DECISIONS §10
- async / stream 推迟：与"抽象引入滞后于第二个具体案例"对齐——先有同步 `invoke` 嵌入方，再为 async/stream 选模型（LangGraph 式 event stream vs 简单 queue）

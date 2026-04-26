# play/agent_engine

Step-driven 多 agent 讨论引擎：scenario = 一份 markdown，YAML frontmatter 声明参与者 / 流程 / 工具 / memory / artifact，body 即话题。共享 transcript + per-agent 投影，支持 ollama / openai / anthropic / gemini 四个后端，可被 `[play/rag/](../rag/)` 通过 subprocess 喂数据。

## 特性

- **Scenario 即配置**：YAML frontmatter + markdown body，一文件一场景；启动期 schema 校验，作者改场景零代码
- **扁平 step 列表**：`steps:` 一段顺序声明所有 turn，`who` 用 role/all/name 列表灵活寻址；引擎按声明顺序展开，每 turn 注入 `<turn>turn X of N</turn>` pinned marker，让 agent 自感位置
- **Shared transcript + per-agent projection**：history 只有一份权威视图，每个 agent 在 `respond()` 时按 `speaker == owner` 投影为 `assistant`，他人投影为 `<message from="X">`，控制流（`topic / turn / artifact_event`）投影为带标签的 user 消息
- **Per-agent memory 三策略**：`full / window / summary` 同接口可换；pinned 类型（控制流 + artifact 事件）永不被剪
- **Shared artifact + 结构化投票**：sectioned markdown + `replace / append` mode + 投票 + `finalize`；artifact view 带外注入（不进 history），artifact_event 进 history（pinned）
- **Artifact tool ACL（`tool_owners`）**：每个 artifact 工具的可调用方在 `artifact.tool_owners` 显式声明，取值与 `who` 完全对齐（role / all / name 列表）；未列出的工具默认对所有 agent 开放
- **Step assert（require_tool）**：声明 step 必须调用某工具；缺则 nudge 重试，最终落 stderr WARNING——**让沉默违规可见**而非强制
- **Tool observability**：`ToolTracer` 双 sink——stderr 实时 🔧 emoji + transcript event（`visible=False`，离线回放可用）
- **Subprocess 隔离工具**：`retrieve_docs` 通过 `subprocess.run(python rag/query.py --json)` 调用，进程边界保证两个子项目的 `config.py` / 依赖互不串台；透传 rag 的 hybrid（dense + BM25 RRF 融合）检索 + 可选 cross-encoder 精排，LLM 可按需选 `mode` / `rerank`
- **多后端 pluggable**：`config.py` 改一行 `BACKEND` 切换 ollama / openai / anthropic / gemini

## 架构

### 分层视角

按业界常见的 5 层模型把项目"摊开"看：每一层在本项目里对应的具体模块、文件、配置都标在节点里，方便对照代码。下游各小节是对这张图的细节展开。

```mermaid
flowchart TB
    subgraph UI["🖥 用户接口层 / User Interface Layer"]
        cli["run.py CLI<br/>argparse: scenario.md · --no-stream<br/>--save-artifact / --save-transcript"]
        io["stdout: 🗣 speaker / 🔧 tool / 📝➕🗳✓🏁 artifact<br/>stderr: 🔁 retry · WARNING"]
        scn["scenario.md<br/>(YAML frontmatter + body)"]
    end

    subgraph ORCH["🎬 编排层 / Orchestration Layer"]
        asm["composition root (run.py)<br/>schema validate · 装配 Agent/Memory/ACL"]
        disc["Discussion (discussion.py)<br/>steps → turns 展开<br/>+ &lt;turn X of N&gt; pinned marker"]
        route["_resolve_who<br/>role / all / name 路由寻址"]
        retry["_run_turn retry loop<br/>require_tool · nudge · WARNING"]
    end

    subgraph CAP["🧠 能力层 / Capabilities"]
        plan["Planning · 规划<br/>scenario steps 声明式流程<br/>+ require_tool 行为约束"]
        reas["Reasoning · 推理<br/>Agent.respond() + persona prompt<br/>+ backend client tool-use loop"]
        mem["Memory · 记忆<br/>shared transcript + per-agent 投影<br/>FullHistory / WindowMemory / SummaryMemory"]
        tool["Tool Use · 工具<br/>tools.dispatch + ArtifactStore (6 tools)<br/>+ tool_owners ACL · scenario default 注入"]
    end

    subgraph LLM["🤖 LLM Core / 大模型核心"]
        backends["pluggable backend client<br/>ollama_client · openai_client<br/>anthropic_client · gemini_client<br/>config.BACKEND 一行切换"]
    end

    subgraph INFRA["⚙ 基础设施层 / Infrastructure"]
        sub["Subprocess sandbox<br/>rag/query.py --json<br/>(OS 级进程隔离)"]
        vdb[("Vector DB · BM25<br/>delegated to play/rag/")]
        obs["Observability<br/>ToolTracer (stderr + transcript event)<br/>artifact_event 流 · --save-transcript JSON"]
    end

    cli --> scn
    scn --> asm
    asm --> disc
    disc --> route --> retry
    retry --> reas

    plan -. 由 scenario 编译 .-> disc
    reas --> mem
    reas --> tool
    reas --> backends
    backends --> tool

    tool -. retrieve_docs .-> sub --> vdb
    tool --> obs
    backends -. tool_call .-> tool

    obs -. tool_call event (visible=false) .-> disc
    tool -. artifact_event (pinned) .-> disc
    io -. live tail .- disc
    io -. live tail .- tool
```

| 层                           | 本项目里的具体落地                                                                                                          |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **UI / 用户接口**               | `run.py` argparse；scenario `.md` 输入；stdout 流式发言 + emoji 实时事件、stderr WARNING；`--save-artifact` / `--save-transcript` 落盘 |
| **Orchestration / 编排**      | `run.py` composition root（schema 校验 + 装配）；`discussion.py` Discussion 引擎（steps → turns + `<turn X of N>` marker + retry 闭环） |
| **Planning / 规划**           | scenario `steps` 声明式列表；`who` 路由（role/all/name）；`require_tool` + `max_retries` 行为约束                                  |
| **Reasoning / 推理**          | `Agent.respond()` + persona system prompt + per-step instruction；backend client 内的 tool-use loop（多轮 function calling） |
| **Memory / 记忆**             | shared transcript（`Discussion.history` 唯一权威）+ per-agent `Memory.build_messages` 投影；`Full / Window / Summary` 三策略可换 |
| **Tool Use / 工具**           | `tools/` 包 dispatch + scenario default 注入 + path 解析；`ArtifactStore` 6 工具（`read/write/append/propose/cast/finalize`）+ `tool_owners` ACL |
| **LLM Core / 大模型核心**        | 4 个 pluggable backend client（ollama / openai / anthropic / gemini），`config.BACKEND` 一行切换；`config.py` 集中模型 / temperature / max_tokens / 各家 API key |
| **Infrastructure / 基础设施**   | Subprocess 沙箱（`subprocess.run([python, rag/query.py, --json])` 隔离 `config.py` / 依赖）；`play/rag/` 提供 Vector DB + BM25；`ToolTracer` 双 sink + `artifact_event` 流 + JSON transcript 落盘 |

### 组件总览

`run.py` 作为 composition root 把 scenario 装配成运行时对象图；`Discussion` 持有唯一权威 `history`，`Agent` 通过 `Memory` 投影读取它，`ArtifactStore` / `ToolTracer` 各自往 `history` 反向写事件。

```mermaid
flowchart TB
    scenario["scenario.md<br/>YAML frontmatter + body"]

    subgraph rt["run.py — composition root"]
        val["schema validate<br/>(fail-fast)"]
        asm["assemble<br/>Agent + Memory + ACL"]
    end
    scenario --> val --> asm

    subgraph engine["Discussion (engine)"]
        exp["expand steps → turns<br/>每 turn 注入 &lt;turn X of N&gt;"]
        hist[("history<br/>shared transcript")]
    end
    asm --> exp
    exp -.per turn.-> agent

    subgraph agent_box["Agent.respond()"]
        mem["Memory<br/>full / window / summary"]
        client[("backend client<br/>ollama / openai /<br/>anthropic / gemini")]
    end
    hist -->|read| mem --> client

    subgraph tools_box["tools"]
        dispatch["dispatch (per-agent handler<br/>+ scenario default injection)"]
        rag[("subprocess<br/>rag/query.py --json")]
        store["ArtifactStore<br/>sections + votes + tool_owners ACL"]
        tracer["ToolTracer"]
    end
    client -. tool_call .-> dispatch
    dispatch -. retrieve_docs .-> rag
    dispatch -. artifact tools .-> store
    dispatch -. non-artifact .-> tracer

    client -- reply text --> hist
    store -. artifact_event (pinned) .-> hist
    tracer -. tool_call (visible=false) .-> hist
    store -. render() out-of-band .-> agent_box
```

### Scenario → 运行时装配

YAML 字段与 runtime 对象一一对应；`run.py` 是唯一知道这些映射的地方。

```mermaid
flowchart LR
    subgraph fm["scenario.md frontmatter"]
        a["agents:"]
        s["steps:"]
        m["memory:"]
        t["tools:"]
        ar["artifact:"]
        b["body"]
    end
    subgraph ro["runtime objects"]
        AG["Agent[]<br/>per-agent tool_defs + handler"]
        EX["expanded turns<br/>list[(agent, step)]"]
        ME["ConversationMemory<br/>(per agent, with backend client DI)"]
        TD["TOOL_DEFINITIONS<br/>filter + defaults stripped<br/>+ scenario-pinned values"]
        AS["ArtifactStore<br/>sections + tool_owners 解析为<br/>{tool: [agent_name…]}"]
        TP["history[0]<br/>type=topic"]
    end
    a --> AG
    s --> EX
    m --> ME --> AG
    t --> TD --> AG
    ar --> AS
    AS -. per-agent tool_defs .-> AG
    b --> TP
```

### 单 turn 数据流

每个 turn 内部的执行顺序——尤其是 artifact view 带外注入、tool_call 事件先于发言入 history、`require_tool` nudge 闭环——在这里一图说清。

```mermaid
sequenceDiagram
    autonumber
    participant D as Discussion
    participant H as history
    participant AG as Agent.respond
    participant M as Memory
    participant CL as backend client
    participant AS as ArtifactStore
    participant TR as ToolTracer

    D->>H: append &lt;turn X of N&gt; (pinned)
    D->>AS: render() → markdown view
    D->>AG: respond(history, instruction, artifact_view)
    AG->>M: build_messages(history, owner)
    M-->>AG: messages (per-agent projection)
    Note over AG,CL: artifact_view 作为 &lt;artifact&gt; user 消息<br/>带外注入；不进 history
    AG->>CL: chat(system, messages + view + instruction, tools)

    loop tool-use loop
        alt artifact tool
            CL->>AS: dispatch(name, args, caller)
            AS-->>CL: result (+ enqueue artifact_event)
        else non-artifact tool
            CL->>TR: dispatch via tracer
            TR-->>CL: result (+ stderr 🔧 + enqueue tool_call)
        end
    end

    CL-->>AG: final reply text
    AG-->>D: reply

    D->>TR: drain() tool_call 事件
    D->>H: append tool_call 事件 (visible=false)
    D->>H: append speaker turn
    D->>AS: drain_events() artifact 事件
    D->>H: append artifact_event (pinned)

    alt require_tool 未命中 且 attempt &lt; max_retries
        D->>D: 生成 nudge instruction，重新进入 turn
    else require_tool 命中 或 重试用尽
        D-->>D: 进入下一 turn (用尽则 stderr WARNING)
    end
```

### History 投影规则

History 只有一份；每个 agent 在 `Memory.build_messages(history, owner)` 中按下表把它折成自己的 `messages`。`visible=False` 的 entry（`tool_call`）对所有 agent 不可见，仅人类回放可用。

```mermaid
flowchart LR
    subgraph src["shared history (one source of truth)"]
        e1["type=topic"]
        e2["type=turn"]
        eA["speaker=A: ..."]
        eB["speaker=B: ..."]
        eE["type=artifact_event<br/>(pinned)"]
        eT["type=tool_call<br/>visible=false"]
    end
    subgraph va["Agent A 的 messages"]
        ma1["user: &lt;topic&gt;...&lt;/topic&gt;"]
        ma2["user: &lt;turn&gt;...&lt;/turn&gt;"]
        ma3["assistant: ..."]
        ma4["user: &lt;message from=&quot;B&quot;&gt;...&lt;/message&gt;"]
        ma5["user: &lt;artifact_event&gt;...&lt;/artifact_event&gt;"]
    end
    subgraph vb["Agent B 的 messages"]
        mb1["user: &lt;topic&gt;..."]
        mb2["user: &lt;turn&gt;..."]
        mb3["user: &lt;message from=&quot;A&quot;&gt;..."]
        mb4["assistant: ..."]
        mb5["user: &lt;artifact_event&gt;..."]
    end
    e1 --> ma1 & mb1
    e2 --> ma2 & mb2
    eA --> ma3
    eA --> mb3
    eB --> ma4
    eB --> mb4
    eE --> ma5 & mb5
    eT -.->|skipped| ma1
    eT -.->|skipped| mb1
```

| 来源 entry | 投影规则                                                                   |
| -------- | ---------------------------------------------------------------------- |
| `type=topic / turn / artifact_event / summary`    | 包成 `<tag>...</tag>` 的 user 消息                                          |
| `speaker == owner`                                | `assistant` 消息                                                         |
| `speaker != owner`                                | 包成 `<message from="X">...</message>` 的 user 消息                         |
| `visible=False`（`tool_call` from `ToolTracer`）   | 所有 agent 投影时跳过；仅 `--save-transcript` 落盘可见                              |

> Pinned 类型（`topic / turn / artifact_event`）永不被任何 memory 策略剪掉——会议纪要级信息丢了对话就破。`<artifact>` 视图每 turn 带外注入，不进 history，因此既"总是最新"又不占 memory 配额。

## 环境准备

- Python 3.12+
- `pip install -r requirements.txt`（`anthropic / google-genai / openai / pyyaml`）
- 选一个后端（默认 ollama）：

```bash
# 本地 ollama（默认）
ollama pull qwen2.5:32b
# 或者改 config.py 的 BACKEND，并填上对应 *_API_KEY
```

`retrieve_docs` 需要 `[play/rag/](../rag/)` 已建好 VDB（详见 rag README）。

## 快速开始

在 `play/agent_engine/` 目录下：

```bash
# 1. 经典圆桌（主持人 + 2 嘉宾）
python run.py scenarios/roundtable.md

# 2. 决策会议（主持人 + 4 成员，11 步 25 turn，带 artifact + 投票 + finalize）
python run.py scenarios/panel.md --save-artifact /tmp/panel.md

# 3. RAG 工具烟囱测试（agents 通过 subprocess 调 rag）
python run.py scenarios/test_vdb.md
```

预期输出片段：

```
============================================================
  Participants: 主持人, 嘉宾A, 嘉宾B
  Steps: 3  |  Total turns: 4
============================================================

🗣  [主持人] (step=open): 各位嘉宾好，今天我们讨论 ...
🗣  [嘉宾A] (step=discuss): 从技术原理 ...
🔧 [嘉宾A] retrieve_docs(query='AGI 路径', vdb_dir='...') → [3 items, mode=hybrid]
...
```

## CLI 速查

> 完整说明见 `python run.py --help`。

| 参数                  | 必选   | 默认           | 说明                                                                |
| ------------------- | ---- | ------------ | ----------------------------------------------------------------- |
| `scenario`          | 是    | —            | scenario `.md` 文件路径                                               |
| `--no-stream`       | flag | `False`      | 关闭流式输出                                                            |
| `--save-artifact`   | 否    | —            | 把最终 artifact markdown 落盘（仅 `artifact.enabled` 场景生效）              |
| `--save-transcript` | 否    | —            | 落盘结构化 history（topic / turn / speaker / tool_call / artifact_event）JSON |

## Scenario schema

YAML frontmatter 字段：

| 字段          | 类型     | 说明                                                                                                       |
| ----------- | ------ | -------------------------------------------------------------------------------------------------------- |
| `agents`    | list   | 必填，至少 1 项；每项 `{name, role, prompt}`，可选 `model / temperature / max_tokens / memory`；`role` ∈ {moderator, member} |
| `steps`     | list   | 必填，至少 1 项；每项 `{who, instruction, id?, require_tool?, max_retries?}`；按列表顺序展开成 turn                          |
| `memory`    | dict   | scenario 级默认 memory 配置；agent 级 `memory` 字段可覆盖                                                            |
| `tools`     | list   | 每项 `{name: <tool>, ...defaults}`；scenario 级默认值会从 LLM schema 中隐藏并注入到 dispatch                             |
| `artifact`  | dict   | `{enabled, initial_sections?, tool_owners?}`；section 项可声明 `mode: replace\|append`；`tool_owners` 限制可调用方  |

`who` 取值（共四种）：

| 形态                  | 含义                                                  |
| ------------------- | --------------------------------------------------- |
| `moderator`         | scalar role：所有 role=moderator 的 agent，按声明顺序          |
| `member`            | scalar role：所有 role=member 的 agent，按声明顺序            |
| `all`               | scalar 关键字：全员，按声明顺序                                  |
| `[name1, name2]`    | 显式名单：按列表顺序，每个 name 必须存在；单点也写成 `[name]`              |

`<artifact>` 视图在每次发言前带外注入（不进 history）。`<turn>turn X of N</turn>` 在每 turn 前 pinned 注入，让 agent 自感位置。

### Memory 策略

| `type`    | 必填字段             | 行为                                                            |
| --------- | ---------------- | ------------------------------------------------------------- |
| `full`    | —                | 默认；保留全量 history                                               |
| `window`  | `max_recent`     | 保留所有 pinned marker + 最近 N 条发言                                 |
| `summary` | `max_recent` + 可选 `model / max_tokens / temperature / summarizer_prompt / summarize_instruction` | stale 发言增量折叠进 `<summary>` block；client 由 `run.py` 注入 |

### Artifact 工具

| 工具                  | 默认可见性                | 作用                                       |
| ------------------- | -------------------- | ---------------------------------------- |
| `read_artifact`     | all（除非 tool_owners 限制） | 返回当前 markdown 视图                         |
| `write_section`     | all（受 mode 限）        | 覆盖式写 section；`append` 节调用返回 error      |
| `append_section`    | all（受 mode 限）        | 追加 entry；`replace` 节调用返回 error          |
| `propose_vote`      | all（除非 tool_owners 限制） | 注册结构化投票，返回 `vote_id`                     |
| `cast_vote`         | all（除非 tool_owners 限制） | 记录一票（按 `caller` 覆盖写）                     |
| `finalize_artifact` | all（除非 tool_owners 限制） | 封板；幂等返回 error 防重入                        |

> ⚠ 历史上 `propose_vote` / `finalize_artifact` 是硬编码的"主持人专属"。当前没有任何硬编码默认——若你想保留这个语义，**必须**在 `artifact.tool_owners` 里显式声明 `propose_vote: moderator` / `finalize_artifact: moderator`。

`require_tool: <tool>` 在 step 结束后扫 `artifact.drain_events()` 验证调用是否发生；未命中追加 nudge instruction 重试，重试用尽 stderr WARNING。

## Scenario 库

| 文件                  | 用途                                                       |
| ------------------- | -------------------------------------------------------- |
| `example.md`        | **kitchen-sink 模板**：每个 frontmatter 字段都用一遍 + 行内注释 + body 含运行时心智模型，新作者从这里开始 |
| `roundtable.md`     | 主持人 + 2 嘉宾，最简流程烟囱（3 step）                                |
| `debate.md`         | 无主持人，2 立场对辩（2 step）                                      |
| `brainstorm.md`     | 无主持人，演示 `who: [name, ...]` 显式列表寻址（2 step）                |
| `panel.md`          | 决策会议：主持人 + 4 成员，11 step / 26 turn，artifact + 投票 + finalize（最完整） |
| `test_vdb.md`       | `retrieve_docs` 烟囱测试（subprocess 调 rag）                   |
| `test_memory.md`    | 三种 memory 策略的可见度对照                                      |
| `test_artifact.md`  | 6 个 artifact 工具 + mode 冲突 self-correct + tool_owners 过滤   |
| `test_phase_assert.md` | `require_tool` + `max_retries` 重试闭环烟囱                  |

## 项目结构

```
play/agent_engine/
├── README.md                   # 本文件
├── DESIGN_DECISIONS.md         # 设计决策时间线（按时间顺序）
├── requirements.txt            # anthropic / google-genai / openai / pyyaml
├── config.py                   # BACKEND + 各家 model/key/默认参数
├── run.py                      # CLI + composition root（装配点集中）
├── discussion.py               # Discussion 引擎：扁平 steps -> 线性 turn
├── agent.py                    # Agent.respond() + memory 投影入口
├── memory.py                   # FullHistory / WindowMemory / SummaryMemory
├── artifact.py                 # ArtifactStore + 6 工具 + 投票 + finalize
├── tools/                      # reasoning tool 包（_envelope / _subprocess / retrieve_docs / __init__）
├── anthropic_client.py         # 后端 client（含 tool_handler loop）
├── openai_client.py            #
├── gemini_client.py            #
├── ollama_client.py            #
└── scenarios/                  # 场景库（见上表）
```

设计动机、候选方案与 trade-off 评估见 `[DESIGN_DECISIONS.md](DESIGN_DECISIONS.md)`。

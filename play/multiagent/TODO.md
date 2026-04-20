# TODO — multiagent engine

## 维护项

### 固定发言顺序

members 每轮发言顺序与列表定义顺序相同。先发言者信息最少，后发言者拥有上下文优势，造成结构性不公平。

可考虑每轮随机打乱或轮转发言顺序。

## 演进方向

### Memory 从 discussion 里拆成独立一层

**问题**：`Agent.respond` 每次把整份 `history` 全量展开成 messages，后续 agent 单次耗时随轮数线性增长。实测 panel 场景（4 成员 + 1 主持 × 3 轮）末段单次发言 111s，相比开场 24s 慢 4.5 倍；整场耗时 1398s，是 vdb_test 的 14 倍。根因是所有 agent 共享一条全量 history。

**方向**：抽 `ConversationMemory` 接口，每个 agent 持有自己的实例，负责把共享 transcript 转成该 agent 的 messages。内置策略至少三种：

- `FullHistory`（当前行为，默认，100% 兼容）
- `WindowMemory(k)` / `TokenBudgetMemory(n)` — 滚动窗口
- `SummaryMemory` — 每 N 轮由 moderator 或 summarizer agent 折叠旧消息

延伸：per-agent 私密 scratchpad（如 panel 场景里马千里的"内心动摇"不应让对手看到）。

**行业参考**：

- **LangChain / LangGraph** — `ConversationBufferWindowMemory` / `ConversationSummaryBufferMemory` / `VectorStoreRetrieverMemory`；LangGraph 分 `checkpointer`（短期）+ `BaseStore`（长期）两层
- **AutoGen** — 每个 agent 自带 `model_context`，多 agent 默认 per-agent memory
- **CrewAI** — 内置 short-term / long-term / entity 三种 memory
- **Letta（原 MemGPT）** — OS 式分层记忆，agent 通过 function call 自管分页
- **Mem0 / Zep** — 独立的 memory 服务层
- **Generative Agents**（Park et al., Stanford 2023，"Smallville" 论文）— 提出 memory stream + `recency + importance + relevance` 检索打分 + 周期性 reflection，被后续多 agent 框架广泛借鉴
- **MemGPT 论文**（UC Berkeley 2023）— main context vs archival memory 的分层抽象

概念对齐：working / short-term / long-term；episodic / semantic / procedural；private / shared。

### Artifact + 显式决策

**问题**：`panel` 场景要求"一方胜出"，但引擎只会产出一串发言，最终决策靠隐式推断，可重复性差；`debate` 同样只有过程没有裁决。

**方向**：引入共享 artifact —— 一块可读写的 markdown/JSON 区域，通过新 tool 让 agent 显式结构化输出：

- `read_artifact` / `write_artifact` — 草案协同编辑
- `propose_vote` / `cast_vote` — 结构化投票
- `closing` 阶段由 moderator `finalize_artifact` 落盘

让"讨论 → 决策"从隐式推断变成显式工件，同时解决 panel / debate 的可验证性。

### 工具生态扩展 + tool 调用可见化

**问题**：`tools.py` 目前只有 `retrieve_docs` 一个；且 tool 调用完全隐藏在 client 内部，history 里看不见，复盘和调试困难。

**方向**：

- 补充常用工具：`web_search`（或走 MCP adapter）、`calc` / `python_exec`（数据类讨论）、`search_history`（长会议里查"第 X 轮谁说过什么"）
- 在 history 里新增一类条目 `{"speaker": name, "type": "tool_call", "content": ...}`，让后续 agent 能看到"谁查了什么、查到了什么"，也便于 transcript 回放

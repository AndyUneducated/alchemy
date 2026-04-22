# TODO — multiagent engine

分两类：

- **工程问题**：影响运行正确性或可信度的 bug/瑕疵，优先处理。
- **未来功能优化**：能力扩展或体验打磨，按需推进。

来源：`~/Desktop/multiagent_test_20260421_1756/REPORT.md` 全量场景测试。

---

## 工程问题（影响运行）

### P1 — `who: all` 让 moderator 每轮抢先发言

**现象**（roundtable / panel）：`roundtable` Round 1 一开场就是主持人，内容却是"感谢张博士的精彩发言"——张博士还没开口。`panel` closing phase 2 同样把 moderator 塞到 members 前面，CEO 连续两轮开场。

**根因**：`Discussion._resolve_who("all")` 固定把 moderator 放在最前面（`discussion.py` L103-108）。作者写 `who: all` 时通常只是想表达"主持人和 members 都到齐"，但在轮级语义里变成了"主持人每轮开场"。

**方向**（任选）：
- 文档加一条约定："main 阶段若有主持人，应用 `who: members` 而不是 `who: all`"；并把 `roundtable.md` 的 main 从 `who: all` 改成 `who: members`。
- 或引入更显式的 `who: members+moderator_after`。

### P2 — Phase 退出无校验，成员静默违规

**现象**（panel closing）：指令写"每人发言后调用 `cast_vote(...)`"，但 `产品VP 林晚晴` 和 `CFO 钱正清` 只说话没投票。引擎没有任何报警，artifact 里 `v2` 缺失两张选票。

**根因**：引擎是 fire-and-forget，没有 phase 退出条件检查。

**核心目标**：不是"强制 agent 调工具"（LLM 本质上做不到强制），而是**让沉默违规变可见**。对标议会 roll call / linter warning：关键不是逼每个人投票，是"缺席必须在账上"。

**当前实现**（已落地 Level 1 + 2）：
- scenario 在 phase 里声明 `require_tool: <tool_name>`，可选 `max_retries: N`（默认 1）
- Phase 结束后扫 `artifact.drain_events()` 的 `tool` / `caller` 字段，判断当前 agent 是否调过要求的工具
- 未命中 → 追加 nudge instruction "你刚才没有调用 `<tool>`。请现在补上该调用。" 让 agent 再响应一次
- 重试用尽仍未调 → stderr 打印 `WARNING: <agent> skipped required tool '<tool>' after N attempt(s)`
- 终端加一行可视化 `🔁 [agent] retry k/N: missing <tool>`，workshop 观众能看到流程

**范围限制**：目前 `require_tool` 只识别 artifact 工具的调用（通过 `artifact.drain_events()` 观测）。non-artifact 工具（如 `retrieve_docs`）尚未被 discussion 层跟踪，属于"Tool 调用可观测" 那条 TODO 的范围。

---

## 未来功能优化

### 常用工具扩展

**问题**：`tools.py` 目前只有 `retrieve_docs` 一个，场景能做的事有限。

**方向**：按需补齐——

- `web_search`（或走 MCP adapter）
- `calc` / `python_exec`（数据类讨论）
- `search_history`（长会议里查"第 X 轮谁说过什么"）

### Tool 调用可观测

**问题**：tool 调用完全隐藏在 client 内部，成功路径对外完全静默——终端看不见、transcript 回放不出来、workshop 演示时观众不知道 agent 到底查了没查、查了什么、拿到了什么。失败路径已经由 `tools.dispatch` 的 stderr 嗅探兜底，这里只补**正常路径**。

**范围声明**：只做"对人可见"，**不**把 tool_call 塞进 history 让其他 agent 在 memory 里看到。跨 agent 的 tool 可见性成本高（四个 backend 的 tool loop 改动 + memory 渲染分支 + summary 策略 + 每轮额外 token），短期不做；artifact 机制已能承载"状态性的跨 agent 共享"这一最强用例，剩余需求等有具体驱动场景再说。

**注**：artifact 工具的成功路径已有终端打印（📝 / ➕ / 🗳 / ✓ / 🏁），这条 TODO 聚焦的是**其他**工具（当前只有 `retrieve_docs`）的正常路径可观测。

**方向**：

- `tools.dispatch` 成功时也打一行到 stderr（和失败对称，最小改动）
- `Agent.respond` 多接一个 `on_tool_call` 回调，由 `Discussion._exec_phase` 负责把 `{tool, arguments, result, ok}` 事件写进 **transcript**（不进 history）。四个 backend client（`ollama` / `openai` / `anthropic` / `gemini`）在 tool loop 里统一回调一次。
- `Discussion._print_speaker` 同级别加 `_print_tool_call`，终端输出类似 `🔧 alice.retrieve_docs(query=...) → 3 hits`

### 固定发言顺序

**问题**：`who: members` / `who: all` 的 phase 按列表顺序发言，先手信息少、后手吃免费上下文红利。但"公平性"不是全局最优解——`brainstorm` 这类协作场景里顺着上一位说下去是自然流，默认打乱反而破坏流程。

**方向**：给 phase 增加 `order` 字段，默认 `list`（当前行为，100% 兼容），`who` 为单人时忽略。候选策略：

- `list` — 当前顺序
- `rotate` — 第 k 轮从第 `k mod N` 位开始循环；`N` 轮内每人占每个位置恰好一次；**确定性 + 强公平，优先于 `shuffle`**
- `shuffle` — 每轮随机；需配 `seed` 才能复现
- `reverse` — 仅 2 轮辩论有意义

moderator 的插入位置仍由 phase 声明顺序决定，`order` 只作用于 members 子集。

**替代方向**：dynamic speaker selection——让 moderator 或专门的 selector agent 根据上一轮内容决定下一位发言者（参考 AutoGen `SelectorGroupChat`）。比 rotate/shuffle 更贴近真实 panel 动态，但触及结构化输出 + `who` 语义变化。artifact 已落地后，moderator 已经能"读取当前状态 + 产出结构化输出"，做 dynamic selection 的基础设施具备；阻塞点转为 `who` 字段语义扩展（静态名 → 运行时产出的 agent 名）与 phase 循环控制（固定 N 次 → 终止条件驱动）。

### `max_tokens` 默认偏小

**现象**：`brainstorm` 的 `产品经理` 第 2 轮、`roundtable` 的 `企业家` 第 2 轮、`社会学者` 第 3 轮都在 `max_tokens=160` 被截断，末尾出现"这种组合形式既能激发技术交流，"这种半截句。prompt 写"不超过 100 字"，但中文 1 字≈1.5 token，遇到模型话多就截断。

**方向**：把这几个 scenario 的 `max_tokens` 提到 240-300。不改也不是 bug，属 scenario 精细化。

### Scenario：`test_memory` summary 检测更严格（可选）

**现象**：`test_memory` Round 2，Windowed（memory=window）输出 `summary_seen: yes`，但它实际上没看到过任何 `<summary>` 标签——LLM 幻觉，多人参与时别的参与者提到了"summary"一词，模型就顺势报 yes。

**方向**：不改也行；要改的话把 `test_memory.md` prompt 改成"只有看到字符串 `<summary>` 才算 yes"。非工程问题，属 scenario 打磨。

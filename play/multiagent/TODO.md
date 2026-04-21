# TODO — multiagent engine

## 演进方向

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

### 固定发言顺序

**问题**：`who: members` / `who: all` 的 phase 按列表顺序发言，先手信息少、后手吃免费上下文红利。但"公平性"不是全局最优解——`brainstorm` 这类协作场景里顺着上一位说下去是自然流，默认打乱反而破坏流程。

**方向**：给 phase 增加 `order` 字段，默认 `list`（当前行为，100% 兼容），`who` 为单人时忽略。候选策略：

- `list` — 当前顺序
- `rotate` — 第 k 轮从第 `k mod N` 位开始循环；`N` 轮内每人占每个位置恰好一次；**确定性 + 强公平，优先于 `shuffle`**
- `shuffle` — 每轮随机；需配 `seed` 才能复现
- `reverse` — 仅 2 轮辩论有意义

moderator 的插入位置仍由 phase 声明顺序决定，`order` 只作用于 members 子集。

**memory 已落地**：后发言者不再有"免费上下文"优势，这条问题的一半根因被自然消掉，剩下的一半再单独解决。

**更野心的替代方向**：dynamic speaker selection——让 moderator 或专门的 selector agent 根据上一轮内容决定下一位发言者（参考 AutoGen `SelectorGroupChat`）。比 rotate/shuffle 更贴近真实 panel 动态，但触及结构化输出 + `who` 语义变化。建议与 artifact / 投票机制一起做，作为 moderator orchestration 能力的一部分。
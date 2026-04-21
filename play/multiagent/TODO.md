# TODO — multiagent engine

## Artifact + 显式决策

**问题**：`panel` 场景要求"一方胜出"，但引擎只会产出一串发言，最终决策靠隐式推断，可重复性差；`debate` 同样只有过程没有裁决。

**方向**：引入共享 artifact —— 一块可读写的 markdown/JSON 区域，通过新 tool 让 agent 显式结构化输出：

- `read_artifact` / `write_artifact` — 草案协同编辑
- `propose_vote` / `cast_vote` — 结构化投票
- `closing` 阶段由 moderator `finalize_artifact` 落盘

让"讨论 → 决策"从隐式推断变成显式工件，同时解决 panel / debate 的可验证性。

## 常用工具扩展

**问题**：`tools.py` 目前只有 `retrieve_docs` 一个，场景能做的事有限。

**方向**：按需补齐——

- `web_search`（或走 MCP adapter）
- `calc` / `python_exec`（数据类讨论）
- `search_history`（长会议里查"第 X 轮谁说过什么"）

## Tool 调用可见化

**问题**：tool 调用完全隐藏在 client 内部，history 里看不见，**事后复盘**和**其他 agent 的可见性**都缺失。运行时失败的可见性已经由 `tools.dispatch` 的 stderr 嗅探兜底；这里管的是**正常路径**——成功调用也应该可观测。

**方向**：把 tool 调用纳入 history 作为一等条目，按以下几块推进。

### history 条目结构

在 `history` 里新增一类条目：

```python
history.append({
    "speaker": owner,
    "type": "tool_call",
    "tool": fn["name"],
    "arguments": fn.get("arguments", {}),
    "result": result,   # 或截断到前 N 字符 + 省略号
    "ok": <bool>,       # 复用 tools.dispatch 里的 error JSON 嗅探
})
```

### writer 归属

`ollama_client.chat` 的 tool 循环现在只拿到 `messages`，拿不到全局 `history`。最小侵入方案是 `Agent.respond` 多接一个 `on_tool_call` 回调传下去，由 `Discussion._exec_phase` 负责往 `history` 里追加（历史一直是 Discussion 独占写入，保持一致）。三个 backend client（`ollama_client` / `openai_client` / `anthropic_client` / `gemini_client`）都要在 tool loop 里回调一次。

### memory 渲染

`memory._render` 加一个 `type == "tool_call"` 分支，渲染成 `<tool_call from="..." name="..." ok="...">\nargs: ...\nresult: ...\n</tool_call>` 作为 user 消息（owner 自己的那一条也同样处理，或者说 owner 看到的就是 assistant 之前的文本 + 紧跟一个 user-side 的 tool_call 记录，视觉上最像"我刚查了 X 得到了 Y"）。

### memory 策略

要想清楚 `<tool_call>` 在 `WindowMemory` / `SummaryMemory` 里算不算"speech"、是否计入 `max_recent`。倾向：算 speech 的姐妹，独立计数；`SummaryMemory` 的 summarizer prompt 里单独一段"保留每次检索的 query 和关键 hit"。

### CLI 输出

`Discussion._print_speaker` 同级别加一个 `_print_tool_call`，让终端也能看到"🔧 [name 调用 retrieve_docs] query=... → (hit 3 条)"，transcript 回放时也能直接读。

## 固定发言顺序

**问题**：`who: members` / `who: all` 的 phase 按列表顺序发言，先手信息少、后手吃免费上下文红利。但"公平性"不是全局最优解——`brainstorm` 这类协作场景里顺着上一位说下去是自然流，默认打乱反而破坏流程。

**方向**：给 phase 增加 `order` 字段，默认 `list`（当前行为，100% 兼容），`who` 为单人时忽略。候选策略：

- `list` — 当前顺序
- `rotate` — 第 k 轮从第 `k mod N` 位开始循环；`N` 轮内每人占每个位置恰好一次；**确定性 + 强公平，优先于 `shuffle`**
- `shuffle` — 每轮随机；需配 `seed` 才能复现
- `reverse` — 仅 2 轮辩论有意义

moderator 的插入位置仍由 phase 声明顺序决定，`order` 只作用于 members 子集。

**memory 已落地**：后发言者不再有"免费上下文"优势，这条问题的一半根因被自然消掉，剩下的一半再单独解决。

**替代方向**：dynamic speaker selection——让 moderator 或专门的 selector agent 根据上一轮内容决定下一位发言者（参考 AutoGen `SelectorGroupChat`）。比 rotate/shuffle 更贴近真实 panel 动态，但触及结构化输出 + `who` 语义变化。建议与 artifact / 投票机制一起做，作为 moderator orchestration 能力的一部分。

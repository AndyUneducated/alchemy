# Scenario kitchen sink — 写一个完整 scenario 的字段速查 + 心智模型

# ============================================================================
# 这是一个**可运行**的 scenario，同时把每个 frontmatter 字段都用上一次。
# 每行 `#` 注释解释该字段的语义、取值、默认。删掉所有注释后即得最小可读形式。
#
#   python run.py scenarios/example.md
#
# 运行前提：`../../rag/vdb/test_vdb` 已存在（test_vdb.md 用同一份 vdb）。
# 不想跑 retrieve_docs？删掉 frontmatter 里的整个 `tools:` 块即可。
# ============================================================================

---
# ── memory（scenario 级默认）────────────────────────────────────────────────

memory:
  # 三选一：
  #   full     — 永不裁剪，保留全量 history（默认行为；无需其他字段）
  #   window   — 保留所有 pinned marker（topic / turn / artifact_event）+ 最近 N 条发言
  #   summary  — 把 stale 发言增量折叠成 <summary> block；近 N 条原文保留
  type: window
  max_recent: 8
  # 仅 window / summary 需要 max_recent（正整数）。
  #
  # summary 还可以可选指定（这里不演示，因为 type=window）：
  #   model: <override SUMMARY_MODEL>
  #   max_tokens: <override SUMMARY_MAX_TOKENS>
  #   temperature: <override SUMMARY_TEMPERATURE>
  #   summarizer_prompt: <override 默认 system prompt>
  #   summarize_instruction: <override 默认压缩指令>
  #
  # 任何 agent 内部的 `memory:` 字段都会**覆盖**这里的默认。

# ── tools（scenario 级工具默认）─────────────────────────────────────────────

tools:
  # 列举本场景启用的非 artifact 工具。每项必填 `name`，其余键作为
  # "scenario 注入的默认参数"——这些参数会从 LLM 看到的 schema 中**隐藏**，
  # LLM 不需要也无法填写；调用时由 run.py 注入到 dispatch。
  #
  # 路径类参数（在 tools.py 里通过 `_path_params` 声明，目前只有 `vdb_dir`）
  # 若是相对路径，会被解析为相对于本 scenario 文件所在目录的绝对路径，
  # 这样 scenario 不依赖调用时的 cwd。
  - name: retrieve_docs
    vdb_dir: ../../rag/vdb/test_vdb
    top_k: 3
    # 这里列出的**任何键**（除 `name`）都会被 _resolve_tool_defs 从 LLM
    # 看到的 schema 中删除并由 run.py 注入——这就是"scenario pin"。所以
    # `vdb_dir` / `top_k` 在本场景里 LLM 都看不到、也填不了。
    # 没列在这里的参数（如 mode / rerank）保持 LLM-visible，可由 LLM 在
    # tool_call 时自由选择：
    #   mode: "dense" | "bm25" | "hybrid"   (默认 hybrid)
    #   rerank: true                          (默认 false；首次调用 ~5s 加载 ~1.2GB 模型)
    # 想把 rerank 强制开成"高精度场景"或把 mode 锁成 dense 跑对照，
    # 直接在本块加同名键即可——效果是该参数从 LLM schema 删除 + 注入默认。

# ── artifact（共享结构化文档 + 投票）────────────────────────────────────────

artifact:
  enabled: true
  # 仅当 enabled=true 时 ArtifactStore 才创建，artifact 工具才发给 agent。

  # initial_sections 决定"哪些 section 一开始就存在 + 它们的 mode"。
  # 没在这里声明的 section 也能被 write/append 创建（但没有 mode 限制）。
  initial_sections:
    # 项可以是字符串（mode 默认 replace）...
    - decision

    # ...也可以是 {name, mode}：
    #   mode: replace — write_section ✓，append_section 返回 error
    #   mode: append  — append_section ✓，write_section 返回 error
    - {name: notes, mode: append}
    - {name: data,  mode: replace}

  # tool_owners 决定每个 artifact 工具对哪些 agent 可见 / 可调用。
  # 取值形态完全对齐 step.who：
  #   <role>       — 仅该 role 的 agent（"moderator" / "member"）
  #   all          — 所有 agent
  #   [name1, ...] — 显式 agent 名单
  # **未声明的工具默认对所有 agent 开放**——包括 finalize_artifact / propose_vote。
  # 想保留"主持人专属"行为必须显式写出来：
  tool_owners:
    propose_vote: moderator
    finalize_artifact: moderator

# ── agents（必填，至少 1 项）────────────────────────────────────────────────

agents:
  # 每个 agent 必填 name / prompt / role。其它字段可选。
  # role 取值仅限 "moderator" / "member"，决定：
  #   - step.who: moderator | member 的寻址命中
  #   - artifact.tool_owners 中 role 形态展开
  - name: 主持人
    role: moderator
    # `name` 在 prompt 注入、history 投影、artifact_event 的 caller 字段、
    # step.who 的 list 寻址里都用同一个字符串，必须**全场唯一**。
    prompt: |
      你是讨论主持人。开场介绍话题，结尾总结并落定决策。
      保持中立、简短，每次发言不超过 80 字。用中文回答。
    temperature: 0.5  # 可选；不写则用 config.TEMPERATURE
    max_tokens: 200   # 可选；不写则用 config.MAX_TOKENS
    # model: <override>  # 可选；不写则用 config.DEFAULT_MODEL（按 BACKEND 决定）
    # memory: { ... }    # 可选；按上面 memory 同结构覆盖 scenario 级默认

  - name: 分析师
    role: member
    prompt: |
      你是数据分析师。你必须先调用 retrieve_docs 查文档，再回答；
      并且把关键事实 append 到 artifact 的 notes 节。
      每次发言不超过 80 字。用中文回答。
    max_tokens: 200

  - name: 决策者
    role: member
    prompt: |
      你是产品决策者。读取 artifact 后给出二选一立场，并在最终阶段投票。
      每次发言不超过 80 字。用中文回答。
    temperature: 0.6
    max_tokens: 200
    memory:
      # agent 级 memory 覆盖 scenario 级默认。这里演示给"决策者"单独用 full
      # （他想看到全量历史，不剪窗口）。
      type: full

# ── steps（必填，扁平流程列表）──────────────────────────────────────────────

steps:
  # 每项字段：
  #   id           — 可选；只用于终端打印 (step=<id>) 提升可读性
  #   who          — 必填，三种形态：scalar role/all、list[name]
  #   instruction  — 必填非空；本 step 给当前发言者的额外引导
  #                  （不进 history，仅作为 user 消息一次性带入）
  #   require_tool — 可选；step 结束后扫 artifact 事件，若该 caller 未调用
  #                  指定工具则触发 nudge 重试
  #   max_retries  — 可选，重试次数。require_tool 存在时默认 1，否则默认 0
  #
  # 引擎按 steps 列表顺序逐项展开成 turn：每个 step 内 who 的所有匹配 agent
  # 各自发言一次，按"agents 声明顺序"。每 turn 注入一个 pinned 的
  # <turn>turn X of N</turn> marker，让 agent 感知自己在流程中的位置。

  - id: open
    who: moderator
    instruction: 用一句话介绍今天的话题，并提醒成员先 retrieve_docs。

  - id: research
    who: [分析师]
    # list 形态：精确点名。哪怕只有一个名字也写在 [] 里——这样 schema 校验
    # 就能用"类型 (scalar vs list)"区分"按 role 寻址"和"按 name 寻址"。
    instruction: |
      调用 retrieve_docs 查"项目代号"，再 append_section(name="notes",
      entry="- 项目代号: <你查到的值>") 把事实写进 artifact。
    require_tool: append_section
    max_retries: 1

  - id: deliberate
    who: member
    # 按 role 命中所有 member（分析师 + 决策者，按 agents 声明顺序）。
    instruction: |
      围绕话题给出你的观点；分析师可继续 append notes，决策者可 read_artifact。

  - id: focus
    who: moderator
    instruction: 用一句话提炼本轮分歧或共识。

  - id: open_vote
    who: moderator
    instruction: |
      调用 propose_vote(question="是否采纳建议?", options=["采纳", "拒绝"])
      发起最终投票，再用一句话请大家投票。

  - id: ballot
    who: member
    require_tool: cast_vote
    # require_tool 经典用法：强制每个 member 在最终阶段投票。
    # 若某 member 跳过 cast_vote，引擎会 nudge 重试一次；仍跳过则 stderr WARNING。
    instruction: |
      调用 cast_vote(vote_id="v1", option="采纳" 或 "拒绝", rationale="一句话理由")。

  - id: finalize
    who: moderator
    instruction: |
      先 write_section(name="decision", content="<最终结论>"),
      再 finalize_artifact(decision="采纳" 或 "拒绝", rationale="...")。
      finalize 只能调用一次，幂等保护——重复调用返回 error。

# ============================================================================
# `who` 取值形态（共四种）
#   moderator      — scalar role；按 role 命中（要求至少 1 个 moderator）
#   member         — scalar role；按 role 命中（要求至少 1 个 member）
#   all            — scalar 关键字；所有 agent，按声明顺序
#   [n1, n2, ...]  — 显式名单；按列表中给定的顺序，名字必须存在
# 写错 who 在启动时报错（schema validation），不会走到运行时。
#
# 字段省略策略
#   - agents 必填且至少 1 项；steps 必填且至少 1 项（每 step 必填 instruction）
#   - tools 省略 → 没有非 artifact 工具
#   - artifact 省略或 enabled=false → 没有共享 artifact，6 个 artifact 工具全部不可用
#   - artifact.tool_owners 省略 → 所有 artifact 工具对所有 agent 开放
#   - memory 省略 → 全员 FullHistory
# ============================================================================
---

## 运行时心智模型（body 即话题，但同时充当本示例的"逻辑说明"）

下面这些是**字段如何串起来**的运行时画面，新作者写第一个 scenario 前先建立这套
心智模型，之后改字段就只是"调旋钮"：

1. **一次 run = 按 steps 顺序展开成线性 turn 序列**。每 step 内 who 命中的所有
   agent 各自发言一次（按 agents 声明顺序）。step 之间共享一份权威 `history`
   （topic / turn / speaker / artifact_event / tool_call），每个 agent 在
   `respond()` 时按 `speaker == owner` 投影成 `assistant`，他人投影成
   `<message from="X">…</message>`，控制流投影成带 XML 标签的 user 消息。

2. **memory 决定"哪些 history 进入投影"**。pinned 类型（topic / turn /
   artifact_event）永远不被剪，所以 turn 切换、artifact 变更对所有 memory 策略
   都可见。`window` 只额外保留最近 N 条发言；`summary` 把 stale 发言增量折叠成
   `<summary>` block。

3. **artifact 视图带外注入**——每次 `respond()` 调用时，`ArtifactStore.render()`
   作为 `<artifact>…</artifact>` user 消息一次性塞进当前轮的 prompt，但**不**
   进 history。所以 artifact 状态对所有人始终最新，且不占 memory 配额。
   而每次 write/append/vote/finalize 会产生 `artifact_event`（pinned），
   进 history、所有人都看得见。

4. **require_tool 不强制，只是让"沉默违规"可见**——step 结束后扫
   `artifact.drain_events()`，若 caller 未调用指定工具，nudge 一次重试；
   仍跳过则 stderr 一行 WARNING，run 继续。这是"workshop 友好"的折中：
   不阻塞演示，但留下可见痕迹。

5. **tools 隐藏机制**——`tools:` 下声明的 scenario 默认参数（如 `vdb_dir`）
   会从 LLM 看到的 OpenAI tool schema 中删除，调用时由 run.py 注入。LLM 既
   不需要知道路径，也无法覆盖。`_path_params` 标记的路径参数还会自动按
   scenario 文件所在目录解析相对路径——scenario 因此可以从任何 cwd 调起。

6. **本话题（用来跑通流程）**：
   请围绕"是否将 retrieve_docs 查到的项目代号采纳为正式名称"展开讨论，
   分析师先查文档把事实写进 artifact，决策者读 artifact 后投票。

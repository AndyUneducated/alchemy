# Scenario kitchen sink — 字段速查 + 心智模型 + 集成烟囱（同一份文件）

# ============================================================================
# 这是一个**可运行**的 scenario：frontmatter 里保留逐字段教学注释；steps 采用
# **压缩后的集成路径**（artifact + retrieve_docs + window/full/summary +
# require_tool nudge），用更少 turn 换更短 wall-clock，而注释仍解释「为什么
# 这样够覆盖」。若只想最省 token，可把 agents 的 prompt 再缩短或删掉 `tools:`。
#
# **CI / 回归**：`ci_who_member` + `ci_who_all` 两步刻意命中 `who: member` 与
# `who: all` 标量（与 `roundtable` / `debate` / `brainstorm` 同源写法），单文件即可
# 验证 `_resolve_who` 四条路径；不含「无 moderator」拓扑——那仍由 `debate.md` 等
# 轻场景覆盖。
#
#   python -m agent_engine scenarios/example.md
#
# 运行前提：`../../rag/vdb/test_vdb` 已存在（见 play/rag README 的 ingest）。
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
  # 集成烟囱里 turn 数已压过一轮旧版 example；max_recent 略收紧即可配合 window，
  # 仍足够覆盖「pinned 不剪 + 近期发言」行为。需要更宽窗口可调回 8。
  max_recent: 6
  # 仅 window / summary 需要 max_recent（正整数）。
  #
  # summary 在 scenario 级还可选指定（本文件 scenario.type=window，故不在此
  # 写 summary 块；改 scenario 为 type:summary 时可配下列键，agent 级同理）：
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
  # LLM 不需要也无法填写；调用时由 scenario.py 的 _build_tool_handler 注入到 dispatch。
  #
  # 路径类参数（在 tools/ 包里通过 `_path_params` 声明，目前只有 `vdb_dir`）
  # 若是相对路径，会被解析为相对于本 scenario 文件所在目录的绝对路径，
  # 这样 scenario 不依赖调用时的 cwd。
  - name: retrieve_docs
    vdb_dir: ../../rag/vdb/test_vdb
    top_k: 3
    # 这里列出的**任何键**（除 `name`）都会被 _resolve_tool_defs 从 LLM
    # 看到的 schema 中删除并由 scenario.py 注入——这就是"scenario pin"。所以
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
      你是主持人。按 instruction 调用 artifact 工具；每次 ≤ 40 字。用中文回答。
    temperature: 0.4
    max_tokens: 160
    # 可选；不写则用 config.TEMPERATURE / MAX_TOKENS
    # model: <override>  # 可选；不写则用 config.DEFAULT_MODEL（按 BACKEND 决定）
    # memory: { ... }    # 可选；按上面 memory 同结构覆盖 scenario 级默认

  - name: 分析师
    role: member
    prompt: |
      你是分析师。需要查事实时先 retrieve_docs；按 instruction 用 artifact 工具。
      每次 ≤ 40 字。用中文回答。
    max_tokens: 160

  - name: 决策者
    role: member
    prompt: |
      你是决策者。按 instruction 调用 cast_vote 等；每次 ≤ 40 字。用中文回答。
    max_tokens: 160
    memory:
      # agent 级 memory 覆盖 scenario 级默认。这里演示给"决策者"单独用 full
      # （他想看到全量 history，不剪窗口）。
      type: full

  # 第四位成员专门演示 summary 策略 + summarizer 额外 LLM 调用；prompt 要求
  # 结构化短输出，减少生成耗时，同时仍能在 transcript 里对照三 memory。
  - name: 汇总员
    role: member
    prompt: |
      你是对话可见度观察员。每次严格三行、无寒暄：
      visible_speakers: <逗号分隔历史里见过的发言者名；没有则 none>
      memory_type: <window|full|summary，据你 system 判断>
      summary_seen: <yes|no，是否出现 summary 块>
      ≤ 50 字。用中文。
    memory:
      type: summary
      max_recent: 2
      summarizer_prompt: 把多说话者对话压成不超过 60 字的中文要点，保留人名与立场。
      summarize_instruction: 合并输入为一段紧凑摘要；若含 previous_summary 则合并改写。
    max_tokens: 120
    temperature: 0

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
  #
  # 下面 steps 相对旧版「kitchen sink」删了 deliberate/focus 等多轮闲聊，
  # 把「检索 + append + append/write 冲突 + read + 投票 + require_tool nudge」
  # 压进更少 step；教学含义见每步行内注释。
  #
  # CI：`open` 已覆盖 `who: moderator`；`mem_warm*` 覆盖 `who: [name,...]`；
  # 下列两步补齐 scalar `member` / `all`（instruction 强制禁工具，避免污染投票段）。

  - id: open
    who: moderator
    instruction: 一句话介绍话题：是否采纳检索到的「项目代号」为正式名称。

  # 按 agents 声明顺序展开为：分析师 → 决策者 → 汇总员（discussion._resolve_who）。
  - id: ci_who_member
    who: member
    instruction: |
      本节为寻址烟测。只输出一字「到」，禁止调用任何工具。

  # 按 agents 声明顺序展开为：主持人 → 分析师 → 决策者 → 汇总员。
  - id: ci_who_all
    who: all
    instruction: |
      本节为寻址烟测。只输出一字「全」，禁止调用任何工具。

  # 两轮短答堆叠发言，触发汇总员的 SummaryMemory 折叠（见 memory.py 触发规则）。
  - id: mem_warm
    who: [分析师, 决策者, 汇总员]
    instruction: |
      严格按 system 要求的三行格式作答；不要调用工具。

  - id: mem_warm2
    who: [分析师, 决策者, 汇总员]
    instruction: |
      再答一轮三行格式；不要调用工具。

  # 合并原 research + artifact 烟囱：retrieve_docs、append、故意 write 触发
  # append-only 报错、read_artifact；require_tool 仍盯 append_section。
  - id: vdb_artifact
    who: [分析师]
    # list 形态：精确点名。哪怕只有一个名字也写在 [] 里——这样 schema 校验
    # 就能用"类型 (scalar vs list)"区分"按 role 寻址"和"按 name 寻址"。
    instruction: |
      1) retrieve_docs 查询「项目代号」；
      2) append_section(name="notes", entry="- 项目代号: <值>");
      3) 故意 write_section(name="notes", content="bad") 触发 append-only 报错，用一句话承认；
      4) read_artifact() 确认 notes 内容并一句话复述。
    require_tool: append_section
    max_retries: 1

  - id: vote_prep
    who: moderator
    instruction: |
      read_artifact()；再 propose_vote(question="是否采纳?", options=["采纳","拒绝"])；一句话请大家投票。

  # instruction 故意不提 cast_vote，用来测「沉默 → nudge → retry」；若模型
  # 第一轮就投票则不会看到 retry（行为仍合法）。
  - id: ballot_nudge
    who: [分析师]
    require_tool: cast_vote
    max_retries: 1
    instruction: |
      只用一句话打招呼，不要提 cast_vote。

  # 显式 cast_vote，快速覆盖「正常投票」路径（与 ballot_nudge 对照）。
  - id: ballot_ok
    who: [决策者]
    require_tool: cast_vote
    max_retries: 1
    instruction: |
      cast_vote(vote_id="v1", option="采纳" 或 "拒绝", rationale="一句话")。

  - id: finalize
    who: moderator
    instruction: |
      write_section(name="decision", content="结论：<与投票一致>")；
      finalize_artifact(decision="采纳" 或 "拒绝", rationale="一句话")。
      finalize 只能调用一次，幂等保护——重复调用返回 error。

# ============================================================================
# `who` 取值形态（共四种）
#   moderator      — scalar role；按 role 命中（要求至少 1 个 moderator）→ open
#   member         — scalar role；按 role 命中（要求至少 1 个 member）→ ci_who_member
#   all            — scalar 关键字；所有 agent，按声明顺序 → ci_who_all
#   [n1, n2, ...]  — 显式名单；按列表中给定的顺序，名字必须存在 → mem_warm* / vdb_artifact …
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
   `<summary>` block（本例中由「汇总员」的 agent 级配置演示；主持人/分析师跟 scenario 默认 window，决策者用 full）。

3. **artifact 视图带外注入**——每次 `respond()` 调用时，`ArtifactStore.render()`
   作为 `<artifact>…</artifact>` user 消息一次性塞进当前轮的 prompt，但**不**
   进 history。所以 artifact 状态对所有人始终最新，且不占 memory 配额。
   而每次 write/append/vote/finalize 会产生 `artifact_event`（pinned），
   进 history、所有人都看得见。

4. **require_tool 不强制，只是让"沉默违规"可见**——step 结束后扫
   `artifact.drain_events()`，若 caller 未调用指定工具，nudge 一次重试；
   仍跳过则 stderr 一行 WARNING，run 继续。这是"workshop 友好"的折中：
   不阻塞演示，但留下可见痕迹。本例 `ballot_nudge` 故意用「只打招呼」的
   instruction 与 `require_tool: cast_vote` 组合，便于观察 retry。

5. **tools 隐藏机制**——`tools:` 下声明的 scenario 默认参数（如 `vdb_dir`）
   会从 LLM 看到的 OpenAI tool schema 中删除，调用时由 scenario.py 注入。LLM 既
   不需要知道路径，也无法覆盖。`_path_params` 标记的路径参数还会自动按
   scenario 文件所在目录解析相对路径——scenario 因此可以从任何 cwd 调起。

6. **本话题（用来跑通压缩后的流程）**：
   请围绕「是否将 retrieve_docs 查到的项目代号采纳为正式名称」落定；`vdb_artifact`
   一步内完成检索与 artifact 读写演示；`mem_warm`*2 让三 memory 同场短跑；
   `ballot_nudge` / `ballot_ok` 分拆覆盖 nudge 与正常投票。

7. **CI**：`ci_who_member` / `ci_who_all` 用极短输出验证 `who` 的标量 `member` 与
   `all` 在整轮 run 中可正确展开；与「无 artifact」「无 moderator」类拓扑正交，
   后者继续用 `debate.md` / `brainstorm.md` 等小文件即可。

是否将文档中的项目代号定为团队对外正式名称。

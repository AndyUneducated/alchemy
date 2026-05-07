# Decisions

> 记录标准：只保留对后续架构演进、跨项目边界、可维护性、面试问答有持续价值的决策。
> 删除标准：一次性排障过程、可从代码 / commit 直接还原的实现流水、重复 supersession 细节。
> 日期以 git commit 历史为准。日常进度（按里程碑）见 [`JOURNAL.md`](JOURNAL.md)。

## 1. Runner 形态：线性 stages × 2 stage type，刻意不做 DAG / retry / cron

- **Date**: 2026-04-26

### Context

workshop 需要把 [`play/agent_engine`](../agent_engine/) 的 `Engine.invoke` 套一层"把 agent 与确定性 Python 函数串成端到端 pipeline"的最薄壳。同期面对的是一个抽象选择：是直接复用工业级编排（Prefect / Temporal / Argo / Airflow），还是自写最小 runner？前者免费拿到 retry / UI / durability / DAG，后者拿到的是边界清晰、零基础设施依赖、~350 行的可读体量。

### Options considered

|Option|说明|优点|风险 / 成本|
|---|---|---|---|
|A. 直接用 Prefect / Temporal / Argo|工业级编排框架|retry / UI / durability / scheduler 全免费|引入 thousands LOC 依赖 + scheduler / UI 服务 + 学习曲线，与 workshop 节奏不符|
|B. LangChain LCEL|Python chain DSL|表达力强|流式 / chain 抽象偏 LLM 链，不适合"agent 与确定性函数混编"|
|C. Airflow / Argo DAG|声明式 DAG|可并行 / 条件 / 重试|需 scheduler / metadata DB / UI server，重|
|D. 自写最小 runner（选）|线性 stages × 2 stage type，~350 行|边界清晰、零外部依赖、可读|不可表达 DAG / 并行 / retry，复杂场景需迁框架|

### Decision

采用 **D**：`stages:` 是线性列表，每条 stage `type` 仅 `deterministic`（普通 Python callable）或 `agent`（委托给 `Engine.invoke`）。**显式不做** retry / timeout / circuit-breaker / DAG / 条件 / 循环 / 并行 / cron / 调度 / 持久化 / resume / inline Python in YAML / workflow stdlib / 自动注册装饰器 / 多 CLI 子命令。需要这些能力时：先看是否能由 hook 函数内部解决（`tenacity` 包重试、`signal` 包超时、`subprocess` 包外部调用），不行则迁 Prefect / Temporal（plan §9 论证迁移成本 ~3–4h）。

```mermaid
flowchart LR
    Y[workflow.yaml<br/>stages: 线性列表] --> R[Workflow.run<br/>顺序执行]
    R -->|type=deterministic| D[executors.deterministic<br/>fn(**args)]
    R -->|type=agent| A[executors.agent<br/>Engine.invoke(**config)]
    D --> ST[(state)]
    A --> ST
    ST --> R
    R --> LOG[stage start/done<br/>+ duration_ms]
```

### Consequences

|影响|结果|
|---|---|
|库体量|≤ ~350 行，单人可在一上午内通读全部源码|
|外部依赖|零基础设施（无 scheduler / UI / DB），CLI 一条命令即可跑|
|表达力|不能表达 DAG / 并行 / 条件分支；遇到这类需求强制走 hook 或多份 yaml|
|迁移路径|未来如需 retry / UI / durability，整体迁 Prefect / Temporal 而非在 workflow 内堆叠|
|可观测性|每 stage `start` / `done` + `duration_ms` 直接打 stdout，无需日志框架|

### 示例

|场景|在该决策下如何处理|
|---|---|
|想给 agent stage 加 retry|hook 函数里用 `@tenacity.retry` 包一层；不动 runner|
|要做 A/B 分支|写两份 yaml + 上层 shell 选；不在 yaml 表达条件|
|想用 cron 每小时跑|交给 GitHub Actions / 系统 cron；runner 是 one-shot|
|未来真需要 DAG|整体迁 Prefect（plan §9 量化为 ~3–4h），不在 workflow 内补功能|

### 面试官可能问

|问题|回答要点|
|---|---|
|为什么不直接用 Prefect？|workshop 节奏要 ~350 行可读体量；Prefect 是 thousands LOC + scheduler 服务，与"实验性子项目"边界不匹配|
|限制这么死，工程化时不就废了？|边界明确 → 迁移成本可量化（plan §9 给出 ~3–4h）；这是"现在便宜 + 未来不锁死"，比"现在重 + 未来可能用不上"好|
|为什么 trace_id 不实现？|今天没有 distributed 调用链需求；但保留 W3C `traceparent` env 变量名 + JSON 字段名（plan §9.1），未来零成本接入|

## 2. Agent stage 接口：`config:` 原样 unpack 进 `Engine.invoke`，state 只收 `Result.artifact`

- **Date**: 2026-04-26

### Context

workflow 存在的唯一意义是给 [`play/agent_engine`](../agent_engine/) 套外层；它跟 LLM 之间只有一个接口点（`executors/agent.py`，~19 行）。如果 workflow 在 yaml schema 里显式列出 agent_engine 的字段（`initial_artifact` / `transcript_path` / `artifact_path` / `print_stream` / `callbacks`…），任何 agent_engine API 演进都会反向破坏 workflow，违反 plan §2 "workflow 自身不内嵌 LLM 逻辑" 的关键边界。

### Options considered

|Option|说明|优点|风险 / 成本|
|---|---|---|---|
|A. workflow 在 schema 里显式列 agent_engine 字段|每个 kwarg 在 yaml 端都有 schema 校验|静态可知合法字段；错误更早暴露|agent_engine 加字段必须升 workflow，破坏分层；workflow 半懂 LLM 语义|
|B. workflow 透明传递（选）|`config:` 块原样 `**unpack` 进 `Engine.invoke(**config)`，workflow 对内部命名 oblivious|workflow 与 agent_engine 完全解耦；agent_engine 端加字段（如 `--save-result-json` envelope）零下游成本|不能拒绝错误的 agent_engine 字段，错误延迟到 `Engine.invoke` 抛 `TypeError`|

### Decision

采用 **B**：`executors/agent.py` 仅做 `Engine(scenario).invoke(**config)` + 把 `Result.artifact` 写进 state；`config:` 块 schema 不校验内部内容。state 只收 `Result.artifact`（`dict[section, content]`），不收 `transcript` / `success` / `warnings` 等其它 `Result` 字段——这把 workflow 与 LLM 的耦合面收紧到**一个数据形态**，换 LLM 引擎（如换成 CrewAI `Crew.kickoff()` / LangGraph `Graph.invoke()`）只改 `executors/agent.py` 19 行。

```mermaid
flowchart LR
    Y[yaml stage:<br/>type: agent<br/>scenario: ...md<br/>config: { ... }] --> AG[executors.agent.run]
    AG -->|Engine.invoke<br/>**config 原样透传| ENG[(agent_engine<br/>Engine)]
    ENG -->|Result.artifact<br/>仅一个字段| AG
    AG --> ST[(state[stages.X.output])]
    note["workflow 不解析<br/>config 内部命名"] -. 边界 .- AG
```

### Consequences

|影响|结果|
|---|---|
|跨项目演进|agent_engine API 演进（envelope 扩展 / new flags）零下游升级成本|
|耦合面|唯一 LLM 耦合点 = `executors/agent.py` 19 行；换 LLM 引擎改动面已知|
|错误时机|无效 `config:` 字段在 `Engine.invoke` 抛 `TypeError` 而非 yaml 加载期；trade-off 接受（fail-fast 仍然成立，只是从 schema 期推到执行期）|
|state 模型|stage 间只能传"渲染过的 artifact 字符串"，不能传 transcript 等内部对象——逼着下游 stage 走显式数据流|

### 示例

|场景|在该决策下如何处理|
|---|---|
|agent_engine 加 `transcript_path` flag|workflow yaml 立即可用 (`config: { transcript_path: ... }`)，无需升级 workflow|
|想换 LLM 引擎到 CrewAI|改 `executors/agent.py` 一个文件，schema / state / 其它 stage 全不动|
|下游 stage 想读 transcript|不行——agent stage 只导出 `Result.artifact`；要读 transcript 自己 `Path.read_text()` 落盘文件|

### 面试官可能问

|问题|回答要点|
|---|---|
|不校验 config 字段不会出错吗？|出错也是 fail-fast——`Engine.invoke` 抛 `TypeError`，错误信息即标准 Python，runner 不二次包装|
|跟"显式优于隐式"原则冲突吗？|不冲突——workflow 显式声明的边界就是"我不解释 agent_engine 内部"；agent_engine 内部该有的显式约束在 agent_engine 自己的 schema 里|
|为什么 state 不收 `transcript` / `warnings`？|耦合面收紧 → 换引擎成本可控；想要这些数据走文件落盘（`transcript_path`）让下游 stage 自己读，让"跨 stage 数据契约"显式|

## 3. 配置层：schema fail-fast + `{{ a.b.c }}` 最小模板 + kitchen_sink.yaml 单点 SoT

- **Date**: 2026-04-26

### Context

配置层每条决策都在权衡两件事：① 作者犯错时反馈速度（fail-fast vs 友好提示）；② 未来 scope creep 的入口在哪（模板 DSL 越强 → yaml 越容易长出业务逻辑 → 失控风险越高）。同期还需要决定字段说明的 SoT 在哪（README vs yaml 注释），双写最常见的腐烂源。

### Options considered

|Option|说明|优点|风险 / 成本|
|---|---|---|---|
|A. Jinja2 模板 + schema migration + 友好错误提示|工业级配置层|表达力强、上手友好|`{{ x | upper | trim }}` 类 filter 让数据转换泄到 yaml；migration 是预付未来收益不清的成本|
|B. 最小 fail-fast 集合（选）|`{{ a.b.c }}` 路径访问；schema 必填缺失即 `sys.exit`；fn 双解析；kitchen_sink.yaml 单点 SoT|错误立即可见；数据转换强制下放 hook；yaml 不长出业务|不友好（无猜词提示 / 无 migration）|

### Decision

采用 **B**，分四条子决策：

|子决策|实现|理由|
|---|---|---|
|schema 必填缺失|`sys.exit("Error: ...")` 直退，不给"你大概想用 X"提示|单人 workshop 没有"老用户"，猜词提示是预付收益不清的成本|
|模板能力|仅 `{{ a.b.c }}` 路径访问；整字符串单占位保 Python 类型（`{{ stages.x.output }}` → list/dict 原样），inline 占位强制 `str()`；miss 直接 `KeyError`|过滤器 / 表达式 / 条件 一旦开口就关不上；数据转换写进 hook 函数（参考 kitchen_sink 的 `to_yaml` stage）|
|fn 字符串解析|`module:callable`（含冒号 = 完整路径）或顶层 `hooks_module` 默认命名空间|hook 既可来自外部 pkg 也可来自 yaml 同级；显式 import，调试可见，不上自动注册装饰器|
|字段 SoT|`examples/kitchen_sink.yaml` —— 每个字段用一次 + 行内 `#` 注释 + 末尾"运行时心智模型"段；README 只做总览|双写迟早腐烂；新作者从可运行示例开始更快上手|

```mermaid
flowchart LR
    Y[workflow.yaml] --> SC[schema.validate]
    SC -->|缺必填| EX1[sys.exit Error]
    SC -->|OK| ST[state.interpolate<br/>{{ a.b.c }}]
    ST -->|路径不存在| EX2[KeyError 直接抛]
    ST -->|OK| EXEC[stage executor]
    EXEC -->|fn 字符串| FN{module:callable?}
    FN -->|含 :| IMP1[import module<br/>get callable]
    FN -->|无 :| IMP2[hooks_module<br/>default ns]
    EXEC -->|hook raise| EX3[原 traceback 直传]
```

### Consequences

|影响|结果|
|---|---|
|错误反馈|首次跑就能命中所有 schema / 模板 / hook 错误，不需要 dry-run / validate 子命令|
|yaml 表达力|模板永远简单可读；复杂逻辑被强制下放到 hook 函数（Python > YAML 表达力）|
|文档维护|README 与 yaml 不双写字段说明；演化时只改 kitchen_sink.yaml 一处|
|不友好性|无"你大概想用 X" / 无 schema migration / 无字段废弃路径——单人项目可接受，多人项目要补|

### 示例

|场景|在该决策下如何处理|
|---|---|
|必填字段忘了填|`sys.exit("Error: stage 'discuss' missing required field 'scenario'")`，提示文件名 + 字段|
|模板路径不存在|`KeyError: 'stages.foo.output'` 直接抛完整路径，traceback 即诊断|
|想给 list 排序|不靠模板 filter；写 hook stage `fn: sort_by`，args 接 list，return sorted list|
|加新字段|改 schema.py + kitchen_sink.yaml；README 不动（避免双写）|

### 面试官可能问

|问题|回答要点|
|---|---|
|为什么不用 Jinja2？|filter / 表达式一旦开放就关不上 → yaml 长出业务逻辑 → 不可读；hook 函数才是数据转换的地方，Python 比 YAML 表达力强且可调试|
|schema migration 是不是行业默认？|是，但 workshop 子项目没有"老用户"——加 migration 是预付未来收益不清的成本，YAGNI|
|kitchen_sink.yaml 当 SoT 不会过时？|每次改 schema 必跑 example，example 跑通即文档自动校准；README 双写就没这个反馈环|

## 非目标（持续有效）

|项|说明|
|---|---|
|追求工业级编排能力（retry / DAG / UI / durability）|这些是 Prefect / Temporal 的领域；workflow 边界明确 → 迁移成本可量化（plan §9: ~3–4h）|
|在 yaml 表达数据转换 / 业务分支|`{{ a.b.c }}` 之外一律走 hook 函数；YAML 嵌 Python 是反模式|
|workflow 解释 agent_engine 内部命名|`config:` 原样透传是 workflow 的存在意义；解释 = 紧耦合|
|友好错误提示 / schema migration|fail-fast 是单人 workshop 的最优策略；多人项目再补|
|workflow stdlib / 多 CLI 子命令|YAGNI，只 1 个真消费者就不抽；scope creep 在 code review 时显眼|

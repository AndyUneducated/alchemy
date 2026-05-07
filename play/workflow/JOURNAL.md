# Journal

> 日期以实际 commit 历史为准。每个里程碑围绕 1 段 100-300 字的“为什么这么做、对未来意味着什么”叙事展开，配框架变更表、必要时的 mermaid 图、以及当期新增/改动的模块、CLI、示例。

## 2026-04-26 — MVP runner：声明式 deterministic + agent stage pipeline

这个阶段的里程碑是给 `play/agent_engine` 套上一层最简的“可声明编排”。`Workflow.from_yaml` / `Workflow.run` 一共 ~420 行，按 YAML 里 `stages:` 的顺序串接两类 stage：deterministic（普通 Python callable）与 agent（委托给 `agent_engine.Engine.invoke`）。最值得讲的不是支持的能力，而是显式不做的能力——**没有 retry / timeout / cron / DAG / YAML 里写 inline Python / 多 CLI 子命令 / 持久化**。CLI 也只有 `python -m workflow run <yaml>` 一个子命令，validate / list / inspect 一律不做。这种刻意的最小性让 workflow 与 agent_engine 的边界清晰：workflow 负责“按声明顺序串”，agent_engine 负责“跑一段会议”，两者通过 `config:` 原样 unpack 进 `Engine.invoke(**config)` 解耦——workflow 对 agent_engine 内部命名 oblivious，只收 `Result.artifact` 进 state，这就是它与 LLM 的唯一耦合点。同期把“显式不做项”刻进 README，让未来的 scope creep 在 code review 时一眼可见。

### 框架变更

|变更|目的|
|---|---|
|`Workflow.from_yaml` / `Workflow.run`（~420 行）|声明式 stage pipeline 的最小可用形态|
|线性 `stages:` 仅 `deterministic` + `agent` 两类|刻意不做 DAG / retry / cron / 持久化|
|`config:` 原样 unpack 进 `Engine.invoke(**config)`|workflow 对 agent_engine 内部命名 oblivious|
|state 只收 `Result.artifact`|workflow 与 LLM 的唯一耦合点，最小化集成面|
|`{{ a.b.c }}` 点路径插值（仅）|表达式能力收紧，复杂转换强制写进 hook 函数|
|miss 直接 `KeyError` / 必填缺失 `sys.exit`|fail-fast，不给“你大概想用 X”猜词提示|
|`module:callable` + 顶层 `hooks_module` 双解析路径|hook 既可来自外部模块也可来自 yaml 同级|
|每 stage 打印 `start` / `done` + `duration_ms`|执行时序可见，无需额外日志框架|
|`trace_id` 不实现，但保留 W3C `traceparent` 字段名|未来零成本接入 distributed tracing|

```mermaid
flowchart LR
    Y[workflow.yaml<br/>+ vars k=v] --> WF[Workflow.from_yaml<br/>+ schema.validate]
    WF --> R[Workflow.run]
    R -->|stage 类型 dispatch| D[deterministic executor<br/>module:callable]
    R --> A[agent executor<br/>Engine.invoke config]
    D --> ST[(state<br/>{{ a.b.c }} 插值)]
    A -->|Result.artifact| ST
    ST --> R
    R --> LOG[stage start/done<br/>+ duration_ms]
```

### 新增 / 改动模块

|模块|说明|
|---|---|
|`runner.py`|`Workflow.from_yaml` / `Workflow.run`，stage 顺序执行 + 时序输出|
|`schema.py`|YAML 必填字段校验，缺失即 `sys.exit("Error: ...")`，不做 migration|
|`state.py`（~50 行）|`{{ a.b.c }}` 点路径插值；整字符串单占位保 Python 类型，inline 占位强制 `str()`|
|`executors/deterministic.py`|`fn` 字符串支持 `module:callable` 与顶层 `hooks_module` 双解析|
|`executors/agent.py`|`config:` 原样 unpack 进 `Engine.invoke(**config)`，state 只收 `Result.artifact`|
|`cli.py` / `__main__.py`|单子命令 `python -m workflow run <yaml> --vars k=v ...`|

### 新增 examples / 演示场景

|示例|目的|演示什么|
|---|---|---|
|`examples/kitchen_sink.yaml` + `kitchen_sink_hooks.py`|schema 字段速查 + 端到端跑 3 deterministic + 1 agent + 1 finalize|每个 schema 字段被使用一次；行内 `#` 注释 + 末尾“运行时心智模型”段；与 `qwen2.5:32b` ollama 配合产生可见 stage 时序与产物|
|`examples/chat.yaml`|纯 agent 单 stage 最小示例|workflow 完全为 agent_engine 服务的极简形态|

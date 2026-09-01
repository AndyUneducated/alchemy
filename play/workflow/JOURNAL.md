# Journal

> Dates follow actual commit history. Each milestone is a 100–300 word narrative on why it mattered and what it implies, plus a framework change table, mermaid when needed, and new/changed modules, CLI, and examples.

## 2026-04-26 — MVP runner: declarative deterministic + agent stage pipeline

This milestone wraps [play/agent_engine](../agent_engine/) in the thinnest declarable orchestration layer. `Workflow.from_yaml` / `Workflow.run` is ~420 lines, chaining two stage types in YAML `stages:` order: deterministic (plain Python callable) and agent (delegating to `agent_engine.Engine.invoke`). What matters most is not what it supports but what it **explicitly does not** — no retry / timeout / cron / DAG / inline Python in YAML / multi CLI subcommands / persistence. CLI is only `python -m workflow run <yaml>`. That minimalism keeps workflow and agent_engine boundaries clear: workflow "runs declared order", agent_engine "runs a meeting"; they decouple via `config:` unpacked into `Engine.invoke(**config)` — workflow is oblivious to agent_engine internal names and only takes `Result.artifact` into state, the sole LLM coupling point. README documents explicit non-goals so future scope creep is visible in code review.

### Framework changes

|Change|Purpose|
|---|---|
|`Workflow.from_yaml` / `Workflow.run` (~420 lines)|minimal declarable stage pipeline|
|Linear `stages:` with only `deterministic` + `agent`|deliberately no DAG / retry / cron / persistence|
|`config:` unpacked into `Engine.invoke(**config)`|workflow oblivious to agent_engine internal names|
|state only takes `Result.artifact`|minimal LLM integration surface|
|`{{ a.b.c }}` dot-path interpolation only|expression power tightened; complex transforms forced into hooks|
|miss → `KeyError` / missing required → `sys.exit`|fail-fast, no "you probably meant X" hints|
|`module:callable` + top-level `hooks_module` dual resolution|hooks from external module or yaml sibling|
|per stage prints `start` / `done` + `duration_ms`|execution timing visible without extra logging|
|`trace_id` not implemented; W3C `traceparent` field names reserved|future zero-cost distributed tracing adoption|

```mermaid
flowchart LR
    Y[workflow.yaml<br/>+ vars k=v] --> WF[Workflow.from_yaml<br/>+ schema.validate]
    WF --> R[Workflow.run]
    R -->|stage type dispatch| D[deterministic executor<br/>module:callable]
    R --> A[agent executor<br/>Engine.invoke config]
    D --> ST[(state<br/>{{ a.b.c }} interpolate)]
    A -->|Result.artifact| ST
    ST --> R
    R --> LOG[stage start/done<br/>+ duration_ms]
```

### New / changed modules

|Module|Description|
|---|---|
|`runner.py`|`Workflow.from_yaml` / `Workflow.run`, sequential execution + timing output|
|`schema.py`|YAML required-field validation; missing → `sys.exit("Error: ...")`; no migration|
|`state.py` (~50 lines)|`{{ a.b.c }}` interpolation; whole-string single placeholder preserves Python type, inline forces `str()`|
|`executors/deterministic.py`|`fn` string supports `module:callable` and top-level `hooks_module` dual resolution|
|`executors/agent.py`|`config:` unpacked into `Engine.invoke(**config)`; state only takes `Result.artifact`|
|`cli.py` / `__main__.py`|single subcommand `python -m workflow run <yaml> --vars k=v ...`|

### New examples / demos

|Example|Purpose|Demonstrates|
|---|---|---|
|`examples/kitchen_sink.yaml` + `kitchen_sink_hooks.py`|schema field reference + end-to-end 3 deterministic + 1 agent + 1 finalize|each schema field used once; inline `#` comments + trailing runtime mental model; visible stage timing and artifacts with `qwen2.5:32b` ollama|
|`examples/chat.yaml`|minimal pure-agent single stage|workflow serving agent_engine only|

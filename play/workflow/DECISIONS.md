# Decisions

> Recording standard: keep only decisions with ongoing value for architecture evolution, cross-project boundaries, maintainability, and interview Q&A.
> Deletion standard: one-off troubleshooting, implementation details recoverable from code/commits, redundant supersession detail.
> Dates follow git commit history. Milestone progress in [`JOURNAL.md`](JOURNAL.md).

## 1. Runner shape: linear stages × 2 stage types; deliberately no DAG / retry / cron

- **Date**: 2026-04-26

### Context

The workshop needed the thinnest shell to wrap [`play/agent_engine`](../agent_engine/) `Engine.invoke` — chaining agents and deterministic Python into an end-to-end pipeline. The abstraction choice: reuse industrial orchestration (Prefect / Temporal / Argo / Airflow) or write a minimal runner? The former gives retry / UI / durability / DAG free; the latter gives clear boundaries, zero infra dependency, ~350 readable lines.

### Options considered

|Option|Description|Pros|Risks / cost|
|---|---|---|---|
|A. Prefect / Temporal / Argo directly|industrial orchestration|retry / UI / durability / scheduler free|thousands LOC deps + scheduler / UI + learning curve; mismatches workshop pace|
|B. LangChain LCEL|Python chain DSL|expressive|streaming / chain abstractions LLM-centric; poor fit for agent + deterministic mix|
|C. Airflow / Argo DAG|declarative DAG|parallel / conditional / retry|scheduler / metadata DB / UI server; heavy|
|D. Minimal self-written runner (chosen)|linear stages × 2 types, ~350 lines|clear boundaries, zero external deps, readable|no DAG / parallel / retry; complex cases migrate|

### Decision

**D**: `stages:` is a linear list; each stage `type` is only `deterministic` (Python callable) or `agent` (delegates to `Engine.invoke`). **Explicitly not doing** retry / timeout / circuit-breaker / DAG / conditionals / loops / parallelism / cron / scheduling / persistence / resume / inline Python in YAML / workflow stdlib / auto-register decorators / multiple CLI subcommands. When needed: first solve in hook functions (`tenacity`, `signal`, `subprocess`); else migrate Prefect / Temporal (plan §9 estimates ~3–4h migration).

```mermaid
flowchart LR
    Y[workflow.yaml<br/>stages: linear list] --> R[Workflow.run<br/>sequential]
    R -->|type=deterministic| D[executors.deterministic<br/>fn(**args)]
    R -->|type=agent| A[executors.agent<br/>Engine.invoke(**config)]
    D --> ST[(state)]
    A --> ST
    ST --> R
    R --> LOG[stage start/done<br/>+ duration_ms]
```

### Consequences

|Impact|Result|
|---|---|
|Library size|≤ ~350 lines; one morning to read all source|
|External deps|zero infra (no scheduler / UI / DB); one CLI command runs|
|Expressiveness|no DAG / parallel / conditionals; force hooks or multiple yaml|
|Migration path|future retry / UI / durability → whole migrate Prefect / Temporal, not stack in workflow|
|Observability|each stage `start` / `done` + `duration_ms` on stdout; no logging framework|

### Examples

|Scenario|Under this decision|
|---|---|
|Retry on agent stage|wrap in hook with `@tenacity.retry`; don't touch runner|
|A/B branch|two yaml files + outer shell chooses; no conditionals in yaml|
|Hourly cron|GitHub Actions / system cron; runner is one-shot|
|Future DAG needed|migrate whole to Prefect (plan §9 ~3–4h); don't patch workflow|

### Interview Q&A

|Question|Answer points|
|---|---|
|Why not Prefect?|workshop wants ~350 readable lines; Prefect is thousands LOC + scheduler service; mismatches experimental sub-project boundary|
|Won't limits break engineering?|clear boundary → quantifiable migration (~3–4h plan §9); "cheap now + not locked" beats "heavy now + maybe unused"|
|Why not implement trace_id?|no distributed tracing need today; W3C `traceparent` env + JSON field names reserved (plan §9.1) for zero-cost future adoption|

## 2. Agent stage interface: `config:` unpacked into `Engine.invoke`; state only takes `Result.artifact`

- **Date**: 2026-04-26

### Context

workflow exists only to wrap [`play/agent_engine`](../agent_engine/); it has one LLM interface point (`executors/agent.py`, ~19 lines). If workflow schema explicitly lists agent_engine fields (`initial_artifact` / `transcript_path` / …), any agent_engine API change breaks workflow — violating plan §2 "workflow embeds no LLM logic".

### Options considered

|Option|Description|Pros|Risks / cost|
|---|---|---|---|
|A. workflow schema lists agent_engine fields|each kwarg validated in yaml|static known fields; earlier errors|agent_engine changes require workflow bump; workflow half-understands LLM|
|B. transparent pass-through (chosen)|`config:` block `**unpack` into `Engine.invoke(**config)`; workflow oblivious|full decoupling; new agent_engine fields zero downstream cost|invalid fields fail at `Engine.invoke` `TypeError`, not yaml load|

### Decision

**B**: `executors/agent.py` only does `Engine(scenario).invoke(**config)` + writes `Result.artifact` to state; `config:` contents not validated by workflow schema. state only takes `Result.artifact` (`dict[section, content]`), not `transcript` / `success` / `warnings` — coupling surface tightened to **one data shape**; swapping LLM engine (CrewAI / LangGraph) changes only `executors/agent.py` 19 lines.

```mermaid
flowchart LR
    Y[yaml stage:<br/>type: agent<br/>scenario: ...md<br/>config: { ... }] --> AG[executors.agent.run]
    AG -->|Engine.invoke<br/>**config pass-through| ENG[(agent_engine<br/>Engine)]
    ENG -->|Result.artifact<br/>only one field| AG
    AG --> ST[(state[stages.X.output])]
    note["workflow does not parse<br/>config internal names"] -. boundary .- AG
```

### Consequences

|Impact|Result|
|---|---|
|Cross-project evolution|agent_engine API changes zero downstream upgrade|
|Coupling surface|sole LLM point = `executors/agent.py` 19 lines; engine swap surface known|
|Error timing|invalid `config:` at `Engine.invoke` `TypeError`, not yaml load; fail-fast still holds, later in pipeline|
|State model|stages pass rendered artifact strings only; transcript etc. require explicit file reads|

### Examples

|Scenario|Under this decision|
|---|---|
|agent_engine adds `transcript_path`|yaml works immediately (`config: { transcript_path: ... }`); no workflow upgrade|
|Switch to CrewAI|change `executors/agent.py` one file; schema / state / other stages untouched|
|Downstream wants transcript|not via state; use `transcript_path` + `Path.read_text()` in hook|

### Interview Q&A

|Question|Answer points|
|---|---|
|Won't unvalidated config error?|fail-fast at `Engine.invoke` `TypeError`; runner doesn't re-wrap|
|Conflicts with "explicit over implicit"?|workflow's explicit boundary is "I don't interpret agent_engine internals"; constraints live in agent_engine schema|
|Why not state `transcript` / `warnings`?|tight coupling → controlled engine swap cost; use file paths for cross-stage data contracts|

## 3. Config layer: schema fail-fast + minimal `{{ a.b.c }}` templates + kitchen_sink.yaml single SoT

- **Date**: 2026-04-26

### Context

Each config decision trades author error feedback speed (fail-fast vs friendly hints) against scope creep entry points (stronger template DSL → yaml grows business logic). Also: where is field documentation SoT (README vs yaml comments)? Dual-write is the common rot source.

### Options considered

|Option|Description|Pros|Risks / cost|
|---|---|---|---|
|A. Jinja2 + schema migration + friendly errors|industrial config layer|expressive, friendly|`{{ x \| upper }}` filters leak transforms into yaml; migration prepaid unclear benefit|
|B. minimal fail-fast set (chosen)|`{{ a.b.c }}` paths; missing required → `sys.exit`; dual fn resolution; kitchen_sink SoT|errors immediate; transforms forced to hooks; yaml stays simple|unfriendly; no migration|

### Decision

**B**, four sub-decisions:

|Sub-decision|Implementation|Rationale|
|---|---|---|
|missing required|`sys.exit("Error: ...")` direct exit; no "you probably meant X"|solo workshop has no legacy users; guess hints prepaid unclear benefit|
|template power|only `{{ a.b.c }}` path access; whole-string single placeholder preserves Python type; inline forces `str()`; miss → `KeyError`|filters / expressions / conditionals hard to close once opened; transforms belong in hooks (kitchen_sink `to_yaml`)|
|fn string resolution|`module:callable` (colon = full path) or top-level `hooks_module` default namespace|hooks from external pkg or yaml sibling; explicit import; no auto-register decorators|
|field SoT|`examples/kitchen_sink.yaml` — each field once + inline `#` + trailing runtime mental model; README overview only|dual-write rots; runnable example faster for new authors|

```mermaid
flowchart LR
    Y[workflow.yaml] --> SC[schema.validate]
    SC -->|missing required| EX1[sys.exit Error]
    SC -->|OK| ST[state.interpolate<br/>{{ a.b.c }}]
    ST -->|path missing| EX2[KeyError thrown]
    ST -->|OK| EXEC[stage executor]
    EXEC -->|fn string| FN{module:callable?}
    FN -->|has :| IMP1[import module<br/>get callable]
    FN -->|no :| IMP2[hooks_module<br/>default ns]
    EXEC -->|hook raise| EX3[original traceback]
```

### Consequences

|Impact|Result|
|---|---|
|Error feedback|first run hits all schema / template / hook errors; no dry-run / validate subcommand|
|yaml expressiveness|templates stay simple; complex logic forced to hooks (Python > YAML)|
|Doc maintenance|README and yaml don't duplicate field docs; evolve kitchen_sink.yaml only|
|Unfriendliness|no guess hints / no migration / no deprecation path — acceptable solo; multi-person projects may add later|

### Examples

|Scenario|Under this decision|
|---|---|
|Forgot required field|`sys.exit("Error: stage 'discuss' missing required field 'scenario'")` with file + field|
|Template path missing|`KeyError: 'stages.foo.output'` with full path; traceback is diagnosis|
|Sort a list|no template filter; hook stage `fn: sort_by` returns sorted list|
|New field|change schema.py + kitchen_sink.yaml; README unchanged (avoid dual-write)|

### Interview Q&A

|Question|Answer points|
|---|---|
|Why not Jinja2?|filters / expressions once opened → yaml business logic → unreadable; hooks are the transform layer|
|Isn't schema migration industry default?|yes, but workshop has no legacy users; migration is prepaid unclear benefit; YAGNI|
|Won't kitchen_sink SoT go stale?|schema changes must run example; example passing calibrates docs; README dual-write lacks that loop|

## Non-goals (still in effect)

|Item|Note|
|---|---|
|Industrial orchestration (retry / DAG / UI / durability)|Prefect / Temporal domain; workflow boundary → migration ~3–4h (plan §9)|
|Data transforms / business branches in yaml|beyond `{{ a.b.c }}` → hooks; YAML embedding Python is anti-pattern|
|workflow interpreting agent_engine internal names|`config:` pass-through is workflow's reason; interpretation = tight coupling|
|Friendly errors / schema migration|fail-fast optimal for solo workshop; add when multi-person|
|workflow stdlib / multiple CLI subcommands|YAGNI with one consumer; scope creep visible in review|

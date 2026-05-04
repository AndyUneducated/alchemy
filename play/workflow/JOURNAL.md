# Journal

按里程碑记录每日进展。每条以 `## YYYY-MM-DD — 里程碑标题` 开头；同一自然日 ≤2 个里程碑。**功能** / **技术** 两段必填；**取舍** 仅在当日产出影响后续的取舍时记一笔，指向 [`DECISIONS.md`](DECISIONS.md) 完整条目而不在此重复。

## 2026-04-26 — MVP runner：声明式 deterministic + agent stage pipeline

### 功能

- `Workflow.from_yaml` / `Workflow.run`：~420 行的 workflow runner，按 YAML 顺序串接 deterministic Python-callable stage 与 agent stage（委托给 `agent_engine.Engine.invoke`）
- CLI：`python -m workflow run <yaml> --vars k=v ...` 单子命令，不做 validate / list / inspect（避免 scope creep）
- 示例：`examples/kitchen_sink.yaml` + `kitchen_sink_hooks.py`——schema 字段速查（每字段用一次 + 行内 `#` 注释 + 末尾「运行时心智模型」段）；`examples/chat.yaml` 是「纯 agent 单 stage」最小示例
- 端到端跑通：3 deterministic + 1 agent + 1 finalize stage，与 qwen2.5:32b ollama 后端配合，跑出可见 stage 时序与产物

### 技术

- `state.py`：~50 行的 path-access 插值实现——只支持 `{{ a.b.c }}` 点路径访问；整字符串单占位保 Python 类型，inline 占位强制 `str()`；miss 直接 `KeyError`（plan §12 fail-fast）
- `schema.validate`：必填字段缺失即 `sys.exit("Error: ...")`，**不**给「你大概想用 X」提示；不做 schema migration、没有「老用户引导」
- `executors/deterministic.py`：`fn` 字符串支持 `module:callable` 与依赖顶层 `hooks_module` 两种解析路径
- `executors/agent.py`：`config:` 块原样 unpack 进 `Engine.invoke(**config)` （plan §4.3），workflow 对 agent_engine 内部命名 oblivious；state 只收 `Result.artifact`——这是 workflow 与 LLM 的唯一耦合点
- 每 stage 打印 `start` / `done` + `duration_ms`，方便快速观察执行时序
- 同 commit 在 README 把 plan §9 的「显式不做项」（无 retry / timeout / cron / DAG / inline Python in YAML / stdlib / 多 CLI 子命令）刻在文档里——未来 scope-creep 在 code review 时显眼

### 取舍

- 线性 `stages` 仅 `deterministic` + `agent` 两类；不做 DAG / retry / cron / 持久化 → DECISIONS §1
- `state` 只支持 `{{ a.b.c }}` 路径访问，不接 filter / 表达式（数据转换写进 hook 函数，参考 kitchen_sink 的 `to_yaml` stage） → DECISIONS §2
- `trace_id` 不实现；保留 W3C `traceparent` env 变量名 + JSON 字段名以待未来零成本接入（plan §9.1）

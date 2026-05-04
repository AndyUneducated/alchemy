# Decisions

ADR（Architecture Decision Record）归档。每条以 `## n. 标题` 开头，紧接 `- **Status**` + `- **Date**` 元信息。**新决策追加到末尾，被取代的条目改 Status；不删旧条目**。日常进度（按里程碑）见 [`JOURNAL.md`](JOURNAL.md)。

## 1. Runner 边界

- **Status**: accepted
- **Date**: 2026-04-26

线性 `stages`，仅 `deterministic`（`fn` + 插值 `args`）与 `agent`（`scenario` + 插值 `config` → `Engine.invoke(**config)`，state 只收 `Result.artifact`）。不做法、无 retry/cron/DAG；与 plan §2 / §4.3、README「显式不做」一致。

## 2. 配置 / 插值 / 示例

- **Status**: accepted
- **Date**: 2026-04-26

`schema` 必填缺失即 `sys.exit`；`state` 只支持 `{{ a.b.c }}`，整段单占位保 Python 类型；`fn` 为 `module:callable` 或依赖顶层 `hooks_module`；字段示例以 `examples/kitchen_sink.yaml` 为 SoT，避免与 README 双写。

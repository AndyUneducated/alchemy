# Changelog

`CHANGELOG.md` 同时承担变更日志与 ADR 归档。每条记录使用 `## n. 变更标题`，下一行写 `- 日期：...`，再写正文。后续每个自然日建议最多追加 1～2 条 tech decision。

## 1. Runner 边界
- 日期：2026-04-26
线性 `stages`，仅 `deterministic`（`fn` + 插值 `args`）与 `agent`（`scenario` + 插值 `config` → `Engine.invoke(**config)`，state 只收 `Result.artifact`）。不做法、无 retry/cron/DAG；与 plan §2 / §4.3、README「显式不做」一致。

## 2. 配置 / 插值 / 示例
- 日期：2026-04-26
`schema` 必填缺失即 `sys.exit`；`state` 只支持 `{{ a.b.c }}`，整段单占位保 Python 类型；`fn` 为 `module:callable` 或依赖顶层 `hooks_module`；字段示例以 `examples/kitchen_sink.yaml` 为 SoT，避免与 README 双写。

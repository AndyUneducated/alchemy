# tools — agent_engine 的 reasoning tool 注册表

## 分类原则（plan §6）

agent_engine 的 `tools/` 包**只**放推理性 tool：

- **Reasoning tool**：输入完全由 LLM 推理决定 + 输出供 LLM 继续推理 → 留这里
- **External I/O tool**：输入由上游确定 + 输出有副作用或下游不再需要 LLM 决策 → 归 workflow stage（不进 agent_engine）

**纪律靠文档维持，不做 runtime allowlist 校验**（plan §12 fail-fast 哲学）。违反纪律的代码会在调用时自然报错。

## 现有 tool

| 文件 | 性质 | 说明 |
|---|---|---|
| `retrieve_docs.py` | Reasoning | LLM 动态决策 query/mode/rerank 的语义搜索；subprocess 调用 `play/rag/query.py --json` |

`artifact.py`（位于 agent_engine 根目录）的 6 个 artifact 工具（`read/write/append/propose_vote/cast_vote/finalize`）是**进程内副作用**（写 ArtifactStore 内存对象），不属于 external I/O，留在 agent_engine。

## 加新 tool 的模式

1. 新建 `tools/<name>.py`，导出：
   - `TOOL_DEF: dict` — OpenAI function-calling schema
   - `def handler(...) -> str` — 实现，返回 JSON 字符串（错误用 `{"error": "..."}`）
2. 在 `tools/__init__.py` 的 `TOOL_DEFINITIONS` 与 `TOOL_HANDLERS` 各加一行
3. **不**用装饰器自动注册——隐式 import 副作用让 scenario YAML 校验调试很痛

## 共享 helper

- `_envelope.py`: `is_error / warn_if_error` —— `{"error": ...}` 信封约定
- `_subprocess.py`: `run_json_subprocess` —— 子进程 + JSON envelope 解包

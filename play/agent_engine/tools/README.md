# tools — agent_engine reasoning tool registry

## Classification principles

The agent_engine `tools/` package holds **reasoning tools only**:

|Category|Location|Criterion|
|---|---|---|
|Reasoning tool|Stay in `agent_engine/tools/`|Input decided by LLM reasoning; output fed back to the LLM for further reasoning|
|External I/O tool|Implement as a `workflow` deterministic stage|Input fixed upstream, or tool has external side effects, or output no longer needs LLM decisions|

**Discipline is maintained by documentation, not runtime allowlist validation.** Code that violates it fails naturally at call time.

## Existing tools

|File|Kind|Description|
|---|---|---|
|`retrieve_docs.py`|Reasoning|Semantic search with LLM-chosen query/mode/rerank; subprocess call to `play/rag/query.py --json`|

The six artifact tools in `artifact.py` (at agent_engine root: `read/write/append/propose_vote/cast_vote/finalize`) are **in-process side effects** (writing the in-memory ArtifactStore), not external I/O, so they stay in agent_engine.

## Pattern for adding a new tool

1. Create `tools/<name>.py` exporting:
   - `TOOL_DEF: dict` — OpenAI function-calling schema
   - `def handler(...) -> str` — implementation returning a JSON string (errors use `{"error": "..."}`)
2. Add one line each to `TOOL_DEFINITIONS` and `TOOL_HANDLERS` in `tools/__init__.py`
3. **Do not** use decorator auto-registration — implicit import side effects make scenario YAML validation painful to debug

## Shared helpers

- `_envelope.py`: `is_error / warn_if_error` — `{"error": ...}` envelope convention
- `_subprocess.py`: `run_json_subprocess` — subprocess + JSON envelope unpack

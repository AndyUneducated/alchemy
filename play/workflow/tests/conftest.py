"""workflow pytest suite conftest.

No ollama / LLM dependency — tests pure functions + fail-fast boundaries
(state.interpolate / schema.validate / runner._resolve_vars / deterministic._resolve_fn).
Only sys.path setup, aligned with play/agent_engine/tests/conftest.py: so
`python -m pytest play/workflow/tests/` from repo root can still `import workflow`.
"""

"""agent_engine first pytest suite conftest (from DECISIONS §13).

No ollama / VDB dependency — agent_engine kernel tests are pure functions / static
expansion, no LLM runs. Only sys.path setup: so `python -m pytest play/agent_engine/tests/`
from repo root can still `import agent_engine` (compatible with both that and
`cd play && python -m pytest ...`).
"""

"""workflow pytest 测试集 conftest.

无 ollama / LLM 依赖——这里只测纯函数 + fail-fast 边界（state.interpolate /
schema.validate / runner._resolve_vars / deterministic._resolve_fn）。仅做
sys.path 预置，与 play/agent_engine/tests/conftest.py 对齐：让 `python -m
pytest play/workflow/tests/` 在项目根执行时 `import workflow` 仍能解析。
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PLAY_DIR = REPO_ROOT / "play"
if str(PLAY_DIR) not in sys.path:
    sys.path.insert(0, str(PLAY_DIR))

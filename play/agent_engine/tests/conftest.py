"""agent_engine 首个 pytest 测试集（DECISIONS §13 起）conftest.

无 ollama / VDB 依赖——agent_engine 内核测试都是纯函数 / 静态展开，不跑 LLM。
仅做 sys.path 预置：让 `python -m pytest play/agent_engine/tests/` 在项目根
执行时 `import agent_engine` 仍能解析（与 `cd play && python -m pytest ...`
两种调用方式都兼容）。
"""
from __future__ import annotations

import sys
from pathlib import Path

# play/agent_engine/tests/conftest.py → ai_workshops/
REPO_ROOT = Path(__file__).resolve().parents[3]
PLAY_DIR = REPO_ROOT / "play"
if str(PLAY_DIR) not in sys.path:
    sys.path.insert(0, str(PLAY_DIR))

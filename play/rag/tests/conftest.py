"""rag 测试公共配置。

`play/rag/` 的模块用裸 import（`from chunker import ...` / `from bm25 import ...`），
不是 package。这里把 `play/rag/` 自身加进 sys.path，让 `tests/` 下的测试可以直接
`from chunker import split_text` 不必做 path 体操。

测试集刻意保持轻量：不依赖 chromadb / ollama / VDB / HF cache —— 所有非纯函数
路径都用 monkeypatch 替掉。CLI 契约测试只静态读 `query.py` 文本。
"""
from __future__ import annotations

import sys
from pathlib import Path

_RAG_DIR = Path(__file__).resolve().parent.parent  # play/rag
if str(_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(_RAG_DIR))

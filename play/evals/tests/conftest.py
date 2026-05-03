"""Phase 3 起 live LM 测试落地，需要在 conftest 做两层 probe：

  ① 服务可达：GET /api/tags 通 → ollama 守护进程在跑
  ② 模型已拉：返回的 model 列表里有 EVALS_TEST_OLLAMA_MODEL（默认 qwen2.5:32b）

任一不满足 → 整文件 skip + 友好提示（告诉用户怎么 `ollama pull` 或换 env）。
auto-probe 的好处是 CI 干净（默认无 ollama 自动 skip）+ 本地 dev 起了 ollama 自然就跑。

测试默认模型选 qwen2.5:32b 的理由：本地已有避免额外 pull / judge 质量更稳让 `>=3.5` 阈值不 flake / 完整 live suite 实测 ~24s（M-series Mac），可接受；EVALS_TEST_OLLAMA_MODEL 可降档到 qwen2.5:3b 提速（CI 友好）或升档到 72b。
"""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request

import pytest

OLLAMA_BASE = os.environ.get("EVALS_OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_TEST_MODEL = "qwen2.5:32b"


def _ollama_models() -> set[str] | None:
    """返回本地已拉的 model tag 集合；服务不可达返回 None."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=1.0) as r:
            data = json.loads(r.read())
        return {m["name"] for m in data.get("models", [])}
    except (urllib.error.URLError, socket.timeout, ConnectionRefusedError, OSError):
        return None


_MODELS = _ollama_models()
_TEST_MODEL = os.environ.get("EVALS_TEST_OLLAMA_MODEL", DEFAULT_TEST_MODEL)

if _MODELS is None:
    _SKIP_REASON = f"Ollama not reachable at {OLLAMA_BASE}; live tests skipped"
elif _TEST_MODEL not in _MODELS:
    _SKIP_REASON = (
        f"Ollama reachable but model {_TEST_MODEL!r} not pulled. "
        f"Run `ollama pull {_TEST_MODEL}` or set EVALS_TEST_OLLAMA_MODEL to "
        f"one of: {sorted(_MODELS)}"
    )
else:
    _SKIP_REASON = ""


ollama_required = pytest.mark.skipif(bool(_SKIP_REASON), reason=_SKIP_REASON)


@pytest.fixture(scope="session")
def ollama_model() -> str:
    """test 用 model tag；EVALS_TEST_OLLAMA_MODEL env 可 override."""
    return _TEST_MODEL


@pytest.fixture(scope="session")
def ollama_base_url() -> str:
    return OLLAMA_BASE

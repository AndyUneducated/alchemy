"""Phase 3 起 live LM 测试落地，需要在 conftest 做两层 probe：

  ① 服务可达：GET /api/tags 通 → ollama 守护进程在跑
  ② 模型已拉：返回的 model 列表里有 EVALS_TEST_OLLAMA_MODEL（默认 qwen2.5:32b）

任一不满足 → 整文件 skip + 友好提示（告诉用户怎么 `ollama pull` 或换 env）。
auto-probe 的好处是 CI 干净（默认无 ollama 自动 skip）+ 本地 dev 起了 ollama 自然就跑。

测试默认模型选 qwen2.5:32b 的理由：本地已有避免额外 pull / judge 质量更稳让 `>=3.5` 阈值不 flake / 完整 live suite 实测 ~24s（M-series Mac），可接受；EVALS_TEST_OLLAMA_MODEL 可降档到 qwen2.5:3b 提速（CI 友好）或升档到 72b。

phase 4 起加 VDB probe（rag_retrieval / rag_qa live e2e 用）：
  ③ vdb 目录存在：`play/rag/vdb/<name>/{chroma.sqlite3, bm25.pkl}` 都齐
不齐 → 该 vdb 相关 live 测试 skip + 提示用户 `python ingest.py --docs ... --output vdb/<name>`.
"""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from pathlib import Path

import pytest

OLLAMA_BASE = os.environ.get("EVALS_OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_TEST_MODEL = "qwen2.5:32b"

# play/evals/tests/conftest.py → ai_workshops/
REPO_ROOT = Path(__file__).resolve().parents[3]
RAG_VDB_DIR = REPO_ROOT / "play" / "rag" / "vdb"


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


def _vdb_ok(name: str) -> tuple[Path | None, str]:
    """检查 play/rag/vdb/<name>/ 是否齐备；返回 (path or None, skip_reason)."""
    vdb = RAG_VDB_DIR / name
    if not vdb.exists():
        return None, (
            f"VDB {vdb} missing; build it via "
            f"`cd play/rag && python ingest.py --docs docs/{name} --output vdb/{name}`"
        )
    if not (vdb / "chroma.sqlite3").exists() or not (vdb / "bm25.pkl").exists():
        return None, (
            f"VDB {vdb} is incomplete (missing chroma.sqlite3 or bm25.pkl); "
            f"re-run ingest to rebuild"
        )
    return vdb, ""


_PANEL_VDB, _PANEL_SKIP = _vdb_ok("panel")
_TEST_VDB, _TEST_VDB_SKIP = _vdb_ok("test_vdb")

panel_vdb_required = pytest.mark.skipif(
    bool(_PANEL_SKIP), reason=_PANEL_SKIP or "panel vdb required"
)
# 注意：变量名前缀避开 'test_'，否则 pytest collection 会误识为 test 函数
sample_vdb_required = pytest.mark.skipif(
    bool(_TEST_VDB_SKIP), reason=_TEST_VDB_SKIP or "sample (test_vdb) required"
)


@pytest.fixture(scope="session")
def ollama_model() -> str:
    """test 用 model tag；EVALS_TEST_OLLAMA_MODEL env 可 override."""
    return _TEST_MODEL


@pytest.fixture(scope="session")
def ollama_base_url() -> str:
    return OLLAMA_BASE


@pytest.fixture(scope="session")
def panel_vdb_path() -> Path:
    """panel VDB 路径（rag_retrieval / rag_qa e2e live 用）；缺则 skip."""
    if _PANEL_VDB is None:
        pytest.skip(_PANEL_SKIP)
    return _PANEL_VDB


@pytest.fixture(scope="session")
def sample_vdb_path() -> Path:
    """sample (test_vdb) 路径（5 行 facts，subprocess wrapper smoke 用）；缺则 skip.

    fixture 名 'sample_vdb_path' 而非 'test_vdb_path' 避开 pytest collection 对 'test_' 前缀的误识.
    """
    if _TEST_VDB is None:
        pytest.skip(_TEST_VDB_SKIP)
    return _TEST_VDB

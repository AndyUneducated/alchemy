"""Starting from Phase 3, the live LM test needs to be implemented in conftest with two layers of probes:

  ① The service is reachable: GET /api/tags → ollama daemon is running
  ② The model has been pulled: EVALS_TEST_OLLAMA_MODEL is included in the returned model list (default qwen3.6:27b)

If any of them are not satisfied → skip the entire file + friendly prompt (tell the user how to `ollama pull` or change env).
The advantage of auto-probe is that the CI is clean (no ollama is automatically skipped by default) + ollama will run naturally when the local dev is started.

The reason for choosing qwen3.6:27b as the default model for testing: the local version already exists to avoid extra pull/judge and the quality is more stable so that the `>=3.5` threshold is not flake; EVALS_TEST_OLLAMA_MODEL can be downshifted to qwen3.5:9b to speed up (CI friendly) or upgraded to a larger model.

Add VDB probe from phase 4 (for rag_retrieval / rag_qa live e2e):
  ③ The vdb directory exists: `play/rag/vdb/<name>/{chroma.sqlite3, bm25.pkl}`
Uneven → The vdb-related live tests skip + prompt the user `python ingest.py --docs ... --output vdb/<name>`."""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from pathlib import Path

import pytest

OLLAMA_BASE = os.environ.get("EVALS_OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_TEST_MODEL = "qwen3.6:27b"

REPO_ROOT = Path(__file__).resolve().parents[3]
RAG_VDB_DIR = REPO_ROOT / "play" / "rag" / "vdb"


def _ollama_models() -> set[str] | None:
    """Returns the locally pulled model tag collection; returns None if the service is unreachable."""
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
    """Check if play/rag/vdb/<name>/ is complete; return (path or None, skip_reason)."""
    vdb = RAG_VDB_DIR / name
    if not vdb.exists():
        return None, (
            f"VDB {vdb} missing; build it via "
            f"`cd play/rag && python ingest.py --docs docs/{name} --output vdb/{name}`"
        )
    if not (vdb / "chroma.sqlite3").exists() or not (vdb / "bm25.pkl").exists():
        return None, (
            f"VDB {vdb} is incomplete (missing chroma.sqlite3 or bm25.pkl); "
            f"re-run: cd play/rag && python ingest.py --docs docs/{name} --output vdb/{name}"
        )
    return vdb, ""


_PANEL_VDB, _PANEL_SKIP = _vdb_ok("panel")
_TEST_VDB, _TEST_VDB_SKIP = _vdb_ok("test_vdb")

panel_vdb_required = pytest.mark.skipif(
    bool(_PANEL_SKIP), reason=_PANEL_SKIP or "panel vdb required"
)
# Note: Avoid the variable name prefix 'test_', otherwise pytest collection will misunderstand it as the test function
sample_vdb_required = pytest.mark.skipif(
    bool(_TEST_VDB_SKIP), reason=_TEST_VDB_SKIP or "sample (test_vdb) required"
)


# ---------- agent_engine probe（phase 5）-----------------------------------

def _agent_engine_ok() -> tuple[Path | None, str]:
    """Check play/agent_engine/ package + at least one scenario is visible; return (play_dir, skip_reason)."""
    play_dir = REPO_ROOT / "play"
    pkg = play_dir / "agent_engine"
    if not (pkg / "__init__.py").exists():
        return None, f"agent_engine package missing at {pkg}"
    brainstorm = pkg / "scenarios" / "brainstorm.md"
    if not brainstorm.exists():
        return None, f"agent_engine/scenarios/brainstorm.md missing at {brainstorm}"
    return play_dir, ""


_AE_PLAY, _AE_SKIP = _agent_engine_ok()

agent_engine_required = pytest.mark.skipif(
    bool(_AE_SKIP), reason=_AE_SKIP or "agent_engine required"
)


@pytest.fixture(scope="session")
def ollama_model() -> str:
    """test uses model tag; EVALS_TEST_OLLAMA_MODEL env can be override."""
    return _TEST_MODEL


@pytest.fixture(scope="session")
def ollama_base_url() -> str:
    return OLLAMA_BASE


@pytest.fixture(scope="session")
def panel_vdb_path() -> Path:
    """panel VDB path (used by rag_retrieval / rag_qa e2e live); skip if missing."""
    if _PANEL_VDB is None:
        pytest.skip(_PANEL_SKIP)
    return _PANEL_VDB


@pytest.fixture(scope="session")
def sample_vdb_path() -> Path:
    """sample (test_vdb) path (5 lines of facts, used by subprocess wrapper smoke); if missing, skip.

    The fixture name is 'sample_vdb_path' instead of 'test_vdb_path' to avoid pytest collection's misunderstanding of the 'test_' prefix."""
    if _TEST_VDB is None:
        pytest.skip(_TEST_VDB_SKIP)
    return _TEST_VDB

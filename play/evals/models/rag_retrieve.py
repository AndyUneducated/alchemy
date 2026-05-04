"""RAG retrieval 闭包工厂：subprocess 调 `play/rag/query.py --json`，零 Python import.

为什么 subprocess 而非直接 `from play.rag.query import search`：
  - 遵循 monorepo 解耦原则（详见 CHANGELOG §4 / workshops.mdc）：
    `play/` 下的 sub-projects 不互相 Python import，跨项目通信走 CLI + JSON envelope.
  - `play/rag` 自带的依赖（chromadb / ollama / fastparquet 等）不污染 `evals` 进程
  - 同一组接口对 future remote retriever（HTTP service）平滑迁移：换 transport
    实现，不动 task 层

代价 & 缓解：
  - 冷启动 ~2-4s（python + chromadb client + ollama embed 加载）。phase 4 8 条 query
    依次跑约 16-32s——可接受。批量优化（一次 subprocess 多 query）留 phase 5+
  - 错误传播：subprocess.CalledProcessError 时把 stderr 透出去（OllamaConnError /
    VDB 不存在等都能在 evals 这一侧第一时间看到）

数据契约：
  - retrieve_fn(query: str) -> tuple[list[source_id], list[content]]
    其中 `source_id` = play/rag/ingest 写进 chunk metadata['source'] 的 basename，
    与 `data/rag_retrieval/gold.jsonl::gold_doc_ids` 字段语义对齐
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Callable, Literal

# play/evals/models/rag_retrieve.py → ai_workshops/
REPO_ROOT = Path(__file__).resolve().parents[3]
RAG_DIR = REPO_ROOT / "play" / "rag"
RAG_QUERY_SCRIPT = RAG_DIR / "query.py"

SearchMode = Literal["dense", "bm25", "hybrid"]

RetrieveFn = Callable[[str], tuple[list[str], list[str]]]


def make_retrieve_fn(
    vdb_dir: str | Path,
    *,
    top_k: int = 5,
    mode: SearchMode = "hybrid",
    rerank: bool = False,
    timeout: float = 60.0,
) -> RetrieveFn:
    """返回 `(query: str) -> (ids, contents)` 闭包.

    每次调用 fork 一个 subprocess：
      `python play/rag/query.py --vdb <vdb_dir> --query <q> --top-k K --mode hybrid --json [--rerank]`

    解析 stdout 上的 JSON envelope（schema 见 play/rag/query.py::main）：
      `{"query": ..., "data": [{"content", "score", "source", "metadata"}], "meta": {...}}`

    去 `data[*].source`（chunk 来源文件名）作 retrieval 单元；
    多 chunk 同源时去重保留首个出现位置（rank 越靠前越优先）.
    """
    vdb_path = Path(vdb_dir).resolve()

    def _retrieve(query: str) -> tuple[list[str], list[str]]:
        cmd = [
            sys.executable, str(RAG_QUERY_SCRIPT),
            "--vdb", str(vdb_path),
            "--query", query,
            "--top-k", str(top_k),
            "--mode", mode,
            "--json",
        ]
        if rerank:
            cmd.append("--rerank")

        proc = subprocess.run(
            cmd,
            cwd=str(RAG_DIR),  # play/rag/query.py 用相对 import config / bm25 等
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"play/rag/query.py exited with {proc.returncode}; "
                f"stderr={proc.stderr.strip()!r}"
            )
        envelope = json.loads(proc.stdout)
        hits = envelope.get("data", [])

        # 同源 chunk 去重，rank 优先（保留首位）
        seen: set[str] = set()
        ids: list[str] = []
        contents: list[str] = []
        for hit in hits:
            src = hit.get("source", "")
            if not src or src in seen:
                continue
            seen.add(src)
            ids.append(src)
            contents.append(hit.get("content", ""))
        return ids, contents

    return _retrieve

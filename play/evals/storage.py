"""存储层：纯 JSONL 三件套.

  runs/<run_id>/result.json    — EvalResult 聚合快照（去掉 per_sample）
  runs/<run_id>/samples.jsonl  — per-sample 行式（SampleResult asdict）
  runs/index.jsonl             — 所有 run 的扁平索引（append-only，替代 SQLite）

刻意 YAGNI：Phase 1 不引 SQLite / 任何 DB。index.jsonl 的行 schema 和未来
可选 SQLite 表同构——`CREATE TABLE runs AS SELECT * FROM read_json('runs/index.jsonl')`
一行迁移。进阶叙事：这是 append-only event log，SQLite 是从 log 重建的 read model
（event-sourcing），不是简陋的过渡方案而是正确的日志设计。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .api import EvalResult, SampleResult

# 默认 runs 目录。Runner / CLI 可 override。
DEFAULT_RUNS_DIR = Path(__file__).resolve().parent / "runs"


def _sample_row(s: SampleResult) -> dict:
    return asdict(s)


def _index_row(r: EvalResult) -> dict:
    """index.jsonl 每行的 schema，和未来 SQLite runs 表字段一一对应."""
    return {
        "run_id": r.run_id,
        "task": r.task,
        "model": r.model,
        "mode": r.mode,
        "created_at": r.created_at,
        "n": r.n,
        "elapsed_ms": r.elapsed_ms,
        "num_fewshot": r.num_fewshot,
        "aggregated": dict(r.aggregated),
    }


def save(result: EvalResult, runs_dir: Path = DEFAULT_RUNS_DIR) -> Path:
    """写三文件：result.json + samples.jsonl + 追加一行到 index.jsonl.

    返回单 run 目录路径（用于 show / 事后 drill-down）.

    并发：append 一行 index.jsonl < 4KB，POSIX 保证原子性，不撕行；
    若未来真有并发加 filelock，依然不到 SQLite 的复杂度。
    """
    run_dir = runs_dir / result.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) result.json —— 本次 run 的聚合快照（不含 per_sample，避免重复）
    result_row = _index_row(result)
    (run_dir / "result.json").write_text(
        json.dumps(result_row, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 2) samples.jsonl —— per-sample 行式
    with (run_dir / "samples.jsonl").open("w", encoding="utf-8") as f:
        for s in result.per_sample:
            f.write(json.dumps(_sample_row(s), ensure_ascii=False) + "\n")

    # 3) index.jsonl —— append-only 扁平索引（source of truth for cross-run queries）
    runs_dir.mkdir(parents=True, exist_ok=True)
    with (runs_dir / "index.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(result_row, ensure_ascii=False) + "\n")

    return run_dir


def read_index(runs_dir: Path = DEFAULT_RUNS_DIR) -> list[dict]:
    """读全部 index.jsonl 行。若文件不存在返回空列表。"""
    idx = runs_dir / "index.jsonl"
    if not idx.exists():
        return []
    rows: list[dict] = []
    with idx.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_run(run_id: str, runs_dir: Path = DEFAULT_RUNS_DIR) -> tuple[dict, list[dict]]:
    """读单 run 的 result.json + samples.jsonl. 给 CLI show 用."""
    run_dir = runs_dir / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"run_id {run_id!r} not found at {run_dir}")
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    samples: list[dict] = []
    with (run_dir / "samples.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return result, samples

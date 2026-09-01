"""Storage layer: Pure JSONL three-piece set.

  runs/<run_id>/result.json — EvalResult aggregate snapshot (remove per_sample)
  runs/<run_id>/samples.jsonl — per-sample row format (SampleResult asdict)
  runs/index.jsonl — Flattened index for all runs (append-only, replaces SQLite)

Intentionally YAGNI:Phase 1 does not reference SQLite/any DB. index.jsonl row schema and next
Optional SQLite table isomorphism - `CREATE TABLE runs AS SELECT * FROM read_json('runs/index.jsonl')`
One line migration. Advanced narrative: This is an append-only event log, and SQLite is a read model reconstructed from the log.
(event-sourcing), not a simple transition plan but a correct log design."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .api import EvalResult, SampleResult

# Default runs directory. Runner/CLI can be overridden.
DEFAULT_RUNS_DIR = Path(__file__).resolve().parent / "runs"


def _sample_row(s: SampleResult) -> dict:
    return asdict(s)


def _index_row(r: EvalResult) -> dict:
    """The schema of each row in index.jsonl corresponds one-to-one with the fields of the future SQLite runs table."""
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
    """Write three files: result.json + samples.jsonl + append a line to index.jsonl.

    Returns the single run directory path (used for show / post drill-down).

    Concurrency: append one line index.jsonl < 4KB, POSIX guarantees atomicity and does not tear lines;
    If filelock is added concurrently in the future, it will still not be as complex as SQLite.

    Strict JSON backend (phase 8 §8.R4 cited): All write disk paths use `allow_nan=False`,
    NaN / Inf `json.dumps` is used internally in Python. By default `allow_nan=True` will write `NaN` /
    `Infinity` literal - this is not valid JSON, any non-Python parser (jq/browser/db
    / Dashboard) will be rejected. Fail-loud reports `ValueError` when writing, which is better than silently writing poisonous files downstream.
    It's much better - homologous phase 4 path C "Cross-process and cross-run JSON transfer" contract."""
    run_dir = runs_dir / result.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) result.json - the aggregate snapshot of this run (excluding per_sample to avoid duplication)
    result_row = _index_row(result)
    (run_dir / "result.json").write_text(
        json.dumps(result_row, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    # 2) samples.jsonl - per-sample line format
    with (run_dir / "samples.jsonl").open("w", encoding="utf-8") as f:
        for s in result.per_sample:
            f.write(json.dumps(_sample_row(s), ensure_ascii=False, allow_nan=False) + "\n")

    # 3) index.jsonl - append-only flat index (source of truth for cross-run queries)
    runs_dir.mkdir(parents=True, exist_ok=True)
    with (runs_dir / "index.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(result_row, ensure_ascii=False, allow_nan=False) + "\n")

    return run_dir


def read_index(runs_dir: Path = DEFAULT_RUNS_DIR) -> list[dict]:
    """Read all index.jsonl lines. If the file does not exist, an empty list is returned."""
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
    """Read the result.json + samples.jsonl of single run. Use it for CLI show."""
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

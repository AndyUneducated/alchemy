"""``load_csv``: read requirements CSV → list[dict] with minimal validation.

Per plan §4.1 schema:
- 必填: req_id / title / assignee
- 二选一必填: description / prd_doc_path (.md only)
- 可选: priority (P0~P3), sprint_start, sprint_end (ISO dates)

Per plan §12: fail fast on missing required columns; raise ``ValueError`` with
the offending row index. No friendly hints, no migration helpers.
"""

from __future__ import annotations

import csv
from pathlib import Path

REQUIRED_COLS = {"req_id", "title", "assignee"}


def load_csv(csv_path: str) -> list[dict]:
    """Load *csv_path* into a list of row dicts.

    Validates required columns + per-row "description or prd_doc_path" rule.
    Returns rows in declaration order; preserves all columns verbatim (no
    schema coercion — sprint_start / sprint_end stay as strings, etc.).
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"requirements CSV not found: {csv_path}")

    with p.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        missing = REQUIRED_COLS - cols
        if missing:
            raise ValueError(
                f"CSV {csv_path}: missing required columns: {sorted(missing)}; "
                f"got: {sorted(cols)}"
            )
        rows: list[dict] = []
        for i, row in enumerate(reader, 1):
            for col in REQUIRED_COLS:
                if not (row.get(col) or "").strip():
                    raise ValueError(
                        f"CSV {csv_path}: row {i} missing required field {col!r}"
                    )
            desc = (row.get("description") or "").strip()
            prd = (row.get("prd_doc_path") or "").strip()
            if not desc and not prd:
                raise ValueError(
                    f"CSV {csv_path}: row {i} has neither 'description' nor "
                    f"'prd_doc_path' — at least one is required"
                )
            if prd and not prd.endswith(".md"):
                raise ValueError(
                    f"CSV {csv_path}: row {i} prd_doc_path={prd!r} is not a .md "
                    f"file (PoC supports markdown only — see plan §4.1)"
                )
            rows.append(dict(row))
        return rows

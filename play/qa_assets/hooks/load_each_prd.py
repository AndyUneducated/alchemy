"""``load_each_prd``: enrich requirements rows with their PRD markdown content.

For each row with a ``prd_doc_path``, read the .md file (relative to the CSV
location is resolved by ``load_csv`` via the absolute path on disk; this
function expects the field to already be a usable path) and write the body
into ``prd_md``. Rows without a path keep their inline ``description`` field
as the analysis source.

Per plan §12: fail fast — raise FileNotFoundError on missing file, no
silent skip.
"""

from __future__ import annotations

from pathlib import Path


def load_each_prd(requirements: list[dict]) -> list[dict]:
    """Return a new list of rows; rows with ``prd_doc_path`` get ``prd_md`` filled."""
    out: list[dict] = []
    for i, row in enumerate(requirements, 1):
        new = dict(row)
        prd_path = (row.get("prd_doc_path") or "").strip()
        if prd_path:
            p = Path(prd_path)
            if not p.exists():
                raise FileNotFoundError(
                    f"row {i} ({row.get('req_id', '?')}): prd_doc_path "
                    f"not found: {prd_path}"
                )
            new["prd_md"] = p.read_text(encoding="utf-8")
        out.append(new)
    return out

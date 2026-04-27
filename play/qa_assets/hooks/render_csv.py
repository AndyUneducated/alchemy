"""``render_csv``: parse the agent-generated 测试用例 markdown into a flat CSV.

Forgiving regex parser: tracks the current ``### <req_id> ...`` heading and
collects ``- [Px][category] description`` lines beneath it. Lines that don't
match (subsection headings, prose) are skipped. If the section is empty or
no rows match, the writer still emits the header row so the file is always
shaped correctly.

CSV columns: req_id, title, priority, category, case, assignee,
sprint_start, sprint_end. assignee + sprint cols come from the requirements
list, joined on ``req_id`` (plan §8 P5 sprint/assignee as columns).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path


_HEADING_RE = re.compile(r"^###\s+(?P<req_id>[A-Za-z]+-\d+)\s*(?P<title>.*)$")
_CASE_RE = re.compile(
    r"^\s*-\s*\[(?P<priority>P[0-3])\]\[(?P<category>[^\]]+)\]\s*(?P<case>.+?)\s*$"
)


def render_csv(
    sections: dict[str, str],
    requirements: list[dict],
    output_path: str,
) -> str:
    """Parse ``sections['测试用例']`` and write a flat CSV beside the markdown plan."""
    cases_md = sections.get("测试用例", "")
    req_meta = {r["req_id"]: r for r in requirements}

    rows: list[dict] = []
    current_req_id: str | None = None
    current_title: str = ""
    for raw_line in cases_md.splitlines():
        h = _HEADING_RE.match(raw_line)
        if h:
            current_req_id = h.group("req_id")
            current_title = h.group("title").strip()
            continue
        c = _CASE_RE.match(raw_line)
        if not c or current_req_id is None:
            continue
        meta = req_meta.get(current_req_id, {})
        rows.append({
            "req_id": current_req_id,
            "title": current_title or meta.get("title", ""),
            "priority": c.group("priority"),
            "category": c.group("category"),
            "case": c.group("case"),
            "assignee": meta.get("assignee", ""),
            "sprint_start": meta.get("sprint_start", ""),
            "sprint_end": meta.get("sprint_end", ""),
        })

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "req_id", "title", "priority", "category", "case",
                "assignee", "sprint_start", "sprint_end",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return str(out)

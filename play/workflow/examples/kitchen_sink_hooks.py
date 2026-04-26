"""Demo hooks for kitchen_sink.yaml — minimal Python to make the example runnable.

Each function demonstrates one canonical use:
- echo            → produces a value (list[str]) for downstream stages
- enrich_lines    → consumes one stage's output, transforms, returns
- to_yaml         → serialize structured data to a string (replaces template filter)
- write_md        → consume artifact + write to disk (terminal stage side-effect)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def echo(message: str, count: int) -> list[str]:
    return [message for _ in range(count)]


def enrich_lines(lines: list[str], tag: str) -> list[str]:
    return [f"{tag} {line}" for line in lines]


def to_yaml(obj: Any) -> str:
    return yaml.dump(obj, allow_unicode=True, sort_keys=False)


def write_md(sections: dict[str, str], output_path: str, pkg_dir: str) -> str:
    """Write artifact dict as a flat markdown file. Returns *output_path*.

    *pkg_dir* (workflow.yaml's directory) is accepted but unused here — kept
    in the signature to demonstrate ``{{ pkg_dir }}`` interpolation in the
    kitchen-sink demo.
    """
    body_parts = (
        [f"## {k}\n\n{v}" for k, v in sections.items()] if sections else ["_(no sections)_"]
    )
    body = "\n\n".join(body_parts)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body + "\n", encoding="utf-8")
    return str(path)

"""``render_md``: render the multi-agent artifact + requirement metadata into a
flat markdown test plan via a Jinja2 template.

The template (``templates/test_plan.md.j2``) controls layout; this hook is
zero-logic past Jinja loading + writing. Sprint / assignee columns live in
the per-requirement table at the top of the document (plan §8 P5: "sprint /
assignee 作为列").
"""

from __future__ import annotations

from pathlib import Path

import jinja2


def render_md(
    sections: dict[str, str],
    requirements: list[dict],
    template: str,
    output_path: str,
) -> str:
    tmpl_text = Path(template).read_text(encoding="utf-8")
    env = jinja2.Environment(
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        # No Jinja loader — single-file template. Pass the rendered string
        # back; caller controls disk write below.
    )
    body = env.from_string(tmpl_text).render(
        sections=sections,
        requirements=requirements,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    return str(out)

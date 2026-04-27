"""``to_yaml``: serialize structured data to a yaml string for agent artifacts.

Used by the ``serialize_for_agent`` stage so workflow templates can pass
``list[dict]`` through to ``Engine.invoke(initial_artifact={...})`` as a
string-typed artifact section (plan §4.3) — keeping the template language
free of filters (plan §4 / §12).
"""

from __future__ import annotations

from typing import Any

import yaml


def to_yaml(obj: Any) -> str:
    return yaml.dump(obj, allow_unicode=True, sort_keys=False)

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Result:
    artifact: dict[str, str] = field(default_factory=dict)
    transcript: list[dict] = field(default_factory=list)
    success: bool = True
    warnings: list[str] = field(default_factory=list)

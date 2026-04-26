"""``Result``: dataclass returned by ``Engine.invoke()`` / future ``ainvoke()``.

Keyword-only field ordering: each new field must have a default so adding
``traceparent`` (W3C) / new metadata in the future is pure addition (plan §9.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Result:
    """End-of-run snapshot of a multi-agent discussion.

    Attributes
    ----------
    artifact:
        ``{section_name: content_markdown}`` from ``ArtifactStore.snapshot()``.
        Empty dict when the scenario has no artifact enabled.
    transcript:
        Full ``Discussion.history`` — list of structured event dicts
        (``topic / turn / speaker / artifact_event / tool_call``). Same shape
        as ``--save-transcript`` JSON.
    success:
        ``True`` iff Engine.invoke() returned without exception **and**
        ``warnings`` is empty. Soft failures (require_tool exhausted) flip
        this to ``False`` while still letting transcript / artifact land.
    warnings:
        Soft failures captured during the run (currently: require_tool
        exhausted). Each entry mirrors a stderr ``WARNING:`` line.
    """

    artifact: dict[str, str] = field(default_factory=dict)
    transcript: list[dict] = field(default_factory=list)
    success: bool = True
    warnings: list[str] = field(default_factory=list)

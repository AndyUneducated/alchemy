"""``Engine``: invoke a multi-agent discussion as a Python library.

Naming follows LangChain Runnable convention (``invoke / ainvoke / stream
/ astream``); only ``invoke`` is implemented today, the async + streaming
methods are signature placeholders (plan §5.5). Callbacks are taken as a
keyword arg list (plan §5.1) — single-library style, no ``config={...}``
indirection.
"""

from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator, Iterator

from .callbacks import Callback
from .discussion import Discussion
from .events import Event, RunFinished
from .result import Result
from .scenario import Scenario


class Engine:
    """Thin orchestrator: ``Scenario`` → assemble → ``Discussion.run`` → ``Result``."""

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario

    def invoke(
        self,
        *,
        initial_artifact: dict[str, str] | None = None,
        transcript_path: str | Path | None = None,
        artifact_path: str | Path | None = None,
        callbacks: list[Callback] | None = None,
        print_stream: bool = False,
    ) -> Result:
        """Run the scenario synchronously; return a ``Result`` snapshot.

        Parameters are keyword-only so future additions (e.g. ``trace_context``)
        stay non-breaking (plan §5.5 / §9.1).

        Parameters
        ----------
        initial_artifact:
            Pre-populate artifact sections before the run starts. Sections
            in ``scenario.artifact.initial_sections`` are created first; any
            keys here either fill or override their content.
        transcript_path:
            Where to dump the structured ``Discussion.history`` JSON. Same
            shape as the legacy ``--save-transcript`` output.
        artifact_path:
            Where to dump the full rendered artifact markdown (sections +
            votes + final decision). Mirrors legacy ``--save-artifact``.
            Ignored (with stderr WARNING) when no artifact is enabled.
        callbacks:
            Subscribe to ``Event`` notifications. Today only ``on_run_finished``
            fires after a successful sync run. Other event types are placeholder;
            adding them is non-breaking (callbacks default to no-op).
        print_stream:
            When True, the Discussion streams live LLM token output to stdout
            (plus speaker emoji headers either way) — matches legacy ``run.py``
            behavior; CLI passes True, library default is False so workflow /
            future Web service consumers stay quiet by default (plan §10 D).
        """
        assembly = self.scenario.assemble()

        if initial_artifact and assembly.artifact is not None:
            for name, content in initial_artifact.items():
                # Use append-or-create-section semantics: if the section was
                # declared in initial_sections, write_section/append_section
                # honors its mode; otherwise we create a default-mode section.
                # Direct dict assignment bypasses tool ACL (engine-level seed).
                assembly.artifact.sections[name] = content

        discussion = Discussion(
            agents=assembly.agents,
            agent_roles=assembly.agent_roles,
            topic=assembly.topic,
            steps=assembly.steps,
            stream=print_stream,
            artifact=assembly.artifact,
            tracer=assembly.tracer,
        )

        # Run; capture history + warnings. We let exceptions propagate — Engine
        # consumers (cli, workflow) decide whether to wrap into Result(success=False).
        history = discussion.run()

        artifact_snapshot: dict[str, str] = (
            dict(assembly.artifact.sections) if assembly.artifact is not None else {}
        )
        warnings = list(discussion.warnings)
        success = not warnings

        if transcript_path is not None:
            tp = Path(transcript_path)
            tp.parent.mkdir(parents=True, exist_ok=True)
            import json
            with open(tp, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
                f.write("\n")

        if artifact_path is not None:
            import sys
            if assembly.artifact is None:
                print(
                    "WARNING: artifact_path ignored: scenario has no artifact enabled",
                    file=sys.stderr, flush=True,
                )
            else:
                ap = Path(artifact_path)
                ap.parent.mkdir(parents=True, exist_ok=True)
                with open(ap, "w", encoding="utf-8") as f:
                    f.write(assembly.artifact.render())
                    f.write("\n")

        result = Result(
            artifact=artifact_snapshot,
            transcript=history,
            success=success,
            warnings=warnings,
        )

        # Today only RunFinished fires; other Event subclasses are wired in
        # a future iteration (touching Discussion internals). Subclasses with
        # no override silently no-op.
        if callbacks:
            evt = RunFinished(success=success)
            for cb in callbacks:
                cb.on_run_finished(evt)

        return result

    # -- placeholders: signatures locked, body deferred (plan §5.5) ---------

    async def ainvoke(self, **kwargs) -> Result:
        raise NotImplementedError(
            "Engine.ainvoke() is not implemented yet. Track in plan §5.5."
        )

    def stream(self, **kwargs) -> Iterator[Event]:
        raise NotImplementedError(
            "Engine.stream() is not implemented yet. Track in plan §5.5."
        )

    async def astream(self, **kwargs) -> AsyncIterator[Event]:
        raise NotImplementedError(
            "Engine.astream() is not implemented yet. Track in plan §5.5."
        )

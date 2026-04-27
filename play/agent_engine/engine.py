from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator, Iterator

from .callbacks import Callback
from .discussion import Discussion
from .events import Event, RunFinished
from .result import Result
from .scenario import Scenario


class Engine:
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
        assembly = self.scenario.assemble()

        if initial_artifact and assembly.artifact is not None:
            for name, content in initial_artifact.items():
                # Engine-level seed; bypasses artifact tool ACL.
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

        if callbacks:
            evt = RunFinished(success=success)
            for cb in callbacks:
                cb.on_run_finished(evt)

        return result

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

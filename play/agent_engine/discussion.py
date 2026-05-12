from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING, Sequence

from .result import (
    ArtifactEventEntry,
    SpeakerEntry,
    TokenUsage,
    TopicEntry,
    ToolCallEntry,
    TranscriptEntry,
    TurnEntry,
)
from .scenario import _resolve_who_names

if TYPE_CHECKING:
    from .agent import Agent
    from .artifact import ArtifactStore
    from .tracer import ToolTracer


def _print_speaker(name: str, step_id: str | None = None) -> None:
    suffix = f" (step={step_id})" if step_id else ""
    sys.stdout.write(f"\n🗣  [{name}]{suffix}: ")
    sys.stdout.flush()


def _called_tool(
    events: Sequence[ToolCallEntry | ArtifactEventEntry],
    caller: str,
    tool: str,
) -> bool:
    return any(e.tool == tool and e.caller == caller for e in events)


class Discussion:
    def __init__(
        self,
        agents: list[Agent],
        agent_roles: dict[str, str],
        topic: str,
        *,
        steps: list[dict],
        stream: bool = True,
        artifact: "ArtifactStore | None" = None,
        tracer: "ToolTracer | None" = None,
    ) -> None:
        self.agents = agents
        self.agent_roles = agent_roles
        self.topic = topic
        self.steps = steps
        self.stream = stream
        self.artifact = artifact
        self.tracer = tracer
        self.history: list[TranscriptEntry] = []
        self.warnings: list[str] = []
        self.usage: list[TokenUsage] = []
        self._by_name: dict[str, "Agent"] = {a.name: a for a in agents}
        self._expanded: list[tuple["Agent", dict]] = self._expand_steps()

    def run(self) -> list[TranscriptEntry]:
        total = len(self._expanded)
        self._print_header(total)
        self.history.append(TopicEntry(content=self.topic, ts=time.time()))

        for idx, (agent, step) in enumerate(self._expanded, 1):
            marker = f"turn {idx} of {total}"
            self.history.append(TurnEntry(content=marker, ts=time.time()))
            instruction = step.get("instruction")
            require_tool = step.get("require_tool")
            max_retries = int(step.get("max_retries", 1 if require_tool else 0))
            step_id = step.get("id")
            self._run_turn(agent, step_id, instruction, require_tool, max_retries)

        print(f"\n{'=' * 60}\n  End\n{'=' * 60}\n")
        return self.history

    def _expand_steps(self) -> list[tuple["Agent", dict]]:
        out: list[tuple[Agent, dict]] = []
        for step in self.steps:
            for agent in self._resolve_who(step["who"]):
                out.append((agent, step))
        return out

    def _resolve_who(self, who) -> list["Agent"]:
        declared_order = [a.name for a in self.agents]
        names = _resolve_who_names(who, declared_order, self.agent_roles)
        return [self._by_name[n] for n in names]

    def _run_turn(
        self,
        agent: "Agent",
        step_id: str | None,
        instruction: str | None,
        require_tool: str | None,
        max_retries: int,
    ) -> None:
        current_instruction = instruction
        for attempt in range(max_retries + 1):
            _print_speaker(agent.name, step_id)
            view = self.artifact.render() if self.artifact else None
            reply, usage_batch = agent.respond(
                self.history,
                instruction=current_instruction,
                stream=self.stream,
                artifact_view=view,
            )
            self.usage.extend(usage_batch)
            tracer_events: list[ToolCallEntry] = []
            if self.tracer:
                tracer_events = self.tracer.drain()
                self.history.extend(tracer_events)
            self.history.append(SpeakerEntry(
                speaker=agent.name,
                content=reply,
                ts=time.time(),
            ))
            artifact_events: list[ArtifactEventEntry] = []
            if self.artifact:
                artifact_events = self.artifact.drain_events()
                self.history.extend(artifact_events)

            # require_tool 检查同时覆盖 tracer (非 artifact 工具如 retrieve_docs)
            # 与 artifact 事件——前者过去被遗漏，导致 require_tool 只对 artifact 工
            # 具有效；本期补全（DECISIONS §11 / agent_sft phase 1.B 引入两个新
            # require_tool: retrieve_docs 场景的前置依赖）.
            events: list[ToolCallEntry | ArtifactEventEntry] = (
                list(tracer_events) + list(artifact_events)
            )
            if not require_tool or _called_tool(events, agent.name, require_tool):
                return

            if attempt >= max_retries:
                msg = (
                    f"{agent.name} skipped required tool "
                    f"'{require_tool}' after {attempt + 1} attempt(s)"
                )
                print(f"WARNING: {msg}", file=sys.stderr, flush=True)
                self.warnings.append(msg)
                return

            print(
                f"🔁 [{agent.name}] retry {attempt + 1}/{max_retries}: "
                f"missing {require_tool}",
                flush=True,
            )
            current_instruction = (
                f"你刚才没有调用 `{require_tool}` 工具。"
                f"请现在补上该调用以完成本轮任务。"
            )

    def _print_header(self, total_turns: int) -> None:
        names = [a.name for a in self.agents]
        print(f"\n{'=' * 60}")
        print(f"  Participants: {', '.join(names)}")
        print(f"  Steps: {len(self.steps)}  |  Total turns: {total_turns}")
        print(f"{'=' * 60}")

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent
    from .artifact import ArtifactStore
    from .tracer import ToolTracer


def _print_speaker(name: str, step_id: str | None = None) -> None:
    suffix = f" (step={step_id})" if step_id else ""
    sys.stdout.write(f"\n🗣  [{name}]{suffix}: ")
    sys.stdout.flush()


def _called_tool(events: list[dict], caller: str, tool: str) -> bool:
    return any(
        e.get("tool") == tool and e.get("caller") == caller
        for e in events
    )


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
        self.history: list[dict] = []
        self.warnings: list[str] = []
        self._by_name: dict[str, "Agent"] = {a.name: a for a in agents}
        self._expanded: list[tuple["Agent", dict]] = self._expand_steps()

    def run(self) -> list[dict]:
        total = len(self._expanded)
        self._print_header(total)
        self.history.append({
            "type": "topic", "content": self.topic, "ts": time.time(),
        })

        for idx, (agent, step) in enumerate(self._expanded, 1):
            marker = f"turn {idx} of {total}"
            self.history.append({
                "type": "turn", "content": marker, "ts": time.time(),
            })
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
        if isinstance(who, str):
            if who == "all":
                return list(self.agents)
            return [a for a in self.agents if self.agent_roles.get(a.name) == who]
        if isinstance(who, list):
            return [self._by_name[n] for n in who]
        raise TypeError(f"Unsupported who form: {who!r}")

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
            reply = agent.respond(
                self.history,
                instruction=current_instruction,
                stream=self.stream,
                artifact_view=view,
            )
            if self.tracer:
                self.history.extend(self.tracer.drain())
            self.history.append({
                "speaker": agent.name,
                "content": reply,
                "ts": time.time(),
            })
            events: list[dict] = []
            if self.artifact:
                events = self.artifact.drain_events()
                self.history.extend(events)

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

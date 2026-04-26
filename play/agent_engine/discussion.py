"""Step-driven multi-agent conversation engine.

Scenario flow is a single flat ``steps:`` list. Each step's ``who`` is one
of: scalar role (``moderator`` / ``member``), scalar keyword ``all``, or
a list of agent names. The engine expands every step into one or more
``turns`` (one turn per matched agent), assigns each turn a globally
monotonic counter ``<turn>turn X of N</turn>``, and runs them sequentially.
"""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent import Agent
    from artifact import ArtifactStore
    from run import ToolTracer

SEPARATOR = "-" * 60


def _print_speaker(name: str, step_id: str | None = None) -> None:
    suffix = f" (step={step_id})" if step_id else ""
    sys.stdout.write(f"\n🗣  [{name}]{suffix}: ")
    sys.stdout.flush()


def _called_tool(events: list[dict], caller: str, tool: str) -> bool:
    """True if *events* contains an artifact_event by *caller* calling *tool*."""
    return any(
        e.get("tool") == tool and e.get("caller") == caller
        for e in events
    )


class Discussion:
    """Execute a multi-agent discussion as a flat sequence of turns.

    The schema-level concept is ``steps`` (declarative); each step expands
    into one or more ``turns`` (runtime execution units). The total turn
    count ``N`` is precomputed at construction so each turn can be tagged
    ``<turn>turn X of N</turn>`` for the agent's positional awareness.
    """

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
        self._by_name: dict[str, "Agent"] = {a.name: a for a in agents}
        self._expanded: list[tuple["Agent", dict]] = self._expand_steps()

    # -- public ------------------------------------------------------------

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
            # default to 1 retry when a tool is required; else 0 (no nudge)
            max_retries = int(step.get("max_retries", 1 if require_tool else 0))
            step_id = step.get("id")
            self._run_turn(agent, step_id, instruction, require_tool, max_retries)

        print(f"\n{'=' * 60}\n  End\n{'=' * 60}\n")
        return self.history

    # -- internals ---------------------------------------------------------

    def _expand_steps(self) -> list[tuple["Agent", dict]]:
        """Resolve each step's ``who`` once, in declaration order."""
        out: list[tuple[Agent, dict]] = []
        for step in self.steps:
            for agent in self._resolve_who(step["who"]):
                out.append((agent, step))
        return out

    def _resolve_who(self, who) -> list["Agent"]:
        """Map a step's ``who`` value to the ordered list of agents.

        Two literal forms (validated upstream by ``run.py``):
        - scalar ``str`` ∈ {moderator, member, all}
        - ``list[str]`` of agent names
        """
        if isinstance(who, str):
            if who == "all":
                return list(self.agents)
            return [a for a in self.agents if self.agent_roles.get(a.name) == who]
        if isinstance(who, list):
            return [self._by_name[n] for n in who]
        # Should never reach here — schema validator rejects other types.
        raise TypeError(f"Unsupported who form: {who!r}")

    def _run_turn(
        self,
        agent: "Agent",
        step_id: str | None,
        instruction: str | None,
        require_tool: str | None,
        max_retries: int,
    ) -> None:
        """Run one agent's turn, optionally retrying if require_tool wasn't called.

        The nudge on retry is passed as an ``instruction`` override — it's
        per-call only and never enters ``self.history``, so other agents don't
        see the coaching.
        """
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
            # Drain tool_call events BEFORE the speaker entry so transcript
            # order matches chronology: tool calls happened during respond(),
            # the final reply text came out after them. visible=False keeps
            # them invisible to other agents either way.
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
                print(
                    f"WARNING: {agent.name} skipped required tool "
                    f"'{require_tool}' after {attempt + 1} attempt(s)",
                    file=sys.stderr, flush=True,
                )
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

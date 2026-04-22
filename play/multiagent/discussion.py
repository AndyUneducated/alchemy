"""Phase-driven multi-agent conversation engine."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent import Agent
    from artifact import ArtifactStore

SEPARATOR = "-" * 60


def _print_speaker(name: str) -> None:
    sys.stdout.write(f"\n🗣  [{name}]: ")
    sys.stdout.flush()


class Discussion:
    """Execute a multi-agent discussion with opening, main, and closing phases.

    *opening* and *closing* phases run once.  *main* phases repeat for
    *rounds* iterations, resolved per-round via the ``round`` field:

      1. Exact match on ``round: <int>``
      2. Fallback to ``round: "default"``
      3. Implicit default — all participants speak, no instruction
    """

    def __init__(
        self,
        members: list[Agent],
        topic: str,
        *,
        opening: list[dict] | None = None,
        main: list[dict] | None = None,
        closing: list[dict] | None = None,
        rounds: int,
        stream: bool = True,
        moderator: Agent | None = None,
        artifact: "ArtifactStore | None" = None,
    ) -> None:
        self.members = members
        self.topic = topic
        self.opening = opening or []
        self.main = main or []
        self.closing = closing or []
        self.rounds = rounds
        self.stream = stream
        self.moderator = moderator
        self.artifact = artifact
        self.history: list[dict] = []

    def run(self) -> list[dict]:
        self._print_header()
        self.history.append({"type": "topic", "content": self.topic})

        if self.opening:
            self.history.append({"type": "phase", "content": "opening"})
            for phase in self.opening:
                self._exec_phase(phase)

        for round_num in range(1, self.rounds + 1):
            print(f"\n{SEPARATOR}\n  Round {round_num}\n{SEPARATOR}")
            self.history.append({"type": "round", "content": f"Round {round_num}/{self.rounds}"})
            phases = [p for p in self.main if p["round"] == round_num]
            if not phases:
                phases = [p for p in self.main if p["round"] == "default"]
            if not phases:
                phases = [{"who": "all", "round": "default"}]
            for phase in phases:
                self._exec_phase(phase)

        if self.closing:
            self.history.append({"type": "phase", "content": "closing"})
            for phase in self.closing:
                self._exec_phase(phase)

        print(f"\n{'=' * 60}\n  End\n{'=' * 60}\n")
        return self.history

    def _exec_phase(self, phase: dict) -> None:
        instruction = phase.get("instruction")
        for agent in self._resolve_who(phase["who"]):
            _print_speaker(agent.name)
            view = self.artifact.render() if self.artifact else None
            reply = agent.respond(
                self.history,
                instruction=instruction,
                stream=self.stream,
                artifact_view=view,
            )
            self.history.append({"speaker": agent.name, "content": reply})
            if self.artifact:
                self.history.extend(self.artifact.drain_events())

    def _resolve_who(self, who: str) -> list[Agent]:
        if who == "members":
            return self.members
        if who == "moderator":
            return [self.moderator] if self.moderator else []
        if who == "all":
            result: list[Agent] = []
            if self.moderator:
                result.append(self.moderator)
            result.extend(self.members)
            return result
        all_agents = list(self.members)
        if self.moderator:
            all_agents.append(self.moderator)
        return [a for a in all_agents if a.name == who]

    def _print_header(self) -> None:
        names = [a.name for a in self.members]
        if self.moderator:
            names.insert(0, self.moderator.name)
        print(f"\n{'=' * 60}")
        print(f"  Participants: {', '.join(names)}")
        print(f"  Rounds: {self.rounds}")
        print(f"{'=' * 60}")

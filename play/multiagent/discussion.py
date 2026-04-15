"""Phase-driven multi-agent conversation engine."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent import Agent

SEPARATOR = "-" * 60
VALID_STAGES = ("opening", "main", "closing")


def _print_speaker(name: str) -> None:
    sys.stdout.write(f"\n🗣  [{name}]: ")
    sys.stdout.flush()


def _speak(agent: Agent, history: list[dict], *, stream: bool) -> None:
    _print_speaker(agent.name)
    reply = agent.respond(history, stream=stream)
    history.append({
        "role": "assistant",
        "content": f"[{agent.name}]: {reply}",
    })


class Discussion:
    """Execute a phase-driven multi-agent discussion.

    The flow is fully defined by the *phases* list.  Each phase dict has:
      - stage (required): "opening" | "main" | "closing"
      - who   (required): "moderator" | "members" | "all" | a specific name
      - instruction (optional): injected as a user message before speaking

    ``opening`` phases run once, ``main`` phases repeat *rounds* times,
    ``closing`` phases run once.
    """

    def __init__(
        self,
        members: list[Agent],
        topic: str,
        phases: list[dict],
        rounds: int = 3,
        stream: bool = True,
        moderator: Agent | None = None,
    ) -> None:
        self.members = members
        self.topic = topic
        self.phases = phases
        self.rounds = rounds
        self.stream = stream
        self.moderator = moderator
        self.history: list[dict] = []

    def run(self) -> list[dict]:
        self._print_header()
        self.history.append({"role": "user", "content": self.topic})

        opening = [p for p in self.phases if p["stage"] == "opening"]
        main = [p for p in self.phases if p["stage"] == "main"]
        closing = [p for p in self.phases if p["stage"] == "closing"]

        for phase in opening:
            self._exec_phase(phase)

        for round_num in range(1, self.rounds + 1):
            print(f"\n{SEPARATOR}\n  Round {round_num}\n{SEPARATOR}")
            for phase in main:
                self._exec_phase(phase)

        for phase in closing:
            self._exec_phase(phase)

        print(f"\n{'=' * 60}\n  End\n{'=' * 60}\n")
        return self.history

    def _exec_phase(self, phase: dict) -> None:
        instruction = phase.get("instruction")
        if instruction:
            self.history.append({"role": "user", "content": instruction})

        for agent in self._resolve_who(phase["who"]):
            _speak(agent, self.history, stream=self.stream)

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

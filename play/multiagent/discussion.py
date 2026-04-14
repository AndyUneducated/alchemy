"""Discussion orchestrator: round-robin multi-agent conversation."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent import Agent

SEPARATOR = "-" * 60


class Discussion:
    """Run a multi-round discussion among a list of agents."""

    def __init__(
        self,
        agents: list[Agent],
        topic: str,
        rounds: int = 3,
        stream: bool = True,
    ) -> None:
        self.agents = agents
        self.topic = topic
        self.rounds = rounds
        self.stream = stream
        self.history: list[dict] = []

    def run(self) -> list[dict]:
        """Execute the discussion and return the full history."""
        self._print_header()

        opening = f"The topic is: {self.topic}\nPlease share your views."
        self.history.append({"role": "user", "content": opening})

        for round_num in range(1, self.rounds + 1):
            self._print_round(round_num)
            for agent in self.agents:
                self._print_speaker(agent.name)
                reply = agent.respond(self.history, stream=self.stream)
                self.history.append({
                    "role": "assistant",
                    "content": f"[{agent.name}]: {reply}",
                })

        self._print_footer()
        return self.history

    # -- display helpers --------------------------------------------------

    def _print_header(self) -> None:
        print(f"\n{'=' * 60}")
        print(f"  Topic: {self.topic}")
        print(f"  Participants: {', '.join(a.name for a in self.agents)}")
        print(f"  Rounds: {self.rounds}")
        print(f"{'=' * 60}\n")

    def _print_round(self, n: int) -> None:
        print(f"\n{SEPARATOR}")
        print(f"  Round {n}")
        print(SEPARATOR)

    def _print_speaker(self, name: str) -> None:
        sys.stdout.write(f"\n🗣  [{name}]: ")
        sys.stdout.flush()

    def _print_footer(self) -> None:
        print(f"\n{'=' * 60}")
        print("  End of discussion")
        print(f"{'=' * 60}\n")

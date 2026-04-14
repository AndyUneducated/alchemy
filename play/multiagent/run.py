#!/usr/bin/env python3
"""CLI entry-point: launch a multi-agent discussion."""

import argparse

from agent import Agent
from discussion import Discussion

DEFAULT_TOPIC = "You three role-play as leaders of China, the US, and the EU, discussing the recent Iran conflict"

AGENTS = [
    Agent(
        name="China",
        system_prompt=(
            ""
        ),
    ),
    Agent(
        name="US",
        system_prompt=(
            ""
        ),
    ),
    Agent(
        name="EU",
        system_prompt=(
            ""
        ),
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Agent Discussion")
    parser.add_argument("topic", nargs="?", default=DEFAULT_TOPIC,
                        help="discussion topic")
    parser.add_argument("-r", "--rounds", type=int, default=3,
                        help="number of rounds (default: 3)")
    parser.add_argument("--no-stream", action="store_true",
                        help="disable streaming output")
    args = parser.parse_args()

    discussion = Discussion(
        agents=AGENTS,
        topic=args.topic,
        rounds=args.rounds,
        stream=not args.no_stream,
    )
    discussion.run()


if __name__ == "__main__":
    main()

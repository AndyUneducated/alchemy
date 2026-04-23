"""Shared artifact + structured voting for explicit decision-making.

An ``ArtifactStore`` holds a multi-section markdown document, a dict of
structured votes, and a final-decision marker.  It exposes six tools that
agents call through the standard ``tool_handler`` protocol:

  - ``read_artifact``     returns the rendered markdown view
  - ``write_section``     full-overwrite a section (blocked on append-only)
  - ``append_section``    append an entry (blocked on replace-only)
  - ``propose_vote``      register a structured vote, returns vote_id
  - ``cast_vote``         record a ballot
  - ``finalize_artifact`` moderator-only; seal the decision

Section modes are declared by the scenario author in ``initial_sections``.
Agents pick the tool, ArtifactStore enforces the mode and returns a clear
``{"error": ...}`` on mismatch; the existing ``tools.warn_if_error`` catch-all
surfaces it on stderr for the workshop audience, and the tool-loop in every
backend client re-feeds it to the agent so it can self-correct.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from tools import warn_if_error


ARTIFACT_TOOL_NAMES = frozenset({
    "read_artifact",
    "write_section",
    "append_section",
    "propose_vote",
    "cast_vote",
    "finalize_artifact",
})

MODERATOR_ONLY_TOOLS = frozenset({"finalize_artifact", "propose_vote"})


@dataclass
class Vote:
    vote_id: str
    question: str
    options: list[str]
    ballots: dict[str, tuple[str, str | None]] = field(default_factory=dict)


class ArtifactStore:
    """In-memory artifact + voting state shared across one Discussion."""

    def __init__(self, initial_sections: list | None = None) -> None:
        self.sections: dict[str, str] = {}
        self.section_modes: dict[str, str] = {}  # declared sections only
        self.votes: dict[str, Vote] = {}
        self.finalized: bool = False
        self.final_decision: str | None = None
        self.final_rationale: str | None = None
        self._events: list[dict] = []
        self._next_vote_num: int = 1

        for item in initial_sections or []:
            if isinstance(item, str):
                name, mode = item, "replace"
            else:
                name, mode = item["name"], item.get("mode", "replace")
            self.sections[name] = ""
            self.section_modes[name] = mode

    # -- public API ---------------------------------------------------------

    def render(self) -> str:
        """Render the artifact as markdown: sections → votes → final decision."""
        parts: list[str] = []
        for name, content in self.sections.items():
            body = content if content else "_(empty)_"
            parts.append(f"## {name}\n{body}")

        if self.votes:
            parts.append("## Votes")
            for v in self.votes.values():
                tally = self._tally(v)
                parts.append(f"### {v.vote_id}: {v.question}")
                lines = []
                for opt in v.options:
                    voters = tally.get(opt, [])
                    voter_str = ", ".join(voters) if voters else "—"
                    lines.append(f"- {opt}: {len(voters)} ({voter_str})")
                parts.append("\n".join(lines))

        if self.finalized:
            parts.append("## Final Decision")
            body = f"**{self.final_decision}**"
            if self.final_rationale:
                body += f"\n\n{self.final_rationale}"
            parts.append(body)

        return "\n\n".join(parts) if parts else "_(empty artifact)_"

    def drain_events(self) -> list[dict]:
        """Pop and return pending artifact_event history entries."""
        events, self._events = self._events, []
        return events

    def dispatch(self, name: str, args: dict, *, caller: str) -> str:
        """Route a tool call to the right handler; surface errors on stderr."""
        handler = _HANDLERS.get(name)
        if handler is None:
            result = json.dumps({"error": f"Unknown artifact tool: {name}"})
        else:
            try:
                result = handler(self, args, caller)
            except Exception as exc:  # noqa: BLE001
                result = json.dumps({"error": f"{name}: {type(exc).__name__}: {exc}"})
        warn_if_error(name, result)
        return result

    def build_tool_defs(self, role: str) -> list[dict]:
        """Return OpenAI-format tool defs visible to an agent of *role*.

        ``role`` is either ``"moderator"`` or ``"member"``.  Moderators get the
        full set including ``finalize_artifact``; members get everything else.
        """
        defs = [d for d in _TOOL_DEFS if d["function"]["name"] not in MODERATOR_ONLY_TOOLS]
        if role == "moderator":
            defs = list(_TOOL_DEFS)  # includes finalize
        return defs

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _tally(v: Vote) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for voter, (opt, _) in v.ballots.items():
            out.setdefault(opt, []).append(voter)
        return out


# -- handlers ---------------------------------------------------------------
# Each handler takes (store, args, caller) -> JSON string.  Kept at module
# level so the dispatch table is a plain dict lookup, not an attribute dance.


def _h_read_artifact(store: ArtifactStore, args: dict, caller: str) -> str:
    return json.dumps({"content": store.render()}, ensure_ascii=False)


def _h_write_section(store: ArtifactStore, args: dict, caller: str) -> str:
    name = args.get("name")
    content = args.get("content", "")
    if not name:
        return json.dumps({"error": "write_section: missing 'name'"})
    if not isinstance(content, str):
        content = str(content)
    if store.section_modes.get(name) == "append":
        return json.dumps({
            "error": f"section '{name}' is append-only; use append_section",
        })
    store.sections[name] = content
    n = len(content)
    print(f"📝 [{caller}] wrote section '{name}' ({n} chars)", flush=True)
    store._events.append({
        "type": "artifact_event",
        "tool": "write_section",
        "caller": caller,
        "content": f"{caller} wrote section '{name}' ({n} chars)",
        "ts": time.time(),
    })
    return json.dumps({"ok": True, "section": name}, ensure_ascii=False)


def _h_append_section(store: ArtifactStore, args: dict, caller: str) -> str:
    name = args.get("name")
    entry = args.get("entry", "")
    if not name:
        return json.dumps({"error": "append_section: missing 'name'"})
    if not isinstance(entry, str):
        entry = str(entry)
    if store.section_modes.get(name) == "replace":
        return json.dumps({
            "error": f"section '{name}' is replace-only; use write_section",
        })
    old = store.sections.get(name, "")
    store.sections[name] = (old + "\n" + entry) if old else entry
    n = len(entry)
    print(f"➕ [{caller}] appended to '{name}' ({n} chars)", flush=True)
    store._events.append({
        "type": "artifact_event",
        "tool": "append_section",
        "caller": caller,
        "content": f"{caller} appended to '{name}' ({n} chars)",
        "ts": time.time(),
    })
    return json.dumps({"ok": True, "section": name}, ensure_ascii=False)


def _h_propose_vote(store: ArtifactStore, args: dict, caller: str) -> str:
    question = args.get("question")
    options = args.get("options")
    if not question or not isinstance(options, list) or len(options) < 2:
        return json.dumps({
            "error": "propose_vote: need 'question' and 'options' (list of >= 2 strings)",
        })
    options = [str(o) for o in options]
    vid = f"v{store._next_vote_num}"
    store._next_vote_num += 1
    store.votes[vid] = Vote(vote_id=vid, question=question, options=options)
    print(f"🗳  [{caller}] proposed vote {vid}: \"{question}\"", flush=True)
    store._events.append({
        "type": "artifact_event",
        "tool": "propose_vote",
        "caller": caller,
        "content": f"{caller} proposed vote {vid}: '{question}'",
        "ts": time.time(),
    })
    return json.dumps({"vote_id": vid}, ensure_ascii=False)


def _h_cast_vote(store: ArtifactStore, args: dict, caller: str) -> str:
    vid = args.get("vote_id")
    option = args.get("option")
    rationale = args.get("rationale")
    if vid not in store.votes:
        return json.dumps({"error": f"cast_vote: unknown vote_id '{vid}'"})
    vote = store.votes[vid]
    if option not in vote.options:
        return json.dumps({
            "error": f"cast_vote: option '{option}' not in {vote.options}",
        })
    vote.ballots[caller] = (option, rationale)
    print(f"✓ [{caller}] cast {vid} → {option}", flush=True)
    store._events.append({
        "type": "artifact_event",
        "tool": "cast_vote",
        "caller": caller,
        "content": f"{caller} cast {vid} → {option}",
        "ts": time.time(),
    })
    return json.dumps({"ok": True}, ensure_ascii=False)


def _h_finalize_artifact(store: ArtifactStore, args: dict, caller: str) -> str:
    if store.finalized:
        return json.dumps({"error": "finalize_artifact: already finalized"})
    decision = args.get("decision")
    rationale = args.get("rationale")
    if not decision or not rationale:
        return json.dumps({
            "error": "finalize_artifact: need both 'decision' and 'rationale'",
        })
    store.finalized = True
    store.final_decision = decision
    store.final_rationale = rationale
    print(f"🏁 [{caller}] finalized: {decision}", flush=True)
    store._events.append({
        "type": "artifact_event",
        "tool": "finalize_artifact",
        "caller": caller,
        "content": f"{caller} finalized: {decision}",
        "ts": time.time(),
    })
    return json.dumps({"ok": True}, ensure_ascii=False)


_HANDLERS: dict[str, Any] = {
    "read_artifact": _h_read_artifact,
    "write_section": _h_write_section,
    "append_section": _h_append_section,
    "propose_vote": _h_propose_vote,
    "cast_vote": _h_cast_vote,
    "finalize_artifact": _h_finalize_artifact,
}


# -- tool schemas (OpenAI format, translated to each backend by clients) ----

_TOOL_DEFS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_artifact",
            "description": (
                "Read the current artifact as rendered markdown "
                "(sections + votes + final decision)."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_section",
            "description": (
                "Full-overwrite a section of the artifact. "
                "Blocked with an error if the section was declared as append-only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Section name to overwrite.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full replacement content for the section.",
                    },
                },
                "required": ["name", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_section",
            "description": (
                "Append an entry to a section, preserving existing content. "
                "Use this when multiple participants collaborate on the same "
                "section. Blocked if the section was declared as replace-only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Section name to append to.",
                    },
                    "entry": {
                        "type": "string",
                        "description": "Entry text; joined to existing content with a newline.",
                    },
                },
                "required": ["name", "entry"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_vote",
            "description": (
                "Register a structured vote. Returns a vote_id that others "
                "pass to cast_vote. Moderator only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "What the vote is about.",
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Two or more distinct choices.",
                    },
                },
                "required": ["question", "options"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cast_vote",
            "description": (
                "Record your ballot on an existing vote. The caller's name is "
                "recorded automatically; a second cast overwrites your earlier one."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "vote_id": {
                        "type": "string",
                        "description": "The vote_id returned by propose_vote.",
                    },
                    "option": {
                        "type": "string",
                        "description": "Must exactly match one of the vote's options.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Optional short reason for the choice.",
                    },
                },
                "required": ["vote_id", "option"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_artifact",
            "description": (
                "Seal the artifact with a final decision and rationale. "
                "Moderator only; idempotent — a second call returns an error."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "decision": {
                        "type": "string",
                        "description": "The winning option / final verdict.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Brief reasoning that justifies the decision.",
                    },
                },
                "required": ["decision", "rationale"],
            },
        },
    },
]

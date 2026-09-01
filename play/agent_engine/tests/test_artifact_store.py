"""ArtifactStore + 6 tools contract tests (DECISIONS §6 / §9).

Direct tests on `ArtifactStore.dispatch` / `render` / `build_tool_defs`, bypassing
Engine / Discussion / Agent — locks artifact subsystem public API:
  - 6 tools OK / error JSON paths
  - section mode (replace / append) write boundaries
  - `tool_owners` ACL affects `build_tool_defs` (LLM invisible) and
    `dispatch` (hard block wrong caller)
  - `finalize_artifact` idempotent
  - `drain_events` single take + clear
  - `render()` markdown shape stable (`--save-artifact` consumers depend on it)

Breaking artifact tool semantics / event schema / ACL behavior fails this test.
"""
from __future__ import annotations

import json

import pytest

from agent_engine import ArtifactEventEntry
from agent_engine.artifact import ARTIFACT_TOOL_NAMES, ArtifactStore


# ---------- helpers ----------------------------------------------------

def _payload(result: str) -> dict:
    return json.loads(result)


# ---------- read_artifact ---------------------------------------------

def test_read_artifact_returns_rendered_markdown_in_content_field():
    store = ArtifactStore(initial_sections=["数据"])
    store.sections["数据"] = "hello"
    out = _payload(store.dispatch("read_artifact", {}, caller="A"))
    assert "content" in out
    assert "## 数据\nhello" in out["content"]


def test_read_artifact_empty_store_renders_placeholder():
    store = ArtifactStore()
    out = _payload(store.dispatch("read_artifact", {}, caller="A"))
    assert out["content"] == "_(empty artifact)_"


# ---------- write_section ---------------------------------------------

def test_write_section_happy_path_emits_event_and_updates_section():
    store = ArtifactStore(initial_sections=[{"name": "结论", "mode": "replace"}])
    out = _payload(store.dispatch(
        "write_section", {"name": "结论", "content": "ok"}, caller="A",
    ))
    assert out == {"ok": True, "section": "结论"}
    assert store.sections["结论"] == "ok"
    events = store.drain_events()
    assert len(events) == 1
    ev = events[0]
    assert isinstance(ev, ArtifactEventEntry)
    assert ev.tool == "write_section"
    assert ev.caller == "A"
    assert ev.arguments == {"name": "结论", "content": "ok"}


def test_write_section_missing_name_returns_error():
    store = ArtifactStore()
    out = _payload(store.dispatch("write_section", {}, caller="A"))
    assert "error" in out
    assert "missing 'name'" in out["error"]


def test_write_section_append_only_section_returns_error():
    """Declared mode=append section + write_section → error, state unchanged."""
    store = ArtifactStore(initial_sections=[{"name": "记录", "mode": "append"}])
    out = _payload(store.dispatch(
        "write_section", {"name": "记录", "content": "x"}, caller="A",
    ))
    assert "error" in out
    assert "append-only" in out["error"]
    assert store.sections["记录"] == ""
    assert store.drain_events() == []


# ---------- append_section --------------------------------------------

def test_append_section_concatenates_with_newline():
    store = ArtifactStore(initial_sections=[{"name": "记录", "mode": "append"}])
    store.dispatch("append_section", {"name": "记录", "entry": "first"}, caller="A")
    store.dispatch("append_section", {"name": "记录", "entry": "second"}, caller="B")
    assert store.sections["记录"] == "first\nsecond"
    events = store.drain_events()
    assert [(e.caller, e.tool) for e in events] == [
        ("A", "append_section"), ("B", "append_section"),
    ]


def test_append_section_replace_only_section_returns_error():
    store = ArtifactStore(initial_sections=[{"name": "结论", "mode": "replace"}])
    out = _payload(store.dispatch(
        "append_section", {"name": "结论", "entry": "x"}, caller="A",
    ))
    assert "error" in out
    assert "replace-only" in out["error"]


# ---------- propose_vote / cast_vote ----------------------------------

def test_propose_vote_returns_unique_monotonic_vote_ids():
    store = ArtifactStore()
    a = _payload(store.dispatch("propose_vote", {
        "question": "Q1", "options": ["yes", "no"],
    }, caller="M"))
    b = _payload(store.dispatch("propose_vote", {
        "question": "Q2", "options": ["a", "b"],
    }, caller="M"))
    assert a["vote_id"] == "v1"
    assert b["vote_id"] == "v2"
    assert set(store.votes.keys()) == {"v1", "v2"}


def test_propose_vote_rejects_too_few_options():
    store = ArtifactStore()
    out = _payload(store.dispatch("propose_vote", {
        "question": "Q", "options": ["only-one"],
    }, caller="M"))
    assert "error" in out


def test_cast_vote_records_ballot_and_supports_overwrite():
    store = ArtifactStore()
    vid = _payload(store.dispatch("propose_vote", {
        "question": "Q", "options": ["yes", "no"],
    }, caller="M"))["vote_id"]
    store.drain_events()
    store.dispatch("cast_vote", {"vote_id": vid, "option": "yes"}, caller="A")
    store.dispatch("cast_vote", {"vote_id": vid, "option": "no"}, caller="A")
    vote = store.votes[vid]
    assert vote.ballots == {"A": ("no", None)}, "second cast overwrites first"


def test_cast_vote_rejects_unknown_vote_id():
    store = ArtifactStore()
    out = _payload(store.dispatch(
        "cast_vote", {"vote_id": "nope", "option": "yes"}, caller="A",
    ))
    assert "error" in out


def test_cast_vote_rejects_unknown_option():
    store = ArtifactStore()
    vid = _payload(store.dispatch("propose_vote", {
        "question": "Q", "options": ["yes", "no"],
    }, caller="M"))["vote_id"]
    out = _payload(store.dispatch(
        "cast_vote", {"vote_id": vid, "option": "maybe"}, caller="A",
    ))
    assert "error" in out


# ---------- finalize_artifact -----------------------------------------

def test_finalize_artifact_seals_and_is_idempotent():
    store = ArtifactStore()
    first = _payload(store.dispatch("finalize_artifact", {
        "decision": "采纳", "rationale": "OK",
    }, caller="M"))
    assert first == {"ok": True}
    assert store.finalized is True
    assert store.final_decision == "采纳"

    second = _payload(store.dispatch("finalize_artifact", {
        "decision": "改口", "rationale": "x",
    }, caller="M"))
    assert "error" in second
    assert "already finalized" in second["error"]
    assert store.final_decision == "采纳"


def test_finalize_artifact_requires_decision_and_rationale():
    store = ArtifactStore()
    no_dec = _payload(store.dispatch(
        "finalize_artifact", {"rationale": "x"}, caller="M",
    ))
    no_rat = _payload(store.dispatch(
        "finalize_artifact", {"decision": "y"}, caller="M",
    ))
    assert "error" in no_dec
    assert "error" in no_rat
    assert store.finalized is False


# ---------- tool_owners ACL -------------------------------------------

def test_build_tool_defs_filters_by_tool_owners_acl():
    """`tool_owners` restriction: unauthorized caller's `build_tool_defs` omits tool —
    LLM never sees it in schema."""
    store = ArtifactStore(tool_owners={
        "finalize_artifact": ["M"],
        "propose_vote": ["M"],
    })
    moderator_defs = {d["function"]["name"] for d in store.build_tool_defs("M")}
    member_defs = {d["function"]["name"] for d in store.build_tool_defs("A")}
    assert "finalize_artifact" in moderator_defs
    assert "propose_vote" in moderator_defs
    assert "finalize_artifact" not in member_defs
    assert "propose_vote" not in member_defs
    # undeclared tools default to visible to all agents
    assert "read_artifact" in member_defs
    assert "cast_vote" in member_defs


def test_dispatch_blocks_caller_not_in_tool_owners():
    """ACL is two-layer defense: even if LLM fabricates tool_call, dispatch blocks it."""
    store = ArtifactStore(tool_owners={"finalize_artifact": ["M"]})
    out = _payload(store.dispatch(
        "finalize_artifact",
        {"decision": "x", "rationale": "y"},
        caller="A",
    ))
    assert "error" in out
    assert "not in tool_owners" in out["error"]
    assert store.finalized is False


# ---------- drain / unknown tool / render -----------------------------

def test_drain_events_returns_and_clears_buffered_events():
    store = ArtifactStore(initial_sections=["x"])
    store.dispatch("write_section", {"name": "x", "content": "1"}, caller="A")
    store.dispatch("write_section", {"name": "x", "content": "2"}, caller="A")
    first = store.drain_events()
    second = store.drain_events()
    assert len(first) == 2
    assert second == []


def test_dispatch_unknown_tool_returns_error():
    store = ArtifactStore()
    out = _payload(store.dispatch("nope", {}, caller="A"))
    assert "error" in out
    assert "Unknown artifact tool" in out["error"]


def test_artifact_tool_names_constant_matches_handlers():
    """`ARTIFACT_TOOL_NAMES` is how `scenario.py` routes artifact tools;
    add/remove artifact tool without syncing constant fails here immediately."""
    store = ArtifactStore()
    for name in ARTIFACT_TOOL_NAMES:
        out = store.dispatch(name, {}, caller="A")
        payload = json.loads(out)
        assert payload != {"error": f"Unknown artifact tool: {name}"}, (
            f"{name} listed in ARTIFACT_TOOL_NAMES but has no dispatch handler"
        )


def test_render_includes_sections_votes_and_final_decision():
    """`--save-artifact` on-disk format locked: consumers expect markdown like
    `## <section>\n<body>` + `## Votes` block + `## Final Decision`."""
    store = ArtifactStore(initial_sections=["数据"])
    store.sections["数据"] = "value"
    vid = _payload(store.dispatch(
        "propose_vote", {"question": "Q?", "options": ["yes", "no"]}, caller="M",
    ))["vote_id"]
    store.dispatch("cast_vote", {"vote_id": vid, "option": "yes"}, caller="A")
    store.dispatch("finalize_artifact", {
        "decision": "采纳", "rationale": "因为",
    }, caller="M")
    md = store.render()
    assert "## 数据\nvalue" in md
    assert "## Votes" in md
    assert f"### {vid}: Q?" in md
    assert "yes: 1 (A)" in md
    assert "no: 0" in md
    assert "## Final Decision" in md
    assert "**采纳**" in md
    assert "因为" in md


@pytest.mark.parametrize("tool", sorted(ARTIFACT_TOOL_NAMES))
def test_each_artifact_tool_appears_in_build_tool_defs_when_unrestricted(tool):
    """Store without ACL exposes all 6 artifact tools to any caller."""
    store = ArtifactStore()
    names = {d["function"]["name"] for d in store.build_tool_defs("anyone")}
    assert tool in names

"""metrics/trajectory.py unit layer: 5 closure-factory metric + math helpers.

The goal of the test is not to "prove that the Levenshtein algorithm itself is correct", but to weld it to death:
  ① Factory-produced callable accepts (Doc, Response) protocol
  ② Pull data from doc.metadata['trajectory'] / doc.metadata (phase 5 contract coupling point)
  ③ Boundary (trajectory missing/gold empty/double empty vacuous match) goes 0/1 gracefully downgraded
  ④ Numerical correctness on known toy data (perfect / partial / wrong / garbage four states)
  ⑤ wrong_decision story can be reproduced on synthetic data: tool is all correct but task_success=0"""

from __future__ import annotations

from evals.api import Doc, Response
from evals.metrics.trajectory import (
    argument_correctness,
    levenshtein,
    multiset_f1,
    normalized_lev_match,
    predicate_decision_in_options,
    predicate_speakers_covered,
    task_success,
    tool_call_set_f1,
    trajectory_coverage,
    trajectory_match,
)


# ---------- Math helpers (10 items) -----------------------------------------------

def test_multiset_f1_double_empty_is_one():
    """Double empty set → vacuous match 1.0 (no requirement = perfect score)."""
    assert multiset_f1([], []) == 1.0


def test_multiset_f1_one_empty_is_zero():
    """One side is empty and the other is not empty → 0.0 (precision or recall must be 0)."""
    assert multiset_f1(["a"], []) == 0.0
    assert multiset_f1([], ["a"]) == 0.0


def test_multiset_f1_perfect():
    """multiset is exactly equivalent → 1.0; duplicate elements counter can be caught."""
    assert multiset_f1(["a", "b", "a"], ["b", "a", "a"]) == 1.0


def test_multiset_f1_partial_known():
    """pred=[a,b,c] gold=[a,b,d] → TP=2，p=r=2/3 → F1=2/3."""
    f1 = multiset_f1(["a", "b", "c"], ["a", "b", "d"])
    assert abs(f1 - 2 / 3) < 1e-9


def test_levenshtein_empty():
    assert levenshtein([], []) == 0
    assert levenshtein(["a", "b"], []) == 2
    assert levenshtein([], ["x", "y", "z"]) == 3


def test_levenshtein_single_substitute():
    """[a,b,c] → [a,X,c] distance 1 (one substitution)."""
    assert levenshtein(["a", "b", "c"], ["a", "X", "c"]) == 1


def test_levenshtein_swap_costs_two():
    """Neighbor exchange = 2 ops (no transposition; consistent with naive Levenshtein)."""
    assert levenshtein(["a", "b"], ["b", "a"]) == 2


def test_normalized_lev_match_double_empty_is_one():
    assert normalized_lev_match([], []) == 1.0


def test_normalized_lev_match_identical():
    assert normalized_lev_match(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_normalized_lev_match_completely_different():
    """Same length but completely different → 1 - 3/3 = 0.0."""
    assert normalized_lev_match(["a", "b", "c"], ["x", "y", "z"]) == 0.0


# ---------- closure factory + Doc/Response protocol (13 items)---------------------

def _doc(metadata: dict, doc_id: str = "d1") -> Doc:
    return Doc(id=doc_id, input="topic", target=None, metadata=metadata)


_RESP = Response(doc_id="d1")


def test_task_success_predicate_true():
    pred = lambda _doc: True  # noqa: E731
    assert task_success(pred)(_doc({}), _RESP) == 1.0


def test_task_success_predicate_false():
    pred = lambda _doc: False  # noqa: E731
    assert task_success(pred)(_doc({}), _RESP) == 0.0


def test_task_success_swallows_predicate_exception():
    """Predicate bugs should not blow up the entire batch - conservatively count 0."""
    def boom(_d): raise RuntimeError("boom")
    assert task_success(boom)(_doc({}), _RESP) == 0.0


def test_tool_call_set_f1_perfect():
    gold = [{"tool": "t1", "caller": "A"}, {"tool": "t2", "caller": "B"}]
    pred = [{"tool": "t1", "caller": "A"}, {"tool": "t2", "caller": "B"}]
    d = _doc({"gold_tool_calls": gold, "trajectory": {"tool_calls": pred}})
    assert tool_call_set_f1()(d, _RESP) == 1.0


def test_tool_call_set_f1_partial():
    """3-elem multiset, pred only hits 2 → F1 = 2*2/3*2/3 / (2/3+2/3) = 2/3."""
    gold = [
        {"tool": "t1", "caller": "A"},
        {"tool": "t2", "caller": "B"},
        {"tool": "t3", "caller": "C"},
    ]
    pred = [
        {"tool": "t1", "caller": "A"},
        {"tool": "t2", "caller": "B"},
        {"tool": "t9", "caller": "Z"},
    ]
    d = _doc({"gold_tool_calls": gold, "trajectory": {"tool_calls": pred}})
    assert abs(tool_call_set_f1()(d, _RESP) - 2 / 3) < 1e-9


def test_argument_correctness_subset_match():
    """gold only pins {name='X'}, pred adds {content='...'} → should be judged as hit (⊆ subset matching)."""
    gold = [{"tool": "write", "caller": "A", "arguments": {"name": "X"}}]
    pred = [{"tool": "write", "caller": "A", "arguments": {"name": "X", "content": "long"}}]
    d = _doc({"gold_tool_calls": gold, "trajectory": {"tool_calls": pred}})
    assert argument_correctness()(d, _RESP) == 1.0


def test_argument_correctness_value_mismatch_misses():
    gold = [{"tool": "write", "arguments": {"name": "X"}}]
    pred = [{"tool": "write", "arguments": {"name": "Y"}}]
    d = _doc({"gold_tool_calls": gold, "trajectory": {"tool_calls": pred}})
    assert argument_correctness()(d, _RESP) == 0.0


def test_argument_correctness_empty_gold_is_one():
    """No gold requirement → 1.0 (consistent with multiset_f1 double-null convention)."""
    d = _doc({"gold_tool_calls": [], "trajectory": {"tool_calls": []}})
    assert argument_correctness()(d, _RESP) == 1.0


def test_trajectory_match_identical_seq():
    d = _doc({
        "gold_tool_seq": ["a", "b", "c"],
        "trajectory": {"tool_seq": ["a", "b", "c"]},
    })
    assert trajectory_match()(d, _RESP) == 1.0


def test_trajectory_match_one_substitution():
    """[a,b,c] vs [a,X,c]：lev=1, max=3 → 1 - 1/3 = 2/3."""
    d = _doc({
        "gold_tool_seq": ["a", "b", "c"],
        "trajectory": {"tool_seq": ["a", "X", "c"]},
    })
    assert abs(trajectory_match()(d, _RESP) - 2 / 3) < 1e-9


def test_trajectory_coverage_callers_full():
    """required = (cast_vote, A) ∪ (cast_vote, B); pred covers both → 1.0."""
    d = _doc({
        "required_callers": {"cast_vote": ["A", "B"]},
        "trajectory": {"tool_calls": [
            {"tool": "cast_vote", "caller": "A"},
            {"tool": "cast_vote", "caller": "B"},
        ]},
    })
    assert trajectory_coverage(kind="callers")(d, _RESP) == 1.0


def test_trajectory_coverage_callers_partial():
    """4 requests, pred only 1 → 1/4."""
    d = _doc({
        "required_callers": {"cast_vote": ["A", "B", "C", "D"]},
        "trajectory": {"tool_calls": [{"tool": "cast_vote", "caller": "A"}]},
    })
    assert trajectory_coverage(kind="callers")(d, _RESP) == 0.25


def test_trajectory_coverage_speakers_kind():
    """kind='speakers': extract speakers from transcript and compare them with expected_speakers.

    Starting from §16, the speaker entry in the transcript must contain an explicit `type=="speaker"` tag."""
    d = _doc({
        "expected_speakers": ["前端", "后端", "PM"],
        "trajectory": {"transcript": [
            {"type": "speaker", "speaker": "前端", "content": "..."},
            {"type": "speaker", "speaker": "PM", "content": "..."},
        ]},
    })
    assert abs(trajectory_coverage(kind="speakers")(d, _RESP) - 2 / 3) < 1e-9


# ---------- predicates (4 items) --------------------------------------------------

def test_predicate_decision_in_options_pass():
    d = _doc({
        "expected_decision_options": ["保留", "关停"],
        "trajectory": {"artifact": {"x": "y"}, "decision": "关停"},
    })
    assert predicate_decision_in_options(d) is True


def test_predicate_decision_in_options_wrong_decision():
    """wrong_decision Story core: finalize is complete but decision is not in the whitelist → False."""
    d = _doc({
        "expected_decision_options": ["保留", "关停"],
        "trajectory": {"artifact": {"x": "y"}, "decision": "暂缓"},
    })
    assert predicate_decision_in_options(d) is False


def test_predicate_decision_in_options_no_artifact():
    """artifact missing → False (finalize has not been adjusted)."""
    d = _doc({
        "expected_decision_options": ["保留", "关停"],
        "trajectory": {"artifact": {}, "decision": "关停"},
    })
    assert predicate_decision_in_options(d) is False


def test_predicate_speakers_covered_perfect():
    d = _doc({
        "expected_speakers": ["前端", "后端", "PM"],
        "trajectory": {
            "success": True,
            "transcript": [
                {"type": "speaker", "speaker": "前端", "content": "..."},
                {"type": "speaker", "speaker": "后端", "content": "..."},
                {"type": "speaker", "speaker": "PM", "content": "..."},
            ],
        },
    })
    assert predicate_speakers_covered(d) is True


def test_predicate_speakers_covered_warnings_kill_success():
    """Everyone speaks but success=False (warnings present) → False."""
    d = _doc({
        "expected_speakers": ["A", "B"],
        "trajectory": {
            "success": False,
            "transcript": [
                {"type": "speaker", "speaker": "A", "content": "..."},
                {"type": "speaker", "speaker": "B", "content": "..."},
            ],
        },
    })
    assert predicate_speakers_covered(d) is False


# ---------- Graceful degradation due to lack of data (3 items)---------------------------------------------

def test_metrics_handle_missing_trajectory_metadata():
    """The old doc does not have a trajectory field → Each metric degrades gracefully; no raise."""
    d = _doc({})
    # Gold is also missing → regarded as no requirement; vacant 1.0
    assert tool_call_set_f1()(d, _RESP) == 1.0
    assert argument_correctness()(d, _RESP) == 1.0
    assert trajectory_match()(d, _RESP) == 1.0
    assert trajectory_coverage()(d, _RESP) == 1.0


def test_metrics_garbage_traj_with_gold():
    """Gold has requirements but pred trajectory is empty → each metric falls to 0 / 0."""
    d = _doc({
        "gold_tool_calls": [{"tool": "t1", "caller": "A"}],
        "gold_tool_seq": ["t1"],
        "required_callers": {"t1": ["A"]},
        "trajectory": {"tool_calls": [], "tool_seq": []},
    })
    assert tool_call_set_f1()(d, _RESP) == 0.0
    assert argument_correctness()(d, _RESP) == 0.0
    assert trajectory_match()(d, _RESP) == 0.0
    assert trajectory_coverage()(d, _RESP) == 0.0


def test_trajectory_coverage_invalid_kind_raises():
    """kind is misspelled → fail-fast, not silently counts 0."""
    import pytest
    with pytest.raises(ValueError, match="trajectory_coverage"):
        trajectory_coverage(kind="bogus")

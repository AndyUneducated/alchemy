"""metrics/trajectory.py 单元层：5 个 closure-factory metric + 数学 helpers.

测试目标不是"证明 Levenshtein 算法本身正确"，而是焊死：
  ① 工厂生产的 callable 接受 (Doc, Response) 协议
  ② 从 doc.metadata['trajectory'] / doc.metadata 拉数据（phase 5 契约耦合点）
  ③ 边界（trajectory 缺失 / gold 空 / 双空 vacuous match）走 0/1 优雅降级
  ④ 已知玩具数据上的数值正确性（perfect / partial / wrong / garbage 四态）
  ⑤ wrong_decision 故事在合成数据上能被复刻：tool 全对但 task_success=0
"""

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


# ---------- 数学 helpers（10 条）-------------------------------------------

def test_multiset_f1_double_empty_is_one():
    """双空集 → vacuous match 1.0（无要求 = 满分）."""
    assert multiset_f1([], []) == 1.0


def test_multiset_f1_one_empty_is_zero():
    """一边空一边非空 → 0.0（precision 或 recall 必为 0）."""
    assert multiset_f1(["a"], []) == 0.0
    assert multiset_f1([], ["a"]) == 0.0


def test_multiset_f1_perfect():
    """multiset 完全等价 → 1.0；重复元素 counter 抓得到."""
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
    """[a,b,c] → [a,X,c] 距离 1（一次替换）."""
    assert levenshtein(["a", "b", "c"], ["a", "X", "c"]) == 1


def test_levenshtein_swap_costs_two():
    """相邻交换 = 2 ops（无 transposition；与朴素 Levenshtein 一致）."""
    assert levenshtein(["a", "b"], ["b", "a"]) == 2


def test_normalized_lev_match_double_empty_is_one():
    assert normalized_lev_match([], []) == 1.0


def test_normalized_lev_match_identical():
    assert normalized_lev_match(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_normalized_lev_match_completely_different():
    """长度同但全不同 → 1 - 3/3 = 0.0."""
    assert normalized_lev_match(["a", "b", "c"], ["x", "y", "z"]) == 0.0


# ---------- closure factory + Doc/Response 协议（13 条）---------------------

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
    """谓词 bug 不应把整 batch 拉爆——保守计 0."""
    def boom(_d): raise RuntimeError("boom")
    assert task_success(boom)(_doc({}), _RESP) == 0.0


def test_tool_call_set_f1_perfect():
    gold = [{"tool": "t1", "caller": "A"}, {"tool": "t2", "caller": "B"}]
    pred = [{"tool": "t1", "caller": "A"}, {"tool": "t2", "caller": "B"}]
    d = _doc({"gold_tool_calls": gold, "trajectory": {"tool_calls": pred}})
    assert tool_call_set_f1()(d, _RESP) == 1.0


def test_tool_call_set_f1_partial():
    """3-elem multiset，pred 只命中 2 → F1 = 2*2/3*2/3 / (2/3+2/3) = 2/3."""
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
    """gold 仅 pin {name='X'}，pred 加了 {content='...'} → 应判 hit（⊆ 子集匹配）."""
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
    """无 gold 要求 → 1.0（与 multiset_f1 双空规约一致）."""
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
    """required = (cast_vote, A) ∪ (cast_vote, B)；pred 涵盖两者 → 1.0."""
    d = _doc({
        "required_callers": {"cast_vote": ["A", "B"]},
        "trajectory": {"tool_calls": [
            {"tool": "cast_vote", "caller": "A"},
            {"tool": "cast_vote", "caller": "B"},
        ]},
    })
    assert trajectory_coverage(kind="callers")(d, _RESP) == 1.0


def test_trajectory_coverage_callers_partial():
    """4 名要求，pred 仅 1 名 → 1/4."""
    d = _doc({
        "required_callers": {"cast_vote": ["A", "B", "C", "D"]},
        "trajectory": {"tool_calls": [{"tool": "cast_vote", "caller": "A"}]},
    })
    assert trajectory_coverage(kind="callers")(d, _RESP) == 0.25


def test_trajectory_coverage_speakers_kind():
    """kind='speakers'：从 transcript 抽 speakers 与 expected_speakers 比对.

    §16 起 transcript 内 speaker entry 必含显式 `type=="speaker"` 标签.
    """
    d = _doc({
        "expected_speakers": ["前端", "后端", "PM"],
        "trajectory": {"transcript": [
            {"type": "speaker", "speaker": "前端", "content": "..."},
            {"type": "speaker", "speaker": "PM", "content": "..."},
        ]},
    })
    assert abs(trajectory_coverage(kind="speakers")(d, _RESP) - 2 / 3) < 1e-9


# ---------- predicates（4 条）----------------------------------------------

def test_predicate_decision_in_options_pass():
    d = _doc({
        "expected_decision_options": ["保留", "关停"],
        "trajectory": {"artifact": {"x": "y"}, "decision": "关停"},
    })
    assert predicate_decision_in_options(d) is True


def test_predicate_decision_in_options_wrong_decision():
    """wrong_decision 故事核心：finalize 完整但 decision 不在白名单 → False."""
    d = _doc({
        "expected_decision_options": ["保留", "关停"],
        "trajectory": {"artifact": {"x": "y"}, "decision": "暂缓"},
    })
    assert predicate_decision_in_options(d) is False


def test_predicate_decision_in_options_no_artifact():
    """artifact 缺 → False（finalize 没调过）."""
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
    """全员发言但 success=False（warnings present）→ False."""
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


# ---------- 缺数据优雅降级（3 条）------------------------------------------

def test_metrics_handle_missing_trajectory_metadata():
    """老 doc 没有 trajectory 字段 → 各 metric 优雅降级；无 raise."""
    d = _doc({})
    # gold 也都缺 → 视为无要求；vacuous 1.0
    assert tool_call_set_f1()(d, _RESP) == 1.0
    assert argument_correctness()(d, _RESP) == 1.0
    assert trajectory_match()(d, _RESP) == 1.0
    assert trajectory_coverage()(d, _RESP) == 1.0


def test_metrics_garbage_traj_with_gold():
    """gold 有要求但 pred trajectory 空 → 各 metric 都掉到 0 / 0."""
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
    """kind 拼写错 → fail-fast，不 silently 计 0."""
    import pytest
    with pytest.raises(ValueError, match="trajectory_coverage"):
        trajectory_coverage(kind="bogus")

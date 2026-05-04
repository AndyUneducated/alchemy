"""Phase 4 契约扩展锁定：Doc.target / SampleResult.artifacts.

围绕两个 dataclass 改动写"形状 + 默认 + 反序列化"三类断言，避免日后误把
`target` 收紧回 str 或抹掉 `artifacts` 字段——破坏 RAG / 未来 agent task 的
"语义诚实"与"per-sample 非标量产物分桶"两条约束.
"""

from __future__ import annotations

from dataclasses import asdict

from evals.api import Doc, SampleResult


def test_doc_target_accepts_none():
    """Doc.target = None 必须能直接构造（rag_retrieval 用例）."""
    d = Doc(id="r1", input="who is X?", target=None)
    assert d.target is None
    # asdict 不抛 + None 落到序列化里
    assert asdict(d)["target"] is None


def test_doc_target_default_is_none():
    """target 是 keyword-default，省略时为 None——`Doc(id, input)` 直接合法."""
    d = Doc(id="r2", input="q?")
    assert d.target is None


def test_doc_target_str_still_supported():
    """老 task 显式传 str 不破——sentiment / mt / qa_open 全部走这条."""
    d = Doc(id="s1", input="x", target="positive")
    assert d.target == "positive"


def test_sample_result_artifacts_default_empty_dict():
    """老调用点不传 artifacts 必须 default 到空 dict（向后兼容根基）."""
    sr = SampleResult(doc_id="x", prediction="p", target="t", metrics={"acc": 1.0})
    assert sr.artifacts == {}
    # asdict 也带上 artifacts 字段（落 samples.jsonl 不丢）
    assert asdict(sr)["artifacts"] == {}


def test_sample_result_artifacts_carries_non_scalar():
    """artifacts 装 list[str] / dict 等非标量——这是它存在的核心理由."""
    sr = SampleResult(
        doc_id="x",
        prediction="",
        target="",
        metrics={},
        artifacts={"pred_ids": ["d1", "d2"], "gold_ids": ["d2"]},
    )
    assert sr.artifacts["pred_ids"] == ["d1", "d2"]
    assert sr.artifacts["gold_ids"] == ["d2"]
    # JSONL 落盘也 ok（asdict 浅拷贝即可）
    assert asdict(sr)["artifacts"]["pred_ids"] == ["d1", "d2"]


def test_sample_result_metrics_still_only_scalar():
    """metrics 仍是 dict[str, float] 用法（类型不强制，但语义契约保留）."""
    sr = SampleResult(doc_id="x", prediction="p", target="t", metrics={"em": 1.0, "rouge_l": 0.42})
    assert all(isinstance(v, float) for v in sr.metrics.values())

"""Phase 4 contract lock: Doc.target/SampleResult.artifacts.

Write three types of assertions "shape + default + deserialization" around the two dataclasses to avoid misunderstandings in the future.
`target` tightens back to str or erases `artifacts` field - breaks RAG / agent task
There are two constraints: "semantic honesty" and "per-sample non-scalar product bucketing".

From phase 6 onwards, Usage / Response.usage / EvalResult.aggregated nested form of similar locking will be added."""

from __future__ import annotations

from dataclasses import asdict

from evals.api import Doc, EvalResult, Response, SampleResult, Usage


def test_doc_target_accepts_none():
    """Doc.target = None must be directly constructible (rag_retrieval use case)."""
    d = Doc(id="r1", input="who is X?", target=None)
    assert d.target is None
    # asdict does not throw + None falls into serialization
    assert asdict(d)["target"] is None


def test_doc_target_default_is_none():
    """target is keyword-default, and is None when omitted - `Doc(id, input)` is directly legal."""
    d = Doc(id="r2", input="q?")
    assert d.target is None


def test_doc_target_str_still_supported():
    """The old task's explicit transmission of str is not broken—sentiment / mt / qa_open all go this way."""
    d = Doc(id="s1", input="x", target="positive")
    assert d.target == "positive"


def test_sample_result_artifacts_default_empty_dict():
    """If you do not pass artifacts, you must default to an empty dict (the basis for inserting the schema without losing fields)."""
    sr = SampleResult(doc_id="x", prediction="p", target="t", metrics={"acc": 1.0})
    assert sr.artifacts == {}
    # asdict also brings the artifacts field (dropping samples.jsonl will not be lost)
    assert asdict(sr)["artifacts"] == {}


def test_sample_result_artifacts_carries_non_scalar():
    """artifacts holds non-scalars such as list[str] / dict - this is the core reason for its existence."""
    sr = SampleResult(
        doc_id="x",
        prediction="",
        target="",
        metrics={},
        artifacts={"pred_ids": ["d1", "d2"], "gold_ids": ["d2"]},
    )
    assert sr.artifacts["pred_ids"] == ["d1", "d2"]
    assert sr.artifacts["gold_ids"] == ["d2"]
    # JSONL is also ok (shallow copy of asdict is enough)
    assert asdict(sr)["artifacts"]["pred_ids"] == ["d1", "d2"]


def test_sample_result_metrics_still_only_scalar():
    """metrics still uses dict[str, float] (the type is not enforced, but the semantic contract is preserved)."""
    sr = SampleResult(doc_id="x", prediction="p", target="t", metrics={"em": 1.0, "rouge_l": 0.42})
    assert all(isinstance(v, float) for v in sr.metrics.values())


# ---------- phase 6: Usage / Response.usage / EvalResult.aggregated Nesting relaxation ----------

def test_usage_default_is_none_fields():
    """Both Usage fields default to None - ensuring that unfilled scenarios (MockLM / old ollama) do not explode the structure."""
    u = Usage()
    assert u.tokens_in is None
    assert u.tokens_out is None
    # asdict is not thrown + the two fields fall into serialization
    d = asdict(u)
    assert d == {"tokens_in": None, "tokens_out": None}


def test_response_usage_default_is_none():
    """`Response(doc_id, text)` must default to usage=None (minimum construction form - MockLM / score path does not report usage)."""
    r = Response(doc_id="x", text="hello")
    assert r.usage is None
    assert r.latency_ms is None
    # asdict also brings new fields (no loss when placing the order)
    d = asdict(r)
    assert d["usage"] is None
    assert d["latency_ms"] is None


def test_response_usage_carries_token_counts():
    """OllamaLM real path: usage install (tokens_in, tokens_out) and then embed Response."""
    r = Response(
        doc_id="x",
        text="hi",
        latency_ms=123.4,
        usage=Usage(tokens_in=10, tokens_out=5),
    )
    assert r.usage.tokens_in == 10
    assert r.usage.tokens_out == 5
    # asdict nested expansion
    d = asdict(r)
    assert d["usage"] == {"tokens_in": 10, "tokens_out": 5}
    assert d["latency_ms"] == 123.4


def test_eval_result_aggregated_accepts_nested_subgroup():
    """EvalResult.aggregated type relaxed to dict[str, Any] - supports phase 6 efficiency nested subgroups.
    audit §1.1 add cost_usd.mean, §1.2 add latency_ms.max, §1.5 tokens.total use int."""
    r = EvalResult(
        task="dummy",
        model="mock:gold",
        mode="run",
        n=1,
        aggregated={
            "accuracy": 1.0,
            "efficiency": {
                "latency_ms": {"mean": 1.0, "p50": 1.0, "p95": 1.0, "max": 1.0},
                "tokens_in": {"total": 0, "mean": 0.0},
                "tokens_out": {"total": 0, "mean": 0.0},
                "cost_usd": {"total": 0.0, "mean": 0.0},
            },
        },
        per_sample=(),
        run_id="r",
        created_at="2026-05-04T00:00:00",
        elapsed_ms=0.0,
    )
    assert r.aggregated["accuracy"] == 1.0
    assert r.aggregated["efficiency"]["latency_ms"]["p50"] == 1.0
    assert r.aggregated["efficiency"]["latency_ms"]["max"] == 1.0
    assert r.aggregated["efficiency"]["cost_usd"]["mean"] == 0.0
    # asdict nested deep copy (dropping result.json without losing any layers)
    d = asdict(r)
    assert d["aggregated"]["efficiency"]["cost_usd"]["total"] == 0.0
    assert d["aggregated"]["efficiency"]["tokens_in"]["total"] == 0  # int semantics reserved


def test_eval_result_aggregated_still_accepts_flat_only():
    """The score path aggregated is also legal for full tiling (nesting is not forced when there is no cross-cutting dim)."""
    r = EvalResult(
        task="t",
        model="m",
        mode="score",
        n=0,
        aggregated={"accuracy": 0.5, "f1_macro": 0.4},
        per_sample=(),
        run_id="r",
        created_at="2026-05-04T00:00:00",
        elapsed_ms=0.0,
    )
    assert "efficiency" not in r.aggregated
    assert r.aggregated["accuracy"] == 0.5

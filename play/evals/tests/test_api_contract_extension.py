"""Phase 4 契约扩展锁定：Doc.target / SampleResult.artifacts.

围绕两个 dataclass 改动写"形状 + 默认 + 反序列化"三类断言，避免日后误把
`target` 收紧回 str 或抹掉 `artifacts` 字段——破坏 RAG / 未来 agent task 的
"语义诚实"与"per-sample 非标量产物分桶"两条约束.

phase 6 起追加 Usage / Response.usage / EvalResult.aggregated 嵌套放宽的同类锁定.
"""

from __future__ import annotations

from dataclasses import asdict

from evals.api import Doc, EvalResult, Response, SampleResult, Usage


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


# ---------- phase 6: Usage / Response.usage / EvalResult.aggregated 嵌套放宽 ----------

def test_usage_default_is_none_fields():
    """Usage 两字段都默认 None——保证未填写场景（MockLM / 老 ollama）不爆构造."""
    u = Usage()
    assert u.tokens_in is None
    assert u.tokens_out is None
    # asdict 不抛 + 两字段落到序列化里
    d = asdict(u)
    assert d == {"tokens_in": None, "tokens_out": None}


def test_response_usage_default_is_none():
    """老调用点 `Response(doc_id, text)` 必须默认 usage=None（向后兼容根基）."""
    r = Response(doc_id="x", text="hello")
    assert r.usage is None
    assert r.latency_ms is None
    # asdict 也带上新字段（落盘不丢）
    d = asdict(r)
    assert d["usage"] is None
    assert d["latency_ms"] is None


def test_response_usage_carries_token_counts():
    """OllamaLM 真实路径：usage 装 (tokens_in, tokens_out) 再嵌入 Response."""
    r = Response(
        doc_id="x",
        text="hi",
        latency_ms=123.4,
        usage=Usage(tokens_in=10, tokens_out=5),
    )
    assert r.usage.tokens_in == 10
    assert r.usage.tokens_out == 5
    # asdict 嵌套展开
    d = asdict(r)
    assert d["usage"] == {"tokens_in": 10, "tokens_out": 5}
    assert d["latency_ms"] == 123.4


def test_eval_result_aggregated_accepts_nested_subgroup():
    """EvalResult.aggregated 类型放宽到 dict[str, Any]——支持 phase 6 efficiency 嵌套子组.
    audit §1.1 加 cost_usd.mean、§1.2 加 latency_ms.max、§1.5 tokens.total 用 int.
    """
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
    # asdict 嵌套深拷贝（落 result.json 不丢任何层）
    d = asdict(r)
    assert d["aggregated"]["efficiency"]["cost_usd"]["total"] == 0.0
    assert d["aggregated"]["efficiency"]["tokens_in"]["total"] == 0  # int 语义保留


def test_eval_result_aggregated_still_accepts_flat_only():
    """老 score 路径 aggregated 全平铺仍合法（向后兼容；schema 放宽不破老 result.json）."""
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

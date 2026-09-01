"""parse_model_spec unit test (zero net).

Only validate spec string → LM typemap. Does not actually call generate_until.

According to plan §2.5: 6 assertions, including openai / anthropic's respective explicit NotImplementedError——
The error messages of the two providers may drift separately (if anthropic supports it in the future), the respective locks will be more stable.

Plus 3 _build_task_with_optional_deps dispatch assertions (phase 3 CLI integrity patch,
score / run share the same helper, so only the helper's own behavior is locked, and monkeypatch on cmd_* is not repeated):
  - judge_model=None: Return to the original task without judge_lm
  - qa_open + judge_model: return the injected version of QAOpen(judge_lm=...)
  - Other task + judge_model: Immediate SystemExit (without touching LM)"""

from __future__ import annotations

import pytest

from evals import tasks  # noqa: F401 — trigger @register_task
from evals.cli import _build_task_with_optional_deps, parse_model_spec
from evals.models.mock import MockLM
from evals.models.ollama import OllamaLM
from evals.registry import get_task
from evals.tasks.qa_open import QAOpen


@pytest.fixture
def task():
    return get_task("sentiment_clf")


def test_parse_spec_mock_still_works(task):
    """The mock:* path of phase 1 does not return."""
    lm = parse_model_spec("mock:gold", task)
    assert isinstance(lm, MockLM)
    assert lm.name == "mock:gold"


def test_parse_spec_ollama_returns_ollama_lm(task):
    """ollama:<model> resolves to OllamaLM; name falls into `ollama:<model>` into EvalResult.model."""
    lm = parse_model_spec("ollama:qwen3.6:27b", task)
    assert isinstance(lm, OllamaLM)
    assert lm.name == "ollama:qwen3.6:27b"
    assert lm.model == "qwen3.6:27b"


def test_parse_spec_ollama_with_base_url_override(task, monkeypatch):
    """EVALS_OLLAMA_BASE_URL env can override the default base_url (no need to change the spec syntax)."""
    monkeypatch.setenv("EVALS_OLLAMA_BASE_URL", "http://other:11434")
    lm = parse_model_spec("ollama:qwen3.6:27b", task)
    assert isinstance(lm, OllamaLM)
    assert lm.base_url == "http://other:11434"


# ---------- @seed=K suffix (agent_sft phase 1 multi-seed wiring) ----------

def test_parse_spec_ollama_with_seed_suffix(task):
    """`ollama:<model>@seed=K` → OllamaLM(seed=K); name reserves the @seed=K suffix for EvalResult.model to distinguish."""
    lm = parse_model_spec("ollama:qwen3.5:9b-instruct@seed=42", task)
    assert isinstance(lm, OllamaLM)
    assert lm.model == "qwen3.5:9b-instruct"  # @seed= suffix stripped
    assert lm.seed == 42
    assert lm.name == "ollama:qwen3.5:9b-instruct@seed=42"  # But model_label remains


def test_parse_spec_ollama_without_seed_keeps_default_zero(task):
    """None @seed= suffix → OllamaLM default seed=0, name does not contain suffix (default form of bare spec)."""
    lm = parse_model_spec("ollama:qwen3.5:9b-instruct", task)
    assert lm.seed == 0
    assert lm.name == "ollama:qwen3.5:9b-instruct"
    assert "@seed=" not in lm.name


def test_parse_spec_ollama_with_seed_zero_explicit(task):
    """Explicit `@seed=0` also writes name (so that multi-seed bash loops with seed=0 can still be distinguished from bare spec)."""
    lm = parse_model_spec("ollama:qwen3.5:9b-instruct@seed=0", task)
    assert lm.seed == 0
    assert lm.name == "ollama:qwen3.5:9b-instruct@seed=0"


def test_parse_spec_invalid_seed_raises(task):
    """`@seed=abc` non-integer → ValueError (same fail-fast path as unknown provider)."""
    with pytest.raises(ValueError, match="invalid seed"):
        parse_model_spec("ollama:qwen3.5:9b@seed=abc", task)


def test_parse_spec_seed_suffix_on_mock_raises(task):
    """`mock:gold@seed=42` → ValueError; mock uses its own `mock:noisy:<noise>:<seed>` syntax."""
    with pytest.raises(ValueError, match="seed=K suffix"):
        parse_model_spec("mock:gold@seed=42", task)


def test_parse_spec_openai_explicit_not_implemented(task):
    """openai:* → Explicit NotImplementedError, the error message indicates that phase 3 is not enabled."""
    with pytest.raises(NotImplementedError, match="phase 3"):
        parse_model_spec("openai:gpt-4o-mini", task)


def test_parse_spec_anthropic_explicit_not_implemented(task):
    """anthropic:* → Explicit NotImplementedError (locked separately from openai, error messages may drift separately)."""
    with pytest.raises(NotImplementedError, match="phase 3"):
        parse_model_spec("anthropic:claude-3-haiku", task)


def test_parse_spec_unknown_provider_raises(task):
    """Unknown provider → ValueError (not to be confused with NotImplementedError)."""
    with pytest.raises(ValueError):
        parse_model_spec("weirdprovider:foo", task)


# ---------- _build_task_with_optional_deps dispatch (common to score/run) ----------

def test_build_task_no_judge_returns_plain_qa_open():
    """judge_model=None → Take get_task trivial construction, task._judge_lm is None."""
    t = _build_task_with_optional_deps("qa_open", judge_model_spec=None)
    assert isinstance(t, QAOpen)
    assert t._judge_lm is None


def test_build_task_with_judge_injects_judge_lm():
    """qa_open + judge_model spec → Rebuild the injected version of QAOpen(judge_lm=...)."""
    t = _build_task_with_optional_deps("qa_open", judge_model_spec="mock:gold")
    assert isinstance(t, QAOpen)
    assert t._judge_lm is not None


def test_build_task_judge_on_non_qa_open_raises_systemexit():
    """non qa_open + judge_model → SystemExit (fail-fast instead of silently ignored)."""
    with pytest.raises(SystemExit, match="qa_open|rag_qa"):
        _build_task_with_optional_deps("sentiment_clf", judge_model_spec="mock:gold")


# ---------- Phase 4 dispatch (RAG / safety parameter) ----------

from evals.tasks.rag_qa import RagQA  # noqa: E402
from evals.tasks.rag_retrieval import RagRetrieval  # noqa: E402
from evals.tasks.safety import Safety  # noqa: E402


def test_build_rag_retrieval_with_vdb_injects_retrieve_fn():
    """rag_retrieval + --vdb → inject retrieve_fn(callable)."""
    t = _build_task_with_optional_deps(
        "rag_retrieval", vdb="/tmp/fake_vdb", retrieve_top_k=3, retrieve_mode="dense",
    )
    assert isinstance(t, RagRetrieval)
    assert t._retrieve_fn is not None
    assert callable(t._retrieve_fn)
    assert t._top_k == 3


def test_build_rag_retrieval_without_vdb_returns_naked_task():
    """rag_retrieval None --vdb (score path usage) → task body, retrieve_fn=None."""
    t = _build_task_with_optional_deps("rag_retrieval")
    assert isinstance(t, RagRetrieval)
    assert t._retrieve_fn is None


def test_build_rag_retrieval_with_judge_raises_systemexit():
    """rag_retrieval + --judge-model → SystemExit (rag_retrieval has no LM-side output to judge)."""
    with pytest.raises(SystemExit, match="rag_retrieval"):
        _build_task_with_optional_deps("rag_retrieval", judge_model_spec="mock:gold")


def test_build_rag_qa_with_vdb_and_judge_injects_both():
    """rag_qa + --vdb + --judge-model → retrieve_fn + judge_lm double injection."""
    t = _build_task_with_optional_deps(
        "rag_qa",
        vdb="/tmp/fake_vdb",
        judge_model_spec="mock:gold",
    )
    assert isinstance(t, RagQA)
    assert t._retrieve_fn is not None
    assert t._judge_lm is not None


def test_build_rag_qa_without_judge_lexical_only():
    """rag_qa + --vdb none --judge-model → lexical baseline only."""
    t = _build_task_with_optional_deps("rag_qa", vdb="/tmp/fake_vdb")
    assert isinstance(t, RagQA)
    assert t._retrieve_fn is not None
    assert t._judge_lm is None


def test_build_qa_open_with_vdb_raises_systemexit():
    """qa_open + --vdb → SystemExit (qa_open does not connect to RAG flag)."""
    with pytest.raises(SystemExit, match="qa_open|rag"):
        _build_task_with_optional_deps("qa_open", vdb="/tmp/fake_vdb")


def test_build_sentiment_clf_with_vdb_raises_systemexit():
    """Non-RAG task + --vdb → SystemExit(fail-fast)."""
    with pytest.raises(SystemExit, match="rag"):
        _build_task_with_optional_deps("sentiment_clf", vdb="/tmp/fake_vdb")


def test_build_safety_with_judge_injects_judge_lm():
    """safety + --judge-model → return injected version of Safety(judge_lm=...)."""
    t = _build_task_with_optional_deps("safety", judge_model_spec="mock:gold")
    assert isinstance(t, Safety)
    assert t._judge_lm is not None


def test_build_safety_with_vdb_raises_systemexit():
    """safety + --vdb → SystemExit (safety non-retrieval task)."""
    with pytest.raises(SystemExit, match="safety|retrieval"):
        _build_task_with_optional_deps("safety", vdb="/tmp/fake_vdb")


# ---------- Phase 8 IAA dispatch (iaa_nominal / iaa_ordinal no new flag, same shape as sentiment_clf)

from evals.tasks.iaa_nominal import IaaNominal  # noqa: E402
from evals.tasks.iaa_ordinal import IaaOrdinal  # noqa: E402


def test_build_iaa_nominal_naked_returns_task():
    """iaa_nominal has no flag → returns to naked task (IAA task does not connect to judge / vdb, and has the same shape as sentiment_clf)."""
    t = _build_task_with_optional_deps("iaa_nominal")
    assert isinstance(t, IaaNominal)


def test_build_iaa_ordinal_naked_returns_task():
    """iaa_ordinal Same as iaa_nominal."""
    t = _build_task_with_optional_deps("iaa_ordinal")
    assert isinstance(t, IaaOrdinal)


def test_build_iaa_with_judge_raises_systemexit():
    """iaa_nominal + --judge-model → SystemExit (IAA task does not accept judge; judge LM as annotator
    Teaching narrative deferred same as phase 5 §8 ADR)."""
    with pytest.raises(SystemExit, match="judge"):
        _build_task_with_optional_deps("iaa_nominal", judge_model_spec="mock:gold")


def test_build_iaa_with_vdb_raises_systemexit():
    """iaa_ordinal + --vdb → SystemExit (IAA non-retrieval task)."""
    with pytest.raises(SystemExit, match="vdb|rag"):
        _build_task_with_optional_deps("iaa_ordinal", vdb="/tmp/fake_vdb")


# ---------- Phase 6 _fmt_kv nested printing (CLI landing point for aggregated efficiency subgroup) ----------

from evals.cli import _fmt_kv, _fmt_row  # noqa: E402


def test_fmt_kv_flat_scalar_unchanged():
    """Old phase 1-5 tiling index (k, float) → "k=v.4f" (same as original _fmt_row bytes)."""
    assert _fmt_kv("accuracy", 0.875) == ["accuracy=0.8750"]
    assert _fmt_kv("f1_macro", 1.0) == ["f1_macro=1.0000"]


def test_fmt_kv_nested_subgroup_uses_dot_path():
    """phase 6 nesting: efficiency.latency_ms.p50=... form (HELM-style path, cross-run friendly)."""
    out = _fmt_kv("efficiency", {"latency_ms": {"p50": 12.5, "p95": 50.0}})
    assert "efficiency.latency_ms.p50=12.5000" in out
    assert "efficiency.latency_ms.p95=50.0000" in out


def test_fmt_row_includes_efficiency_keys_when_present():
    """_fmt_row end-to-end for index row: aggregated with efficiency subgroup also prints dot-path form."""
    row = {
        "run_id": "r1",
        "task": "sentiment_clf",
        "mode": "run",
        "model": "mock:gold",
        "n": 30,
        "aggregated": {
            "accuracy": 1.0,
            "efficiency": {"latency_ms": {"p50": 0.0, "p95": 0.0, "mean": 0.0}},
        },
    }
    s = _fmt_row(row)
    assert "accuracy=1.0000" in s
    assert "efficiency.latency_ms.p50=0.0000" in s


# ---------- audit §1.7: Nested subgroups all 0 collapsed to <not measured> ----------

from evals.cli import _is_all_zero_nested, _print_aggregated  # noqa: E402


def test_is_all_zero_nested_true_for_nested_zeros():
    """All 0 nesting (mock path efficiency subgroup shape) → True."""
    eff = {
        "latency_ms": {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0},
        "tokens_in": {"total": 0, "mean": 0.0},
        "tokens_out": {"total": 0, "mean": 0.0},
        "cost_usd": {"total": 0.0, "mean": 0.0},
    }
    assert _is_all_zero_nested(eff) is True


def test_is_all_zero_nested_false_when_any_nonzero():
    """Any leaf is non-0 → False (real LM does not fold when it runs out of true data)."""
    eff = {
        "latency_ms": {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0},
        "tokens_in": {"total": 178, "mean": 59.33},  # Not 0
        "tokens_out": {"total": 0, "mean": 0.0},
        "cost_usd": {"total": 0.0, "mean": 0.0},
    }
    assert _is_all_zero_nested(eff) is False


def test_print_aggregated_collapses_zero_efficiency(capsys):
    """mock path efficiency all 0 → collapsed into a single line `<not measured (no LM signal)>`,
    Replaced the visually misleading 11-line 0 placeholder ("latency=0.0000" looks like ultra-low latency rather than unmeasured)."""
    agg = {
        "accuracy": 1.0,
        "efficiency": {
            "latency_ms": {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0},
            "tokens_in": {"total": 0, "mean": 0.0},
            "tokens_out": {"total": 0, "mean": 0.0},
            "cost_usd": {"total": 0.0, "mean": 0.0},
        },
    }
    _print_aggregated(agg)
    out = capsys.readouterr().out
    assert "accuracy" in out and "1.0000" in out
    assert "<not measured" in out
    assert "efficiency.latency_ms" not in out  # No longer expand 11 lines


def test_print_aggregated_expands_nonzero_efficiency(capsys):
    """real LM does not collapse when running out real data, and expands according to dot-path (the signal path is not broken)."""
    agg = {
        "accuracy": 1.0,
        "efficiency": {
            "latency_ms": {"mean": 899.3, "p50": 687.0, "p95": 1274.7, "max": 1339.9},
            "tokens_in": {"total": 178, "mean": 59.33},
            "tokens_out": {"total": 12, "mean": 4.0},
            "cost_usd": {"total": 0.000152, "mean": 0.0000507},
        },
    }
    _print_aggregated(agg)
    out = capsys.readouterr().out
    assert "<not measured" not in out
    assert "efficiency.latency_ms.p50" in out
    assert "efficiency.latency_ms.max" in out  # audit §1.2 New
    assert "efficiency.cost_usd.mean" in out  # audit §1.1 New


def test_print_aggregated_does_not_collapse_zero_task_metric(capsys):
    """The task's own indicator is not folded even if it is 0 (accuracy=0 is a real signal, not "not measured");
    Folding only takes effect on nested subgroups."""
    agg = {"accuracy": 0.0, "f1_macro": 0.0}
    _print_aggregated(agg)
    out = capsys.readouterr().out
    assert "accuracy" in out and "0.0000" in out
    assert "<not measured" not in out


# ---------- phase 7 audit P1: trait protocol/content class all 0, not folded ----------

from evals.cli import _should_fold_when_all_zero  # noqa: E402


def test_trait_efficiency_folds_when_all_zero():
    """efficiency is call class, trait True, all 0 folds (mock path is equivalent to "unmeasured")."""
    assert _should_fold_when_all_zero("efficiency") is True


def test_trait_unknown_dim_defaults_to_fold():
    """Unregistered dim defaults to True folding (new cross-cutting must explicitly declare trait=False in its own module if it wants to exit folding)."""
    assert _should_fold_when_all_zero("nonexistent_dim") is True


# After wave 3 (DECISIONS §7.2) withdraws safety cross-cutting AOP, safety no longer appears in
# aggregated top-level nested subgroup (safety task own task-specific 4 stat tile),
# The folding protocol is no longer applicable to safety - the safety non-folding test group of the original phase 7 audit P1 has been deleted.


# ---------- wave 3 §7.3: efficiency.judge nested two-level folding ----------

def test_print_aggregated_folds_efficiency_judge_subgroup_when_all_zero(capsys):
    """The efficiency top level is not all 0 (the task part has a numerical value) but the judge subgroup is all 0 → the judge subgroup is collapsed separately
    `efficiency.judge: <not measured>` Single line; does not affect task part dot-path rendering.

    Scenario: task receives judge_lm but judge LM does not report latency/tokens (such as mock), and task LM
    Normally report efficiency."""
    agg = {
        "efficiency": {
            "latency_ms": {"mean": 100.0, "p50": 100.0, "p95": 100.0, "max": 100.0},
            "tokens_in": {"total": 100, "mean": 10.0},
            "tokens_out": {"total": 50, "mean": 5.0},
            "cost_usd": {"total": 0.001, "mean": 0.0001},
            "judge": {  # All 0: mock judge / task is also in this state when judge_lm is not connected.
                "latency_ms": {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0},
                "tokens_in": {"total": 0, "mean": 0.0},
                "tokens_out": {"total": 0, "mean": 0.0},
                "cost_usd": {"total": 0.0, "mean": 0.0},
            },
        },
    }
    _print_aggregated(agg)
    out = capsys.readouterr().out
    # The task part is still expanded by dot-path
    assert "efficiency.latency_ms.mean" in out
    assert "efficiency.tokens_in.total" in out
    # judge subgroup single line folding
    assert "efficiency.judge" in out
    assert "<not measured" in out
    # dot-path of judge subgroup should not be expanded (only one line after collapse)
    assert "efficiency.judge.latency_ms" not in out
    assert "efficiency.judge.tokens_in" not in out


def test_print_aggregated_does_not_fold_efficiency_judge_when_nonzero(capsys):
    """efficiency.judge subgroup has numeric value → dot-path fully expanded (not collapsed)."""
    agg = {
        "efficiency": {
            "latency_ms": {"mean": 100.0, "p50": 100.0, "p95": 100.0, "max": 100.0},
            "tokens_in": {"total": 100, "mean": 10.0},
            "tokens_out": {"total": 50, "mean": 5.0},
            "cost_usd": {"total": 0.001, "mean": 0.0001},
            "judge": {
                "latency_ms": {"mean": 200.0, "p50": 200.0, "p95": 200.0, "max": 200.0},
                "tokens_in": {"total": 50, "mean": 25.0},
                "tokens_out": {"total": 10, "mean": 5.0},
                "cost_usd": {"total": 0.0005, "mean": 0.00025},
            },
        },
    }
    _print_aggregated(agg)
    out = capsys.readouterr().out
    # The judge subgroup is fully expanded
    assert "efficiency.judge.latency_ms.mean" in out
    assert "efficiency.judge.tokens_in.total" in out
    assert "<not measured" not in out  # The whole is not folded


# ---------- phase 7 audit P2: None placeholder → CLI <n/a> rendering ----------

def test_fmt_kv_none_value_renders_as_na():
    """None placeholder (safety judge_safety_score is not connected to judge_lm) → `<n/a>` instead of 0.0000.
    Explicitly distinguished from "true 0" (as in the refusal_rate=0 garbage path)."""
    assert _fmt_kv("judge_safety_score", None) == ["judge_safety_score=<n/a>"]


def test_fmt_kv_nested_none_in_subgroup_renders_as_na():
    """None in nested subgroups are also rendered as `<n/a>`, and the dot-path is preserved."""
    out = _fmt_kv("safety", {"refusal_rate": 0.5, "judge_safety_score": None})
    assert "safety.refusal_rate=0.5000" in out
    assert "safety.judge_safety_score=<n/a>" in out


def test_print_aggregated_renders_safety_with_mixed_real_and_na(capsys):
    """End-to-end: safety subgroup contains 1 true + 3 None, CLI shows true + <n/a> mixed."""
    agg = {
        "safety": {
            "refusal_rate": 0.6667,
            "jailbreak_success_rate": None,
            "over_refusal_rate": None,
            "judge_safety_score": None,
        },
    }
    _print_aggregated(agg)
    out = capsys.readouterr().out
    assert "safety.refusal_rate" in out and "0.6667" in out
    assert "safety.jailbreak_success_rate" in out and "<n/a>" in out
    assert "safety.judge_safety_score" in out and "<n/a>" in out

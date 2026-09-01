"""run path e2e + **dual-mode equivalence test**.

test_run_gold_equals_score_perfect is the anchor of the entire architecture:
It puts the sentence "score and run share the task.process_results / aggregation tail section"
Became an executable assert. Green = The two paths truly intersect at the task layer, and there are no branches secretly diverging.

Starting from phase 6, the run mode additionally injects `aggregated["efficiency"]` to cross-cut subgroups (score path has no LM
(so it is not injected), so the parity assertion is changed to "subsets of task-specific keys are equal" + explicit assertion
score does not contain the efficiency subgroup; architectural equivalence is preserved: efficiency is the cross-cutting AOP delta,
It is not a fork at the end of the task path."""

from __future__ import annotations

from pathlib import Path

from evals.models.mock import MockLM
from evals.runner import evaluate_run, evaluate_score
from evals.tasks.mt import MT
from evals.tasks.sentiment_clf import SentimentClf

PRED_DIR = Path(__file__).resolve().parent.parent / "data" / "sentiment" / "predictions"
MT_PRED_DIR = Path(__file__).resolve().parent.parent / "data" / "mt" / "predictions"


# wave 3 (DECISIONS §7.2) withdraws safety cross-cutting AOP; only efficiency remains cross-cutting
# Nested subgroups (infrastructure indicators, in the same spirit as the HELM efficiency dimension, consistent with the industry).
_CROSS_CUTTING_SUBGROUPS = {"efficiency"}


def _task_agg(agg: dict) -> dict:
    """Strip away cross-cutting subgroups, leaving only task-specific top-level metrics."""
    return {k: v for k, v in agg.items() if k not in _CROSS_CUTTING_SUBGROUPS}


def _task_metrics(metrics: dict) -> dict:
    """Strip away the cross-cutting nested subgroup (efficiency/safety) of phase 7 §7.D nested, leaving only task-specific metrics.
    The run path writes the cross-cutting subgroup (0 placeholder or real value), the score path only writes the content class subgroup (safety),
    After stripping off the task layer parity still exists."""
    return {k: v for k, v in metrics.items() if k not in _CROSS_CUTTING_SUBGROUPS}


def test_run_gold_runs_full_accuracy():
    task = SentimentClf()
    docs = list(task.docs())
    lm = MockLM(mode="gold", docs=docs)
    r = evaluate_run(task, lm)
    assert r.mode == "run"
    assert r.model == "mock:gold"
    assert r.n == 30
    assert _task_agg(r.aggregated) == {"accuracy": 1.0, "f1_macro": 1.0, "cohens_kappa": 1.0}
    # Phase 6 cross-cutting subgroups: MockLM does not report → the subgroup key values ​​​​are all 0 (schema always exists)
    assert "efficiency" in r.aggregated
    assert r.aggregated["efficiency"]["latency_ms"]["p50"] == 0.0
    assert r.aggregated["efficiency"]["cost_usd"]["total"] == 0.0


def test_run_gold_equals_score_perfect():
    """Architecture commitment: evaluate_run(task, MockLM(gold)) ≡ evaluate_score(task, perfect.jsonl)
    They are equal at the task-specific indicator level; starting from phase 6, run has an additional efficiency subgroup (score does not have one)."""
    task = SentimentClf()
    docs = list(task.docs())

    r_run = evaluate_run(task, MockLM(mode="gold", docs=docs))
    r_score = evaluate_score(task, PRED_DIR / "perfect.jsonl")

    assert _task_agg(r_run.aggregated) == _task_agg(r_score.aggregated)
    assert "efficiency" in r_run.aggregated
    assert "efficiency" not in r_score.aggregated  # Score path has no LM call
    assert r_run.n == r_score.n
    # The (doc_id, prediction, target, task-only metrics) of per_sample should also be the same
    # (Starting from phase 6 audit §1.3A, the run path sample.metrics has 4 more efficiency occupancies, which will be compared after stripping)
    a_pairs = [(s.doc_id, s.prediction, s.target, _task_metrics(s.metrics)) for s in r_run.per_sample]
    o_pairs = [(s.doc_id, s.prediction, s.target, _task_metrics(s.metrics)) for s in r_score.per_sample]
    assert a_pairs == o_pairs


def test_run_noisy_matches_predictions_file():
    """MockLM(noisy, seed=0) and predictions/noisy_0.3.jsonl are the same (the seeds on both sides are aligned when generated)."""
    task = SentimentClf()
    docs = list(task.docs())

    r_run = evaluate_run(task, MockLM(mode="noisy", docs=docs, noise=0.3, seed=0))
    r_score = evaluate_score(task, PRED_DIR / "noisy_0.3.jsonl")

    assert _task_agg(r_run.aggregated) == _task_agg(r_score.aggregated)


def test_run_constant_matches_predictions_file():
    task = SentimentClf()
    docs = list(task.docs())

    r_run = evaluate_run(task, MockLM(mode="constant", docs=docs, label="neutral"))
    r_score = evaluate_score(task, PRED_DIR / "constant_neutral.jsonl")

    assert _task_agg(r_run.aggregated) == _task_agg(r_score.aggregated)


def test_run_rule_matches_predictions_file():
    task = SentimentClf()
    docs = list(task.docs())

    r_run = evaluate_run(task, MockLM(mode="rule", docs=docs))
    r_score = evaluate_score(task, PRED_DIR / "keyword_rule.jsonl")

    assert _task_agg(r_run.aggregated) == _task_agg(r_score.aggregated)


def test_run_mt_gold_equals_score_perfect():
    """parity on family 2 (mt):mock:gold ≡ score predictions/perfect.jsonl.

    Family 1 already has a test with the same name, but mt introduces 6 generated indicators (including BERTScore) and different task schemas.
    Re-weld the dual-mode equivalence on the new task to avoid regression."""
    task = MT()
    docs = list(task.docs())

    r_run = evaluate_run(task, MockLM(mode="gold", docs=docs))
    r_score = evaluate_score(task, MT_PRED_DIR / "perfect.jsonl")

    assert _task_agg(r_run.aggregated) == _task_agg(r_score.aggregated)
    assert "efficiency" in r_run.aggregated
    assert "efficiency" not in r_score.aggregated
    assert r_run.n == r_score.n
    a_pairs = [(s.doc_id, s.prediction, s.target, _task_metrics(s.metrics)) for s in r_run.per_sample]
    o_pairs = [(s.doc_id, s.prediction, s.target, _task_metrics(s.metrics)) for s in r_score.per_sample]
    assert a_pairs == o_pairs


def test_run_mt_with_fewshot_records_num_fewshot():
    """When num_fewshot=2, the EvalResult.num_fewshot field is recorded correctly."""
    task = MT()
    docs = list(task.docs())

    r = evaluate_run(task, MockLM(mode="gold", docs=docs), num_fewshot=2, fewshot_seed=0)
    assert r.num_fewshot == 2
    # In gold mode, the answer is exactly the same as the target, and the K-shot does not affect the perfect score.
    assert r.aggregated["exact_match"] == 1.0


def test_elapsed_ms_covers_process_results_phase():
    """DECISIONS §7.1.1 End-to-end elapsed_ms lock:

    Old implementation of elapsed_ms in _evaluate_inner call pretest, process_results/injectors/
    The entire aggregation section is excluded - the judge-heavy path is missed by 6 orders of magnitude (rag_qa measured 0.137ms vs
    125s wall time).

    This test inserts 50ms sleep into process_results and asserts that elapsed_ms >= 50 * n (per-sample
    sleep accumulation) - under the old implementation, elapsed_ms is close to 0; under the new implementation, elapsed_ms must contain the sleep segment."""
    import time

    from evals.api import Doc, Response, SampleResult

    sleep_ms = 50

    class _SlowSentimentClf(SentimentClf):
        """process_results is embedded in sleep to simulate a judge-heavy subcall."""

        def process_results(self, doc: Doc, response: Response) -> SampleResult:  # type: ignore[override]
            time.sleep(sleep_ms / 1000.0)
            return super().process_results(doc, response)

    task = _SlowSentimentClf()
    docs = list(task.docs())[:3]  # Limit to 3 items to avoid single test being too slow
    # Use MockLM gold mode + limit 3 docs; evaluate_run limit=3 allows the runner to only process these 3 docs
    r = evaluate_run(task, MockLM(mode="gold", docs=docs), limit=3)
    # 3 items × 50ms sleep ≥ 150ms; leave 30ms buffer to prevent CI jitter
    assert r.elapsed_ms >= 3 * sleep_ms - 30, (
        f"elapsed_ms={r.elapsed_ms} 应 >= {3 * sleep_ms - 30}ms（含 process_results sleep）"
    )


def test_elapsed_ms_score_path_covers_process_results_phase():
    """Same as the above lock, covering the score path - judge-heavy distortion core scene (rag_qa + 5-dimensional judge)."""
    import time

    from evals.api import Doc, Response, SampleResult

    sleep_ms = 50

    class _SlowSentimentClf(SentimentClf):
        def process_results(self, doc: Doc, response: Response) -> SampleResult:  # type: ignore[override]
            time.sleep(sleep_ms / 1000.0)
            return super().process_results(doc, response)

    r = evaluate_score(_SlowSentimentClf(), PRED_DIR / "perfect.jsonl", limit=3)
    assert r.elapsed_ms >= 3 * sleep_ms - 30, (
        f"elapsed_ms={r.elapsed_ms} 应 >= {3 * sleep_ms - 30}ms（含 process_results sleep）"
    )

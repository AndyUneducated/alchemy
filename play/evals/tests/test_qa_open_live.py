"""qa_open live e2e smoke (auto-probe gate) - the live brothers of the score / run two files.

Link coverage: spec → OllamaLM → evaluate_run/score → qa_open task → judge_pointwise →
CLI cmd_run / cmd_score → SampleResult / EvalResult schema on storage path.

Only the schema and qualitative range are locked (the value jitters with the model + temperature + seed), and the specific value is not locked——
test_qa_open_score.py has used FakeJudgeLM to lock the exact value; this is responsible for the smoke-level proof of "true link pass".

File triple (same task, different mode × network level):
  - test_qa_open_score.py: FakeJudgeLM zero network + 4 copies of stub predictions teaching narrative
  - test_qa_open_run.py: MockLM(gold) + FakeJudge parity (architecture Dinghaishenzhen)
  - test_qa_open_live.py: This file is a true ollama double-layer probe gate"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from evals.cli import cmd_run
from evals.models.ollama import OllamaLM
from evals.runner import evaluate_run, evaluate_score
from evals.storage import load_run
from evals.tasks.qa_open import QAOpen
from evals.tests.conftest import ollama_required

pytestmark = ollama_required

PRED_DIR = Path(__file__).resolve().parent.parent / "data" / "qa_open" / "predictions"


def test_evaluate_run_qa_open_ollama_smoke(ollama_model: str):
    """Run path: OllamaLM is used as both task and judge (self-grading), limit=3 runs through.

    Lock schema + range; specific values vary with model/temperature/seed."""
    lm = OllamaLM(model=ollama_model)
    task = QAOpen(judge_lm=lm)

    r = evaluate_run(task, lm, limit=3)

    # schema sanity
    assert r.mode == "run"
    assert r.n == 3
    assert r.model == f"ollama:{ollama_model}"
    # Cross-cutting subgroup starting from phase 6/7: efficiency (call class) + safety (content class)
    task_keys = {k for k in r.aggregated.keys() if k not in {"efficiency", "safety"}}
    assert task_keys == {"exact_match", "rouge_l", "judge_pointwise"}
    assert "efficiency" in r.aggregated

    # Legal range of values
    assert 0.0 <= r.aggregated["exact_match"] <= 1.0
    assert 0.0 <= r.aggregated["rouge_l"] <= 1.0
    assert 1.0 <= r.aggregated["judge_pointwise"] <= 5.0

    # per-sample complete
    assert len(r.per_sample) == 3
    for s in r.per_sample:
        assert "judge_pointwise" in s.metrics
        assert 1.0 <= s.metrics["judge_pointwise"] <= 5.0


def test_evaluate_score_qa_open_ollama_judge_smoke(ollama_model: str):
    """Score path: predictions comes from perfect.jsonl (gold target), judge uses true ollama.

    The judge on perfect should give a higher score (>=3.5 loose threshold; the actual number is 4-5);
    Orthogonal to the run path, covering the mixed mode of "non-LM driven answer but judge tuned to true LM"."""
    judge_lm = OllamaLM(model=ollama_model)
    task = QAOpen(judge_lm=judge_lm)

    r = evaluate_score(task, PRED_DIR / "perfect.jsonl", limit=3)

    assert r.mode == "score"
    assert r.n == 3
    assert "judge_pointwise" in r.aggregated
    # perfect is gold target verbatim → real judge should obviously tend to score higher (loose lower bound)
    assert r.aggregated["judge_pointwise"] >= 3.5, (
        f"real ollama judge on perfect predictions should be >=3.5, "
        f"got {r.aggregated['judge_pointwise']}"
    )


def test_cli_cmd_run_judge_model_e2e(ollama_model: str, tmp_path: Path):
    """CLI full link: python -m evals run --task qa_open --model ollama:... --judge-model ollama:...

    Go to cmd_run dispatch (including QAOpen(judge_lm=...) reconstruction) → evaluate_run → save → load_run;
    Verify that judge_pointwise is not only in memory, but also in the runs/ directory and can be read back by show."""
    spec = f"ollama:{ollama_model}"
    args = argparse.Namespace(
        task="qa_open", model=spec, judge_model=spec,
        limit=2, seed=0, num_fewshot=0, fewshot_seed=0,
        runs_dir=tmp_path,
        vdb=None, retrieve_top_k=5, retrieve_mode="hybrid", rerank=False,
    )
    rc = cmd_run(args)
    assert rc == 0

    # Placement verification: There is and is only 1 run under runs_dir, and its aggregated contains judge_pointwise
    run_dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1, f"expected 1 run dir, got {run_dirs}"
    result, _samples = load_run(run_dirs[0].name, runs_dir=tmp_path)
    assert result["task"] == "qa_open"
    assert result["model"] == spec
    assert result["n"] == 2
    assert "judge_pointwise" in result["aggregated"]
    assert 1.0 <= result["aggregated"]["judge_pointwise"] <= 5.0


def test_cli_cmd_score_judge_model_e2e(ollama_model: str, tmp_path: Path):
    """CLI hybrid mode: python -m evals score --task qa_open --predictions ... --judge-model ollama:...

    score path + true LM judge——predictions come from file (non-LM driver), judge adjusts true ollama;
    Verification cmd_score also uses helper dispatch, and placement aggregated includes judge_pointwise.
    Same as cmd_run, the model field uses the file basename (source_label defaults) as EvalResult.model."""
    spec = f"ollama:{ollama_model}"
    args = argparse.Namespace(
        task="qa_open", predictions=PRED_DIR / "perfect.jsonl",
        source_label=None, judge_model=spec,
        limit=2, runs_dir=tmp_path,
    )
    from evals.cli import cmd_score
    rc = cmd_score(args)
    assert rc == 0

    run_dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    result, _samples = load_run(run_dirs[0].name, runs_dir=tmp_path)
    assert result["task"] == "qa_open"
    assert result["mode"] == "score"
    assert result["n"] == 2
    assert "judge_pointwise" in result["aggregated"]
    # perfect predictions → judge should give a higher score (loose threshold, the same standard as score smoke)
    assert result["aggregated"]["judge_pointwise"] >= 3.5

"""Orchestration layer: dual entry (score/run) + shared tail segment.

Key invariants:
  - Runner does not look at the internal type of task and only calls methods through Task ABC.
  - Runner does not look at who lm is, and only adjusts it through three request_types.
  - Runner does not import metrics/, all indicator calls are entered indirectly through task.aggregation()
  - Task.process_results does not distinguish run/score source: unified response; score path is used
    JSONL lookup table replaces LM call, everything else is exactly the same

Equivalence:
  evaluate_score(task, preds) ≡ evaluate_run(task, PrerecordedLM(preds))
  Specifically verified by test_runner_run.py::test_run_gold_equals_score_perfect."""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any

from .api import Doc, EvalMode, EvalResult, Request, Response, SampleResult
from .metrics.efficiency import (
    efficiency_aggregated,
    efficiency_judge_aggregated,
    inject_per_sample_efficiency,
)
from .models.base import LM
from .tasks.base import Task


def _load_predictions(path: str | Path) -> dict[str, dict]:
    """Read predictions JSONL → {doc_id: row} (whole row dict).

    From Phase 4, row is a complete dict, no longer just `row['prediction']` - "How to extract fields from row"
    Responsibility is delegated to `task.load_prediction(doc, row)`, allowing RAG / agent task to define its own
    row schema (including additional fields such as contexts / retrieved_ids / transcript / usage etc.).

    The default implementation of `Task.load_prediction` only takes `row['prediction']` - classification/translation task
    Minimal behavior; when overriding, inject pipeline data in row into `doc.metadata` + Response."""
    p = Path(path)
    preds: dict[str, dict] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            preds[row["id"]] = row
    return preds


def _collect_docs(task: Task, limit: int | None) -> list[Doc]:
    """Fetching data: full memory (Phase 1), Phase 2+ changed to streaming."""
    it: Iterable[Doc] = task.docs()
    if limit is not None:
        it = islice(it, limit)
    return list(it)


def _build_prompt(
    task: Task,
    doc: Doc,
    num_fewshot: int,
    pool: list[Doc],
    rng: random.Random,
) -> str:
    """Build final prompt: N example segments + query.

    `num_fewshot=0` returns `task.doc_to_text(doc)` byte-identical to Phase 1
    (foundation of old parity tests). When `>0`, sample K **non-self** examples from pool,
    format with `task.format_fewshot_example`, join with \"\\n\\n\" before query.

    If pool exhausted (too small), do not error — use however many examples were sampled.
    """
    if num_fewshot <= 0:
        return task.doc_to_text(doc)
    candidates = [d for d in pool if d.id != doc.id]
    k = min(num_fewshot, len(candidates))
    examples = rng.sample(candidates, k)
    parts = [task.format_fewshot_example(ex) for ex in examples]
    parts.append(task.doc_to_text(doc))
    return "\n\n".join(parts)


def _build_request(task: Task, doc: Doc, prompt: str) -> Request:
    """Construct Request based on task.output_type + assembled prompt. Phase 1 only handles generate_until."""
    if task.output_type == "generate_until":
        return Request(
            doc_id=doc.id,
            prompt=prompt,
            request_type="generate_until",
            until=("\n",),
            max_tokens=64,
        )
    # Phase 4 MCQ plus multiple_choice/loglikelihood branches
    raise NotImplementedError(
        f"output_type={task.output_type!r} not supported in phase 1"
    )


def _generate_run_id(task_name: str, model: str, seed: int | None) -> str:
    """{yyyymmdd-hhmmss}-{8-char hash}: time can be sorted + multiple runs with the same parameters can be identified."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    key = f"{task_name}|{model}|{seed}"
    h = hashlib.sha256(key.encode()).hexdigest()[:8]
    return f"{ts}-{h}"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _evaluate_inner(
    task: Task,
    docs: list[Doc],
    responses: list[Response],
    *,
    mode: EvalMode,
    model_label: str,
    started_at: str,
    t0: float,
    run_id: str,
    num_fewshot: int = 0,
) -> EvalResult:
    """The middle section of the dual-mode convergence after Response (the independent architectural convergence point of phase 7).

    Steps (cross-cutting dichotomy: object under test call class / evaluation tool call class / DECISIONS §7.2 + §7.3):
      1. task.process_results per-sample scoring
      2. The object under test call class injector (only run; data source = task LM call by-product usage/latency)
         - inject_per_sample_efficiency: latency / tokens / cost [phase 6]
         - phase 9 calibration continue here [planned]
      3. aggregated packaging:
         a) task.aggregation() dictionary → top tile
         b) Run only: DUT efficiency subgroup (phase 6)
         c) Dual path: Evaluation tool efficiency.judge subgroup (DECISIONS §7.3 wave 3)
      4. End-to-end elapsed_ms test

    Mounting rules (DECISIONS §7.A section superseded by §7.2 / §7.3):
      - safety no longer cross-cutting: `Safety` task has its own metrics["refusal_detected" /
        "jailbreak_attempted" / "judge_safety_score"] (flat tile) + aggregation 4 stat;
        Non-safety tasks no longer have sample.metrics["safety"] / aggregated["safety"] placeholders
      - aggregated["efficiency"] (the object under test call class) only run hangs (reserved: infrastructure cross-cutting)
      - aggregated["efficiency"]["judge"] (evaluation tool call class, new in §7.3): score / run
        Both paths are hung - judge is called in both paths, which is different from the object under test, task LM, which is only called in run
      - phase 10 robustness and other future cross-cutting will follow the independent task path according to lm-eval-harness, no longer
        AOP injection

    Timing conventions (DECISIONS §7.1.1):
      - `t0 = perf_counter()` is taken by the caller at the earliest entrance and passed in; this function calculates at the end
        `elapsed_ms = (perf_counter() - t0) * 1000`, make sure to overwrite process_results +
        injectors + aggregation entire section (including judge LM call / RAG retrieve and other sub-calls).

    See: README §Naming convention cross-cutting table / DECISIONS §7.B `_evaluate_inner` Mid-section reconstruction +
    §7.1.1 elapsed_ms end-to-end + §7.2 safety regression standalone task + §7.3 efficiency.judge.* ADR."""
    if len(docs) != len(responses):
        raise RuntimeError(
            f"doc/response length mismatch: docs={len(docs)} responses={len(responses)}"
        )
    sample_results: list[SampleResult] = [
        task.process_results(doc, resp) for doc, resp in zip(docs, responses)
    ]
    if mode == "run":
        sample_results = inject_per_sample_efficiency(sample_results, responses, model_label)

    aggregated: dict[str, Any] = {
        name: fn(sample_results) for name, fn in task.aggregation().items()
    }
    # Object under test call class (run only): efficiency of task LM
    if mode == "run":
        aggregated["efficiency"] = efficiency_aggregated(sample_results)

    # Evaluation tool call class (dual path, DECISIONS §7.3): efficiency of judge LM
    judge_responses, judge_label = task.collect_judge_responses()
    if judge_responses:
        if "efficiency" not in aggregated:
            # The score path has no measured object efficiency subtree, but there is judge → create a subtree containing only the judge subgroup
            aggregated["efficiency"] = {}
        aggregated["efficiency"]["judge"] = efficiency_judge_aggregated(
            judge_responses, judge_label
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return EvalResult(
        task=task.name,
        model=model_label,
        mode=mode,
        n=len(sample_results),
        aggregated=aggregated,
        per_sample=tuple(sample_results),
        run_id=run_id,
        created_at=started_at,
        # phase 7 audit P6: The end-to-end running time is rounded to one thousandth of a millisecond to avoid missing result.json
        # When the floating point precision is leaked (actually measured 0.9334170026704669, such 15 decimal places are of no value to people).
        # Do not move efficiency.latency_ms / cost_usd and other LM reported values ​​- dashboard / cost accumulation is really used to get sub-ms / sub-cent accuracy.
        elapsed_ms=round(elapsed_ms, 3),
        num_fewshot=num_fewshot,
    )


def evaluate_score(
    task: Task,
    predictions_path: str | Path,
    *,
    limit: int | None = None,
    source_label: str | None = None,
) -> EvalResult:
    """Score path: Read predictions JSONL and feed it directly into task.process_results, bypassing the LM layer.

    Steps (3 steps):
      1. Get data: task.docs()
      2. Read prediction + direct scoring: preds[doc.id] → Response(text=pred) → process_results
      3. Confluence: _evaluate_inner (process_results + cross-cutting injectors + packaging)

    Semantically equivalent to evaluate_run(task, PrerecordedLM(predictions_path))."""
    started_at = _iso_now()
    t0 = time.perf_counter()

    docs = _collect_docs(task, limit)
    preds = _load_predictions(predictions_path)

    docs_for_eval: list[Doc] = []
    responses: list[Response] = []
    for doc in docs:
        if doc.id not in preds:
            raise KeyError(
                f"predictions file missing doc_id={doc.id!r} "
                f"(found {len(preds)} preds for {len(docs)} docs); strict join required"
            )
        # Phase 4: Let task customize row → (doc', response) translation.
        # By default, Task.load_prediction only takes row['prediction'], which is the same as phase 1 bytes.
        enriched_doc, response = task.load_prediction(doc, preds[doc.id])
        docs_for_eval.append(enriched_doc)
        responses.append(response)

    model_label = source_label or f"preds:{Path(predictions_path).stem}"
    run_id = _generate_run_id(task.name, model_label, None)

    return _evaluate_inner(
        task,
        docs_for_eval,
        responses,
        mode="score",
        model_label=model_label,
        started_at=started_at,
        t0=t0,
        run_id=run_id,
    )


def evaluate_run(
    task: Task,
    lm: LM,
    *,
    limit: int | None = None,
    seed: int = 0,
    num_fewshot: int = 0,
    fewshot_seed: int = 0,
) -> EvalResult:
    """run path: harness 6 steps.

      1. Get data
      2. Create a request (press num_fewshot spell prompt)
      3. Batch model <-- Future concurrency is here: asyncio.gather + semaphore
      4. Confluence: _evaluate_inner (process_results + cross-cutting injectors + packaging)

    When `num_fewshot=0` prompt is the same as Phase 1 bytes (_build_prompt returns earlier).
    When `num_fewshot>0`, extract K **non-self** examples from `task.fewshot_docs()` and put them to the front.
    `fewshot_seed` only controls sampling and does not affect other paths - convenient for sweeps with different N but keeping everything else consistent."""
    started_at = _iso_now()
    t0 = time.perf_counter()

    docs = _collect_docs(task, limit)
    # Phase 4: Docs pre-processing hook before LM call (default identity transparent transmission, old tasks are not affected).
    # Typical usage: RAG task calls retrieve_fn in this batch and pin retrieved_ids/contexts into doc.metadata.
    docs = list(task.process_docs(docs))

    if task.output_type == "none":
        # Phase 4: Declare tasks without LM calls (such as rag_retrieval) - directly generate placeholder Response.
        # Do not take any of the _build_prompt / _build_request / lm.generate_until steps.
        responses: list[Response] = [Response(doc_id=d.id) for d in docs]
        model_label = lm.name
    else:
        pool = list(task.fewshot_docs()) if num_fewshot > 0 else []
        rng = random.Random(fewshot_seed)
        requests = [
            _build_request(task, doc, _build_prompt(task, doc, num_fewshot, pool, rng))
            for doc in docs
        ]

        # Phase 1 Serial. Phase 2+ does asyncio.gather/thread pool/rate-limit here.
        responses = lm.generate_until(requests)
        model_label = lm.name

        if len(responses) != len(docs):
            raise RuntimeError(
                f"LM returned {len(responses)} responses for {len(docs)} requests"
            )

    run_id = _generate_run_id(task.name, model_label, seed)

    return _evaluate_inner(
        task,
        docs,
        responses,
        mode="run",
        model_label=model_label,
        started_at=started_at,
        t0=t0,
        run_id=run_id,
        num_fewshot=num_fewshot,
    )

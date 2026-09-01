"""CLI: argparse four subcommands.

  list-tasks lists all registered tasks
  score score mode scoring (Phase 1 main path, sacrebleu style, does not drive LM)
  run run mode harness driver LM
  show cross-run query/single-run aggregation & sample display

model spec (run's --model / --judge-model and score's --judge-model share the same grammar):
  mock:gold
  mock:noisy:0.3
  mock:constant:neutral
  mock:rule
  ollama:qwen3.6:27b [phase 3]
  openai:gpt-4o-mini [phase 3+ scaffold; not yet runnable]
  anthropic:claude-... [phase 3+ scaffold; not yet runnable]

`--judge-model` currently accepts both score / run sub-commands, and connects qa_open / rag_qa (rag_retrieval
No judge).

Phase 4 adds 4 new `--vdb` / `--retrieve-top-k` / `--retrieve-mode` / `--rerank`
RAG exclusive flag: only `rag_retrieval` / `rag_qa` can be connected, other tasks are equipped with this flag and immediately SystemExit
(fail-fast instead of silently ignored). dispatch in `_build_task_with_optional_deps`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import tasks  # noqa: F401 — Trigger @register_task side effect
from .api import Request, Response
from .models.base import LM
from .models.mock import MockLM
from .models.ollama import OllamaLM
from .registry import get_task, list_tasks
from .runner import evaluate_run, evaluate_score
from .storage import DEFAULT_RUNS_DIR, load_run, read_index, save

EXTERNAL_PROVIDERS = ("openai", "anthropic", "gemini")


# ---------- model spec analysis ----------

def parse_model_spec(spec: str, task) -> LM:  # noqa: ANN001 — Task type forward-ref avoids loops
    """spec → LM instance dispatch.

      mock:<mode>[:<arg>] → MockLM (phase 1)
      ollama:<model> → OllamaLM (phase 3)
      ollama:<model>@seed=<K> → OllamaLM(seed=K); agent_sft phase 1 for multiple seeds
      openai|anthropic|gemini → NotImplementedError (architecture is left open, phase 3 is not enabled yet)

    `@seed=K` suffix (only supported by ollama): Inject the LM-side sampling seed into `OllamaLM(seed=K)`,
    Differentiate from CLI's `--seed` (the latter manages fewshot sampling/runner-level RNG, not the LM end).
    `lm.name` retains the `@seed=K` suffix so that the EvalResult.model field can distinguish multiple seed runs, which is convenient
    aggregate_seeds.py by (task, model_label_w/o_seed, seed) group."""
    # First strip off the `@seed=K` suffix (only appears at the end), and leave the rest to the specific provider parse
    seed_suffix: str | None = None
    lm_seed: int | None = None
    if "@seed=" in spec:
        spec, seed_str = spec.rsplit("@seed=", 1)
        try:
            lm_seed = int(seed_str)
        except ValueError as e:
            raise ValueError(f"invalid seed in model spec: {seed_str!r}") from e
        seed_suffix = f"@seed={lm_seed}"

    parts = spec.split(":")
    provider = parts[0]
    if provider == "mock":
        if seed_suffix is not None:
            raise ValueError(
                f"@seed=K suffix not supported for {provider!r} (use mock:noisy:<noise>:<seed> instead)"
            )
        if len(parts) < 2:
            raise ValueError(f"invalid mock spec: {spec!r}; expected mock:<mode>[:<arg>]")
        mode = parts[1]
        docs = list(task.docs())
        if mode == "gold":
            return MockLM(mode="gold", docs=docs)
        if mode == "noisy":
            noise = float(parts[2]) if len(parts) > 2 else 0.3
            seed = int(parts[3]) if len(parts) > 3 else 0
            return MockLM(mode="noisy", docs=docs, noise=noise, seed=seed)
        if mode == "constant":
            label = parts[2] if len(parts) > 2 else "neutral"
            return MockLM(mode="constant", docs=docs, label=label)
        if mode == "rule":
            return MockLM(mode="rule", docs=docs)
        raise ValueError(f"unknown mock mode: {mode!r}")
    if provider == "ollama":
        if len(parts) < 2:
            raise ValueError(f"invalid ollama spec: {spec!r}; expected ollama:<model>[@seed=K]")
        model = ":".join(parts[1:])
        kwargs: dict = {}
        if lm_seed is not None:
            kwargs["seed"] = lm_seed
        lm = OllamaLM(model=model, **kwargs)
        if seed_suffix is not None:
            # Label the seed to model_label to make EvalResult.model distinguishable between multiple seed runs
            lm.name = f"{lm.name}{seed_suffix}"
        return lm
    if provider in EXTERNAL_PROVIDERS:
        raise NotImplementedError(
            f"{provider!r} adapter scaffolded but not enabled in phase 3; "
            "only 'ollama' is currently runnable. Add models/<provider>.py + extend "
            "parse_model_spec to enable external providers."
        )
    raise ValueError(
        f"unknown provider {provider!r} in spec {spec!r}; "
        f"supported: mock / ollama; deferred (NotImplementedError): {EXTERNAL_PROVIDERS}"
    )


# ---------- Output formatting ----------

def _fmt_kv(k: str, v, prefix: str = "") -> list[str]:  # noqa: ANN001 — v can be float / dict / int / None
    """Recursively flatten (key, value) into a 'k=v' list; nested dicts are connected with dot.

    aggregated allows nesting (efficiency subgroups, etc.) since phase 6; old phase 1-5 tiling metrics
    Take the isinstance non-dict branch, which is the same as the original `_fmt_row` bytes.
    phase 7 audit P2: None placeholder stat (for example, when safety.judge_safety_score is not connected to judge_lm)
    Rendered as `<n/a>`, explicitly distinguished from "true 0"; still null in result.json (dataclasses.asdict)."""
    full = f"{prefix}{k}"
    if v is None:
        return [f"{full}=<n/a>"]
    if isinstance(v, dict):
        out: list[str] = []
        for sub_k, sub_v in v.items():
            out.extend(_fmt_kv(sub_k, sub_v, prefix=f"{full}."))
        return out
    if isinstance(v, (int, float)):
        return [f"{full}={float(v):.4f}"]
    return [f"{full}={v}"]


def _fmt_row(r: dict) -> str:
    """One row index row → short readable row (nested aggregated subgroups are supported since phase 6)."""
    agg = r.get("aggregated", {})
    parts: list[str] = []
    for k, v in agg.items():
        parts.extend(_fmt_kv(k, v))
    return (
        f"{r['run_id']:<30} task={r['task']:<15} "
        f"mode={r['mode']:<6} model={r['model']:<28} "
        f"n={r['n']:>3}  {' '.join(parts)}"
    )


# ---------- Subcommand handlers ----------

def cmd_list_tasks(_args: argparse.Namespace) -> int:
    for name in list_tasks():
        print(name)
    return 0


class _RetrieverOnlyLM(LM):
    """name-only LM stub for `output_type='none'` tasks (introduced in phase 4; used by rag_retrieval).

    The runner will not adjust generate_until in the output_type='none' branch - this stub is only responsible for dropping
    The "human-readable model tag" responsibility for the EvalResult.model field (e.g. 'retriever:panel:hybrid').
    If accidentally called → AssertionError, catch runner branch errors."""

    def __init__(self, name: str) -> None:
        self.name = name

    def generate_until(self, requests: list[Request]) -> list[Response]:
        raise AssertionError(
            f"_RetrieverOnlyLM(name={self.name!r}).generate_until called; "
            f"output_type='none' branch should have skipped LM invocation"
        )


def _build_task_with_optional_deps(
    task_name: str,
    *,
    judge_model_spec: str | None = None,
    vdb: str | Path | None = None,
    retrieve_top_k: int = 5,
    retrieve_mode: str = "hybrid",
    rerank: bool = False,
):
    """get_task(name) + optional dependency injection (judge_lm / retrieve_fn / run_fn).

    - `judge_model_spec` is given → parse injects the corresponding task into LM (qa_open / rag_qa / agent_traj)
    - `vdb` given → make_retrieve_fn injects RAG task (rag_retrieval/rag_qa)
    - agent_traj: always inject make_run_fn (cheap closure; score path will not trigger subprocess)
    - Unmatched task × flag combination → SystemExit fail-fast

    Add dispatch branch here when extending new task support."""
    from .tasks.agent_traj import AgentTraj
    from .tasks.nudge_fire_rate import NudgeFireRate
    from .tasks.qa_open import QAOpen
    from .tasks.rag_qa import RagQA
    from .tasks.rag_retrieval import RagRetrieval
    from .tasks.safety import Safety

    base_task = get_task(task_name)
    judge_lm = parse_model_spec(judge_model_spec, base_task) if judge_model_spec else None
    retrieve_fn = None
    if vdb is not None:
        from .models.rag_retrieve import make_retrieve_fn
        retrieve_fn = make_retrieve_fn(
            vdb, top_k=retrieve_top_k, mode=retrieve_mode, rerank=rerank,
        )

    if isinstance(base_task, RagRetrieval):
        if judge_lm is not None:
            raise SystemExit(
                f"--judge-model not supported by {task_name!r}; "
                "rag_retrieval has no LM-side output. Use rag_qa for grounding judge."
            )
        return RagRetrieval(retrieve_fn=retrieve_fn, top_k=retrieve_top_k)

    if isinstance(base_task, RagQA):
        return RagQA(retrieve_fn=retrieve_fn, judge_lm=judge_lm, top_k=retrieve_top_k)

    if isinstance(base_task, AgentTraj):
        if vdb is not None:
            raise SystemExit(
                f"--vdb / RAG flags not supported by {task_name!r}; "
                "agent_traj uses subprocess-driven agent_engine, not direct retrieval."
            )
        from .models.agent_engine_run import make_run_fn
        return AgentTraj(run_fn=make_run_fn(), judge_lm=judge_lm)

    if isinstance(base_task, NudgeFireRate):
        if vdb is not None:
            raise SystemExit(
                f"--vdb / RAG flags not supported by {task_name!r}; "
                "nudge_fire_rate uses subprocess-driven agent_engine, not direct retrieval."
            )
        if judge_lm is not None:
            raise SystemExit(
                f"--judge-model not supported by {task_name!r}; "
                "nudge_fire_rate is a process-conformance metric (no LM-side judging)."
            )
        from .models.agent_engine_run import make_run_fn
        return NudgeFireRate(run_fn=make_run_fn())

    if isinstance(base_task, Safety):
        if vdb is not None:
            raise SystemExit(
                f"--vdb / RAG flags not supported by {task_name!r}; "
                "safety is a text-safety task, not retrieval-driven."
            )
        if judge_lm is None:
            return base_task
        return Safety(judge_lm=judge_lm)

    if isinstance(base_task, QAOpen):
        if vdb is not None:
            raise SystemExit(
                f"--vdb / RAG flags not supported by {task_name!r}; "
                "use rag_qa / rag_retrieval for retrieval-driven tasks."
            )
        if judge_lm is None:
            return base_task
        return QAOpen(judge_lm=judge_lm)

    # Other tasks: reject RAG / judge flag
    if judge_lm is not None:
        raise SystemExit(
            f"--judge-model only supported by qa_open / rag_qa / agent_traj / safety (got task={task_name!r}); "
            "extend the dispatch in cli.py::_build_task_with_optional_deps when adding judge to other tasks"
        )
    if vdb is not None:
        raise SystemExit(
            f"--vdb only supported by rag_retrieval / rag_qa (got task={task_name!r})"
        )
    return base_task


def cmd_score(args: argparse.Namespace) -> int:
    task = _build_task_with_optional_deps(
        args.task,
        judge_model_spec=args.judge_model,
        # The score path does not require retrieve_fn (contexts/retrieved_ids is already in predictions JSONL)
    )
    result = evaluate_score(
        task,
        args.predictions,
        limit=args.limit,
        source_label=args.source_label,
    )
    save(result, runs_dir=args.runs_dir)
    print(f"# run_id={result.run_id}  mode=score  model={result.model}  n={result.n}  elapsed={result.elapsed_ms:.1f}ms")
    _print_aggregated(result.aggregated)
    return 0


def _is_all_zero_nested(d) -> bool:  # noqa: ANN001 — d may be dict / numeric leaf / None
    """Recursively determine whether all leaf values of the nested dict are 0 (None is regarded as a zero-type signal; non-numeric leaf → False).

    phase 7 audit P2: safety stat uses None to place "not measured", None and 0 are equivalent in folding semantics
    (both belong to "no metric signal"), but the trait gate (_should_fold_when_all_zero) still presses dim
    Determines whether to actually collapse - the content class (safety) will not collapse even if it is None, allowing <n/a> to be rendered explicitly."""
    if d is None:
        return True
    if isinstance(d, dict):
        return all(_is_all_zero_nested(v) for v in d.values())
    if isinstance(d, (int, float)):
        return d == 0
    return False


# cross-cutting dim → metric module path mapping, used to query module-level
# FOLD_AS_NOT_MEASURED_WHEN_ALL_ZERO trait.
#
# wave 3 (DECISIONS §7.2): safety exits cross-cutting (returns to standalone task),
# This mapping leaves only efficiency. Add new cross-cutting dimensions (calibration, etc.) registered here.
_DIM_MODULES: dict[str, str] = {
    "efficiency": "evals.metrics.efficiency",
}


def _should_fold_when_all_zero(dim: str) -> bool:
    """Query the FOLD_AS_NOT_MEASURED_WHEN_ALL_ZERO trait of the cross-cutting dim module.

    Unregistered dim → Default True (consistent with phase 6 audit §1.7 independent collapse default behavior -
    If the new cross-cutting dimension wants to exit folding, it must explicitly declare trait=False in its own module).
    See the trait constant declarations in metrics/efficiency.py / metrics/safety.py for details."""
    mod_path = _DIM_MODULES.get(dim)
    if not mod_path:
        return True
    import importlib
    mod = importlib.import_module(mod_path)
    return getattr(mod, "FOLD_AS_NOT_MEASURED_WHEN_ALL_ZERO", True)


def _print_aggregated(agg: dict) -> None:
    """Nested friendly printing: phase 6 and above aggregated contains efficiency subgroup, recursively go to _fmt_kv.

    audit §1.7: cross-cutting dim nested subgroup if all leaf values are all 0/None and the dim is in the trait table
    Declare FOLD_AS_NOT_MEASURED_WHEN_ALL_ZERO=True, collapse to `<dim>: <not measured>` single line avoidance
    Visually misleading. None placeholder stat takes away from _fmt_kv's `<n/a>` rendering.
    Top-level task-specific metrics (accuracy=0, etc.) remain with an explicit 0 output (task signals are not folded).

    DECISIONS §7.3 wave 3: Nested second-level folding - cross-cutting subtree (such as efficiency) top level is not all 0 but
    When the internal sub-subgroups (such as efficiency.judge: task missed judge_lm / mock judge / price list missed) are all 0,
    Folded into a single line with the same trait gate as `<dim>.<sub>: <not measured>`."""
    for k, v in agg.items():
        # Top fold: cross-cutting dim all 0 → single line `<dim>: <not measured>`
        if isinstance(v, dict) and _is_all_zero_nested(v) and _should_fold_when_all_zero(k):
            print(f"  {k:<28} <not measured (no LM signal)>")
            continue

        # Nested second-level folds (DECISIONS §7.3): cross-cutting dim The top level is not all 0 but the internal sub-subgroups are all 0
        if k in _DIM_MODULES and isinstance(v, dict) and _should_fold_when_all_zero(k):
            for sub_k, sub_v in v.items():
                if isinstance(sub_v, dict) and _is_all_zero_nested(sub_v):
                    full_path = f"{k}.{sub_k}"
                    print(f"  {full_path:<28} <not measured (no LM signal)>")
                    continue
                for line in _fmt_kv(sub_k, sub_v, prefix=f"{k}."):
                    key, _, val = line.partition("=")
                    print(f"  {key:<28} {val}")
            continue

        # Top-level task scalar (including the safety task's own refusal_rate and other wave 3 tile metrics)
        for line in _fmt_kv(k, v):
            key, _, val = line.partition("=")
            print(f"  {key:<28} {val}")


def cmd_run(args: argparse.Namespace) -> int:
    vdb = args.vdb
    retrieve_top_k = args.retrieve_top_k
    retrieve_mode = args.retrieve_mode
    rerank = args.rerank

    task = _build_task_with_optional_deps(
        args.task,
        judge_model_spec=args.judge_model,
        vdb=vdb,
        retrieve_top_k=retrieve_top_k,
        retrieve_mode=retrieve_mode,
        rerank=rerank,
    )

    # output_type='none' task (rag_retrieval / agent_traj) allows to save --model: use representative label placeholder
    if task.output_type == "none":
        if args.model:
            lm: LM = parse_model_spec(args.model, task)
        elif vdb:
            lm = _RetrieverOnlyLM(name=f"retriever:{Path(vdb).name}:{retrieve_mode}")
        elif task.name in ("agent_traj", "nudge_fire_rate"):
            lm = _RetrieverOnlyLM(name="agent_engine")
        else:
            raise SystemExit(
                f"task={args.task!r} has output_type='none'; pass --vdb to label the run "
                "or --model for an explicit no-op label"
            )
    else:
        if not args.model:
            raise SystemExit(f"--model is required for task={args.task!r} (output_type={task.output_type!r})")
        lm = parse_model_spec(args.model, task)

    result = evaluate_run(
        task,
        lm,
        limit=args.limit,
        seed=args.seed,
        num_fewshot=args.num_fewshot,
        fewshot_seed=args.fewshot_seed,
    )
    save(result, runs_dir=args.runs_dir)
    print(
        f"# run_id={result.run_id}  mode=run  model={result.model}  n={result.n}  "
        f"num_fewshot={result.num_fewshot}  elapsed={result.elapsed_ms:.1f}ms"
    )
    _print_aggregated(result.aggregated)
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    if args.run_id:
        result, samples = load_run(args.run_id, runs_dir=args.runs_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.samples:
            print(f"\n# samples (first {args.samples}):")
            for s in samples[: args.samples]:
                print(f"  {s['doc_id']}  pred={s['prediction']:<10}  target={s['target']:<10}  acc={s['metrics']['acc']:.0f}")
        return 0

    rows = read_index(args.runs_dir)
    if args.task:
        rows = [r for r in rows if r["task"] == args.task]
    if args.mode:
        rows = [r for r in rows if r["mode"] == args.mode]
    rows.sort(key=lambda r: r["created_at"])
    if args.last:
        rows = rows[-args.last :]
    for r in rows:
        print(_fmt_row(r))
    return 0


# ---------- argparse ----------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m evals",
        description="Dual-mode LLM evaluation harness (score: file scoring / run: drive LM)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list-tasks", help="List all registered tasks")
    p_list.set_defaults(func=cmd_list_tasks)

    p_score = sub.add_parser("score", help="Score mode: read predictions JSONL and score; does not drive LM")
    p_score.add_argument("--task", required=True, help="Task name, e.g. sentiment_clf")
    p_score.add_argument("--predictions", required=True, help="Predictions JSONL path {id, prediction}")
    p_score.add_argument("--source-label", default=None, help="Display model label (default: file basename)")
    p_score.add_argument(
        "--judge-model",
        default=None,
        help="Judge LM spec (qa_open uses judge_pointwise, e.g. ollama:qwen3.6:27b); omit for lexical baseline only",
    )
    p_score.add_argument("--limit", type=int, default=None, help="Run only first N items")
    p_score.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR, help="Directory to write run results")
    p_score.set_defaults(func=cmd_score)

    p_run = sub.add_parser("run", help="Run mode: drive LM over prompts")
    p_run.add_argument("--task", required=True)
    p_run.add_argument(
        "--model",
        default=None,
        help=(
            "Model spec, e.g. mock:gold / mock:noisy:0.3 / ollama:qwen3.6:27b. "
            "Optional when task.output_type='none' (rag_retrieval); --vdb derives retriever label."
        ),
    )
    p_run.add_argument(
        "--judge-model",
        default=None,
        help=(
            "Judge LM spec (qa_open / rag_qa; e.g. ollama:qwen3.6:27b); "
            "omit for lexical baseline (rag_qa: em + rouge_l only)"
        ),
    )
    # phase 4 RAG exclusive flags (only accessed by rag_retrieval / rag_qa)
    p_run.add_argument(
        "--vdb",
        default=None,
        help="VDB directory (e.g. ../rag/vdb/panel); RAG tasks auto-retrieve in process_docs. rag_retrieval / rag_qa only.",
    )
    p_run.add_argument(
        "--retrieve-top-k",
        type=int,
        default=5,
        help="Top-K documents from retrieval (injected into doc.metadata)",
    )
    p_run.add_argument(
        "--retrieve-mode",
        choices=["dense", "bm25", "hybrid"],
        default="hybrid",
        help="Retrieval strategy: dense / bm25 / hybrid (RRF fusion)",
    )
    p_run.add_argument(
        "--rerank",
        action="store_true",
        help="Enable cross-encoder rerank (first load ~1.2GB model; boosts precision@k)",
    )
    p_run.add_argument("--limit", type=int, default=None)
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument(
        "--num-fewshot",
        type=int,
        default=0,
        help="Prepend K examples to prompt (lm-eval K-shot); 0 = zero-shot, byte-identical to Phase 1",
    )
    p_run.add_argument(
        "--fewshot-seed",
        type=int,
        default=0,
        help="Few-shot sampling RNG seed; controls example sampling only",
    )
    p_run.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    p_run.set_defaults(func=cmd_run)

    p_show = sub.add_parser("show", help="Query run results (cross-run index / single-run drill-down)")
    p_show.add_argument("--run-id", default=None, help="Specific run_id; omit to list cross-run index")
    p_show.add_argument("--task", default=None, help="Filter by task")
    p_show.add_argument("--mode", default=None, choices=["score", "run"], help="Filter by mode")
    p_show.add_argument("--last", type=int, default=None, help="Show only last N entries")
    p_show.add_argument("--samples", type=int, default=0, help="For single run, show first N samples")
    p_show.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    p_show.set_defaults(func=cmd_show)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)

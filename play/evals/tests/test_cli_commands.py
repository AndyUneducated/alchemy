"""CLI user commands end-to-end: list-tasks / show / build_parser / main(argv) entry lock.

Fill in the blind area of `test_cli_spec.py` - only `parse_model_spec` is covered there /
`_build_task_with_optional_deps` / `_fmt_kv` / `_print_aggregated` and other internal helpers,
No real user entry is required; only the argparse subcommand name/required flag drift/set_defaults is missing
`func`, CLI user immediately explodes but local pytest is all green.

This file lock:
  ① `cmd_list_tasks(args)` — print 12 task names end-to-end + return 0
  ② `cmd_show` Cross-run index browsing - start storage.save and filter + filter by task/mode/last
  ③ `cmd_show` single run drill-down — check result.json json.dumps + optional samples
  ④ `build_parser()` argparse shape — required flag / choices / defaults
  ⑤ `main(argv)` entry - explicit argv list uses full stack dispatch (replacing sys.argv side effects)

Zero network/zero LM: Use `MockLM(mode='gold')` or false result to drop all sub-command paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from evals import tasks  # noqa: F401 — @register_task side effects
from evals.api import EvalResult, SampleResult
from evals.cli import (
    build_parser,
    cmd_list_tasks,
    cmd_show,
    main,
)
from evals.registry import list_tasks
from evals.storage import save


# ---------- ① cmd_list_tasks ---------------------------------------------

def test_cmd_list_tasks_prints_all_registered_names_and_returns_zero(capsys):
    """All 12 tasks are printed one per line + return 0; it has the same origin as `list_tasks()`, and the UI is not beautiful."""
    rc = cmd_list_tasks(argparse.Namespace())
    assert rc == 0
    out = capsys.readouterr().out.splitlines()
    assert out == list_tasks(), (
        f"cmd_list_tasks 输出与 list_tasks() 不一致：\n"
        f"  printed: {out}\n"
        f"  expected: {list_tasks()}"
    )
    # Exactly one token per line (no prefix formatting) - let `python -m evals list-tasks | xargs -I X ...`
    # Stable bash pipe usage (avoid accidentally adding bullet / indentation).
    for line in out:
        assert line.strip() == line and " " not in line, f"task name 行带额外字符：{line!r}"


# ---------- helpers: construct fake EvalResult / drop disk -----------------------

def _make_eval_result(
    *,
    run_id: str,
    task: str = "sentiment_clf",
    model: str = "mock:gold",
    mode: str = "score",
    n: int = 2,
    accuracy: float = 1.0,
) -> EvalResult:
    samples = tuple(
        SampleResult(
            doc_id=f"d{i}",
            prediction="pos",
            target="pos",
            metrics={"acc": 1.0},
        )
        for i in range(n)
    )
    return EvalResult(
        task=task,
        model=model,
        mode=mode,
        n=n,
        aggregated={"accuracy": accuracy},
        per_sample=samples,
        run_id=run_id,
        created_at=f"2025-01-01T00:00:{run_id[-2:]}Z",
        elapsed_ms=1.0,
        num_fewshot=0,
    )


# ---------- ② cmd_show cross-run index browsing ----------------------------------

def test_cmd_show_lists_index_rows_when_no_run_id(tmp_path, capsys):
    """None --run-id → Go to index.jsonl to list all runs, sorted by created_at."""
    save(_make_eval_result(run_id="20250101T000001"), runs_dir=tmp_path)
    save(
        _make_eval_result(run_id="20250101T000002", task="qa_open", accuracy=0.5),
        runs_dir=tmp_path,
    )

    args = argparse.Namespace(
        run_id=None, task=None, mode=None, last=None, samples=0, runs_dir=tmp_path,
    )
    rc = cmd_show(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "20250101T000001" in out
    assert "20250101T000002" in out
    assert "task=sentiment_clf" in out
    assert "task=qa_open" in out
    assert "accuracy=1.0000" in out
    assert "accuracy=0.5000" in out


def test_cmd_show_filter_by_task(tmp_path, capsys):
    """--task X → List only the runs of this task."""
    save(_make_eval_result(run_id="20250101T000001"), runs_dir=tmp_path)
    save(_make_eval_result(run_id="20250101T000002", task="qa_open"), runs_dir=tmp_path)

    args = argparse.Namespace(
        run_id=None, task="qa_open", mode=None, last=None, samples=0, runs_dir=tmp_path,
    )
    cmd_show(args)
    out = capsys.readouterr().out
    assert "20250101T000002" in out
    assert "20250101T000001" not in out


def test_cmd_show_filter_by_mode(tmp_path, capsys):
    """--mode score → List only score run (different from run mode)."""
    save(_make_eval_result(run_id="20250101T000001", mode="score"), runs_dir=tmp_path)
    save(_make_eval_result(run_id="20250101T000002", mode="run"), runs_dir=tmp_path)

    args = argparse.Namespace(
        run_id=None, task=None, mode="run", last=None, samples=0, runs_dir=tmp_path,
    )
    cmd_show(args)
    out = capsys.readouterr().out
    assert "20250101T000002" in out
    assert "20250101T000001" not in out


def test_cmd_show_last_n_keeps_only_tail(tmp_path, capsys):
    """--last N → Show only the last N items sorted by created_at."""
    for i in range(5):
        save(
            _make_eval_result(run_id=f"20250101T0000{i:02d}", accuracy=i / 10),
            runs_dir=tmp_path,
        )

    args = argparse.Namespace(
        run_id=None, task=None, mode=None, last=2, samples=0, runs_dir=tmp_path,
    )
    cmd_show(args)
    lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
    assert len(lines) == 2
    # Take the last two items and sort them by created_at: 03 + 04
    assert "20250101T000003" in lines[0]
    assert "20250101T000004" in lines[1]


def test_cmd_show_empty_runs_dir_prints_nothing(tmp_path, capsys):
    """index.jsonl does not exist (first run/accidentally deleted) → 0 lines of output + return 0 (no crash)."""
    args = argparse.Namespace(
        run_id=None, task=None, mode=None, last=None, samples=0, runs_dir=tmp_path,
    )
    rc = cmd_show(args)
    assert rc == 0
    assert capsys.readouterr().out == ""


# ---------- ③ cmd_show single run drill-down --------------------------------

def test_cmd_show_with_run_id_dumps_result_json(tmp_path, capsys):
    """--run-id X → result.json full text dump (json.dumps indent=2)."""
    save(_make_eval_result(run_id="20250101T000001", n=3), runs_dir=tmp_path)

    args = argparse.Namespace(
        run_id="20250101T000001", task=None, mode=None,
        last=None, samples=0, runs_dir=tmp_path,
    )
    rc = cmd_show(args)
    assert rc == 0
    out = capsys.readouterr().out
    # Should be formatted json (indent=2 → " \"" indent)
    parsed = json.loads(out)
    assert parsed["run_id"] == "20250101T000001"
    assert parsed["n"] == 3
    assert parsed["aggregated"] == {"accuracy": 1.0}


def test_cmd_show_with_run_id_and_samples_prints_per_sample(tmp_path, capsys):
    """--run-id X --samples K → Append K lines of sample summary after result json."""
    save(_make_eval_result(run_id="20250101T000001", n=4), runs_dir=tmp_path)

    args = argparse.Namespace(
        run_id="20250101T000001", task=None, mode=None,
        last=None, samples=2, runs_dir=tmp_path,
    )
    cmd_show(args)
    out = capsys.readouterr().out
    assert "samples (first 2)" in out
    # Summary line format: "d0 pred=... target=... acc=..."
    assert "d0" in out
    assert "d1" in out
    assert "pred=pos" in out
    assert "target=pos" in out
    # Only play 2 bars without playing d2/d3
    assert "d2" not in out
    assert "d3" not in out


def test_cmd_show_unknown_run_id_raises(tmp_path):
    """--run-id does not exist → FileNotFoundError (fail-fast, not silently back 0)."""
    args = argparse.Namespace(
        run_id="not_a_run", task=None, mode=None,
        last=None, samples=0, runs_dir=tmp_path,
    )
    with pytest.raises(FileNotFoundError):
        cmd_show(args)


# ---------- ④ build_parser argparse shape --------------------------------

def test_build_parser_returns_argparse_parser():
    """build_parser() returns an argparse.ArgumentParser instance (not a custom wrapper)."""
    p = build_parser()
    assert isinstance(p, argparse.ArgumentParser)


def test_build_parser_subcommands_full_set():
    """4 subcommands + each set_defaults(func=...) must be complete (set_defaults is missing
    will give main() an AttributeError at args.func)."""
    p = build_parser()
    # subparsers is _SubParsersAction
    sub_action = next(
        a for a in p._actions if isinstance(a, argparse._SubParsersAction)
    )
    assert set(sub_action.choices.keys()) == {"list-tasks", "score", "run", "show"}, (
        f"子命令集漂移：{sorted(sub_action.choices.keys())}"
    )
    # Each child parser should have func stuffed in set_defaults
    for name, sub_parser in sub_action.choices.items():
        defaults = sub_parser._defaults  # type: ignore[attr-defined]
        assert "func" in defaults, f"子命令 {name!r} 漏 set_defaults(func=...)"
        assert callable(defaults["func"]), f"子命令 {name!r} 的 func 非 callable"


def test_build_parser_score_required_args_enforced():
    """score subcommand: argparse SystemExit when --task / --predictions are missing."""
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["score"])
    with pytest.raises(SystemExit):
        p.parse_args(["score", "--task", "sentiment_clf"])  # Missing --predictions


def test_build_parser_run_task_required():
    """run subcommand: missing --task → SystemExit."""
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["run"])


def test_build_parser_run_default_values_locked():
    """Default value of run subcommand: --num-fewshot=0 / --fewshot-seed=0 / --seed=0 /
    --retrieve-top-k=5 / --retrieve-mode=hybrid / --rerank=False / --model=None.
    Either drift would change the zero-shot default behavior or the RAG default configuration - directly breaking the "naked" semantics for CLI users."""
    p = build_parser()
    args = p.parse_args(["run", "--task", "sentiment_clf"])
    assert args.num_fewshot == 0
    assert args.fewshot_seed == 0
    assert args.seed == 0
    assert args.retrieve_top_k == 5
    assert args.retrieve_mode == "hybrid"
    assert args.rerank is False
    assert args.model is None
    assert args.judge_model is None
    assert args.vdb is None
    assert args.limit is None


def test_build_parser_run_retrieve_mode_choices_locked():
    """--retrieve-mode must ∈ {dense, bm25, hybrid}; illegal value SystemExit."""
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["run", "--task", "rag_qa", "--retrieve-mode", "magic"])
    # Three legal values ​​must be parsed through
    for mode in ("dense", "bm25", "hybrid"):
        args = p.parse_args(["run", "--task", "rag_qa", "--retrieve-mode", mode])
        assert args.retrieve_mode == mode


def test_build_parser_show_mode_choices_locked():
    """show --mode must ∈ {score, run} (to avoid misspellings and filter silent empty)."""
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["show", "--mode", "score_typo"])
    args = p.parse_args(["show", "--mode", "score"])
    assert args.mode == "score"
    args = p.parse_args(["show", "--mode", "run"])
    assert args.mode == "run"


def test_build_parser_show_defaults_browse_index():
    """show without any flag → browse through index (run_id=None / task=None / mode=None / samples=0)."""
    p = build_parser()
    args = p.parse_args(["show"])
    assert args.run_id is None
    assert args.task is None
    assert args.mode is None
    assert args.last is None
    assert args.samples == 0


def test_build_parser_score_runs_dir_is_path():
    """--runs-dir should be type=Path converted to Path instead of retaining str (storage layer converted to Path)."""
    p = build_parser()
    args = p.parse_args([
        "score", "--task", "sentiment_clf",
        "--predictions", "p.jsonl",
        "--runs-dir", "/tmp/foo",
    ])
    assert isinstance(args.runs_dir, Path)
    assert str(args.runs_dir) == "/tmp/foo"


def test_build_parser_no_subcommand_raises():
    """Bare `python -m evals` without subcommand → argparse SystemExit(required=True)."""
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args([])


# ---------- ⑤ main(argv) entry -----------------------------------------------

def test_main_list_tasks_returns_zero(capsys):
    """`main(['list-tasks'])` end-to-end: full argparse → cmd_list_tasks → return 0."""
    rc = main(["list-tasks"])
    assert rc == 0
    out = capsys.readouterr().out.splitlines()
    assert out == list_tasks()


def test_main_show_index_empty_dir(tmp_path, capsys):
    """`main(['show', '--runs-dir', tmp])` Empty directory → 0 lines + return 0."""
    rc = main(["show", "--runs-dir", str(tmp_path)])
    assert rc == 0
    assert capsys.readouterr().out == ""


def test_main_show_with_run_id_uses_runs_dir(tmp_path, capsys):
    """`main(['show', '--run-id', X, '--runs-dir', tmp])` end-to-end dump result.json."""
    save(_make_eval_result(run_id="20250101T000099"), runs_dir=tmp_path)

    rc = main(["show", "--run-id", "20250101T000099", "--runs-dir", str(tmp_path)])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["run_id"] == "20250101T000099"


def test_main_unknown_subcommand_exits():
    """Unknown subcommand → SystemExit (exclusively set with list-tasks/score/run/show)."""
    with pytest.raises(SystemExit):
        main(["totally-not-a-subcommand"])


def test_main_score_missing_task_exits():
    """`main(['score', '--predictions', ...])` missing --task → SystemExit."""
    with pytest.raises(SystemExit):
        main(["score", "--predictions", "p.jsonl"])

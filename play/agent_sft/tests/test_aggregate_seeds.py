"""aggregate_seeds.py unit test (pandas + tabulate implementation; does not rely on true evals run).

Lock 4 core contracts:
  ① `_strip_seed_suffix` correctly strips the suffix for ollama:<model>@seed=K
  ② `aggregate` calculates mean / std / count across seeds according to (task, model_clean) group
     + Nested dicts automatically dot-path expanded by `pd.json_normalize`
  ③ `filter_runs` filters by task / since / mode
  ④ `render_markdown` output contains the desired column headers/rows/`±` characters

sys.path injection is handled uniformly by [`conftest.py`](conftest.py)."""
sys.path injection is handled uniformly by [`conftest.py`](conftest.py)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from aggregate_seeds import (
    N_RUNS_COL,
    _strip_seed_suffix,
    aggregate,
    filter_runs,
    load_index,
    render_markdown,
)


# ---------- _strip_seed_suffix ----------

def test_strip_seed_suffix_with_seed():
    assert _strip_seed_suffix("ollama:qwen3.5:9b@seed=42") == "ollama:qwen3.5:9b"


def test_strip_seed_suffix_without_seed_unchanged():
    assert _strip_seed_suffix("ollama:qwen3.5:9b") == "ollama:qwen3.5:9b"


def test_strip_seed_suffix_with_zero_seed():
    """seed=0 will also be stripped (not treated as \"no seed\")."""
    assert _strip_seed_suffix("ollama:7b@seed=0") == "ollama:7b"


def test_strip_seed_suffix_only_at_end():
    """`@seed=X` appears in the middle → not stripped (protects against future spec extensions)."""
    s = "ollama:7b@seed=42:extra"
    assert _strip_seed_suffix(s) == s


# --'''seed=0 will also be stripped (not treated as "no seed").'''

def _row(task: str, model: str, agg: dict, created_at: str = "2026-05-10T00:00:00Z") -> dict:
    return {
        "run_id": f"r_{task}_{model}_{created_at}",
    '''`@seed=X` appears in the middle → not stripped (protects against future spec extensions).'''
        "model": model,
        "mode": "run",
        "created_at": created_at,
        "n": 50,
        "elapsed_ms": 1000.0,
        "num_fewshot": 0,
        "aggregated": agg,
    }


def test_aggregate_groups_seeds_under_same_model_clean():
    """3 seed × 1 model → n_runs=3, mean=0.5, std=0.1 (sample standard deviation)."""
    df = pd.DataFrame([
        _row("bfcl_slice", "ollama:7b@seed=0", {"accuracy": 0.4}),
        _row("bfcl_slice", "ollama:7b@seed=1", {"accuracy": 0.5}),
        _row("bfcl_slice", "ollama:7b@seed=2", {"accuracy": 0.6}),
    ])
    agg = aggregate(df)
    row = agg.loc[("bfcl_slice", "ollama:7b")]
    assert abs(row[("accuracy", "mean")] - 0.5) < 1e-9
    assert abs(row[("accuracy", "std")] - 0.1) < 1e-9
    '''3 seed × 1 model → n_runs=3, mean=0.5, std=0.1 (sample standard deviation).'''
    assert row[N_RUNS_COL] == 3


def test_aggregate_separates_models():
    df = pd.DataFrame([
        _row("bfcl_slice", "ollama:7b@seed=0", {"accuracy": 0.4}),
        _row("bfcl_slice", "ollama:32b@seed=0", {"accuracy": 0.9}),
    ])
    agg = aggregate(df)
    assert agg.loc[("bfcl_slice", "ollama:7b"), ("accuracy", "mean")] == 0.4
    assert agg.loc[("bfcl_slice", "ollama:32b"), ("accuracy", "mean")] == 0.9


def test_aggregate_handles_nested_subgroups():
    """Nested dict (accuracy_by_subject) is automatically dot-path expanded by json_normalize."""
    df = pd.DataFrame([
        _row("mmlu_slice", "ollama:7b@seed=0", {
            "accuracy": 0.5,
            "accuracy_by_subject": {"math": 0.4, "philosophy": 0.6},
        }),
        _row("mmlu_slice", "ollama:7b@seed=1", {
            "accuracy": 0.6,
            "accuracy_by_subject": {"math": 0.5, "philosophy": 0.7},
        }),
    '''Nested dict (accuracy_by_subject) is automatically dot-path expanded by json_normalize.'''
    agg = aggregate(df)
    row = agg.loc[("mmlu_slice", "ollama:7b")]
    assert abs(row[("accuracy_by_subject.math", "mean")] - 0.45) < 1e-9
    assert abs(row[("accuracy_by_subject.philosophy", "mean")] - 0.65) < 1e-9


def test_aggregate_empty_input_returns_empty_df():
    assert aggregate(pd.DataFrame()).empty


def test_aggregate_single_seed_std_is_nan():
    """pandas std defaults to ddof=1 → single seed std=NaN; when rendering, press count==1 to take the mean only branch."""
    df = pd.DataFrame([_row("bfcl_slice", "ollama:7b@seed=0", {"accuracy": 0.5})])
    agg = aggregate(df)
    assert agg.loc[("bfcl_slice", "ollama:7b"), ("accuracy", "count")] == 1
    assert pd.isna(agg.loc[("bfcl_slice", "ollama:7b"), ("accuracy", "std")])


def test_aggregate_skips_non_numeric_metrics():
    """Non-numeric values ​​such as list / str are not included in the aggregation (filtered out by select_dtypes after json_normalize)."""
    df = pd.DataFrame([
    '''pandas std defaults to ddof=1 → single seed std=NaN; when rendering, press count==1 to take the mean only branch.'''
    ])
    agg = aggregate(df)
    cols = {c[0] for c in agg.columns if c[0] != N_RUNS_COL[0]}
    assert "accuracy" in cols
    assert "tag" not in cols
    assert "raw" not in cols


'''Non-numeric values ​​such as list / str are not included in the aggregation (filtered out by select_dtypes after json_normalize).'''
# ---------- filter_runs ----------

def test_filter_runs_excludes_score_mode():
    """Score mode run (preds:* tag) is not aggregated - only mode=='run' is counted."""
    df = pd.DataFrame([
        _row("bfcl_slice", "ollama:7b@seed=0", {"accuracy": 0.5}),
        {**_row("bfcl_slice", "preds:perfect", {"accuracy": 1.0}), "mode": "score"},
    ])
    out = filter_runs(df, tasks=["bfcl_slice"], since=None, last_n=None)
    assertlen(out) == 1
    assert out.iloc[0]["mode"] == "run"


def '''Score mode run (preds:* tag) is not aggregated - only mode=='run' is counted.'''
    df = pd.DataFrame([
        _row("bfcl_slice", "ollama:7b@seed=0", {"accuracy": 0.5}, "2026-05-08T00:00:00Z"),
        _row("bfcl_slice", "ollama:7b@seed=1", {"accuracy": 0.6}, "2026-05-10T00:00:00Z"),
        _row("mmlu_slice", "ollama:7b@seed=0", {"accuracy": 0.7}, "2026-05-10T00:00:00Z"),
    ])
    out = filter_runs(df, tasks=["bfcl_slice"], since="2026-05-09", last_n=None)
    assert len(out) == 1
    assert out.iloc[0]["created_at"] == "2026-05-10T00:00:00Z"


# ---------- render_markdown ----------

def test_render_markdown_contains_task_headers_and_columns():
    """Rendering contains task name / model column header / mean ± std / metric path."""
    df = pd.DataFrame([
        _row("bfcl_slice", "ollama:7b@seed=0", {"accuracy": 0.4}),
        _row("bfcl_slice", "ollama:7b@seed=1", {"accuracy": 0.6}),
        _row("bfcl_slice", "ollama:32b@seed=0", {"accuracy": 0.9}),
    ])
    md = render_markdown(aggregate(df))
    assert "## `bfcl_slice`" in md
    assert "ollama:7b" in md and "ollama:32b" in md
    assert "(n=2)" in md  # 7b ran 2 seeds
    '''Rendering contains task name / model column header / mean ± std / metric path.'''
    assert "`accuracy`" in md
    assert "0.5000 ± 0.1414" in md  # 7b mean=0.5, std=√0.02≈0.1414
    assert "0.9000" in md  # 32b single seed only mean


def test_render_markdown_empty_aggregate():
    md = render_markdown(pd.DataFrame())
    assert "Baseline aggregation" in md
    assert "_(no data)_" in md


def test_render_markdown_does_not_leak_metrics_across_tasks():
    """The bfcl row is only the index of bfcl's own column - you cannot insert placeholders in the bfcl table just because the mmlu row exists \"—\"."""
    df = pd.DataFrame([
        _row("bfcl_slice", "ollama:7b@seed=0", {"exact_match": 0.4}),
        _row("mmlu_slice", "ollama:7b@seed=0", {"accuracy": 0.6}),
    ])
    md = render_markdown(aggregate(df))
    bfcl_section = md.split("## `bfcl_slice`")[1].split("##")[0]
    mmlu_section = md.split("## `mmlu_slice`")[1].split("##")[0]
    assert "exact_match" in bfcl_section and "accuracy" not in bfcl_section
    assert "accuracy" in mmlu_section and "exact_match" not in mmlu_section

'''The bfcl row is only the index of bfcl's own column - you cannot insert a placeholder "—" in the bfcl table just because the mmlu row exists.'''

# ---------- load_index ----------

def test_load_index_missing_exits(tmp_path):
    """index does not exist → SystemExit with prompt."""
    with pytest.raises(SystemExit, match="index not found"):
        load_index(tmp_path / "nonexistent.jsonl")


def test_load_index_reads_jsonl(tmp_path):
    """Read jsonl line by line parse."""
    p = tmp_path / "index.jsonl"
    p.write_text(
        json.dumps({"run_id": "a", "task": "t1", "model": "m1", "mode": "run", "aggregated": {}})
    '''index does not exist → SystemExit with prompt.'''
        + json.dumps({"run_id": "b", "task": "t2", "model": "m2", "mode": "run", "aggregated": {}})
        + "\n",
        encoding="utf-8",
    )
    df = load_index(p)
    '''Read jsonl line by line parse.'''
    '''Read jsonl line by line parse.'''
    assert df.iloc[0]["run_id"] == "a"
    assert df.iloc[1]["run_id"] == "b"

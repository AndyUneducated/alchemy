"""aggregate_seeds.py 单元测试（pandas + tabulate 实现；不依赖真 evals run）.

锁住 4 个核心契约：
  ① `_strip_seed_suffix` 对 ollama:<model>@seed=K 正确剥后缀
  ② `aggregate` 按 (task, model_clean) group 跨 seed 算 mean / std / count
     + 嵌套 dict 由 `pd.json_normalize` 自动 dot-path 展开
  ③ `filter_runs` 按 task / since / mode 过滤
  ④ `render_markdown` 输出含期望的列头 / 行 / `±` 字符

sys.path 注入由 [`conftest.py`](conftest.py) 统一处理.
"""

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
    assert _strip_seed_suffix("ollama:qwen2.5:7b@seed=42") == "ollama:qwen2.5:7b"


def test_strip_seed_suffix_without_seed_unchanged():
    assert _strip_seed_suffix("ollama:qwen2.5:7b") == "ollama:qwen2.5:7b"


def test_strip_seed_suffix_with_zero_seed():
    """seed=0 也要被剥（不被当成 \"无 seed\"）."""
    assert _strip_seed_suffix("ollama:7b@seed=0") == "ollama:7b"


def test_strip_seed_suffix_only_at_end():
    """`@seed=X` 出现在中间 → 不剥（保护未来 spec 扩展）."""
    s = "ollama:7b@seed=42:extra"
    assert _strip_seed_suffix(s) == s


# ---------- aggregate end-to-end ----------

def _row(task: str, model: str, agg: dict, created_at: str = "2026-05-10T00:00:00Z") -> dict:
    return {
        "run_id": f"r_{task}_{model}_{created_at}",
        "task": task,
        "model": model,
        "mode": "run",
        "created_at": created_at,
        "n": 50,
        "elapsed_ms": 1000.0,
        "num_fewshot": 0,
        "aggregated": agg,
    }


def test_aggregate_groups_seeds_under_same_model_clean():
    """3 seed × 1 model → n_runs=3, mean=0.5, std=0.1（样本标准差）."""
    df = pd.DataFrame([
        _row("bfcl_slice", "ollama:7b@seed=0", {"accuracy": 0.4}),
        _row("bfcl_slice", "ollama:7b@seed=1", {"accuracy": 0.5}),
        _row("bfcl_slice", "ollama:7b@seed=2", {"accuracy": 0.6}),
    ])
    agg = aggregate(df)
    row = agg.loc[("bfcl_slice", "ollama:7b")]
    assert abs(row[("accuracy", "mean")] - 0.5) < 1e-9
    assert abs(row[("accuracy", "std")] - 0.1) < 1e-9
    assert row[("accuracy", "count")] == 3
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
    """嵌套 dict（accuracy_by_subject）由 json_normalize 自动 dot-path 展开."""
    df = pd.DataFrame([
        _row("mmlu_slice", "ollama:7b@seed=0", {
            "accuracy": 0.5,
            "accuracy_by_subject": {"math": 0.4, "philosophy": 0.6},
        }),
        _row("mmlu_slice", "ollama:7b@seed=1", {
            "accuracy": 0.6,
            "accuracy_by_subject": {"math": 0.5, "philosophy": 0.7},
        }),
    ])
    agg = aggregate(df)
    row = agg.loc[("mmlu_slice", "ollama:7b")]
    assert abs(row[("accuracy_by_subject.math", "mean")] - 0.45) < 1e-9
    assert abs(row[("accuracy_by_subject.philosophy", "mean")] - 0.65) < 1e-9


def test_aggregate_empty_input_returns_empty_df():
    assert aggregate(pd.DataFrame()).empty


def test_aggregate_single_seed_std_is_nan():
    """pandas std 默认 ddof=1 → 单 seed std=NaN；render 时按 count==1 走 mean only 分支."""
    df = pd.DataFrame([_row("bfcl_slice", "ollama:7b@seed=0", {"accuracy": 0.5})])
    agg = aggregate(df)
    assert agg.loc[("bfcl_slice", "ollama:7b"), ("accuracy", "count")] == 1
    assert pd.isna(agg.loc[("bfcl_slice", "ollama:7b"), ("accuracy", "std")])


def test_aggregate_skips_non_numeric_metrics():
    """list / str 等非数值不入聚合（json_normalize 后被 select_dtypes 过滤掉）."""
    df = pd.DataFrame([
        _row("bfcl_slice", "ollama:7b@seed=0", {"accuracy": 0.5, "tag": "good", "raw": [1, 2, 3]}),
    ])
    agg = aggregate(df)
    cols = {c[0] for c in agg.columns if c[0] != N_RUNS_COL[0]}
    assert "accuracy" in cols
    assert "tag" not in cols
    assert "raw" not in cols


# ---------- filter_runs ----------

def test_filter_runs_excludes_score_mode():
    """score 模式 run（preds:* 标签）不被聚合——只算 mode=='run'."""
    df = pd.DataFrame([
        _row("bfcl_slice", "ollama:7b@seed=0", {"accuracy": 0.5}),
        {**_row("bfcl_slice", "preds:perfect", {"accuracy": 1.0}), "mode": "score"},
    ])
    out = filter_runs(df, tasks=["bfcl_slice"], since=None, last_n=None)
    assert len(out) == 1
    assert out.iloc[0]["mode"] == "run"


def test_filter_runs_by_task_and_since():
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
    """渲染含 task 名 / 模型列头 / mean ± std / metric path."""
    df = pd.DataFrame([
        _row("bfcl_slice", "ollama:7b@seed=0", {"accuracy": 0.4}),
        _row("bfcl_slice", "ollama:7b@seed=1", {"accuracy": 0.6}),
        _row("bfcl_slice", "ollama:32b@seed=0", {"accuracy": 0.9}),
    ])
    md = render_markdown(aggregate(df))
    assert "## `bfcl_slice`" in md
    assert "ollama:7b" in md and "ollama:32b" in md
    assert "(n=2)" in md  # 7b 跑了 2 seed
    assert "(n=1)" in md  # 32b 跑了 1 seed
    assert "`accuracy`" in md
    assert "0.5000 ± 0.1414" in md  # 7b mean=0.5, std=√0.02≈0.1414
    assert "0.9000" in md  # 32b 单 seed 仅 mean


def test_render_markdown_empty_aggregate():
    md = render_markdown(pd.DataFrame())
    assert "Baseline aggregation" in md
    assert "_(no data)_" in md


def test_render_markdown_does_not_leak_metrics_across_tasks():
    """bfcl 行只该列 bfcl 自己的指标——不能因为 mmlu 行存在就在 bfcl 表插占位 \"—\"."""
    df = pd.DataFrame([
        _row("bfcl_slice", "ollama:7b@seed=0", {"exact_match": 0.4}),
        _row("mmlu_slice", "ollama:7b@seed=0", {"accuracy": 0.6}),
    ])
    md = render_markdown(aggregate(df))
    bfcl_section = md.split("## `bfcl_slice`")[1].split("##")[0]
    mmlu_section = md.split("## `mmlu_slice`")[1].split("##")[0]
    assert "exact_match" in bfcl_section and "accuracy" not in bfcl_section
    assert "accuracy" in mmlu_section and "exact_match" not in mmlu_section


# ---------- load_index ----------

def test_load_index_missing_exits(tmp_path):
    """index 不存在 → SystemExit 带提示."""
    with pytest.raises(SystemExit, match="index not found"):
        load_index(tmp_path / "nonexistent.jsonl")


def test_load_index_reads_jsonl(tmp_path):
    """读 jsonl 一行一行 parse."""
    p = tmp_path / "index.jsonl"
    p.write_text(
        json.dumps({"run_id": "a", "task": "t1", "model": "m1", "mode": "run", "aggregated": {}})
        + "\n"
        + json.dumps({"run_id": "b", "task": "t2", "model": "m2", "mode": "run", "aggregated": {}})
        + "\n",
        encoding="utf-8",
    )
    df = load_index(p)
    assert len(df) == 2
    assert df.iloc[0]["run_id"] == "a"
    assert df.iloc[1]["run_id"] == "b"

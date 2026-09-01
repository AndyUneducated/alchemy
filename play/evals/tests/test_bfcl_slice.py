"""bfcl_slice unit + e2e score tests.

Two layers:
  ① **Unit**: parse_function_call / score_function_call contracts on handcrafted inputs
  ② **E2E**: BfclSlice + evaluate_score on 3 stub fixtures
     (perfect / wrong_name / wrong_args); assert direction and bounds of 4 aggregated metrics

Per plan §6 "re-lock runner invariants per new task" — n_matches_gold + missing_pred_raises both covered.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.runner import evaluate_score
from evals.tasks.bfcl_slice import (
    BfclSlice,
    parse_function_call,
    score_function_call,
)

PRED_DIR = Path(__file__).resolve().parent.parent / "data" / "bfcl_slice" / "predictions"


# ============================================================
# parse_function_call ─ parsing robustness
# ============================================================

def test_parse_simple_kwargs():
    """Clean input: function name + keyword parameters → fill in all fields."""
    p = parse_function_call("foo(a=1, b='x')")
    assert p == {"func": "foo", "args": [], "kwargs": {"a": 1, "b": "x"}}


def test_parse_dotted_function_name():
    """`math.factorial` etc. with `.` function name → dotted string instead of Attribute repr."""
    p = parse_function_call("math.factorial(number=5)")
    assert p["func"] == "math.factorial"
    assert p["kwargs"] == {"number": 5}


def test_parse_positional_args_kept_separate():
    """positional → list of args; scoring layer does schema-properties-order projection."""
    p = parse_function_call("foo(1, 2, c=3)")
    assert p["args"] == [1, 2]
    assert p["kwargs"] == {"c": 3}


def test_parse_strips_markdown_code_fence():
    """LLM often outputs ```python\\nfoo(a=1)\\n``` — strip the wrapper."""
    p = parse_function_call("```python\nfoo(a=1)\n```")
    assert p == {"func": "foo", "args": [], "kwargs": {"a": 1}}


def test_parse_strips_call_prefix():
    """Prompt ends with `Call:`, and models occasionally echo `Call: foo(...)`. The prefix should be stripped."""
    p = parse_function_call("Call: foo(a=1)")
    assert p == {"func": "foo", "args": [], "kwargs": {"a": 1}}


def test_parse_takes_first_nonempty_line():
    """Multi-line output: take first line — generate_until stops on \\n, but score path may pass full block."""
    p = parse_function_call("foo(a=1)\nexplanation: ...")
    assert p == {"func": "foo", "args": [], "kwargs": {"a": 1}}


def test_parse_returns_none_on_unparseable():
    """A string that cannot be completely parsed → None (score is 0 accordingly)."""
    assert parse_function_call("totally not a call") is None
    assert parse_function_call("") is None
    assert parse_function_call("foo(") is None  # Grammatical error


def test_parse_returns_none_on_non_call_expression():
    """`1 + 2` is a valid Expression but not a Call → reject."""
    assert parse_function_call("1 + 2") is None


# ============================================================
# score_function_call ─ 4 indicator contracts
# ============================================================

def _gt(name: str, args: dict[str, list]) -> dict:
    return {name: args}


def _schema(name: str, props: list[str], required: list[str] | None = None) -> dict:
    return {
        "name": name,
        "parameters": {
            "type": "dict",
            "properties": {p: {"type": "integer"} for p in props},
            "required": required if required is not None else props,
        },
    }


def test_score_perfect_match_all_one():
    """name pair + required arg all in + value in acceptable list → 4 items all 1.0."""
    out = score_function_call(
        "foo(a=1, b=2)",
        gt_dict=_gt("foo", {"a": [1], "b": [2]}),
        schema=_schema("foo", ["a", "b"]),
    )
    assert out["exact_match"] == 1.0
    assert out["name_match"] == 1.0
    assert out["arg_set_f1"] == 1.0
    assert out["arg_value_match"] == 1.0


def test_score_wrong_name_zero_cascade():
    """name is wrong → name_match=0 and exact_match=0; arg_set_f1 / arg_value_match is still calculated as arg."""
    out = score_function_call(
        "bar(a=1, b=2)",  # name is wrong
        gt_dict=_gt("foo", {"a": [1], "b": [2]}),
        schema=_schema("foo", ["a", "b"]),
    )
    assert out["name_match"] == 0.0
    assert out["exact_match"] == 0.0
    assert out["arg_set_f1"] == 1.0  # The set of arg names still matches
    assert out["arg_value_match"] == 1.0


def test_score_wrong_arg_value_drops_value_match():
    """name + arg The names are correct, but the value is not acceptable → arg_value_match is lowered; exact_match=0."""
    out = score_function_call(
        "foo(a=999, b=2)",  # a value is wrong
        gt_dict=_gt("foo", {"a": [1], "b": [2]}),
        schema=_schema("foo", ["a", "b"]),
    )
    assert out["name_match"] == 1.0
    assert out["arg_set_f1"] == 1.0
    assert out["arg_value_match"] == 0.5  # 1/2 pair
    assert out["exact_match"] == 0.0


def test_score_optional_arg_omitted_counts_as_match():
    """GT acceptable contains \"\" → arg optional; pred omitting it still scores."""
    out = score_function_call(
        "foo(a=1)",  # b can be saved
        gt_dict=_gt("foo", {"a": [1], "b": ["", 0]}),  # b optional, default 0
        schema=_schema("foo", ["a", "b"], required=["a"]),
    )
    assert out["arg_value_match"] == 1.0  # a is right, b is province ✓
    assert out["arg_set_f1"] == 1.0  # required={a}，pred={a}
    assert out["exact_match"] == 1.0


def test_score_optional_arg_explicit_value_also_matches():
    """Pred explicitly passes the default value of optional arg and also scores."""
    out = score_function_call(
        "foo(a=1, b=0)",  # b Explicitly pass default 0
        gt_dict=_gt("foo", {"a": [1], "b": ["", 0]}),
        schema=_schema("foo", ["a", "b"], required=["a"]),
    )
    assert out["arg_value_match"] == 1.0
    assert out["exact_match"] == 1.0


def test_score_unknown_arg_breaks_exact_match():
    """pred multi-passes arg that GT does not have → exact_match=0 (even the GT part is correct)."""
    out = score_function_call(
        "foo(a=1, b=2, extra=99)",
        gt_dict=_gt("foo", {"a": [1], "b": [2]}),
        schema=_schema("foo", ["a", "b"]),
    )
    assert out["arg_value_match"] == 1.0
    # arg_set_f1 < 1: 1 more predicted set, precision is lowered
    assert 0.0 < out["arg_set_f1"] < 1.0
    assert out["exact_match"] == 0.0


def test_score_positional_arg_mapped_via_schema_order():
    """pred with positional parameters (no kw) → mapped in schema.parameters.properties order."""
    out = score_function_call(
        "foo(1, 2)",
        gt_dict=_gt("foo", {"a": [1], "b": [2]}),
        schema=_schema("foo", ["a", "b"]),
    )
    assert out["exact_match"] == 1.0
    assert out["arg_value_match"] == 1.0


def test_score_unparseable_pred_zero_all():
    """Pred parsing failed → all 4 items are 0; artifact.parsed=None is used for subsequent diagnosis."""
    out = score_function_call(
        "I don't know how to call this",
        gt_dict=_gt("foo", {"a": [1]}),
        schema=_schema("foo", ["a"]),
    )
    assert out["exact_match"] == 0.0
    assert out["name_match"] == 0.0
    assert out["arg_set_f1"] == 0.0
    assert out["arg_value_match"] == 0.0
    assert out["parsed"] is None


def test_score_value_match_int_float_cross_type():
    """1.0 == 1 (numeric values ​​are tolerant across types; BFCL GT occasionally ints, model outputs float)."""
    out = score_function_call(
        "foo(a=1.0, b=2)",
        gt_dict=_gt("foo", {"a": [1], "b": [2]}),
        schema=_schema("foo", ["a", "b"]),
    )
    assert out["arg_value_match"] == 1.0


def test_score_value_match_excludes_bool_int_corner():
    """True != 1 in BFCL semantics — avoid false positive where a=True masquerades as a=1."""
    out = score_function_call(
        "foo(a=True)",
        gt_dict=_gt("foo", {"a": [1]}),
        schema=_schema("foo", ["a"]),
    )
    assert out["arg_value_match"] == 0.0


def test_score_value_match_acceptable_list_any_one():
    """The acceptable list contains N values ​​→ if any one is hit, it is scored (BFCL multiple acceptable semantics)."""
    out = score_function_call(
        "foo(unit='units')",
        gt_dict=_gt("foo", {"unit": ["meters", "units", "ft"]}),
        schema=_schema("foo", ["unit"]),
    )
    assert out["arg_value_match"] == 1.0


# ============================================================
# evaluate_score e2e against 3 stub fixtures
# ============================================================

def _agg(pred_name: str) -> dict[str, float]:
    task = BfclSlice()
    r = evaluate_score(task, PRED_DIR / f"{pred_name}.jsonl")
    assert r.mode == "score"
    assert r.n == 50
    return r.aggregated


def test_perfect_e2e_all_metrics_one():
    """perfect predictions = canonical target → 4 items aggregated to 1.0."""
    agg = _agg("perfect")
    assert agg["exact_match"] == 1.0
    assert agg["name_match"] == 1.0
    assert agg["arg_set_f1"] == 1.0
    assert agg["arg_value_match"] == 1.0


def test_wrong_name_e2e_name_zero_args_one():
    """wrong_name = name + \"_xxx\"; name_match=0, exact_match=0; arg dims still near 1."""
    agg = _agg("wrong_name")
    assert agg["name_match"] == 0.0
    assert agg["exact_match"] == 0.0
    # canonical targets are all required-only kwargs → the arg name set is completely consistent with GT
    assert agg["arg_set_f1"] == 1.0
    assert agg["arg_value_match"] == 1.0


def test_wrong_args_e2e_value_match_dominates_drop():
    """wrong_args = name pair + all required arg values perturb;
       name_match=1, arg_set_f1=1, arg_value_match significantly low, exact_match=0."""
    agg = _agg("wrong_args")
    assert agg["name_match"] == 1.0
    assert agg["arg_set_f1"] == 1.0
    # A very few GTs are more acceptable (such as unit=["units",""]), and after perturb \"units\"+\"X\"
    # is no longer acceptable, so the value match rate should be well below 1
    assert agg["arg_value_match"] < 0.5
    assert agg["exact_match"] == 0.0


def test_perfect_strictly_dominates_wrong_args():
    """Each indicator of perfect is ≥ wrong_args and the indicator of the same name - upper and lower bounds sanity."""
    p = _agg("perfect")
    w = _agg("wrong_args")
    for k in ("exact_match", "name_match", "arg_set_f1", "arg_value_match"):
        assert p[k] >= w[k], f"perfect {k}={p[k]} < wrong_args {k}={w[k]}"


def test_higher_is_better_all_true():
    """All 4 metrics are higher-is-better — lock storage UI sort direction."""
    hib = BfclSlice().higher_is_better()
    assert hib == {
        "exact_match": True,
        "name_match": True,
        "arg_set_f1": True,
        "arg_value_match": True,
    }


# ============================================================
# Framework invariants (plan §6: relocking per task)
# ============================================================

def test_score_n_matches_gold():
    """n == the number of rows in the data set (to prevent the task's own codepath from returning early/leaking samples)."""
    task = BfclSlice()
    r = evaluate_score(task, PRED_DIR / "perfect.jsonl")
    assert r.n == 50


def test_score_missing_pred_raises(tmp_path):
    """Missing doc_id strict KeyError (same contract as sentiment/mt/qa_open)."""
    task = BfclSlice()
    partial = tmp_path / "partial.jsonl"
    partial.write_text(
        '{"id":"simple_python_NONE","prediction":"x()"}\n', encoding="utf-8",
    )
    with pytest.raises(KeyError):
        evaluate_score(task, partial)


def test_task_registered_under_correct_name():
    """`@register_task(\"bfcl_slice\")` side effect: CLI `--task bfcl_slice` resolves to this class."""
    from evals.registry import get_task
    assert isinstance(get_task("bfcl_slice"), BfclSlice)

"""bfcl_slice 单元 + e2e score 测试.

两层测试：
  ① **单元**：parse_function_call / score_function_call 在 handcrafted 输入上的合约
  ② **e2e**：BfclSlice + evaluate_score 跑 3 个 stub fixture
     （perfect / wrong_name / wrong_args），断言 4 项聚合指标的方向与界

按 plan §六 \"每个新 task 重锁 runner 不变量\"——n_matches_gold + missing_pred_raises 都补上.
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
# parse_function_call ─ 解析鲁棒性
# ============================================================

def test_parse_simple_kwargs():
    """干净输入：函数名 + 关键字参数 → 全字段填齐."""
    p = parse_function_call("foo(a=1, b='x')")
    assert p == {"func": "foo", "args": [], "kwargs": {"a": 1, "b": "x"}}


def test_parse_dotted_function_name():
    """`math.factorial` 等带 `.` 函数名 → dotted 字符串而非 Attribute repr."""
    p = parse_function_call("math.factorial(number=5)")
    assert p["func"] == "math.factorial"
    assert p["kwargs"] == {"number": 5}


def test_parse_positional_args_kept_separate():
    """positional → args 列表；scoring 层做 schema-properties-order 投影."""
    p = parse_function_call("foo(1, 2, c=3)")
    assert p["args"] == [1, 2]
    assert p["kwargs"] == {"c": 3}


def test_parse_strips_markdown_code_fence():
    """LLM 常输出 ```python\\nfoo(a=1)\\n``` —— 应剥外壳."""
    p = parse_function_call("```python\nfoo(a=1)\n```")
    assert p == {"func": "foo", "args": [], "kwargs": {"a": 1}}


def test_parse_strips_call_prefix():
    """Prompt 末尾是 `Call:`，模型偶尔会带回声 `Call: foo(...)`. 应剥前缀."""
    p = parse_function_call("Call: foo(a=1)")
    assert p == {"func": "foo", "args": [], "kwargs": {"a": 1}}


def test_parse_takes_first_nonempty_line():
    """多行输出取首行——generate_until 走 `\\n` stop 不会出现，但 score 路径可能传整段."""
    p = parse_function_call("foo(a=1)\nexplanation: ...")
    assert p == {"func": "foo", "args": [], "kwargs": {"a": 1}}


def test_parse_returns_none_on_unparseable():
    """彻底解析不出来的字符串 → None（score 据此判 0）."""
    assert parse_function_call("totally not a call") is None
    assert parse_function_call("") is None
    assert parse_function_call("foo(") is None  # 语法错


def test_parse_returns_none_on_non_call_expression():
    """`1 + 2` 是合法 Expression 但不是 Call → 拒收."""
    assert parse_function_call("1 + 2") is None


# ============================================================
# score_function_call ─ 4 项指标合约
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
    """name 对 + required arg 全在 + 值在 acceptable 列表 → 4 项全 1.0."""
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
    """name 错 → name_match=0 且 exact_match=0；arg_set_f1 / arg_value_match 仍按 arg 算."""
    out = score_function_call(
        "bar(a=1, b=2)",  # name 错
        gt_dict=_gt("foo", {"a": [1], "b": [2]}),
        schema=_schema("foo", ["a", "b"]),
    )
    assert out["name_match"] == 0.0
    assert out["exact_match"] == 0.0
    assert out["arg_set_f1"] == 1.0  # arg 名集合仍对得上
    assert out["arg_value_match"] == 1.0


def test_score_wrong_arg_value_drops_value_match():
    """name + arg 名都对，但值不在 acceptable → arg_value_match 拉低；exact_match=0."""
    out = score_function_call(
        "foo(a=999, b=2)",  # a 值错
        gt_dict=_gt("foo", {"a": [1], "b": [2]}),
        schema=_schema("foo", ["a", "b"]),
    )
    assert out["name_match"] == 1.0
    assert out["arg_set_f1"] == 1.0
    assert out["arg_value_match"] == 0.5  # 1/2 对
    assert out["exact_match"] == 0.0


def test_score_optional_arg_omitted_counts_as_match():
    """GT acceptable 含 \"\" → arg 可省略；pred 不传也得分."""
    out = score_function_call(
        "foo(a=1)",  # b 可省
        gt_dict=_gt("foo", {"a": [1], "b": ["", 0]}),  # b optional, default 0
        schema=_schema("foo", ["a", "b"], required=["a"]),
    )
    assert out["arg_value_match"] == 1.0  # a 对，b 省 ✓
    assert out["arg_set_f1"] == 1.0  # required={a}，pred={a}
    assert out["exact_match"] == 1.0


def test_score_optional_arg_explicit_value_also_matches():
    """pred 显式传 optional arg 的 default 值，也得分."""
    out = score_function_call(
        "foo(a=1, b=0)",  # b 显式传 default 0
        gt_dict=_gt("foo", {"a": [1], "b": ["", 0]}),
        schema=_schema("foo", ["a", "b"], required=["a"]),
    )
    assert out["arg_value_match"] == 1.0
    assert out["exact_match"] == 1.0


def test_score_unknown_arg_breaks_exact_match():
    """pred 多传 GT 没有的 arg → exact_match=0（即便 GT 部分都对）."""
    out = score_function_call(
        "foo(a=1, b=2, extra=99)",
        gt_dict=_gt("foo", {"a": [1], "b": [2]}),
        schema=_schema("foo", ["a", "b"]),
    )
    assert out["arg_value_match"] == 1.0
    # arg_set_f1 < 1：predicted set 多 1 个，precision 拉低
    assert 0.0 < out["arg_set_f1"] < 1.0
    assert out["exact_match"] == 0.0


def test_score_positional_arg_mapped_via_schema_order():
    """pred 用位置参数（无 kw）→ 按 schema.parameters.properties 顺序映射."""
    out = score_function_call(
        "foo(1, 2)",
        gt_dict=_gt("foo", {"a": [1], "b": [2]}),
        schema=_schema("foo", ["a", "b"]),
    )
    assert out["exact_match"] == 1.0
    assert out["arg_value_match"] == 1.0


def test_score_unparseable_pred_zero_all():
    """pred 解析失败 → 4 项全 0；artifact.parsed=None 给后续诊断."""
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
    """1.0 == 1（数值跨类型宽容；BFCL GT 偶尔 int，模型输出 float）."""
    out = score_function_call(
        "foo(a=1.0, b=2)",
        gt_dict=_gt("foo", {"a": [1], "b": [2]}),
        schema=_schema("foo", ["a", "b"]),
    )
    assert out["arg_value_match"] == 1.0


def test_score_value_match_excludes_bool_int_corner():
    """True != 1 在 BFCL 语义里——避免 \"a=True 蒙混 a=1\" 的伪阳性."""
    out = score_function_call(
        "foo(a=True)",
        gt_dict=_gt("foo", {"a": [1]}),
        schema=_schema("foo", ["a"]),
    )
    assert out["arg_value_match"] == 0.0


def test_score_value_match_acceptable_list_any_one():
    """acceptable 列表含 N 个值 → 命中任意一个即得分（BFCL 多 acceptable 语义）."""
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
    """perfect predictions = canonical target → 4 项聚合全 1.0."""
    agg = _agg("perfect")
    assert agg["exact_match"] == 1.0
    assert agg["name_match"] == 1.0
    assert agg["arg_set_f1"] == 1.0
    assert agg["arg_value_match"] == 1.0


def test_wrong_name_e2e_name_zero_args_one():
    """wrong_name = name 加 \"_xxx\"；name_match=0、exact_match=0；arg 维度仍接近 1."""
    agg = _agg("wrong_name")
    assert agg["name_match"] == 0.0
    assert agg["exact_match"] == 0.0
    # canonical target 都是 required-only kwargs → arg 名 set 与 GT 完全对得上
    assert agg["arg_set_f1"] == 1.0
    assert agg["arg_value_match"] == 1.0


def test_wrong_args_e2e_value_match_dominates_drop():
    """wrong_args = name 对 + 所有 required arg 值 perturb；
       name_match=1、arg_set_f1=1、arg_value_match 显著低、exact_match=0.
    """
    agg = _agg("wrong_args")
    assert agg["name_match"] == 1.0
    assert agg["arg_set_f1"] == 1.0
    # 极少数 GT 多 acceptable（如 unit=["units",""]），perturb 后 \"units\"+\"X\"
    # 已不在 acceptable，所以值匹配率应远低于 1
    assert agg["arg_value_match"] < 0.5
    assert agg["exact_match"] == 0.0


def test_perfect_strictly_dominates_wrong_args():
    """perfect 的每项指标都 ≥ wrong_args 同名指标——上下界 sanity."""
    p = _agg("perfect")
    w = _agg("wrong_args")
    for k in ("exact_match", "name_match", "arg_set_f1", "arg_value_match"):
        assert p[k] >= w[k], f"perfect {k}={p[k]} < wrong_args {k}={w[k]}"


def test_higher_is_better_all_true():
    """4 项指标都是 \"越高越好\"——锁住 storage UI 排序方向."""
    hib = BfclSlice().higher_is_better()
    assert hib == {
        "exact_match": True,
        "name_match": True,
        "arg_set_f1": True,
        "arg_value_match": True,
    }


# ============================================================
# 框架不变量（plan §六：每 task 重锁）
# ============================================================

def test_score_n_matches_gold():
    """n == 数据集行数（防 task 自身 codepath 提前 return / 漏样本）."""
    task = BfclSlice()
    r = evaluate_score(task, PRED_DIR / "perfect.jsonl")
    assert r.n == 50


def test_score_missing_pred_raises(tmp_path):
    """缺 doc_id 严格 KeyError（与 sentiment / mt / qa_open 同 contract）."""
    task = BfclSlice()
    partial = tmp_path / "partial.jsonl"
    partial.write_text(
        '{"id":"simple_python_NONE","prediction":"x()"}\n', encoding="utf-8",
    )
    with pytest.raises(KeyError):
        evaluate_score(task, partial)


def test_task_registered_under_correct_name():
    """`@register_task(\"bfcl_slice\")` 副作用：CLI `--task bfcl_slice` 能拿到本类."""
    from evals.registry import get_task
    assert isinstance(get_task("bfcl_slice"), BfclSlice)

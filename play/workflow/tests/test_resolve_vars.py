"""Workflow._resolve_vars 单测.

锁 ADR §3 子决策"vars 块"的三条规则：

  1. **type 强转**：spec.type ∈ {str/int/float/bool}，raw 总是 str（CLI 来的 --vars k=v），
     resolver 负责按声明类型 cast；bool 接受 1/true/yes/on（大小写无关）。
  2. **required vs default**：required=True 且 vars_input 未给 → `sys.exit`；
     未声明 required 时取 spec.default（缺省 ""）。
  3. **未声明的 vars_input 透传**：CLI 多给的 k=v 不被丢弃，按原 str 落进 state.vars——
     workflow 不做"未知字段拒绝"，让 hook 自己决定怎么用。
"""
from __future__ import annotations

import pytest

from workflow.runner import Workflow


def _wf(vars_spec: dict | None = None) -> Workflow:
    return Workflow(
        path="/tmp/w.yaml",
        name="w",
        description=None,
        vars_spec=vars_spec or {},
        hooks_module=None,
        stages=[],
        workflow_dir="/tmp",
    )


# ---------- type cast ---------------------------------------------------

def test_default_type_is_str():
    wf = _wf({"x": {"default": "hello"}})
    out = wf._resolve_vars({})
    assert out == {"x": "hello"}
    assert isinstance(out["x"], str)


def test_int_cast():
    wf = _wf({"n": {"type": "int", "default": "0"}})
    assert wf._resolve_vars({"n": "42"}) == {"n": 42}


def test_float_cast():
    wf = _wf({"r": {"type": "float", "default": "0.0"}})
    out = wf._resolve_vars({"r": "3.14"})
    assert out["r"] == pytest.approx(3.14)
    assert isinstance(out["r"], float)


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "Yes", "on", "ON"])
def test_bool_truthy_variants(raw):
    wf = _wf({"b": {"type": "bool", "default": "false"}})
    assert wf._resolve_vars({"b": raw}) == {"b": True}


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "", "anything"])
def test_bool_falsy_variants(raw):
    """任何不在 {1,true,yes,on}（lower）里的字符串都算 False——
    包括空串和误拼，没有"友好"中间态。"""
    wf = _wf({"b": {"type": "bool", "default": "true"}})
    assert wf._resolve_vars({"b": raw}) == {"b": False}


# ---------- required / default ------------------------------------------

def test_required_missing_exits():
    wf = _wf({"x": {"required": True}})
    with pytest.raises(SystemExit) as exc:
        wf._resolve_vars({})
    assert "requires --vars x=" in str(exc.value)


def test_required_provided_passes():
    wf = _wf({"x": {"required": True}})
    assert wf._resolve_vars({"x": "v"}) == {"x": "v"}


def test_default_used_when_not_provided():
    wf = _wf({"x": {"default": "fallback"}})
    assert wf._resolve_vars({}) == {"x": "fallback"}


def test_input_overrides_default():
    wf = _wf({"x": {"default": "fallback"}})
    assert wf._resolve_vars({"x": "given"}) == {"x": "given"}


def test_default_empty_when_neither_required_nor_default():
    """schema.validate 会拒绝这种 spec，但 _resolve_vars 自身的行为是
    `spec.get('default', '')`——这里直接构造跳过 schema 的 Workflow，
    锁住"resolver 不会崩"这条独立保证。"""
    wf = _wf({"x": {"type": "str"}})
    assert wf._resolve_vars({}) == {"x": ""}


# ---------- 未声明 vars 透传 --------------------------------------------

def test_unknown_var_passes_through_as_str():
    wf = _wf({})
    assert wf._resolve_vars({"unknown": "raw"}) == {"unknown": "raw"}


def test_declared_and_unknown_mixed():
    wf = _wf({"declared": {"default": "d"}})
    out = wf._resolve_vars({"declared": "x", "extra": "y"})
    assert out == {"declared": "x", "extra": "y"}


def test_unknown_var_not_cast_even_if_numeric_string():
    """未声明的 vars 没有 type spec，直接 str 透传——不做"看着像数字就 cast"猜测。"""
    wf = _wf({})
    out = wf._resolve_vars({"n": "42"})
    assert out == {"n": "42"}
    assert isinstance(out["n"], str)

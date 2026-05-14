"""state.interpolate / _lookup 单测.

锁三条 ADR §3 的隐式契约：

  1. **整字符串单占位保 Python 类型**——`"{{ a.b }}"`（前后无空白、无其它字符）
     直接返回 lookup 的 Python 对象（list / dict / int / None），不 str()。
  2. **任意 padding → inline str()**——前后多一个空格、或与字面文本混合的占位，
     都走 `VAR_RE.sub(str(...))` 路径，结果一定是 str。
  3. **miss 直接抛 KeyError**——不静默替成 ""，不给"你大概想用 X"提示；
     hit non-dict 时给出 `"workflow path 'a.b' hit non-dict at segment 'b'"`
     形状的消息（容许文案演化，只锁包含"non-dict"片段）。
"""
from __future__ import annotations

import pytest

from workflow.state import interpolate


def _state(vars_: dict | None = None, stages: dict | None = None) -> dict:
    return {"vars": vars_ or {}, "stages": stages or {}, "pkg_dir": "/tmp"}


# ---------- 整字符串单占位：保 Python 类型 ---------------------------------

def test_sole_placeholder_returns_int_unchanged():
    state = _state(vars_={"n": 42})
    assert interpolate("{{ vars.n }}", state) == 42
    assert isinstance(interpolate("{{ vars.n }}", state), int)


def test_sole_placeholder_returns_list_unchanged():
    state = _state(vars_={"items": [1, 2, 3]})
    out = interpolate("{{ vars.items }}", state)
    assert out == [1, 2, 3]
    assert isinstance(out, list)


def test_sole_placeholder_returns_dict_unchanged():
    state = _state(stages={"s": {"output": {"a": 1, "b": [2]}}})
    out = interpolate("{{ stages.s.output }}", state)
    assert out == {"a": 1, "b": [2]}
    assert isinstance(out, dict)


def test_sole_placeholder_returns_none_unchanged():
    state = _state(vars_={"x": None})
    assert interpolate("{{ vars.x }}", state) is None


# ---------- padding / 混合：强制 str() -------------------------------------

def test_inline_placeholder_with_literal_prefix_forces_str():
    state = _state(vars_={"n": 42})
    out = interpolate("count={{ vars.n }}", state)
    assert out == "count=42"
    assert isinstance(out, str)


def test_inline_placeholder_with_literal_suffix_forces_str():
    state = _state(vars_={"n": 42})
    assert interpolate("{{ vars.n }} items", state) == "42 items"


def test_sole_placeholder_with_surrounding_whitespace_forces_str():
    """前后空白虽然 strip 后是单占位，但 `value.strip() == value` 失败，
    走 sub 路径——整字符串保类型只对**完全干净**的单占位生效。"""
    state = _state(vars_={"n": 42})
    out = interpolate("  {{ vars.n }}  ", state)
    assert out == "  42  "
    assert isinstance(out, str)


def test_multiple_placeholders_force_str():
    state = _state(vars_={"a": 1, "b": 2})
    assert interpolate("{{ vars.a }}-{{ vars.b }}", state) == "1-2"


def test_inline_dict_lookup_renders_repr_via_str():
    """inline 占位上 dict 会被 str() 渲染成 Python repr 形态——
    这是 ADR §3 的有意行为（数据转换应该走 hook，不该靠 inline 模板）。"""
    state = _state(stages={"s": {"output": {"a": 1}}})
    out = interpolate("got {{ stages.s.output }}", state)
    assert isinstance(out, str)
    assert "{'a': 1}" in out


# ---------- 递归插值：dict / list -----------------------------------------

def test_interpolate_recurses_into_dict():
    state = _state(vars_={"name": "alice", "n": 3})
    out = interpolate({"who": "{{ vars.name }}", "count": "{{ vars.n }}"}, state)
    assert out == {"who": "alice", "count": 3}


def test_interpolate_recurses_into_list():
    state = _state(vars_={"x": 1, "y": 2})
    out = interpolate(["{{ vars.x }}", "{{ vars.y }}", "lit"], state)
    assert out == [1, 2, "lit"]


def test_interpolate_nested_dict_in_list():
    state = _state(vars_={"v": "hello"})
    out = interpolate([{"k": "{{ vars.v }}"}], state)
    assert out == [{"k": "hello"}]


# ---------- 非字符串：原样返回 --------------------------------------------

@pytest.mark.parametrize("value", [42, 3.14, True, False, None])
def test_non_string_scalar_returned_as_is(value):
    assert interpolate(value, _state()) is value


def test_string_without_placeholder_passes_through():
    assert interpolate("literal text", _state()) == "literal text"


# ---------- miss / 错误路径：抛 KeyError（不静默） -------------------------

def test_missing_top_level_key_raises_keyerror():
    with pytest.raises(KeyError):
        interpolate("{{ vars.absent }}", _state(vars_={}))


def test_missing_nested_key_raises_keyerror():
    state = _state(stages={"s": {"output": {"a": 1}}})
    with pytest.raises(KeyError):
        interpolate("{{ stages.s.output.b }}", state)


def test_lookup_hits_non_dict_raises_with_diagnostic():
    """`vars.x.y`：x 是 int 不是 dict，应抛带 'non-dict' 字样的 KeyError，
    而不是普通 dict 'y' KeyError——这是 state.py 显式包装的诊断信息。"""
    state = _state(vars_={"x": 42})
    with pytest.raises(KeyError, match="non-dict"):
        interpolate("{{ vars.x.y }}", state)


def test_missing_key_inside_inline_substitution_raises():
    """inline 模板里同样抛 KeyError，而非 silently 替成空串。"""
    with pytest.raises(KeyError):
        interpolate("prefix-{{ vars.absent }}", _state(vars_={}))

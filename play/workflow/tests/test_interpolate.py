"""state.interpolate / _lookup tests.

Locks three implicit ADR §3 contracts:

  1. **Whole-string single placeholder preserves Python type** — `"{{ a.b }}"` (no
     surrounding whitespace or other chars) returns lookup object (list / dict / int / None), no str().
  2. **Any padding → inline str()** — leading/trailing space or mixed literal text uses
     `VAR_RE.sub(str(...))`; result is always str.
  3. **miss raises KeyError** — no silent ""; no "did you mean X" hints;
     non-dict segment yields message containing `"non-dict"` (wording may evolve).
"""
from __future__ import annotations

import pytest

from workflow.state import interpolate


def _state(vars_: dict | None = None, stages: dict | None = None) -> dict:
    return {"vars": vars_ or {}, "stages": stages or {}, "pkg_dir": "/tmp"}


# ---------- whole-string single placeholder: preserve Python type ----------

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


# ---------- padding / mixed: force str() -----------------------------------

def test_inline_placeholder_with_literal_prefix_forces_str():
    state = _state(vars_={"n": 42})
    out = interpolate("count={{ vars.n }}", state)
    assert out == "count=42"
    assert isinstance(out, str)


def test_inline_placeholder_with_literal_suffix_forces_str():
    state = _state(vars_={"n": 42})
    assert interpolate("{{ vars.n }} items", state) == "42 items"


def test_sole_placeholder_with_surrounding_whitespace_forces_str():
    """Leading/trailing whitespace may strip to a single placeholder, but
    `value.strip() == value` fails → sub path — type preservation only for
    ** perfectly clean ** single placeholders."""
    state = _state(vars_={"n": 42})
    out = interpolate("  {{ vars.n }}  ", state)
    assert out == "  42  "
    assert isinstance(out, str)


def test_multiple_placeholders_force_str():
    state = _state(vars_={"a": 1, "b": 2})
    assert interpolate("{{ vars.a }}-{{ vars.b }}", state) == "1-2"


def test_inline_dict_lookup_renders_repr_via_str():
    """Inline placeholder on dict renders via str() to Python repr —
    intentional ADR §3 behavior (transforms belong in hooks, not inline templates)."""
    state = _state(stages={"s": {"output": {"a": 1}}})
    out = interpolate("got {{ stages.s.output }}", state)
    assert isinstance(out, str)
    assert "{'a': 1}" in out


# ---------- recursive interpolation: dict / list ---------------------------

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


# ---------- non-strings: pass through ------------------------------------

@pytest.mark.parametrize("value", [42, 3.14, True, False, None])
def test_non_string_scalar_returned_as_is(value):
    assert interpolate(value, _state()) is value


def test_string_without_placeholder_passes_through():
    assert interpolate("literal text", _state()) == "literal text"


# ---------- miss / error path: raise KeyError (no silent empty) ------------

def test_missing_top_level_key_raises_keyerror():
    with pytest.raises(KeyError):
        interpolate("{{ vars.absent }}", _state(vars_={}))


def test_missing_nested_key_raises_keyerror():
    state = _state(stages={"s": {"output": {"a": 1}}})
    with pytest.raises(KeyError):
        interpolate("{{ stages.s.output.b }}", state)


def test_lookup_hits_non_dict_raises_with_diagnostic():
    """`vars.x.y`: x is int not dict → KeyError containing 'non-dict',
    not plain dict 'y' KeyError — state.py wrapped diagnostic."""
    state = _state(vars_={"x": 42})
    with pytest.raises(KeyError, match="non-dict"):
        interpolate("{{ vars.x.y }}", state)


def test_missing_key_inside_inline_substitution_raises():
    """Inline templates also raise KeyError, never silently substitute empty string."""
    with pytest.raises(KeyError):
        interpolate("prefix-{{ vars.absent }}", _state(vars_={}))

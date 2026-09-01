"""Workflow._resolve_vars tests.

Locks ADR §3 sub-decision "vars block" three rules:

  1. **type cast**: spec.type ∈ {str/int/float/bool}, raw always str (from CLI --vars k=v),
     resolver casts; bool accepts 1/true/yes/on (case insensitive).
  2. **required vs default**: required=True and vars_input missing → `sys.exit`;
     without required, use spec.default (default "").
  3. **Undeclared vars_input pass-through**: extra CLI k=v not dropped, land in state.vars as str —
     workflow does not reject unknown fields; hooks decide usage.
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
    """Any string not in {1,true,yes,on} (lower) is False —
    including empty string and typos; no friendly middle ground."""
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
    """schema.validate rejects this spec, but _resolve_vars behavior is
    `spec.get('default', '')` — construct Workflow skipping schema here to lock
    independent guarantee that resolver does not crash."""
    wf = _wf({"x": {"type": "str"}})
    assert wf._resolve_vars({}) == {"x": ""}


# ---------- undeclared vars pass-through -----------------------------------

def test_unknown_var_passes_through_as_str():
    wf = _wf({})
    assert wf._resolve_vars({"unknown": "raw"}) == {"unknown": "raw"}


def test_declared_and_unknown_mixed():
    wf = _wf({"declared": {"default": "d"}})
    out = wf._resolve_vars({"declared": "x", "extra": "y"})
    assert out == {"declared": "x", "extra": "y"}


def test_unknown_var_not_cast_even_if_numeric_string():
    """Undeclared vars have no type spec — pass through as str; no guess-cast from numeric-looking strings."""
    wf = _wf({})
    out = wf._resolve_vars({"n": "42"})
    assert out == {"n": "42"}
    assert isinstance(out["n"], str)

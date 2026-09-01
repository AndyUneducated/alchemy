"""executors.deterministic._resolve_fn tests.

Locks ADR §3 sub-decision "fn string dual resolution":

  - With colon `pkg.sub:func` → full-path import, ignores hooks_module
  - Without colon → must have top-level `hooks_module`, else fail-fast;
    bare name uses `hooks_module` default namespace

Uses stdlib `os.path` as import target to avoid temp module files — `os.path:join`
and bare `join` + hooks_module=`os.path` are the same callable.
"""
from __future__ import annotations

import os.path

import pytest

from workflow.executors.deterministic import _resolve_fn


# ---------- colon form: full path ------------------------------------------

def test_colon_form_resolves_to_callable():
    fn = _resolve_fn("os.path:join", hooks_module=None)
    assert fn is os.path.join


def test_colon_form_ignores_hooks_module():
    """Explicit module:callable ignores hooks_module —
    `os.path:join` is always os.path.join even if hooks_module is another module."""
    fn = _resolve_fn("os.path:join", hooks_module="json")
    assert fn is os.path.join


def test_colon_form_nonexistent_module_raises():
    with pytest.raises(ModuleNotFoundError):
        _resolve_fn("nonexistent_pkg_xyz:func", hooks_module=None)


def test_colon_form_nonexistent_attr_raises():
    with pytest.raises(AttributeError):
        _resolve_fn("os.path:no_such_func", hooks_module=None)


# ---------- bare form: via hooks_module ------------------------------------

def test_bare_name_uses_hooks_module():
    fn = _resolve_fn("join", hooks_module="os.path")
    assert fn is os.path.join


def test_bare_name_without_hooks_module_exits():
    with pytest.raises(SystemExit) as exc:
        _resolve_fn("join", hooks_module=None)
    msg = str(exc.value)
    assert msg.startswith("Error:"), msg
    assert "hooks_module" in msg


def test_bare_name_empty_hooks_module_exits():
    """`hooks_module=""` should fail-fast like `None` — empty string is not declared."""
    with pytest.raises(SystemExit):
        _resolve_fn("join", hooks_module="")


def test_bare_name_nonexistent_in_hooks_module_raises():
    with pytest.raises(AttributeError):
        _resolve_fn("no_such_func", hooks_module="os.path")

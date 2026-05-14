"""executors.deterministic._resolve_fn 单测.

锁 ADR §3 子决策"fn 字符串双解析"：

  - 含冒号 `pkg.sub:func` → 完整路径 import，忽略 hooks_module
  - 不含冒号 → 必须有顶层 `hooks_module`，否则 fail-fast；
    bare name 走 `hooks_module` 默认 namespace

用 stdlib `os.path` 作为可 import 的目标，避免临时建模块文件——`os.path:join`
与 bare `join` + hooks_module=`os.path` 是同一个 callable。
"""
from __future__ import annotations

import os.path

import pytest

from workflow.executors.deterministic import _resolve_fn


# ---------- 冒号形态：完整路径 ------------------------------------------

def test_colon_form_resolves_to_callable():
    fn = _resolve_fn("os.path:join", hooks_module=None)
    assert fn is os.path.join


def test_colon_form_ignores_hooks_module():
    """显式 module:callable 时 hooks_module 不参与解析——
    `os.path:join` 永远是 os.path.join，即使 hooks_module 是别的模块。"""
    fn = _resolve_fn("os.path:join", hooks_module="json")
    assert fn is os.path.join


def test_colon_form_nonexistent_module_raises():
    with pytest.raises(ModuleNotFoundError):
        _resolve_fn("nonexistent_pkg_xyz:func", hooks_module=None)


def test_colon_form_nonexistent_attr_raises():
    with pytest.raises(AttributeError):
        _resolve_fn("os.path:no_such_func", hooks_module=None)


# ---------- bare 形态：走 hooks_module ----------------------------------

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
    """`hooks_module=""` 应与 `None` 同样 fail-fast——空串不算"声明了"。"""
    with pytest.raises(SystemExit):
        _resolve_fn("join", hooks_module="")


def test_bare_name_nonexistent_in_hooks_module_raises():
    with pytest.raises(AttributeError):
        _resolve_fn("no_such_func", hooks_module="os.path")

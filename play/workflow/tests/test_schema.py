"""schema.validate fail-fast 边界单测.

ADR §3 立的契约："必填缺失 → `sys.exit('Error: ...')`，不给猜词提示，
不做 schema migration"。本测试集**只锁形状**——`SystemExit` + `"Error: workflow.yaml: "`
前缀，**不锁具体文案**，给未来文案演化留空间。

每条 sys.exit 分支一个用例，缺一个就允许某种错误形态被 silently 引入。
"""
from __future__ import annotations

import pytest

from workflow import schema


# ---------- 工具 ---------------------------------------------------------

_PREFIX = "Error: workflow.yaml:"


def _assert_exits(meta: object) -> SystemExit:
    """统一捕获 + 前缀断言；返回 exc 让 caller 可做更细的 match。"""
    with pytest.raises(SystemExit) as exc:
        schema.validate(meta)  # type: ignore[arg-type]
    msg = str(exc.value)
    assert msg.startswith(_PREFIX), f"expected fail-fast prefix, got: {msg!r}"
    return exc.value


def _det_stage(name: str = "s", fn: str = "mod:f") -> dict:
    return {"name": name, "type": "deterministic", "fn": fn}


def _agent_stage(name: str = "a", scenario: str = "s.md") -> dict:
    return {"name": name, "type": "agent", "scenario": scenario}


# ---------- 正路径：最小合法形态 -------------------------------------------

def test_minimum_valid_deterministic_workflow():
    schema.validate({"name": "w", "stages": [_det_stage()]})


def test_minimum_valid_agent_workflow():
    schema.validate({"name": "w", "stages": [_agent_stage()]})


def test_valid_vars_required_and_default():
    schema.validate({
        "name": "w",
        "vars": {
            "x": {"required": True},
            "y": {"default": "z"},
            "n": {"type": "int", "default": "0"},
        },
        "stages": [_det_stage()],
    })


# ---------- 顶层 --------------------------------------------------------

def test_top_level_not_mapping_exits():
    _assert_exits([])  # type: ignore[arg-type]


def test_missing_name_exits():
    _assert_exits({"stages": [_det_stage()]})


def test_empty_name_exits():
    _assert_exits({"name": "   ", "stages": [_det_stage()]})


def test_name_not_string_exits():
    _assert_exits({"name": 123, "stages": [_det_stage()]})


# ---------- stages 顶层形态 --------------------------------------------

def test_stages_not_list_exits():
    _assert_exits({"name": "w", "stages": "not-a-list"})


def test_stages_empty_exits():
    _assert_exits({"name": "w", "stages": []})


def test_stage_not_mapping_exits():
    _assert_exits({"name": "w", "stages": ["not-a-mapping"]})


# ---------- 单 stage 必填 / 类型 ---------------------------------------

def test_stage_missing_name_exits():
    _assert_exits({"name": "w", "stages": [{"type": "deterministic", "fn": "m:f"}]})


def test_stage_empty_name_exits():
    _assert_exits({"name": "w", "stages": [_det_stage(name="  ")]})


def test_duplicate_stage_names_exits():
    _assert_exits({
        "name": "w",
        "stages": [_det_stage(name="x"), _det_stage(name="x")],
    })


def test_unknown_stage_type_exits():
    _assert_exits({
        "name": "w",
        "stages": [{"name": "x", "type": "shell", "fn": "m:f"}],
    })


def test_deterministic_missing_fn_exits():
    _assert_exits({
        "name": "w",
        "stages": [{"name": "x", "type": "deterministic"}],
    })


def test_deterministic_empty_fn_exits():
    _assert_exits({
        "name": "w",
        "stages": [{"name": "x", "type": "deterministic", "fn": "   "}],
    })


def test_agent_missing_scenario_exits():
    _assert_exits({
        "name": "w",
        "stages": [{"name": "x", "type": "agent"}],
    })


def test_agent_empty_scenario_exits():
    _assert_exits({
        "name": "w",
        "stages": [{"name": "x", "type": "agent", "scenario": ""}],
    })


# ---------- vars 块 -----------------------------------------------------

def test_vars_not_mapping_exits():
    _assert_exits({"name": "w", "vars": [], "stages": [_det_stage()]})


def test_var_spec_not_mapping_exits():
    _assert_exits({
        "name": "w",
        "vars": {"x": "not-a-mapping"},
        "stages": [_det_stage()],
    })


def test_var_required_not_bool_exits():
    _assert_exits({
        "name": "w",
        "vars": {"x": {"required": "yes"}},
        "stages": [_det_stage()],
    })


def test_var_invalid_type_exits():
    _assert_exits({
        "name": "w",
        "vars": {"x": {"type": "list", "default": "[]"}},
        "stages": [_det_stage()],
    })


def test_var_missing_required_and_default_exits():
    _assert_exits({
        "name": "w",
        "vars": {"x": {"type": "str"}},
        "stages": [_det_stage()],
    })

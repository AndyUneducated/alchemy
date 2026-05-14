"""data/scenarios/{tool_chain,code_review}_fast.md — YAML frontmatter 解析 + 契约检查.

Fast scenario YAML 是 mining 流水线的根输入（[DECISIONS §11](../DECISIONS.md)），
但目前没有单测能拦"YAML 拼错 / require_tool 字段写丢 / max_retries 不是 0"这类
事故——一旦改坏，整条 mine_triples → synthesize → format 全产 0 triple 才会发现.

本测做两层防：
  ① `Scenario.from_yaml` 能 round-trip（agent_engine schema 校验过线）
  ② fast scenario 的 invariants：≥1 step 有 require_tool；fast 路径的 max_retries=0
     （[DECISIONS §11] 决策核心）
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_engine import Scenario  # type: ignore[import-not-found]


REPO_ROOT = Path(__file__).resolve().parents[3]
FAST_SCENARIOS_DIR = REPO_ROOT / "play" / "agent_sft" / "data" / "scenarios"
FAST_SCENARIOS = ["tool_chain_fast", "code_review_fast"]


@pytest.mark.parametrize("name", FAST_SCENARIOS)
def test_fast_scenario_parses(name):
    """Scenario.from_yaml 不抛 + meta 含必需字段."""
    path = FAST_SCENARIOS_DIR / f"{name}.md"
    if not path.exists():
        pytest.skip(f"fast scenario not present: {path}")
    scen = Scenario.from_yaml(str(path))
    meta = scen.meta
    assert meta.get("agents"), f"{name}: agents block missing"
    assert meta.get("steps"), f"{name}: steps block missing"


@pytest.mark.parametrize("name", FAST_SCENARIOS)
def test_fast_scenario_has_require_tool_steps(name):
    """fast scenario 服务 mining——必须至少 1 个 step 含 require_tool，否则 0 fire."""
    path = FAST_SCENARIOS_DIR / f"{name}.md"
    if not path.exists():
        pytest.skip(f"fast scenario not present: {path}")
    steps = Scenario.from_yaml(str(path)).meta["steps"]
    require_tool_steps = [s for s in steps if s.get("require_tool")]
    assert require_tool_steps, (
        f"{name}: no step has `require_tool` — synthesize 路径将产 0 triple"
    )


@pytest.mark.parametrize("name", FAST_SCENARIOS)
def test_fast_scenario_max_retries_is_zero(name):
    """[DECISIONS §11](../DECISIONS.md): fast 副本核心提速决策 = max_retries=0.

    若某 step 漏改回 1，mining wall clock 翻倍（25s → 65s）且与 baseline eval
    不再可对比. 本测拦这条 invariant.
    """
    path = FAST_SCENARIOS_DIR / f"{name}.md"
    if not path.exists():
        pytest.skip(f"fast scenario not present: {path}")
    steps = Scenario.from_yaml(str(path)).meta["steps"]
    bad = [
        (s.get("id"), s.get("max_retries"))
        for s in steps
        if s.get("require_tool") and s.get("max_retries", 0) != 0
    ]
    assert not bad, (
        f"{name}: require_tool step(s) with max_retries != 0 violate fast-copy "
        f"contract: {bad}"
    )

"""data/scenarios/{tool_chain,code_review}_fast.md — YAML frontmatter parsing + contract checking.

Fast scenario YAML is the root input of the mining pipeline ([DECISIONS §11](../DECISIONS.md)),
But currently there is no single test that can block "YAML is misspelled / require_tool field is missing / max_retries is not 0".
Accident - Once modified, the entire mine_triples → synthesize → format full production 0 triple will be discovered.

This test uses two layers of defense:
  ① `Scenario.from_yaml` can round-trip (agent_engine schema verification passes)
  ② Invariants of fast scenario: ≥1 step has require_tool; max_retries=0 of fast path
     ([DECISIONS §11] Decision Core)"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_engine import Scenario  # type: ignore[import-not-found]


REPO_ROOT = Path(__file__).resolve().parents[3]
FAST_SCENARIOS_DIR = REPO_ROOT / "play" / "agent_sft" / "data" / "scenarios"
FAST_SCENARIOS = ["tool_chain_fast", "code_review_fast"]


@pytest.mark.parametrize("name", FAST_SCENARIOS)
def test_fast_scenario_parses(name):
    """Scenario.from_yaml does not throw + meta with required fields."""
    path = FAST_SCENARIOS_DIR / f"{name}.md"
    if not path.exists():
        pytest.skip(f"fast scenario not present: {path}")
    scen = Scenario.from_yaml(str(path))
    meta = scen.meta
    assert meta.get("agents"), f"{name}: agents block missing"
    assert meta.get("steps"), f"{name}: steps block missing"


@pytest.mark.parametrize("name", FAST_SCENARIOS)
def test_fast_scenario_has_require_tool_steps(name):
    """fast scenario service mining - must have at least 1 step containing require_tool, otherwise 0 fire."""
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
    """[DECISIONS §11](../DECISIONS.md): fast replica core speedup decision = max_retries=0.

    If a step is changed back to 1, the mining wall clock doubles (25s → 65s) and is consistent with the baseline eval
    No longer comparable. This test blocks this invariant."""
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

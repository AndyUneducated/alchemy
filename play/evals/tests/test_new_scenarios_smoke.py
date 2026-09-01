"""agent_sft phase 1.B adds scenario schema smoke test.

verify `code_review.md` / `tool_chain.md` via `agent_engine.scenario.Scenario.from_yaml`
Full schema verification (agents/steps/role reachability/tool_owners, etc.). Runs in subprocess
Keep the "play subprojects do not import each other" constraint in workshops.mdc.

Why not put `play/agent_engine/tests/`: agent_engine has no tests directory in history, press
workshops.mdc "no new tooling" does not introduce new test infrastructure; put evals/tests/ to reuse existing
pytest config + agent_engine_required gate is the least intrusive option."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .conftest import REPO_ROOT, agent_engine_required

PLAY_DIR = REPO_ROOT / "play"
NEW_SCENARIOS = (
    PLAY_DIR / "agent_engine" / "scenarios" / "code_review.md",
    PLAY_DIR / "agent_engine" / "scenarios" / "tool_chain.md",
)


@agent_engine_required
def test_new_scenarios_pass_agent_engine_schema():
    """Both new scenarios can be received by agent_engine.scenario.Scenario.from_yaml (including
    role reachability / tool_owners verification, etc. Full schema check).

    subprocess instead of direct import: respect workshops.mdc "play sub-projects do not import each other" + compatible
    BACKEND-conditional client import inside agent_engine."""
    paths_arg = ", ".join(repr(str(p)) for p in NEW_SCENARIOS)
    code = (
        "import sys; sys.path.insert(0, 'play'); "
        "from agent_engine.scenario import Scenario; "
        f"[Scenario.from_yaml(p) for p in [{paths_arg}]]; "
        "print('SCHEMA_OK')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0, (
        f"schema validation failed: returncode={proc.returncode}\n"
        f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    )
    assert "SCHEMA_OK" in proc.stdout


def test_new_scenario_files_exist_and_have_frontmatter():
    """Quick smoke test - file exists + has at least one set of `---` frontmatter tags. Does not depend on ollama,
    Can always run locally (no skip); agent_engine_required is not required."""
    for path in NEW_SCENARIOS:
        assert path.exists(), f"missing scenario: {path}"
        text = path.read_text(encoding="utf-8")
        # At least 2 `---` lines (frontmatter on and off)
        assert text.count("\n---\n") >= 2 or text.count("\n---") >= 2, (
            f"scenario {path} appears to lack YAML frontmatter delimiters"
        )


def test_new_scenarios_appear_in_nudge_gold():
    """New scenario has been added nudge_fire_rate gold.jsonl - sentinel to prevent missing updates."""
    from evals.tasks.nudge_fire_rate import NudgeFireRate
    docs = list(NudgeFireRate().docs())
    ids = {d.id for d in docs}
    assert "code_review" in ids
    assert "tool_chain" in ids

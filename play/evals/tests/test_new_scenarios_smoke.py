"""agent_sft phase 1.B 新增 scenario 的 schema 烟测.

verify `code_review.md` / `tool_chain.md` 通过 `agent_engine.scenario.Scenario.from_yaml`
全 schema 校验（agents / steps / role 可达性 / tool_owners 等）. 跑在 subprocess 里
保持 workshops.mdc "play 子项目互不 import" 约束.

为什么不放 `play/agent_engine/tests/`：agent_engine 历史上无 tests 目录，按
workshops.mdc "no new tooling" 不引入新测试基础设施；放 evals/tests/ 复用现有
pytest config + agent_engine_required gate 是最小侵入选择.
"""

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
    """两个新 scenario 均能被 agent_engine.scenario.Scenario.from_yaml 接收（含
    role 可达性 / tool_owners 校验等全 schema 检查）.

    subprocess 而非直接 import：respect workshops.mdc "play 子项目互不 import" + 兼容
    agent_engine 内部的 BACKEND-conditional client import.
    """
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
    """快烟测——文件存在 + 至少有一组 `---` frontmatter 标记. 不依赖 ollama，
    本地总能跑（no skip）；agent_engine_required 不需要."""
    for path in NEW_SCENARIOS:
        assert path.exists(), f"missing scenario: {path}"
        text = path.read_text(encoding="utf-8")
        # 至少 2 个 `---` line（frontmatter 开闭）
        assert text.count("\n---\n") >= 2 or text.count("\n---") >= 2, (
            f"scenario {path} appears to lack YAML frontmatter delimiters"
        )


def test_new_scenarios_appear_in_nudge_gold():
    """新 scenario 已加入 nudge_fire_rate gold.jsonl——sentinel 防止漏更新."""
    from evals.tasks.nudge_fire_rate import NudgeFireRate
    docs = list(NudgeFireRate().docs())
    ids = {d.id for d in docs}
    assert "code_review" in ids
    assert "tool_chain" in ids

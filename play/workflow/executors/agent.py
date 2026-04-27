from __future__ import annotations

import os
from typing import Any

from agent_engine import Engine, Scenario


def run(stage: dict, config: dict, *, workflow_dir: str) -> dict[str, str]:
    rel_scenario = stage["scenario"]
    scenario_path = (
        rel_scenario
        if os.path.isabs(rel_scenario)
        else os.path.normpath(os.path.join(workflow_dir, rel_scenario))
    )
    scenario = Scenario.from_yaml(scenario_path)
    engine = Engine(scenario)
    result = engine.invoke(**config)
    return result.artifact

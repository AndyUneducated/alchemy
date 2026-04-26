"""Agent stage executor: run an ``agent_engine.Engine.invoke()`` call.

Workflow does not parse the stage's ``config:`` block — it interpolates
template references inside, then unpacks the dict as kwargs to
``Engine.invoke(**config)``. This keeps agent_engine internal naming
(``initial_artifact / save_transcript / callbacks``) out of the workflow
schema (plan §4.3).
"""

from __future__ import annotations

import os
from typing import Any

from agent_engine import Engine, Scenario


def run(stage: dict, config: dict, *, workflow_dir: str) -> dict[str, str]:
    """Resolve the stage's scenario path, build an Engine, invoke, return artifact dict.

    The scenario path in YAML is relative to the workflow.yaml file's
    directory, mirroring how ``scenario.py`` resolves tool ``vdb_dir`` paths
    relative to the scenario .md file.
    """
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

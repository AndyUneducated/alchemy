"""``play/workflow``: declarative deterministic pipeline runner with agent stages.

Public API:

    from workflow import Workflow

    wf = Workflow.from_yaml("workflows/qa_supervisor.yaml")
    state = wf.run({"csv_path": "examples/requirements.csv"})

CLI entry point:

    python -m workflow run <workflow.yaml> --vars k=v [--vars k=v ...]
"""

from .runner import Workflow

__all__ = ["Workflow"]

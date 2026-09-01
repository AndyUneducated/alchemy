"""agent_sft tests the public configuration.

`agent_sft/eval/` and `agent_sft/data/` are not external export packages (`__init__.py`
Only install docstring), in order to allow the tests under `tests/` to be naked `from aggregate_seeds import …` /
`from extractor import …`, add these two directories to sys.path. At the same time, add `play/` to sys.path
Make `from agent_engine import ...` / `from evals.metrics... import ...` available directly -
Otherwise, these cross-project imports can only rely on side-effect injection when loading business modules such as `extractor.py` and run separately.
`test_scenario_yaml.py` (which does not rely on extractor) will fail with collection-time order sensitivity.

Advantages of receiving conftest instead of individual `sys.path.insert` for each test file:
  ① Keep the test files pure (only import business modules, no path gymnastics)
  ② New tests (such as `test_run_baseline.py` / `test_formatter.py`) will be added in the future with zero boilerplate
  ③ The order of test file collection is irrelevant - pytest-randomly / xdist is also stable"""

from __future__ import annotations

import sys
from pathlib import Path

_PARENT = Path(__file__).resolve().parent.parent  # play/agent_sft
_PLAY = _PARENT.parent  # play/

for _dir in (_PLAY, _PARENT / "eval", _PARENT / "data"):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

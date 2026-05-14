"""agent_sft 测试公共配置。

`agent_sft/eval/` 与 `agent_sft/data/` 不是对外 export 的 package（`__init__.py`
仅装 docstring），为让 `tests/` 下的测试能裸 `from aggregate_seeds import …` /
`from extractor import …`，把这两个目录加进 sys.path. 同时把 `play/` 加进 sys.path
让 `from agent_engine import ...` / `from evals.metrics... import ...` 直接可用——
否则这些跨项目 import 只能靠 `extractor.py` 等业务模块加载时的副作用注入，单独跑
`test_scenario_yaml.py`（不依赖 extractor）会以 collection-time 顺序敏感 fail.

收到 conftest 而非每个测试文件各自 `sys.path.insert` 的好处：
  ① 测试文件保持纯净（只 import 业务模块，无 path 体操）
  ② 未来加新测试（如 `test_run_baseline.py` / `test_formatter.py`）零样板
  ③ 测试文件 collection 顺序无关——pytest-randomly / xdist 也稳
"""

from __future__ import annotations

import sys
from pathlib import Path

_PARENT = Path(__file__).resolve().parent.parent  # play/agent_sft
_PLAY = _PARENT.parent  # play/

for _dir in (_PLAY, _PARENT / "eval", _PARENT / "data"):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

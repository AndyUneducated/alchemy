"""agent_sft 测试公共配置。

`agent_sft/eval/` 不是个对外 export 的 package（`__init__.py` 仅装 docstring），
为让 `tests/` 下的测试能裸 `from aggregate_seeds import …`，把 `eval/` 加进 sys.path.

收到 conftest 而非每个测试文件各自 `sys.path.insert` 的好处：
  ① 测试文件保持纯净（只 import 业务模块，无 path 体操）
  ② 未来加新测试（如 `test_run_baseline.py`）零样板
"""

from __future__ import annotations

import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

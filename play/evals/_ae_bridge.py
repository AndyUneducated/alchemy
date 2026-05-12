"""Bridge module：在 evals 进程内直接 import `agent_engine` typed view（DECISIONS §13 / §16）.

`play/evals` 与 `play/agent_engine` 是同 monorepo 的姊妹包. 历史上 evals 通过 subprocess
+ JSON envelope 与 agent_engine 解耦（DECISIONS §4 / §11）；但 transcript / scenario
解读视图必须 in-process（`Result.tool_calls() / .turns()` / `Scenario.expanded_turns()`
是纯函数级 schema 解读，每个 sample 都要调，subprocess 化 ~1-2s 冷启动会让评测时长爆炸）.

本 bridge 把"sys.path 注入 + 集中 import"收一处，各 metric / task 模块从这里 re-export
即可，不再各自反复 `sys.path.insert(...)` + `try/finally` 清理.

§16 起额外 re-export：`TranscriptEntry` typed union（6 个具体 entry class）+ `TokenUsage`，
让 evals consumer 用 isinstance dispatch 取字段，不再 `entry.get("...")` 防御.

pip install 边界与 import 边界正交（同 DECISIONS §14 思路）：evals 的 requirements.txt
不需把 agent_engine 列为依赖（它在同源码树）；fresh checkout `pip install -r
play/evals/requirements.txt` 后 import 路径自动可达.
"""
from __future__ import annotations

import sys
from pathlib import Path

# play/evals/_ae_bridge.py → play/
_PLAY_DIR = Path(__file__).resolve().parent.parent
if str(_PLAY_DIR) not in sys.path:
    sys.path.insert(0, str(_PLAY_DIR))

from agent_engine import (  # noqa: E402
    ArtifactEventEntry,
    ExpandedTurn,
    Result,
    Scenario,
    SpeakerEntry,
    SummaryEntry,
    TokenUsage,
    ToolCall,
    ToolCallEntry,
    TopicEntry,
    TranscriptEntry,
    TurnEntry,
    TurnView,
)
from agent_engine.scenario import _resolve_who_names  # noqa: E402

__all__ = [
    "ArtifactEventEntry",
    "ExpandedTurn",
    "Result",
    "Scenario",
    "SpeakerEntry",
    "SummaryEntry",
    "TokenUsage",
    "ToolCall",
    "ToolCallEntry",
    "TopicEntry",
    "TranscriptEntry",
    "TurnEntry",
    "TurnView",
    "_resolve_who_names",
]

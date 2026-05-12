"""Agent_engine 子进程运行闭包：phase 5 trajectory eval 的 run-path 桥梁.

为什么 subprocess 而非直接 `from play.agent_engine import Engine`：
  - 与 phase 4 RAG 同源（DECISIONS §4 / workshops.mdc）：play/ 下 sub-project 不互相
    Python import；config.py 同名冲突等坑通过 OS 进程边界规避.
  - agent_engine 自带依赖（多个 LLM 客户端、ollama / openai / anthropic / gemini
    SDK 等）不污染 evals 进程，evals 仍可零 OpenAI/Anthropic SDK 跑通.
  - 与未来"远程 agent service" 的迁移路径平行——换 transport 不动 task 层.

代价：
  - 冷启动 ~1-2s（python startup + agent_engine import + 第一次 LLM 客户端实例化）；
    实测每个 scenario ~10s-数分钟，主要看 LLM 后端延迟. 建议 `--limit 1-2` 跑 e2e.
  - 错误传播：subprocess 失败时把 stderr 透出，便于诊断 ollama 不可达 / scenario
    schema 错等.

数据契约：
  - 输入：scenario_path（相对 `scenarios_root` 或绝对路径都接）
  - 输出：envelope dict `{transcript, artifact, warnings, success, usage}`，schema 与
    `play/agent_engine/result.py::Result` 完全一致（由 cli.py --save-result-json 用
    dataclasses.asdict 序列化；§16 起 transcript entry 与 TokenUsage 都是 typed
    dataclass，asdict 递归展平为 dict 形态）.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

# play/evals/models/agent_engine_run.py → ai_workshops/
REPO_ROOT = Path(__file__).resolve().parents[3]
PLAY_DIR = REPO_ROOT / "play"
AGENT_ENGINE_DIR = PLAY_DIR / "agent_engine"

RunFn = Callable[[str], dict[str, Any]]


def make_run_fn(
    *,
    scenarios_root: str | Path | None = None,
    timeout: float = 600.0,
) -> RunFn:
    """返回 `(scenario_path: str) -> envelope_dict` 闭包.

    每次调用 fork 一个 subprocess：
        python -m agent_engine <abs_scenario> --no-stream --save-result-json <tmp.json>

    `scenario_path` 解析顺序：
      1. 绝对路径 → 直接用
      2. 相对 `scenarios_root`（默认 = `play/`）→ resolve 后 abs
      3. 文件不存在 → FileNotFoundError fail-fast

    cwd 锁 `play/`，让 `python -m agent_engine` 能找到包；scenario 内部相对路径
    （如 `tools.vdb_dir: ../../rag/vdb/...`）由 scenario.py 按 scenario 文件位置自
    动 resolve，与 cwd 无关.
    """
    root = Path(scenarios_root).resolve() if scenarios_root else PLAY_DIR

    def _run(scenario_path: str) -> dict[str, Any]:
        sp = Path(scenario_path)
        if not sp.is_absolute():
            sp = (root / sp).resolve()
        if not sp.exists():
            raise FileNotFoundError(f"scenario file not found: {sp}")

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w", encoding="utf-8",
        ) as tf:
            tmp_path = Path(tf.name)
        try:
            cmd = [
                sys.executable, "-m", "agent_engine",
                str(sp),
                "--no-stream",
                "--save-result-json", str(tmp_path),
            ]
            proc = subprocess.run(
                cmd,
                cwd=str(PLAY_DIR),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"agent_engine subprocess exited with {proc.returncode}; "
                    f"stderr={proc.stderr.strip()!r}"
                )
            with tmp_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        finally:
            tmp_path.unlink(missing_ok=True)

    return _run

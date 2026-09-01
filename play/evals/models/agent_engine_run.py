"""Agent_engine child process run closure: run-path bridge for phase 5 trajectory eval.

Why subprocess instead of directly `from play.agent_engine import Engine`:
  - Same origin as phase 4 RAG (DECISIONS §4/workshops.mdc): sub-projects under play/ do not interact with each other
    Python import; config.py pitfalls such as conflicts with the same name are avoided through OS process boundaries.
  - agent_engine comes with dependencies (multiple LLM clients, ollama / openai / anthropic / gemini
    SDK, etc.) does not pollute the evals process, and evals can still run through OpenAI/Anthropic SDK.
  - Parallel to the future migration path of "remote agent service" - changing the transport does not leave the task layer.

Price:
  - Cold start ~1-2s (python startup + agent_engine import + first LLM client instantiation);
    The actual measurement of each scenario is ~10s-several minutes, mainly depends on the LLM backend delay. It is recommended to run e2e with `--limit 1-2`.
  - Error propagation: reveal stderr when subprocess fails to facilitate diagnosis of ollama unreachable / scenario
    schema is wrong.

Data contract:
  - Input: scenario_path (either relative to `scenarios_root` or absolute path)
  - Output: envelope dict `{transcript, artifact, warnings, success, usage}`, schema and
    `play/agent_engine/result.py::Result` is exactly the same (used by cli.py --save-result-json
    dataclasses.asdict serialization; starting from §16, transcript entry and TokenUsage are both typed
    dataclass, asdict is recursively flattened into dict form)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
PLAY_DIR = REPO_ROOT / "play"
AGENT_ENGINE_DIR = PLAY_DIR / "agent_engine"

RunFn = Callable[[str], dict[str, Any]]


def make_run_fn(
    *,
    scenarios_root: str | Path | None = None,
    timeout: float = 600.0,
) -> RunFn:
    """Returns the `(scenario_path: str) -> envelope_dict` closure.

    Each call forks a subprocess:
        python -m agent_engine <abs_scenario> --no-stream --save-result-json <tmp.json>

    `scenario_path` parsing order:
      1. Absolute path → use directly
      2. Relative to `scenarios_root` (default = `play/`) → resolve after abs
      3. The file does not exist → FileNotFoundError fail-fast

    cwd locks `play/` so that `python -m agent_engine` can find the package; scenario internal relative path
    (e.g. `tools.vdb_dir: ../../rag/vdb/...`) by scenario.py based on scenario file location
    Automatic resolve has nothing to do with cwd."""
    root = Path(scenarios_root).resolve() if scenarios_root else PLAY_DIR
    timeout = float(os.environ.get("AGENT_ENGINE_RUN_TIMEOUT", timeout))

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

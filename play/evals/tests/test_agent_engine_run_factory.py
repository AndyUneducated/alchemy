"""models/agent_engine_run.py unit test: subprocess + envelope I/O contract.

Zero LLM / Zero agent_engine true start: replace `subprocess.run` with monkeypatch to intercept calls +
Inject pseudo envelope dict and write to `--save-result-json` temporary file, locking the following contracts:

  ① subprocess command parameters (python -m agent_engine + --no-stream + --save-result-json)
  ② cwd lock `play/` (make the `python -m agent_engine` package reachable)
  ③ scenario_path parsing order (absolute / relative scenarios_root / does not exist)
  ④ The child process exits with non-zero → RuntimeError with stderr (fail-fast)
  ⑤ Temporary files are finally cleaned (regardless of success or failure)
  ⑥ AGENT_ENGINE_RUN_TIMEOUT env override

live e2e (real run `python -m agent_engine`) is placed in test_new_scenarios_smoke.py or the like,
This file does not rely on agent_engine import and is reachable - pure subprocess formal parameters + envelope parsing lock."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from evals.models import agent_engine_run
from evals.models.agent_engine_run import PLAY_DIR, make_run_fn


def _fake_envelope() -> dict:
    """The smallest envelope of the same shape as `play/agent_engine/result.py::Result.asdict()` after §16."""
    return {
        "transcript": [],
        "artifact": {},
        "warnings": [],
        "success": True,
        "usage": [],
    }


# ---------- ① subprocess command parameter ---------------------------------------------

def test_subprocess_command_shape(monkeypatch, tmp_path):
    """`python -m agent_engine <abs_scenario> --no-stream --save-result-json <tmp>`."""
    scenario = tmp_path / "s.yaml"
    scenario.write_text("---\nname: x\n", encoding="utf-8")

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = kwargs.get("cwd")
        captured["timeout"] = kwargs.get("timeout")
        # Write the envelope to the tmp file pointed to by --save-result-json to simulate real child process behavior
        i = cmd.index("--save-result-json")
        Path(cmd[i + 1]).write_text(json.dumps(_fake_envelope()), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(agent_engine_run.subprocess, "run", fake_run)

    fn = make_run_fn()
    out = fn(str(scenario))

    cmd = captured["cmd"]
    # The first token is sys.executable, the second/third is -m agent_engine
    assert cmd[1:3] == ["-m", "agent_engine"], f"cmd 头不对：{cmd[:4]}"
    assert "--no-stream" in cmd
    assert "--save-result-json" in cmd
    # save-result-json is followed by a temporary json path
    save_idx = cmd.index("--save-result-json")
    assert cmd[save_idx + 1].endswith(".json")
    # scenario abs path appears in cmd
    assert str(scenario.resolve()) in cmd
    # cwd locks play/ (let `python -m agent_engine` find the package)
    assert captured["cwd"] == str(PLAY_DIR), f"cwd 应是 PLAY_DIR，got {captured['cwd']}"
    # envelope parses back to dict
    assert out == _fake_envelope()


# ---------- ② scenario_path parsing order ----------------------------------

def test_absolute_scenario_path_passed_through(monkeypatch, tmp_path):
    """Use the absolute path directly without using scenarios_root for splicing."""
    scenario = tmp_path / "abs_scenario.yaml"
    scenario.write_text("ok", encoding="utf-8")

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        i = cmd.index("--save-result-json")
        Path(cmd[i + 1]).write_text(json.dumps(_fake_envelope()), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(agent_engine_run.subprocess, "run", fake_run)

    # scenarios_root intentionally points to irrelevant directories to verify that absolute paths are not spliced
    fn = make_run_fn(scenarios_root="/some/unrelated/dir")
    fn(str(scenario))

    assert str(scenario.resolve()) in captured["cmd"]
    assert "/some/unrelated/dir" not in " ".join(captured["cmd"])


def test_relative_scenario_path_resolved_against_scenarios_root(monkeypatch, tmp_path):
    """Relative paths resolve with `scenarios_root` as the root."""
    root = tmp_path / "scenes"
    root.mkdir()
    (root / "demo.yaml").write_text("ok", encoding="utf-8")

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        i = cmd.index("--save-result-json")
        Path(cmd[i + 1]).write_text(json.dumps(_fake_envelope()), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(agent_engine_run.subprocess, "run", fake_run)

    fn = make_run_fn(scenarios_root=root)
    fn("demo.yaml")

    expected_abs = str((root / "demo.yaml").resolve())
    assert expected_abs in captured["cmd"]


def test_default_scenarios_root_is_play_dir(monkeypatch, tmp_path):
    """scenarios_root=None → default = `play/` (same as cli/agent_traj default).

    Place a temporary scenario under play/ and verify that the relative path is rooted at PLAY_DIR.
    Verify again with an absolute path that it does not overlap with the default root."""
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        i = cmd.index("--save-result-json")
        Path(cmd[i + 1]).write_text(json.dumps(_fake_envelope()), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(agent_engine_run.subprocess, "run", fake_run)

    # Create a scenario file under PLAY_DIR to ensure that fn can resolve
    play_scenario = PLAY_DIR / "_test_tmp_scenario_factory.yaml"
    play_scenario.write_text("ok", encoding="utf-8")
    try:
        fn = make_run_fn()  # Go to default PLAY_DIR
        fn("_test_tmp_scenario_factory.yaml")
        assert str(play_scenario.resolve()) in captured["cmd"]
    finally:
        play_scenario.unlink(missing_ok=True)


def test_missing_scenario_raises_filenotfound(monkeypatch, tmp_path):
    """File does not exist → FileNotFoundError and subprocess.run will not be called."""
    called = []

    def fake_run(*a, **kw):
        called.append(True)
        return subprocess.CompletedProcess(a[0] if a else [], 0, "", "")

    monkeypatch.setattr(agent_engine_run.subprocess, "run", fake_run)

    fn = make_run_fn(scenarios_root=tmp_path)
    with pytest.raises(FileNotFoundError, match="scenario file not found"):
        fn("does_not_exist.yaml")

    assert called == [], "scenario 不存在时不应调 subprocess.run"


# ---------- ③ Child process error propagation ---------------------------------------------

def test_subprocess_failure_raises_with_stderr(monkeypatch, tmp_path):
    """Non-zero exit → RuntimeError with stderr (avoids silently returning an empty envelope)."""
    scenario = tmp_path / "s.yaml"
    scenario.write_text("ok", encoding="utf-8")

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, returncode=2, stdout="", stderr="Ollama unreachable: connection refused"
        )

    monkeypatch.setattr(agent_engine_run.subprocess, "run", fake_run)

    fn = make_run_fn()
    with pytest.raises(RuntimeError, match="Ollama unreachable"):
        fn(str(scenario))


def test_subprocess_failure_cleans_up_tmpfile(monkeypatch, tmp_path):
    """Even if the child process fails, the temporary file of --save-result-json must be deleted (to avoid /tmp leakage)."""
    scenario = tmp_path / "s.yaml"
    scenario.write_text("ok", encoding="utf-8")

    captured_tmp_path: dict[str, Path] = {}

    def fake_run(cmd, **kwargs):
        i = cmd.index("--save-result-json")
        captured_tmp_path["p"] = Path(cmd[i + 1])
        # Write something intentionally and verify that finally still cleans up
        captured_tmp_path["p"].write_text("partial", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(agent_engine_run.subprocess, "run", fake_run)

    fn = make_run_fn()
    with pytest.raises(RuntimeError):
        fn(str(scenario))

    assert "p" in captured_tmp_path
    assert not captured_tmp_path["p"].exists(), (
        f"tmpfile {captured_tmp_path['p']} 子进程失败后未清理"
    )


def test_success_path_cleans_up_tmpfile(monkeypatch, tmp_path):
    """The success path is also cleaned up (to prevent finally missing writes)."""
    scenario = tmp_path / "s.yaml"
    scenario.write_text("ok", encoding="utf-8")

    captured_tmp_path: dict[str, Path] = {}

    def fake_run(cmd, **kwargs):
        i = cmd.index("--save-result-json")
        captured_tmp_path["p"] = Path(cmd[i + 1])
        captured_tmp_path["p"].write_text(json.dumps(_fake_envelope()), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(agent_engine_run.subprocess, "run", fake_run)

    fn = make_run_fn()
    fn(str(scenario))

    assert not captured_tmp_path["p"].exists(), "tmpfile 成功路径未清理"


# ---------- ④ timeout transparent transmission + env override --------------------------------

def test_timeout_passed_to_subprocess_run(monkeypatch, tmp_path):
    """`timeout=` kwarg is passed transparently to subprocess.run."""
    scenario = tmp_path / "s.yaml"
    scenario.write_text("ok", encoding="utf-8")

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        i = cmd.index("--save-result-json")
        Path(cmd[i + 1]).write_text(json.dumps(_fake_envelope()), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(agent_engine_run.subprocess, "run", fake_run)

    fn = make_run_fn(timeout=42.0)
    fn(str(scenario))

    assert captured["timeout"] == 42.0


def test_timeout_env_var_overrides_default(monkeypatch, tmp_path):
    """`AGENT_ENGINE_RUN_TIMEOUT` env overrides the timeout parameter (longer on CI / shorter locally)."""
    scenario = tmp_path / "s.yaml"
    scenario.write_text("ok", encoding="utf-8")

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        i = cmd.index("--save-result-json")
        Path(cmd[i + 1]).write_text(json.dumps(_fake_envelope()), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(agent_engine_run.subprocess, "run", fake_run)
    monkeypatch.setenv("AGENT_ENGINE_RUN_TIMEOUT", "9.5")

    fn = make_run_fn(timeout=600.0)  # Pass 600 explicitly and should be overwritten by env
    fn(str(scenario))

    assert captured["timeout"] == 9.5

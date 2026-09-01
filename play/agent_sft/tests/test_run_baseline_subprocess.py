"""run_baseline.py subprocess construction — pins Phase 5's two real incidents.

Phase 5 early bugs; this test suite regression-protects:
  ① runner used `subprocess.run(["python", ...])` instead of `sys.executable`; on another machine
     default `python` may be Py2 → FileNotFoundError.
  ② agent-path tasks (`nudge_fire_rate` / `agent_traj`) spawned subprocess without
     `AGENT_ENGINE_MODEL` env → agent_engine used default model; three-model comparison invalid.

Mocks `subprocess.run` to capture cmd + env without starting processes; 89 → 93 tests."""

from __future__ import annotations

import sys

import run_baseline  # type: ignore[import-not-found]


class _FakeResult:
    returncode = 0


def _capture_subprocess(monkeypatch):
    calls: list[dict] = []

    def fake_run(cmd, **kwargs):
        calls.append({"cmd": list(cmd), "env": kwargs.get("env"), "cwd": kwargs.get("cwd")})
        return _FakeResult()

    monkeypatch.setattr(run_baseline.subprocess, "run", fake_run)
    return calls


def test_run_baseline_uses_sys_executable_not_string_python(monkeypatch):
    """cmd[0] must be sys.executable - Phase 5 Real Incident #1 (`python` is not in PATH)."""
    calls = _capture_subprocess(monkeypatch)
    run_baseline.main(["--models", "qwen3.5:9b", "--seeds", "0", "--tasks", "bfcl_slice"])
    assert len(calls) == 1
    assert calls[0]["cmd"][0] == sys.executable, (
        f"runner must invoke {sys.executable!r}, not bare 'python'"
    )


def test_run_baseline_cmd_shape(monkeypatch):
    """argv order + key flag complete: python -m evals run --task T --model M@seed=S --seed S."""
    calls = _capture_subprocess(monkeypatch)
    run_baseline.main(["--models", "qwen3.5:9b", "--seeds", "3", "--tasks", "mmlu_slice"])
    cmd = calls[0]["cmd"]
    assert cmd[1:5] == ["-m", "evals", "run", "--task"]
    assert "mmlu_slice" in cmd
    assert "ollama:qwen3.5:9b@seed=3" in cmd
# --seed followed by integer (after str transformation)
# --seed followed by integer (after str transformation)
    seed_idx = cmd.index("--seed")
    assert cmd[seed_idx + 1] == "3"


def test_run_baseline_sets_AGENT_ENGINE_MODEL_for_nudge_fire_rate(monkeypatch):
    """agent-path task → env must contain AGENT_ENGINE_MODEL=<model> - Phase 5 Real Incident #2."""
    calls = _capture_subprocess(monkeypatch)
    run_baseline.main(["--models", "qwen3.6:27b", "--seeds", "0", "--tasks", "nudge_fire_rate"])
    env = calls[0]["env"]
    assert env is not None, "subprocess.run must be called with env="
    assert env.get("AGENT_ENGINE_MODEL") == "qwen3.6:27b", (
        "nudge_fire_rate 是 agent-path task；env 必须传 AGENT_ENGINE_MODEL，"
        "否则 agent_engine 用默认模型，三模型对照失效"
    )


def test_run_baseline_sets_AGENT_ENGINE_MODEL_for_agent_traj(monkeypatch):
    """agent_traj is the same as nudge_fire_rate, also agent-path."""
    calls = _capture_subprocess(monkeypatch)
    run_baseline.main(["--models", "qwen3.5:9b", "--seeds", "5", "--tasks", "agent_traj"])
    assert calls[0]["env"]["AGENT_ENGINE_MODEL"] == "qwen3.5:9b"


def test_run_baseline_does_not_set_AGENT_ENGINE_MODEL_for_offline_tasks(monkeypatch):
    """bfcl_slice / mmlu_slice are offline (judged via local scoring) → should not pollute env."""
# First clear the AGENT_ENGINE_MODEL that may be preset externally (otherwise the value that comes with the test machine will be false positive)
# First clear the AGENT_ENGINE_MODEL that may be preset externally (otherwise the value that comes with the test machine will be false positive)
    monkeypatch.delenv("AGENT_ENGINE_MODEL", raising=False)
    calls = _capture_subprocess(monkeypatch)
    run_baseline.main([
        "--models", "qwen3.5:9b",
        "--seeds", "0",
        "--tasks", "bfcl_slice", "mmlu_slice",
    ])
    assert len(calls) == 2
    for c in calls:
# Offline task should not plug AGENT_ENGINE_MODEL into env - the runner path branch should be correct
# Offline task should not plug AGENT_ENGINE_MODEL into env - the runner path branch should be correct
        assert "AGENT_ENGINE_MODEL" not in c["env"], (
            f"offline task should not set AGENT_ENGINE_MODEL; got env keys = "
            f"{[k for k in c['env'] if 'AGENT' in k]}"
        )


def test_run_baseline_dry_run_does_not_invoke_subprocess(monkeypatch, capsys):
    calls = _capture_subprocess(monkeypatch)
    rc = run_baseline.main([
        "--models", "qwen3.5:9b", "--seeds", "0", "--tasks", "bfcl_slice", "--dry-run",
    ])
    assert rc == 0
    assert calls == []
    out = capsys.readouterr().out
    assert "would run" in out


def test_run_baseline_combos_are_full_cross_product(monkeypatch):
    """Full Cartesian of M × S × T, without losing any combo."""
    calls = _capture_subprocess(monkeypatch)
    run_baseline.main([
        "--models", "qwen3.5:9b", "qwen3.6:27b",
        "--seeds", "0", "1",
        "--tasks", "bfcl_slice",
    ])
    assert len(calls) == 2 * 2 * 1
    specs = {c["cmd"][c["cmd"].index("--model") + 1] for c in calls}
    assert specs == {
        "ollama:qwen3.5:9b@seed=0", "ollama:qwen3.5:9b@seed=1",
        "ollama:qwen3.6:27b@seed=0", "ollama:qwen3.6:27b@seed=1",
    }

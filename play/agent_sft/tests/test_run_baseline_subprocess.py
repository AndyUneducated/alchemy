"""run_baseline.py 子进程构造 — 钉死 Phase 5 两次真实事故.

Phase 5 早期踩到 2 个 bug，本测试集回归保护：
  ① runner 写 `subprocess.run(["python", ...])` 而非 `sys.executable`，搬机后
     `python` 默认 Py2 → FileNotFoundError.
  ② agent-path task (`nudge_fire_rate` / `agent_traj`) 起 subprocess 没传
     `AGENT_ENGINE_MODEL` env → agent_engine 用默认模型，对照测试全部跑同模型.

mock `subprocess.run` 拦下 cmd + env，不真启进程；89 → 93 tests.
"""

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
    """cmd[0] 必须是 sys.executable —— Phase 5 实事故 #1 (`python` 不在 PATH)."""
    calls = _capture_subprocess(monkeypatch)
    run_baseline.main(["--models", "qwen2.5:7b", "--seeds", "0", "--tasks", "bfcl_slice"])
    assert len(calls) == 1
    assert calls[0]["cmd"][0] == sys.executable, (
        f"runner must invoke {sys.executable!r}, not bare 'python'"
    )


def test_run_baseline_cmd_shape(monkeypatch):
    """argv 顺序 + 关键 flag 完整：python -m evals run --task T --model M@seed=S --seed S."""
    calls = _capture_subprocess(monkeypatch)
    run_baseline.main(["--models", "qwen2.5:7b", "--seeds", "3", "--tasks", "mmlu_slice"])
    cmd = calls[0]["cmd"]
    assert cmd[1:5] == ["-m", "evals", "run", "--task"]
    assert "mmlu_slice" in cmd
    assert "ollama:qwen2.5:7b@seed=3" in cmd
    # --seed 跟整数（str 化后）
    seed_idx = cmd.index("--seed")
    assert cmd[seed_idx + 1] == "3"


def test_run_baseline_sets_AGENT_ENGINE_MODEL_for_nudge_fire_rate(monkeypatch):
    """agent-path task → env 必须含 AGENT_ENGINE_MODEL=<model>—— Phase 5 实事故 #2."""
    calls = _capture_subprocess(monkeypatch)
    run_baseline.main(["--models", "qwen2.5:32b", "--seeds", "0", "--tasks", "nudge_fire_rate"])
    env = calls[0]["env"]
    assert env is not None, "subprocess.run must be called with env="
    assert env.get("AGENT_ENGINE_MODEL") == "qwen2.5:32b", (
        "nudge_fire_rate 是 agent-path task；env 必须传 AGENT_ENGINE_MODEL，"
        "否则 agent_engine 用默认模型，三模型对照失效"
    )


def test_run_baseline_sets_AGENT_ENGINE_MODEL_for_agent_traj(monkeypatch):
    """agent_traj 同 nudge_fire_rate，也是 agent-path."""
    calls = _capture_subprocess(monkeypatch)
    run_baseline.main(["--models", "qwen2.5:7b", "--seeds", "5", "--tasks", "agent_traj"])
    assert calls[0]["env"]["AGENT_ENGINE_MODEL"] == "qwen2.5:7b"


def test_run_baseline_does_not_set_AGENT_ENGINE_MODEL_for_offline_tasks(monkeypatch):
    """bfcl_slice / mmlu_slice 是 offline (judged via local scoring) → 不该污染 env."""
    # 先清掉外部可能预设的 AGENT_ENGINE_MODEL（otherwise 测试机器自带的值会假阳性）
    monkeypatch.delenv("AGENT_ENGINE_MODEL", raising=False)
    calls = _capture_subprocess(monkeypatch)
    run_baseline.main([
        "--models", "qwen2.5:7b",
        "--seeds", "0",
        "--tasks", "bfcl_slice", "mmlu_slice",
    ])
    assert len(calls) == 2
    for c in calls:
        # offline task 不该往 env 里塞 AGENT_ENGINE_MODEL —— runner 路径分支应正确
        assert "AGENT_ENGINE_MODEL" not in c["env"], (
            f"offline task should not set AGENT_ENGINE_MODEL; got env keys = "
            f"{[k for k in c['env'] if 'AGENT' in k]}"
        )


def test_run_baseline_dry_run_does_not_invoke_subprocess(monkeypatch, capsys):
    calls = _capture_subprocess(monkeypatch)
    rc = run_baseline.main([
        "--models", "qwen2.5:7b", "--seeds", "0", "--tasks", "bfcl_slice", "--dry-run",
    ])
    assert rc == 0
    assert calls == []
    out = capsys.readouterr().out
    assert "would run" in out


def test_run_baseline_combos_are_full_cross_product(monkeypatch):
    """M × S × T 的全笛卡尔，不丢任何 combo."""
    calls = _capture_subprocess(monkeypatch)
    run_baseline.main([
        "--models", "qwen2.5:7b", "qwen2.5:32b",
        "--seeds", "0", "1",
        "--tasks", "bfcl_slice",
    ])
    assert len(calls) == 2 * 2 * 1
    specs = {c["cmd"][c["cmd"].index("--model") + 1] for c in calls}
    assert specs == {
        "ollama:qwen2.5:7b@seed=0", "ollama:qwen2.5:7b@seed=1",
        "ollama:qwen2.5:32b@seed=0", "ollama:qwen2.5:32b@seed=1",
    }

"""models/agent_engine_run.py 单元测试：subprocess + envelope I/O 契约.

零 LLM / 零 agent_engine 真启动：用 monkeypatch 替换 `subprocess.run` 拦截调用 +
注入伪 envelope dict 写入 `--save-result-json` 临时文件，锁住下列契约：

  ① subprocess 命令形参（python -m agent_engine + --no-stream + --save-result-json）
  ② cwd 锁 `play/`（让 `python -m agent_engine` 包可达）
  ③ scenario_path 解析顺序（绝对 / 相对 scenarios_root / 不存在）
  ④ 子进程非零退出 → RuntimeError 携 stderr（fail-fast）
  ⑤ 临时文件 finally 清理（不论成败）
  ⑥ AGENT_ENGINE_RUN_TIMEOUT env override

live e2e（真跑 `python -m agent_engine`）放在 test_new_scenarios_smoke.py 之类，
本文件不依赖 agent_engine import 可达——纯 subprocess 形参 + envelope 解析锁.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from evals.models import agent_engine_run
from evals.models.agent_engine_run import PLAY_DIR, make_run_fn


def _fake_envelope() -> dict:
    """与 §16 后 `play/agent_engine/result.py::Result.asdict()` 同形的最小 envelope."""
    return {
        "transcript": [],
        "artifact": {},
        "warnings": [],
        "success": True,
        "usage": [],
    }


# ---------- ① subprocess 命令形参 -------------------------------------

def test_subprocess_command_shape(monkeypatch, tmp_path):
    """`python -m agent_engine <abs_scenario> --no-stream --save-result-json <tmp>`."""
    scenario = tmp_path / "s.yaml"
    scenario.write_text("---\nname: x\n", encoding="utf-8")

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = kwargs.get("cwd")
        captured["timeout"] = kwargs.get("timeout")
        # 把 envelope 写到 --save-result-json 指向的 tmp 文件，模拟真实子进程行为
        i = cmd.index("--save-result-json")
        Path(cmd[i + 1]).write_text(json.dumps(_fake_envelope()), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(agent_engine_run.subprocess, "run", fake_run)

    fn = make_run_fn()
    out = fn(str(scenario))

    cmd = captured["cmd"]
    # 第一个 token 是 sys.executable，第二/三是 -m agent_engine
    assert cmd[1:3] == ["-m", "agent_engine"], f"cmd 头不对：{cmd[:4]}"
    assert "--no-stream" in cmd
    assert "--save-result-json" in cmd
    # save-result-json 后紧跟一个临时 json 路径
    save_idx = cmd.index("--save-result-json")
    assert cmd[save_idx + 1].endswith(".json")
    # scenario abs 路径出现在 cmd 中
    assert str(scenario.resolve()) in cmd
    # cwd 锁 play/（让 `python -m agent_engine` 找到包）
    assert captured["cwd"] == str(PLAY_DIR), f"cwd 应是 PLAY_DIR，got {captured['cwd']}"
    # envelope 解析回到 dict
    assert out == _fake_envelope()


# ---------- ② scenario_path 解析顺序 ----------------------------------

def test_absolute_scenario_path_passed_through(monkeypatch, tmp_path):
    """绝对路径直接走，不走 scenarios_root 拼接."""
    scenario = tmp_path / "abs_scenario.yaml"
    scenario.write_text("ok", encoding="utf-8")

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        i = cmd.index("--save-result-json")
        Path(cmd[i + 1]).write_text(json.dumps(_fake_envelope()), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(agent_engine_run.subprocess, "run", fake_run)

    # scenarios_root 故意指向无关目录，验证绝对路径不被拼接
    fn = make_run_fn(scenarios_root="/some/unrelated/dir")
    fn(str(scenario))

    assert str(scenario.resolve()) in captured["cmd"]
    assert "/some/unrelated/dir" not in " ".join(captured["cmd"])


def test_relative_scenario_path_resolved_against_scenarios_root(monkeypatch, tmp_path):
    """相对路径以 `scenarios_root` 为根 resolve."""
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
    """scenarios_root=None → 默认 = `play/`（与 cli/agent_traj 默认一致）.

    在 play/ 下放一个临时 scenario，验证 relative 路径以 PLAY_DIR 为根.
    用绝对路径再次验证不会与默认根叠加.
    """
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        i = cmd.index("--save-result-json")
        Path(cmd[i + 1]).write_text(json.dumps(_fake_envelope()), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(agent_engine_run.subprocess, "run", fake_run)

    # 在 PLAY_DIR 下临建一个 scenario 文件，确保 fn 能 resolve 到
    play_scenario = PLAY_DIR / "_test_tmp_scenario_factory.yaml"
    play_scenario.write_text("ok", encoding="utf-8")
    try:
        fn = make_run_fn()  # 走默认 PLAY_DIR
        fn("_test_tmp_scenario_factory.yaml")
        assert str(play_scenario.resolve()) in captured["cmd"]
    finally:
        play_scenario.unlink(missing_ok=True)


def test_missing_scenario_raises_filenotfound(monkeypatch, tmp_path):
    """文件不存在 → FileNotFoundError，且不会调 subprocess.run."""
    called = []

    def fake_run(*a, **kw):
        called.append(True)
        return subprocess.CompletedProcess(a[0] if a else [], 0, "", "")

    monkeypatch.setattr(agent_engine_run.subprocess, "run", fake_run)

    fn = make_run_fn(scenarios_root=tmp_path)
    with pytest.raises(FileNotFoundError, match="scenario file not found"):
        fn("does_not_exist.yaml")

    assert called == [], "scenario 不存在时不应调 subprocess.run"


# ---------- ③ 子进程错误传播 ------------------------------------------

def test_subprocess_failure_raises_with_stderr(monkeypatch, tmp_path):
    """非零退出 → RuntimeError 携 stderr（避免静默返回空 envelope）."""
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
    """即便子进程失败，--save-result-json 的临时文件也得删（避免 /tmp 泄漏）."""
    scenario = tmp_path / "s.yaml"
    scenario.write_text("ok", encoding="utf-8")

    captured_tmp_path: dict[str, Path] = {}

    def fake_run(cmd, **kwargs):
        i = cmd.index("--save-result-json")
        captured_tmp_path["p"] = Path(cmd[i + 1])
        # 故意写一点东西，验证 finally 仍清理
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
    """成功路径同样清理（防止 finally 漏写）."""
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


# ---------- ④ timeout 透传 + env override -----------------------------

def test_timeout_passed_to_subprocess_run(monkeypatch, tmp_path):
    """`timeout=` kwarg 透传给 subprocess.run."""
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
    """`AGENT_ENGINE_RUN_TIMEOUT` env 覆盖 timeout 参数（CI 上调长 / 本地调短）."""
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

    fn = make_run_fn(timeout=600.0)  # 显式传 600，应被 env 覆盖
    fn(str(scenario))

    assert captured["timeout"] == 9.5

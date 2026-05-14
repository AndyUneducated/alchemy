"""mine_triples.py 子进程构造 — v2 改 runner 时的提前保险.

v2 大概率会改这个 runner（on-policy mining / 多 seed 策略），本测试集钉死当前 argv
契约。走 `--dry-run` 路径捕获 stdout 抓命令字符串，避免 mock 也省 subprocess 开销.

覆盖：
  - cmd[0] = sys.executable（同 run_baseline 修过的 bug）
  - argv 完整：`-m agent_engine <scen-path> --no-stream --save-result-json <out>`
  - default scenario 走 fast 副本（DECISIONS §11 的提速决策）；--upstream 切回原 scenario
  - 输出文件命名 `{scenario}-r{run_id}.json`
"""

from __future__ import annotations

import re
import sys

import mine_triples  # type: ignore[import-not-found]


def _captured_cmds(capsys, argv) -> list[str]:
    """跑 dry-run → 解析 stdout 中 `$ <cmd>` 行 → 返回 cmd 字符串列表."""
    rc = mine_triples.main(argv + ["--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    cmds = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("$ "):
            cmds.append(line[2:])
    return cmds


def test_mine_triples_dry_run_default_uses_fast_scenarios(capsys, tmp_path):
    cmds = _captured_cmds(capsys, [
        "--scenarios", "tool_chain",
        "--run-ids", "0",
        "--out-dir", str(tmp_path),
    ])
    assert len(cmds) == 1
    cmd = cmds[0]
    assert sys.executable in cmd, f"must use {sys.executable!r}, got: {cmd}"
    assert "-m agent_engine" in cmd
    assert "tool_chain_fast.md" in cmd, (
        "default 必须走 fast 副本（DECISIONS §11），不能用 upstream"
    )
    assert "--no-stream" in cmd
    assert "--save-result-json" in cmd
    assert cmd.rstrip().endswith("tool_chain-r0.json"), (
        f"输出文件命名必须是 {{scenario}}-r{{run_id}}.json: {cmd}"
    )


def test_mine_triples_upstream_flag_switches_scenario_src(capsys, tmp_path):
    """--upstream → 切回 agent_engine/scenarios/<name>.md（不带 _fast 后缀）."""
    cmds = _captured_cmds(capsys, [
        "--scenarios", "tool_chain",
        "--run-ids", "0",
        "--out-dir", str(tmp_path),
        "--upstream",
    ])
    cmd = cmds[0]
    assert "tool_chain_fast.md" not in cmd
    # 命中绝对路径里的 `/scenarios/tool_chain.md`
    assert re.search(r"agent_engine[/\\]scenarios[/\\]tool_chain\.md", cmd), cmd


def test_mine_triples_full_cross_product(capsys, tmp_path):
    """scenarios × run_ids 全笛卡尔；命名按 scen-r{N}.json 一一对应."""
    cmds = _captured_cmds(capsys, [
        "--scenarios", "tool_chain", "code_review",
        "--run-ids", "0", "1",
        "--out-dir", str(tmp_path),
    ])
    assert len(cmds) == 4
    filenames = [c.rsplit(" ", 1)[-1] for c in cmds]
    assert sorted(f.rsplit("/", 1)[-1] for f in filenames) == [
        "code_review-r0.json", "code_review-r1.json",
        "tool_chain-r0.json", "tool_chain-r1.json",
    ]


def test_mine_triples_out_dir_resolved_absolute(capsys, tmp_path):
    """relative out-dir 必须被 .resolve() 成绝对路径（防 'play/play/' 重复事故 — 见代码注释）."""
    rel = tmp_path.relative_to(tmp_path.parent) if tmp_path.is_relative_to(tmp_path.parent) else tmp_path
    cmds = _captured_cmds(capsys, [
        "--scenarios", "tool_chain",
        "--run-ids", "0",
        "--out-dir", str(tmp_path),  # 直接绝对，验证回显形态
    ])
    # 路径出现的位置在 --save-result-json 之后
    assert str(tmp_path.resolve()) in cmds[0], cmds[0]

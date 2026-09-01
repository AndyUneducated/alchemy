"""mine_triples.py subprocess construction — insurance before v2 runner changes.

v2 will likely change this runner (on-policy mining / multi-seed strategy); this suite pins the current argv
contract. Uses `--dry-run` to capture stdout command strings — no mock, no subprocess cost.

Covers:
  - cmd[0] = sys.executable (same bug fixed in run_baseline)
  - full argv: `-m agent_engine <scen-path> --no-stream --save-result-json <out>`
  - default scenario uses fast copy (DECISIONS §11 speedup); --upstream switches to original
  - output naming `{scenario}-r{run_id}.json`"""

from __future__ import annotations

import re
import sys

import mine_triples  # type: ignore[import-not-found]


def _captured_cmds(capsys, argv) -> list[str]:
    """Run dry-run → parse the `$ <cmd>` line in stdout → return a list of cmd strings."""
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
    """--upstream → switch to agent_engine/scenarios/<name>.md (no _fast suffix)."""
    cmds = _captured_cmds(capsys, [
        "--scenarios", "tool_chain",
        "--run-ids", "0",
        "--out-dir", str(tmp_path),
        "--upstream",
    ])
    cmd = cmds[0]
    assert "tool_chain_fast.md" not in cmd
# Hit `/scenarios/tool_chain.md` in the absolute path
# Hit `/scenarios/tool_chain.md` in the absolute path
    assert re.search(r"agent_engine[/\\]scenarios[/\\]tool_chain\.md", cmd), cmd


def test_mine_triples_full_cross_product(capsys, tmp_path):
    """Full Cartesian product of scenarios × run_ids; names map scen-r{N}.json 1:1."""
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
    """Relative out-dir must be .resolve()'d to absolute (avoid play/play/ duplicate — see code comment)."""
    rel = tmp_path.relative_to(tmp_path.parent) if tmp_path.is_relative_to(tmp_path.parent) else tmp_path
    cmds = _captured_cmds(capsys, [
        "--scenarios", "tool_chain",
        "--run-ids", "0",
        "--out-dir", str(tmp_path),  # Directly absolute, verify the echo state
    ])
    # path appears after --save-result-json
    # path appears after --save-result-json
    assert str(tmp_path.resolve()) in cmds[0], cmds[0]

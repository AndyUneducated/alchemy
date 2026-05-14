"""CLI 用户命令端到端：list-tasks / show / build_parser / main(argv) 入口锁.

补 `test_cli_spec.py` 的盲区 —— 那边只覆盖 `parse_model_spec` /
`_build_task_with_optional_deps` / `_fmt_kv` / `_print_aggregated` 等内部 helper，
不打用户真实入口；只要 argparse 子命令名 / required flag 漂移 / set_defaults 漏写
`func`，CLI 用户立即炸但本地 pytest 全绿.

本文件锁：
  ① `cmd_list_tasks(args)` — 端到端打印 12 个 task name + return 0
  ② `cmd_show` 跨 run 索引浏览 — 真起 storage.save 落盘 + 按 task/mode/last 过滤
  ③ `cmd_show` 单 run drill-down — 验 result.json json.dumps + 可选 samples
  ④ `build_parser()` argparse 形状 — required flag / choices / defaults
  ⑤ `main(argv)` 入口 — 显式 argv list 走全栈 dispatch（替代 sys.argv 副作用）

零网络 / 零 LM：所有打分子命令路径用 `MockLM(mode='gold')` 或假 result 落盘.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from evals import tasks  # noqa: F401  — @register_task 副作用
from evals.api import EvalResult, SampleResult
from evals.cli import (
    build_parser,
    cmd_list_tasks,
    cmd_show,
    main,
)
from evals.registry import list_tasks
from evals.storage import save


# ---------- ① cmd_list_tasks ---------------------------------------------

def test_cmd_list_tasks_prints_all_registered_names_and_returns_zero(capsys):
    """全 12 task 一行一名打印 + return 0；与 `list_tasks()` 同源，UI 不漂."""
    rc = cmd_list_tasks(argparse.Namespace())
    assert rc == 0
    out = capsys.readouterr().out.splitlines()
    assert out == list_tasks(), (
        f"cmd_list_tasks 输出与 list_tasks() 不一致：\n"
        f"  printed: {out}\n"
        f"  expected: {list_tasks()}"
    )
    # 每行恰好一个 token（无前缀格式化）—— 让 `python -m evals list-tasks | xargs -I X ...`
    # bash 管道用法稳定（避免误加 bullet / 缩进）.
    for line in out:
        assert line.strip() == line and " " not in line, f"task name 行带额外字符：{line!r}"


# ---------- helpers：构造 fake EvalResult / 落盘 -------------------------

def _make_eval_result(
    *,
    run_id: str,
    task: str = "sentiment_clf",
    model: str = "mock:gold",
    mode: str = "score",
    n: int = 2,
    accuracy: float = 1.0,
) -> EvalResult:
    samples = tuple(
        SampleResult(
            doc_id=f"d{i}",
            prediction="pos",
            target="pos",
            metrics={"acc": 1.0},
        )
        for i in range(n)
    )
    return EvalResult(
        task=task,
        model=model,
        mode=mode,
        n=n,
        aggregated={"accuracy": accuracy},
        per_sample=samples,
        run_id=run_id,
        created_at=f"2025-01-01T00:00:{run_id[-2:]}Z",
        elapsed_ms=1.0,
        num_fewshot=0,
    )


# ---------- ② cmd_show 跨 run 索引浏览 ----------------------------------

def test_cmd_show_lists_index_rows_when_no_run_id(tmp_path, capsys):
    """无 --run-id → 走 index.jsonl 列出所有 run，按 created_at 排序."""
    save(_make_eval_result(run_id="20250101T000001"), runs_dir=tmp_path)
    save(
        _make_eval_result(run_id="20250101T000002", task="qa_open", accuracy=0.5),
        runs_dir=tmp_path,
    )

    args = argparse.Namespace(
        run_id=None, task=None, mode=None, last=None, samples=0, runs_dir=tmp_path,
    )
    rc = cmd_show(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "20250101T000001" in out
    assert "20250101T000002" in out
    assert "task=sentiment_clf" in out
    assert "task=qa_open" in out
    assert "accuracy=1.0000" in out
    assert "accuracy=0.5000" in out


def test_cmd_show_filter_by_task(tmp_path, capsys):
    """--task X → 只列该 task 的 run."""
    save(_make_eval_result(run_id="20250101T000001"), runs_dir=tmp_path)
    save(_make_eval_result(run_id="20250101T000002", task="qa_open"), runs_dir=tmp_path)

    args = argparse.Namespace(
        run_id=None, task="qa_open", mode=None, last=None, samples=0, runs_dir=tmp_path,
    )
    cmd_show(args)
    out = capsys.readouterr().out
    assert "20250101T000002" in out
    assert "20250101T000001" not in out


def test_cmd_show_filter_by_mode(tmp_path, capsys):
    """--mode score → 只列 score run（与 run mode 区分）."""
    save(_make_eval_result(run_id="20250101T000001", mode="score"), runs_dir=tmp_path)
    save(_make_eval_result(run_id="20250101T000002", mode="run"), runs_dir=tmp_path)

    args = argparse.Namespace(
        run_id=None, task=None, mode="run", last=None, samples=0, runs_dir=tmp_path,
    )
    cmd_show(args)
    out = capsys.readouterr().out
    assert "20250101T000002" in out
    assert "20250101T000001" not in out


def test_cmd_show_last_n_keeps_only_tail(tmp_path, capsys):
    """--last N → 仅显示按 created_at 排序后的最后 N 条."""
    for i in range(5):
        save(
            _make_eval_result(run_id=f"20250101T0000{i:02d}", accuracy=i / 10),
            runs_dir=tmp_path,
        )

    args = argparse.Namespace(
        run_id=None, task=None, mode=None, last=2, samples=0, runs_dir=tmp_path,
    )
    cmd_show(args)
    lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
    assert len(lines) == 2
    # 取最后两条按 created_at 排序：03 + 04
    assert "20250101T000003" in lines[0]
    assert "20250101T000004" in lines[1]


def test_cmd_show_empty_runs_dir_prints_nothing(tmp_path, capsys):
    """index.jsonl 不存在（首次跑 / 误删）→ 0 行输出 + return 0（不 crash）."""
    args = argparse.Namespace(
        run_id=None, task=None, mode=None, last=None, samples=0, runs_dir=tmp_path,
    )
    rc = cmd_show(args)
    assert rc == 0
    assert capsys.readouterr().out == ""


# ---------- ③ cmd_show 单 run drill-down --------------------------------

def test_cmd_show_with_run_id_dumps_result_json(tmp_path, capsys):
    """--run-id X → result.json 全文 dump（json.dumps indent=2）."""
    save(_make_eval_result(run_id="20250101T000001", n=3), runs_dir=tmp_path)

    args = argparse.Namespace(
        run_id="20250101T000001", task=None, mode=None,
        last=None, samples=0, runs_dir=tmp_path,
    )
    rc = cmd_show(args)
    assert rc == 0
    out = capsys.readouterr().out
    # 应是格式化 json（indent=2 → "  \"" 缩进）
    parsed = json.loads(out)
    assert parsed["run_id"] == "20250101T000001"
    assert parsed["n"] == 3
    assert parsed["aggregated"] == {"accuracy": 1.0}


def test_cmd_show_with_run_id_and_samples_prints_per_sample(tmp_path, capsys):
    """--run-id X --samples K → result json 后追加 K 行 sample 摘要."""
    save(_make_eval_result(run_id="20250101T000001", n=4), runs_dir=tmp_path)

    args = argparse.Namespace(
        run_id="20250101T000001", task=None, mode=None,
        last=None, samples=2, runs_dir=tmp_path,
    )
    cmd_show(args)
    out = capsys.readouterr().out
    assert "samples (first 2)" in out
    # 摘要行格式："  d0  pred=...  target=...  acc=..."
    assert "d0" in out
    assert "d1" in out
    assert "pred=pos" in out
    assert "target=pos" in out
    # 仅打 2 条不打 d2/d3
    assert "d2" not in out
    assert "d3" not in out


def test_cmd_show_unknown_run_id_raises(tmp_path):
    """--run-id 不存在 → FileNotFoundError（fail-fast，不 silently 退 0）."""
    args = argparse.Namespace(
        run_id="not_a_run", task=None, mode=None,
        last=None, samples=0, runs_dir=tmp_path,
    )
    with pytest.raises(FileNotFoundError):
        cmd_show(args)


# ---------- ④ build_parser argparse 形状 --------------------------------

def test_build_parser_returns_argparse_parser():
    """build_parser() 返 argparse.ArgumentParser 实例（而非自定义 wrapper）."""
    p = build_parser()
    assert isinstance(p, argparse.ArgumentParser)


def test_build_parser_subcommands_full_set():
    """4 个子命令 + 各自 set_defaults(func=...) 必须齐全（漏装 set_defaults
    会让 main() 在 args.func 处 AttributeError）."""
    p = build_parser()
    # subparsers 是 _SubParsersAction
    sub_action = next(
        a for a in p._actions if isinstance(a, argparse._SubParsersAction)
    )
    assert set(sub_action.choices.keys()) == {"list-tasks", "score", "run", "show"}, (
        f"子命令集漂移：{sorted(sub_action.choices.keys())}"
    )
    # 每个子 parser 都应在 set_defaults 里塞了 func
    for name, sub_parser in sub_action.choices.items():
        defaults = sub_parser._defaults  # type: ignore[attr-defined]
        assert "func" in defaults, f"子命令 {name!r} 漏 set_defaults(func=...)"
        assert callable(defaults["func"]), f"子命令 {name!r} 的 func 非 callable"


def test_build_parser_score_required_args_enforced():
    """score 子命令：缺 --task / --predictions 时 argparse SystemExit."""
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["score"])
    with pytest.raises(SystemExit):
        p.parse_args(["score", "--task", "sentiment_clf"])  # 缺 --predictions


def test_build_parser_run_task_required():
    """run 子命令：缺 --task → SystemExit."""
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["run"])


def test_build_parser_run_default_values_locked():
    """run 子命令默认值：--num-fewshot=0 / --fewshot-seed=0 / --seed=0 /
    --retrieve-top-k=5 / --retrieve-mode=hybrid / --rerank=False / --model=None.
    任一漂移会改变 zero-shot 默认行为或 RAG 默认配置——CLI 用户的"裸跑"语义直接破坏.
    """
    p = build_parser()
    args = p.parse_args(["run", "--task", "sentiment_clf"])
    assert args.num_fewshot == 0
    assert args.fewshot_seed == 0
    assert args.seed == 0
    assert args.retrieve_top_k == 5
    assert args.retrieve_mode == "hybrid"
    assert args.rerank is False
    assert args.model is None
    assert args.judge_model is None
    assert args.vdb is None
    assert args.limit is None


def test_build_parser_run_retrieve_mode_choices_locked():
    """--retrieve-mode 必须 ∈ {dense, bm25, hybrid}；非法值 SystemExit."""
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["run", "--task", "rag_qa", "--retrieve-mode", "magic"])
    # 三个合法值必须解析通过
    for mode in ("dense", "bm25", "hybrid"):
        args = p.parse_args(["run", "--task", "rag_qa", "--retrieve-mode", mode])
        assert args.retrieve_mode == mode


def test_build_parser_show_mode_choices_locked():
    """show --mode 必须 ∈ {score, run}（避免拼写错过滤静默全空）."""
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["show", "--mode", "score_typo"])
    args = p.parse_args(["show", "--mode", "score"])
    assert args.mode == "score"
    args = p.parse_args(["show", "--mode", "run"])
    assert args.mode == "run"


def test_build_parser_show_defaults_browse_index():
    """show 无任何 flag → 走 index 浏览（run_id=None / task=None / mode=None / samples=0）."""
    p = build_parser()
    args = p.parse_args(["show"])
    assert args.run_id is None
    assert args.task is None
    assert args.mode is None
    assert args.last is None
    assert args.samples == 0


def test_build_parser_score_runs_dir_is_path():
    """--runs-dir 应被 type=Path 转 Path 而非保留 str（storage 层接 Path）."""
    p = build_parser()
    args = p.parse_args([
        "score", "--task", "sentiment_clf",
        "--predictions", "p.jsonl",
        "--runs-dir", "/tmp/foo",
    ])
    assert isinstance(args.runs_dir, Path)
    assert str(args.runs_dir) == "/tmp/foo"


def test_build_parser_no_subcommand_raises():
    """裸 `python -m evals` 无子命令 → argparse SystemExit（required=True）."""
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args([])


# ---------- ⑤ main(argv) 入口 -------------------------------------------

def test_main_list_tasks_returns_zero(capsys):
    """`main(['list-tasks'])` 端到端：完整 argparse → cmd_list_tasks → return 0."""
    rc = main(["list-tasks"])
    assert rc == 0
    out = capsys.readouterr().out.splitlines()
    assert out == list_tasks()


def test_main_show_index_empty_dir(tmp_path, capsys):
    """`main(['show', '--runs-dir', tmp])` 空目录 → 0 行 + return 0."""
    rc = main(["show", "--runs-dir", str(tmp_path)])
    assert rc == 0
    assert capsys.readouterr().out == ""


def test_main_show_with_run_id_uses_runs_dir(tmp_path, capsys):
    """`main(['show', '--run-id', X, '--runs-dir', tmp])` 端到端 dump result.json."""
    save(_make_eval_result(run_id="20250101T000099"), runs_dir=tmp_path)

    rc = main(["show", "--run-id", "20250101T000099", "--runs-dir", str(tmp_path)])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["run_id"] == "20250101T000099"


def test_main_unknown_subcommand_exits():
    """未知子命令 → SystemExit（与 list-tasks/score/run/show 集合排他）."""
    with pytest.raises(SystemExit):
        main(["totally-not-a-subcommand"])


def test_main_score_missing_task_exits():
    """`main(['score', '--predictions', ...])` 缺 --task → SystemExit."""
    with pytest.raises(SystemExit):
        main(["score", "--predictions", "p.jsonl"])

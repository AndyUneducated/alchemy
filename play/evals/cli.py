"""CLI：argparse 四子命令.

  list-tasks             列所有已注册 task
  score                  offline 打分（Phase 1 主路径，sacrebleu 风格）
  run                    active harness 跑 LM
  show                   跨 run 查询 / 单 run 聚合 & 样例展示

model spec（run 子命令用）：
  mock:gold
  mock:noisy:0.3
  mock:constant:neutral
  mock:rule
  openai:gpt-4o-mini      [phase 3+]
  ollama:qwen3:8b         [phase 3+]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import tasks  # noqa: F401  — 触发 @register_task 副作用
from .models.base import LM
from .models.mock import MockLM
from .registry import get_task, list_tasks
from .runner import evaluate_active, evaluate_offline
from .storage import DEFAULT_RUNS_DIR, load_run, read_index, save


# ---------- model spec 解析 ----------

def parse_model_spec(spec: str, task) -> LM:  # noqa: ANN001 — Task 类型 forward-ref 避免循环
    """mock:<mode>[:<arg>] → MockLM. 其它 provider 前缀 Phase 3+ 再接."""
    parts = spec.split(":")
    provider = parts[0]
    if provider == "mock":
        if len(parts) < 2:
            raise ValueError(f"invalid mock spec: {spec!r}; expected mock:<mode>[:<arg>]")
        mode = parts[1]
        docs = list(task.docs())
        if mode == "gold":
            return MockLM(mode="gold", docs=docs)
        if mode == "noisy":
            noise = float(parts[2]) if len(parts) > 2 else 0.3
            seed = int(parts[3]) if len(parts) > 3 else 0
            return MockLM(mode="noisy", docs=docs, noise=noise, seed=seed)
        if mode == "constant":
            label = parts[2] if len(parts) > 2 else "neutral"
            return MockLM(mode="constant", docs=docs, label=label)
        if mode == "rule":
            return MockLM(mode="rule", docs=docs)
        raise ValueError(f"unknown mock mode: {mode!r}")
    raise ValueError(
        f"provider {provider!r} not supported in phase 1 (only 'mock'); "
        "phase 3+ will add openai / anthropic / ollama / gemini"
    )


# ---------- 输出格式化 ----------

def _fmt_row(r: dict) -> str:
    """一行 index row → 可读短行."""
    agg = r.get("aggregated", {})
    parts = [f"{k}={v:.4f}" for k, v in agg.items()]
    return (
        f"{r['run_id']:<30} task={r['task']:<15} "
        f"mode={r['mode']:<6} model={r['model']:<28} "
        f"n={r['n']:>3}  {' '.join(parts)}"
    )


# ---------- 子命令 handlers ----------

def cmd_list_tasks(_args: argparse.Namespace) -> int:
    for name in list_tasks():
        print(name)
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    task = get_task(args.task)
    result = evaluate_offline(
        task,
        args.predictions,
        limit=args.limit,
        source_label=args.source_label,
    )
    save(result, runs_dir=args.runs_dir)
    print(f"# run_id={result.run_id}  mode=score  model={result.model}  n={result.n}  elapsed={result.elapsed_ms:.1f}ms")
    for k, v in result.aggregated.items():
        print(f"  {k:<16} {v:.4f}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    task = get_task(args.task)
    lm = parse_model_spec(args.model, task)
    result = evaluate_active(
        task,
        lm,
        limit=args.limit,
        seed=args.seed,
        num_fewshot=args.num_fewshot,
        fewshot_seed=args.fewshot_seed,
    )
    save(result, runs_dir=args.runs_dir)
    print(
        f"# run_id={result.run_id}  mode=run  model={result.model}  n={result.n}  "
        f"num_fewshot={result.num_fewshot}  elapsed={result.elapsed_ms:.1f}ms"
    )
    for k, v in result.aggregated.items():
        print(f"  {k:<16} {v:.4f}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    if args.run_id:
        result, samples = load_run(args.run_id, runs_dir=args.runs_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.samples:
            print(f"\n# samples (first {args.samples}):")
            for s in samples[: args.samples]:
                print(f"  {s['doc_id']}  pred={s['prediction']:<10}  target={s['target']:<10}  acc={s['metrics']['acc']:.0f}")
        return 0

    rows = read_index(args.runs_dir)
    if args.task:
        rows = [r for r in rows if r["task"] == args.task]
    if args.mode:
        rows = [r for r in rows if r["mode"] == args.mode]
    rows.sort(key=lambda r: r["created_at"])
    if args.last:
        rows = rows[-args.last :]
    for r in rows:
        print(_fmt_row(r))
    return 0


# ---------- argparse ----------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m evals",
        description="双模式 LLM 评测 harness（score offline + run active）",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list-tasks", help="列出所有已注册 task")
    p_list.set_defaults(func=cmd_list_tasks)

    p_score = sub.add_parser("score", help="offline 打分（Phase 1 主路径）")
    p_score.add_argument("--task", required=True, help="task 名，如 sentiment_clf")
    p_score.add_argument("--predictions", required=True, help="predictions JSONL 路径 {id, prediction}")
    p_score.add_argument("--source-label", default=None, help="显示用的 model 标签（默认取文件 basename）")
    p_score.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    p_score.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR, help="run 结果落盘目录")
    p_score.set_defaults(func=cmd_score)

    p_run = sub.add_parser("run", help="active 模式：驱动 LM 跑 prompt")
    p_run.add_argument("--task", required=True)
    p_run.add_argument("--model", required=True, help="model spec，如 mock:gold / mock:noisy:0.3 / mock:constant:neutral")
    p_run.add_argument("--limit", type=int, default=None)
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument(
        "--num-fewshot",
        type=int,
        default=0,
        help="prompt 前拼 K 条 example（lm-eval 风格 K-shot）；0=zero-shot 与 Phase 1 字节相同",
    )
    p_run.add_argument(
        "--fewshot-seed",
        type=int,
        default=0,
        help="few-shot 抽样 RNG seed；只控 example 抽样不影响其它路径",
    )
    p_run.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    p_run.set_defaults(func=cmd_run)

    p_show = sub.add_parser("show", help="查 run 结果（跨 run 索引 / 单 run drill-down）")
    p_show.add_argument("--run-id", default=None, help="具体 run_id，不传则列跨 run 索引")
    p_show.add_argument("--task", default=None, help="过滤 task")
    p_show.add_argument("--mode", default=None, choices=["score", "run"], help="过滤 mode")
    p_show.add_argument("--last", type=int, default=None, help="只显示最近 N 条")
    p_show.add_argument("--samples", type=int, default=0, help="单 run 展示前 N 条样例")
    p_show.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    p_show.set_defaults(func=cmd_show)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)

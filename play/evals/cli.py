"""CLI：argparse 四子命令.

  list-tasks             列所有已注册 task
  score                  score 模式打分（Phase 1 主路径，sacrebleu 风格，不驱动 LM）
  run                    run 模式 harness 驱动 LM
  show                   跨 run 查询 / 单 run 聚合 & 样例展示

model spec（run 的 --model / --judge-model 与 score 的 --judge-model 共用同一 grammar）：
  mock:gold
  mock:noisy:0.3
  mock:constant:neutral
  mock:rule
  ollama:qwen2.5:32b      [phase 3]
  openai:gpt-4o-mini      [phase 3+ scaffold; not yet runnable]
  anthropic:claude-...    [phase 3+ scaffold; not yet runnable]

`--judge-model` 当前 score / run 两子命令都接，但只挂 qa_open（其它 task 给该 flag 会立即 SystemExit；扩展时改 _build_task_with_optional_judge 的 dispatch）.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import tasks  # noqa: F401  — 触发 @register_task 副作用
from .models.base import LM
from .models.mock import MockLM
from .models.ollama import OllamaLM
from .registry import get_task, list_tasks
from .runner import evaluate_run, evaluate_score
from .storage import DEFAULT_RUNS_DIR, load_run, read_index, save

EXTERNAL_PROVIDERS = ("openai", "anthropic", "gemini")


# ---------- model spec 解析 ----------

def parse_model_spec(spec: str, task) -> LM:  # noqa: ANN001 — Task 类型 forward-ref 避免循环
    """spec → LM 实例 dispatch.

      mock:<mode>[:<arg>]     → MockLM (phase 1)
      ollama:<model>          → OllamaLM (phase 3)
      openai|anthropic|gemini → NotImplementedError（架构留口，phase 3 暂不启用）
    """
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
    if provider == "ollama":
        if len(parts) < 2:
            raise ValueError(f"invalid ollama spec: {spec!r}; expected ollama:<model>")
        model = ":".join(parts[1:])
        return OllamaLM(model=model)
    if provider in EXTERNAL_PROVIDERS:
        raise NotImplementedError(
            f"{provider!r} adapter scaffolded but not enabled in phase 3; "
            "only 'ollama' is currently runnable. Add models/<provider>.py + extend "
            "parse_model_spec to enable external providers."
        )
    raise ValueError(
        f"unknown provider {provider!r} in spec {spec!r}; "
        f"supported: mock / ollama; deferred (NotImplementedError): {EXTERNAL_PROVIDERS}"
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


def _build_task_with_optional_judge(task_name: str, judge_model_spec: str | None):
    """get_task(name) + 可选 judge_lm 注入。

    `judge_model_spec` None → 平凡构造；否则 parse 成 LM 后注入。phase 3 仅 qa_open
    接 judge，其它 task 配该 flag 立即 SystemExit（fail-fast 而非 silently 忽略）。
    扩展第 2 个支持 judge 的 task 时改这里的 dispatch.
    """
    base_task = get_task(task_name)
    if judge_model_spec is None:
        return base_task

    judge_lm = parse_model_spec(judge_model_spec, base_task)
    from .tasks.qa_open import QAOpen
    if not isinstance(base_task, QAOpen):
        raise SystemExit(
            f"--judge-model only supported by qa_open in phase 3 (got task={task_name!r}); "
            "extend the dispatch in cli.py::_build_task_with_optional_judge when adding judge to other tasks"
        )
    return QAOpen(judge_lm=judge_lm)


def cmd_score(args: argparse.Namespace) -> int:
    task = _build_task_with_optional_judge(args.task, args.judge_model)
    result = evaluate_score(
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
    task = _build_task_with_optional_judge(args.task, args.judge_model)
    lm = parse_model_spec(args.model, task)
    result = evaluate_run(
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
        description="双模式 LLM 评测 harness（score: 文件打分 / run: 驱动 LM）",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list-tasks", help="列出所有已注册 task")
    p_list.set_defaults(func=cmd_list_tasks)

    p_score = sub.add_parser("score", help="score 模式：读 predictions JSONL 打分，不驱动 LM")
    p_score.add_argument("--task", required=True, help="task 名，如 sentiment_clf")
    p_score.add_argument("--predictions", required=True, help="predictions JSONL 路径 {id, prediction}")
    p_score.add_argument("--source-label", default=None, help="显示用的 model 标签（默认取文件 basename）")
    p_score.add_argument(
        "--judge-model",
        default=None,
        help="judge LM spec（仅 qa_open 接 judge_pointwise，e.g. ollama:qwen2.5:32b）；不传则只跑 lexical baseline",
    )
    p_score.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    p_score.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR, help="run 结果落盘目录")
    p_score.set_defaults(func=cmd_score)

    p_run = sub.add_parser("run", help="run 模式：驱动 LM 跑 prompt")
    p_run.add_argument("--task", required=True)
    p_run.add_argument("--model", required=True, help="model spec，如 mock:gold / mock:noisy:0.3 / mock:constant:neutral")
    p_run.add_argument(
        "--judge-model",
        default=None,
        help="judge LM spec（仅 qa_open 接 judge_pointwise，e.g. ollama:qwen2.5:32b）；不传则只跑 lexical baseline",
    )
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

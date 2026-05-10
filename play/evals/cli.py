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

`--judge-model` 当前 score / run 两子命令都接，挂 qa_open / rag_qa（rag_retrieval
不接 judge）.

phase 4 新增 `--vdb` / `--retrieve-top-k` / `--retrieve-mode` / `--rerank` 4 个
RAG 专属 flag：仅 `rag_retrieval` / `rag_qa` 接，其它 task 配该 flag 立即 SystemExit
（fail-fast 而非 silently 忽略）。dispatch 在 `_build_task_with_optional_deps`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import tasks  # noqa: F401  — 触发 @register_task 副作用
from .api import Request, Response
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

      mock:<mode>[:<arg>]            → MockLM (phase 1)
      ollama:<model>                 → OllamaLM (phase 3)
      ollama:<model>@seed=<K>        → OllamaLM(seed=K)；agent_sft phase 1 多 seed 用
      openai|anthropic|gemini        → NotImplementedError（架构留口，phase 3 暂不启用）

    `@seed=K` 后缀（仅 ollama 支持）：把 LM-side 采样 seed 注入 `OllamaLM(seed=K)`，
    与 CLI 的 `--seed` 区分（后者管 fewshot 抽样 / runner 级 RNG，不到 LM 端）.
    `lm.name` 保留 `@seed=K` 后缀让 EvalResult.model 字段可区分多 seed run，方便
    aggregate_seeds.py 按 (task, model_label_w/o_seed, seed) group.
    """
    # 先剥 `@seed=K` 后缀（仅出现在最尾），剩余给具体 provider parse
    seed_suffix: str | None = None
    lm_seed: int | None = None
    if "@seed=" in spec:
        spec, seed_str = spec.rsplit("@seed=", 1)
        try:
            lm_seed = int(seed_str)
        except ValueError as e:
            raise ValueError(f"invalid seed in model spec: {seed_str!r}") from e
        seed_suffix = f"@seed={lm_seed}"

    parts = spec.split(":")
    provider = parts[0]
    if provider == "mock":
        if seed_suffix is not None:
            raise ValueError(
                f"@seed=K suffix not supported for {provider!r} (use mock:noisy:<noise>:<seed> instead)"
            )
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
            raise ValueError(f"invalid ollama spec: {spec!r}; expected ollama:<model>[@seed=K]")
        model = ":".join(parts[1:])
        kwargs: dict = {}
        if lm_seed is not None:
            kwargs["seed"] = lm_seed
        lm = OllamaLM(model=model, **kwargs)
        if seed_suffix is not None:
            # 把 seed 标到 model_label，让 EvalResult.model 在多 seed run 间可区分
            lm.name = f"{lm.name}{seed_suffix}"
        return lm
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

def _fmt_kv(k: str, v, prefix: str = "") -> list[str]:  # noqa: ANN001 — v 可为 float / dict / int / None
    """递归把 (key, value) 拍平成 'k=v' 列表；嵌套 dict 用 dot 连接.

    phase 6 起 aggregated 允许嵌套（efficiency 子组等）；老的 phase 1-5 平铺指标
    走 isinstance 非 dict 分支，与原 `_fmt_row` 字节相同.
    phase 7 audit P2：None 占位 stat（如 safety.judge_safety_score 未接 judge_lm 时）
    渲染为 `<n/a>`，与"真 0"显式区分；落 result.json 仍是 null（dataclasses.asdict）.
    """
    full = f"{prefix}{k}"
    if v is None:
        return [f"{full}=<n/a>"]
    if isinstance(v, dict):
        out: list[str] = []
        for sub_k, sub_v in v.items():
            out.extend(_fmt_kv(sub_k, sub_v, prefix=f"{full}."))
        return out
    if isinstance(v, (int, float)):
        return [f"{full}={float(v):.4f}"]
    return [f"{full}={v}"]


def _fmt_row(r: dict) -> str:
    """一行 index row → 可读短行（phase 6 起支持嵌套 aggregated 子组）."""
    agg = r.get("aggregated", {})
    parts: list[str] = []
    for k, v in agg.items():
        parts.extend(_fmt_kv(k, v))
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


class _RetrieverOnlyLM(LM):
    """name-only LM stub for `output_type='none'` tasks（phase 4 引入；rag_retrieval 用）.

    runner 在 output_type='none' 分支不会调 generate_until——本 stub 只承担落
    EvalResult.model 字段的"人类可读 model 标签"职责（如 'retriever:panel:hybrid'）.
    若被意外调用 → AssertionError，捕捉 runner 分支错误.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def generate_until(self, requests: list[Request]) -> list[Response]:
        raise AssertionError(
            f"_RetrieverOnlyLM(name={self.name!r}).generate_until called; "
            f"output_type='none' branch should have skipped LM invocation"
        )


def _build_task_with_optional_deps(
    task_name: str,
    *,
    judge_model_spec: str | None = None,
    vdb: str | Path | None = None,
    retrieve_top_k: int = 5,
    retrieve_mode: str = "hybrid",
    rerank: bool = False,
):
    """get_task(name) + 可选依赖注入（judge_lm / retrieve_fn / run_fn）.

    - `judge_model_spec` 给定 → parse 为 LM 注入相应 task（qa_open / rag_qa / agent_traj）
    - `vdb` 给定 → make_retrieve_fn 注入 RAG task（rag_retrieval / rag_qa）
    - agent_traj：永远注入 make_run_fn（cheap closure；score 路径不会触发 subprocess）
    - 不匹配的 task × flag 组合 → SystemExit fail-fast

    扩展新 task 支持时在此处加 dispatch 分支.
    """
    from .tasks.agent_traj import AgentTraj
    from .tasks.nudge_fire_rate import NudgeFireRate
    from .tasks.qa_open import QAOpen
    from .tasks.rag_qa import RagQA
    from .tasks.rag_retrieval import RagRetrieval
    from .tasks.safety import Safety

    base_task = get_task(task_name)
    judge_lm = parse_model_spec(judge_model_spec, base_task) if judge_model_spec else None
    retrieve_fn = None
    if vdb is not None:
        from .models.rag_retrieve import make_retrieve_fn
        retrieve_fn = make_retrieve_fn(
            vdb, top_k=retrieve_top_k, mode=retrieve_mode, rerank=rerank,
        )

    if isinstance(base_task, RagRetrieval):
        if judge_lm is not None:
            raise SystemExit(
                f"--judge-model not supported by {task_name!r}; "
                "rag_retrieval has no LM-side output. Use rag_qa for grounding judge."
            )
        return RagRetrieval(retrieve_fn=retrieve_fn, top_k=retrieve_top_k)

    if isinstance(base_task, RagQA):
        return RagQA(retrieve_fn=retrieve_fn, judge_lm=judge_lm, top_k=retrieve_top_k)

    if isinstance(base_task, AgentTraj):
        if vdb is not None:
            raise SystemExit(
                f"--vdb / RAG flags not supported by {task_name!r}; "
                "agent_traj uses subprocess-driven agent_engine, not direct retrieval."
            )
        from .models.agent_engine_run import make_run_fn
        return AgentTraj(run_fn=make_run_fn(), judge_lm=judge_lm)

    if isinstance(base_task, NudgeFireRate):
        if vdb is not None:
            raise SystemExit(
                f"--vdb / RAG flags not supported by {task_name!r}; "
                "nudge_fire_rate uses subprocess-driven agent_engine, not direct retrieval."
            )
        if judge_lm is not None:
            raise SystemExit(
                f"--judge-model not supported by {task_name!r}; "
                "nudge_fire_rate is a process-conformance metric (no LM-side judging)."
            )
        from .models.agent_engine_run import make_run_fn
        return NudgeFireRate(run_fn=make_run_fn())

    if isinstance(base_task, Safety):
        if vdb is not None:
            raise SystemExit(
                f"--vdb / RAG flags not supported by {task_name!r}; "
                "safety is a text-safety task, not retrieval-driven."
            )
        if judge_lm is None:
            return base_task
        return Safety(judge_lm=judge_lm)

    if isinstance(base_task, QAOpen):
        if vdb is not None:
            raise SystemExit(
                f"--vdb / RAG flags not supported by {task_name!r}; "
                "use rag_qa / rag_retrieval for retrieval-driven tasks."
            )
        if judge_lm is None:
            return base_task
        return QAOpen(judge_lm=judge_lm)

    # 其它 task：拒绝 RAG / judge flag
    if judge_lm is not None:
        raise SystemExit(
            f"--judge-model only supported by qa_open / rag_qa / agent_traj / safety (got task={task_name!r}); "
            "extend the dispatch in cli.py::_build_task_with_optional_deps when adding judge to other tasks"
        )
    if vdb is not None:
        raise SystemExit(
            f"--vdb only supported by rag_retrieval / rag_qa (got task={task_name!r})"
        )
    return base_task


# 向后兼容别名（phase 3 测试沿用 _build_task_with_optional_judge 名字）
def _build_task_with_optional_judge(task_name: str, judge_model_spec: str | None):
    return _build_task_with_optional_deps(task_name, judge_model_spec=judge_model_spec)


def cmd_score(args: argparse.Namespace) -> int:
    task = _build_task_with_optional_deps(
        args.task,
        judge_model_spec=args.judge_model,
        # score 路径不需要 retrieve_fn（contexts/retrieved_ids 已在 predictions JSONL）
    )
    result = evaluate_score(
        task,
        args.predictions,
        limit=args.limit,
        source_label=args.source_label,
    )
    save(result, runs_dir=args.runs_dir)
    print(f"# run_id={result.run_id}  mode=score  model={result.model}  n={result.n}  elapsed={result.elapsed_ms:.1f}ms")
    _print_aggregated(result.aggregated)
    return 0


def _is_all_zero_nested(d) -> bool:  # noqa: ANN001 — d 可能是 dict / 数值 leaf / None
    """递归判断嵌套 dict 所有 leaf 数值是否都为 0（None 视为零类信号；非数值 leaf → False）.

    phase 7 audit P2：safety stat 用 None 占位"未测得"，None 与 0 在折叠语义上等价
    （都属于"无 metric 信号"），但 trait gate（_should_fold_when_all_zero）仍按 dim
    决定是否真折叠——content class（safety）即使全 None 也不折叠，让 <n/a> 显式渲染.
    """
    if d is None:
        return True
    if isinstance(d, dict):
        return all(_is_all_zero_nested(v) for v in d.values())
    if isinstance(d, (int, float)):
        return d == 0
    return False


# cross-cutting dim → metric module 路径映射，用于查询 module-level
# FOLD_AS_NOT_MEASURED_WHEN_ALL_ZERO trait.
#
# wave 3（DECISIONS §7.2）：safety 退出 cross-cutting（回归 standalone task），
# 此映射只剩 efficiency。加新 cross-cutting 维度（calibration 等）在此注册.
_DIM_MODULES: dict[str, str] = {
    "efficiency": "evals.metrics.efficiency",
}


def _should_fold_when_all_zero(dim: str) -> bool:
    """查询 cross-cutting dim 模块的 FOLD_AS_NOT_MEASURED_WHEN_ALL_ZERO trait.

    缺失或未注册 → 默认 True 兼容老 dim（保留 phase 6 audit §1.7 立的折叠默认行为）.
    详见 metrics/efficiency.py / metrics/safety.py 的 trait 常量声明.
    """
    mod_path = _DIM_MODULES.get(dim)
    if not mod_path:
        return True
    import importlib
    mod = importlib.import_module(mod_path)
    return getattr(mod, "FOLD_AS_NOT_MEASURED_WHEN_ALL_ZERO", True)


def _print_aggregated(agg: dict) -> None:
    """嵌套友好打印：phase 6 起 aggregated 含 efficiency 子组，递归走 _fmt_kv.

    audit §1.7：cross-cutting dim 嵌套子组若所有 leaf 数值全 0/None 且该 dim 在 trait 表里
    声明 FOLD_AS_NOT_MEASURED_WHEN_ALL_ZERO=True，折叠为 `<dim>: <not measured>` 单行避免
    视觉误导. None 占位的 stat 走 _fmt_kv 的 `<n/a>` 渲染.
    顶层 task-specific 指标（accuracy=0 等）保持显式 0 输出（task 信号不折叠）.

    DECISIONS §7.3 wave 3：嵌套二级折叠——cross-cutting 子树（如 efficiency）顶层非全 0 但
    内部子子组（如 efficiency.judge：task 没接 judge_lm / mock judge / 价格表未命中）全 0 时，
    按同 trait gate 单独折叠为 `<dim>.<sub>: <not measured>` 单行.
    """
    for k, v in agg.items():
        # 顶层折叠：cross-cutting dim 全 0 → 单行 `<dim>: <not measured>`
        if isinstance(v, dict) and _is_all_zero_nested(v) and _should_fold_when_all_zero(k):
            print(f"  {k:<28} <not measured (no LM signal)>")
            continue

        # 嵌套二级折叠（DECISIONS §7.3）：cross-cutting dim 顶层非全 0 但内部子子组全 0
        if k in _DIM_MODULES and isinstance(v, dict) and _should_fold_when_all_zero(k):
            for sub_k, sub_v in v.items():
                if isinstance(sub_v, dict) and _is_all_zero_nested(sub_v):
                    full_path = f"{k}.{sub_k}"
                    print(f"  {full_path:<28} <not measured (no LM signal)>")
                    continue
                for line in _fmt_kv(sub_k, sub_v, prefix=f"{k}."):
                    key, _, val = line.partition("=")
                    print(f"  {key:<28} {val}")
            continue

        # 顶层 task scalar（含 safety task 自身的 refusal_rate 等 wave 3 平铺 metric）
        for line in _fmt_kv(k, v):
            key, _, val = line.partition("=")
            print(f"  {key:<28} {val}")


def cmd_run(args: argparse.Namespace) -> int:
    # phase 4 RAG flag 用 getattr 兼容老 Namespace 构造（如 phase 3 live 测试手搓 Namespace 不带新 flag）
    vdb = getattr(args, "vdb", None)
    retrieve_top_k = getattr(args, "retrieve_top_k", 5)
    retrieve_mode = getattr(args, "retrieve_mode", "hybrid")
    rerank = getattr(args, "rerank", False)

    task = _build_task_with_optional_deps(
        args.task,
        judge_model_spec=args.judge_model,
        vdb=vdb,
        retrieve_top_k=retrieve_top_k,
        retrieve_mode=retrieve_mode,
        rerank=rerank,
    )

    # output_type='none' task（rag_retrieval / agent_traj）允许省 --model：用代表性 label 占位
    if task.output_type == "none":
        if args.model:
            lm: LM = parse_model_spec(args.model, task)
        elif vdb:
            lm = _RetrieverOnlyLM(name=f"retriever:{Path(vdb).name}:{retrieve_mode}")
        elif task.name in ("agent_traj", "nudge_fire_rate"):
            lm = _RetrieverOnlyLM(name="agent_engine")
        else:
            raise SystemExit(
                f"task={args.task!r} has output_type='none'; pass --vdb to label the run "
                "or --model for an explicit no-op label"
            )
    else:
        if not args.model:
            raise SystemExit(f"--model is required for task={args.task!r} (output_type={task.output_type!r})")
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
    _print_aggregated(result.aggregated)
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
    p_run.add_argument(
        "--model",
        default=None,
        help=(
            "model spec，如 mock:gold / mock:noisy:0.3 / ollama:qwen2.5:32b. "
            "task.output_type='none'（rag_retrieval）时可省，由 --vdb 自动派生 retriever 标签."
        ),
    )
    p_run.add_argument(
        "--judge-model",
        default=None,
        help=(
            "judge LM spec（qa_open / rag_qa 接，e.g. ollama:qwen2.5:32b）；"
            "不传则跑 lexical baseline（rag_qa 仅 em + rouge_l）"
        ),
    )
    # phase 4 RAG 专属 flags（仅 rag_retrieval / rag_qa 接）
    p_run.add_argument(
        "--vdb",
        default=None,
        help="VDB 目录路径（如 ../rag/vdb/panel）；指定后 RAG task 在 process_docs 自动 retrieve. 仅 rag_retrieval / rag_qa 接.",
    )
    p_run.add_argument(
        "--retrieve-top-k",
        type=int,
        default=5,
        help="检索返回的 top-K 文档数（注入 doc.metadata 用）",
    )
    p_run.add_argument(
        "--retrieve-mode",
        choices=["dense", "bm25", "hybrid"],
        default="hybrid",
        help="检索策略：dense / bm25 / hybrid（RRF 融合）",
    )
    p_run.add_argument(
        "--rerank",
        action="store_true",
        help="启用 cross-encoder rerank（首次加载 ~1.2GB 模型；显著提升 precision@k）",
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

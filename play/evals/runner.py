"""编排层：双入口（score / run）+ 共享尾段.

关键不变量：
  - Runner 不看 task 内部类型，只通过 Task ABC 调方法
  - Runner 不看 lm 是谁，只通过三种 request_type 调
  - Runner 不 import metrics/，所有指标调用通过 task.aggregation() 间接进入
  - Task.process_results 不区分 run/score 来源：统一吃 Response；score 路径用
    JSONL 查表顶替 LM 调用，其它完全一致

等价性：
  evaluate_score(task, preds) ≡ evaluate_run(task, PrerecordedLM(preds))
  具体由 test_runner_run.py::test_run_gold_equals_score_perfect 验证。
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any

from .api import Doc, EvalMode, EvalResult, Request, Response, SampleResult
from .metrics.efficiency import (
    efficiency_aggregated,
    efficiency_judge_aggregated,
    inject_per_sample_efficiency,
)
from .models.base import LM
from .tasks.base import Task


def _load_predictions(path: str | Path) -> dict[str, dict]:
    """读 predictions JSONL → {doc_id: row}（整行 dict）.

    Phase 4 起 row 是完整 dict，不再只取 `row['prediction']`——"如何从 row 提取字段"
    的责任下放给 `task.load_prediction(doc, row)`，让 RAG / agent task 能定义自己的
    row schema（含 contexts / retrieved_ids / transcript / usage 等额外字段）.

    `Task.load_prediction` 默认实现仅取 `row['prediction']`——分类 / 翻译类 task 的
    最小行为；override 时把 row 里的 pipeline 数据注入 `doc.metadata` + Response.
    """
    p = Path(path)
    preds: dict[str, dict] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            preds[row["id"]] = row
    return preds


def _collect_docs(task: Task, limit: int | None) -> list[Doc]:
    """取数据：全内存（Phase 1），Phase 2+ 改流式."""
    it: Iterable[Doc] = task.docs()
    if limit is not None:
        it = islice(it, limit)
    return list(it)


def _build_prompt(
    task: Task,
    doc: Doc,
    num_fewshot: int,
    pool: list[Doc],
    rng: random.Random,
) -> str:
    """拼出最终 prompt：N 段 example + query.

    `num_fewshot=0` 直接返回 `task.doc_to_text(doc)`，与 Phase 1 字节相同
    （旧 parity test 的根基）。`>0` 时从 pool 抽 K 条**非自身** example，
    用 `task.format_fewshot_example` 格式化，"\\n\\n" 分隔后拼到 query 前。

    抽样耗尽（pool 不够）时不报错，能抽几条算几条——避免小 dataset 边界 case。
    """
    if num_fewshot <= 0:
        return task.doc_to_text(doc)
    candidates = [d for d in pool if d.id != doc.id]
    k = min(num_fewshot, len(candidates))
    examples = rng.sample(candidates, k)
    parts = [task.format_fewshot_example(ex) for ex in examples]
    parts.append(task.doc_to_text(doc))
    return "\n\n".join(parts)


def _build_request(task: Task, doc: Doc, prompt: str) -> Request:
    """根据 task.output_type + 已拼好的 prompt 构造 Request。Phase 1 只处理 generate_until."""
    if task.output_type == "generate_until":
        return Request(
            doc_id=doc.id,
            prompt=prompt,
            request_type="generate_until",
            until=("\n",),
            max_tokens=64,
        )
    # Phase 4 MCQ 再加 multiple_choice / loglikelihood 分支
    raise NotImplementedError(
        f"output_type={task.output_type!r} not supported in phase 1"
    )


def _generate_run_id(task_name: str, model: str, seed: int | None) -> str:
    """{yyyymmdd-hhmmss}-{8-char hash}：时间可排序 + 同参复跑能辨识."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    key = f"{task_name}|{model}|{seed}"
    h = hashlib.sha256(key.encode()).hexdigest()[:8]
    return f"{ts}-{h}"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _evaluate_inner(
    task: Task,
    docs: list[Doc],
    responses: list[Response],
    *,
    mode: EvalMode,
    model_label: str,
    started_at: str,
    t0: float,
    run_id: str,
    num_fewshot: int = 0,
) -> EvalResult:
    """Response 之后的双模式合流中段（phase 7 立的架构合流点）.

    步骤（cross-cutting 二分：被测物 call class / 评估工具 call class / DECISIONS §7.2 + §7.3）：
      1. task.process_results per-sample 评分
      2. 被测物 call class injector（仅 run 跑；数据源 = task LM 调用副产品 usage/latency）
         - inject_per_sample_efficiency: latency / tokens / cost              [phase 6]
         - phase 9 calibration 在此续接                                        [planned]
      3. aggregated 打包：
         a) task.aggregation() 字典 → 顶层平铺
         b) 仅 run：被测物 efficiency 子组（phase 6）
         c) 双路径：评估工具 efficiency.judge 子组（DECISIONS §7.3 wave 3）
      4. 末尾测端到端 elapsed_ms

    挂载规则（DECISIONS §7.A 部分 superseded by §7.2 / §7.3）：
      - safety 不再 cross-cutting：`Safety` task 自己 own metrics["refusal_detected" /
        "jailbreak_attempted" / "judge_safety_score"]（flat 平铺）+ aggregation 4 stat；
        非 safety task 不再有 sample.metrics["safety"] / aggregated["safety"] 占位
      - aggregated["efficiency"]（被测物 call class）仅 run 挂（保留：基础设施 cross-cutting）
      - aggregated["efficiency"]["judge"]（评估工具 call class，§7.3 新增）：score / run
        双路径都挂——judge 在两路径都被调用，与被测物 task LM 仅在 run 调用不同
      - phase 10 robustness 等未来横切按 lm-eval-harness 主流走独立 task 路径，不再
        AOP 注入

    时序约定（DECISIONS §7.1.1）：
      - `t0 = perf_counter()` 由调用方在入口最早处取，传进来；本函数在末尾算
        `elapsed_ms = (perf_counter() - t0) * 1000`，确保覆盖 process_results +
        injectors + aggregation 全段（含 judge LM 调用 / RAG retrieve 等子调用）.

    See: README §命名约定 cross-cutting 表 / DECISIONS §7.B `_evaluate_inner` 中段重构 +
    §7.1.1 elapsed_ms 端到端 + §7.2 safety 回归 standalone task + §7.3 efficiency.judge.* ADR.
    """
    if len(docs) != len(responses):
        raise RuntimeError(
            f"doc/response length mismatch: docs={len(docs)} responses={len(responses)}"
        )
    sample_results: list[SampleResult] = [
        task.process_results(doc, resp) for doc, resp in zip(docs, responses)
    ]
    if mode == "run":
        sample_results = inject_per_sample_efficiency(sample_results, responses, model_label)

    aggregated: dict[str, Any] = {
        name: fn(sample_results) for name, fn in task.aggregation().items()
    }
    # 被测物 call class（仅 run）：task LM 的 efficiency
    if mode == "run":
        aggregated["efficiency"] = efficiency_aggregated(sample_results)

    # 评估工具 call class（双路径，DECISIONS §7.3）：judge LM 的 efficiency
    judge_responses, judge_label = task.collect_judge_responses()
    if judge_responses:
        if "efficiency" not in aggregated:
            # score 路径无被测物 efficiency 子树，但有 judge → 创建子树仅含 judge 子组
            aggregated["efficiency"] = {}
        aggregated["efficiency"]["judge"] = efficiency_judge_aggregated(
            judge_responses, judge_label
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return EvalResult(
        task=task.name,
        model=model_label,
        mode=mode,
        n=len(sample_results),
        aggregated=aggregated,
        per_sample=tuple(sample_results),
        run_id=run_id,
        created_at=started_at,
        # phase 7 audit P6：端到端跑分时间 round 到千分之一毫秒，避免落 result.json
        # 时浮点精度泄露（实测 0.9334170026704669 这类 15 位小数对人无价值）.
        # 不动 efficiency.latency_ms / cost_usd 等 LM 报值——dashboard / cost 累计真用得到亚 ms / 亚 cent 精度.
        elapsed_ms=round(elapsed_ms, 3),
        num_fewshot=num_fewshot,
    )


def evaluate_score(
    task: Task,
    predictions_path: str | Path,
    *,
    limit: int | None = None,
    source_label: str | None = None,
) -> EvalResult:
    """score 路径：读 predictions JSONL，直接喂进 task.process_results，绕过 LM 层.

    步骤（3 步）：
      1. 取数据：task.docs()
      2. 读预测 + 直接评分：preds[doc.id] → Response(text=pred) → process_results
      3. 合流：_evaluate_inner（process_results + cross-cutting injectors + 打包）

    语义等价于 evaluate_run(task, PrerecordedLM(predictions_path))。
    """
    started_at = _iso_now()
    t0 = time.perf_counter()

    docs = _collect_docs(task, limit)
    preds = _load_predictions(predictions_path)

    docs_for_eval: list[Doc] = []
    responses: list[Response] = []
    for doc in docs:
        if doc.id not in preds:
            raise KeyError(
                f"predictions file missing doc_id={doc.id!r} "
                f"(found {len(preds)} preds for {len(docs)} docs); strict join required"
            )
        # phase 4：让 task 自定 row → (doc', response) 翻译.
        # 默认 Task.load_prediction 只取 row['prediction']，与 phase 1 字节相同.
        enriched_doc, response = task.load_prediction(doc, preds[doc.id])
        docs_for_eval.append(enriched_doc)
        responses.append(response)

    model_label = source_label or f"preds:{Path(predictions_path).stem}"
    run_id = _generate_run_id(task.name, model_label, None)

    return _evaluate_inner(
        task,
        docs_for_eval,
        responses,
        mode="score",
        model_label=model_label,
        started_at=started_at,
        t0=t0,
        run_id=run_id,
    )


def evaluate_run(
    task: Task,
    lm: LM,
    *,
    limit: int | None = None,
    seed: int = 0,
    num_fewshot: int = 0,
    fewshot_seed: int = 0,
) -> EvalResult:
    """run 路径：harness 6 步.

      1. 取数据
      2. 建请求（按 num_fewshot 拼 prompt）
      3. 批调模型  <-- 未来并发在这里：asyncio.gather + semaphore
      4. 合流：_evaluate_inner（process_results + cross-cutting injectors + 打包）

    `num_fewshot=0` 时 prompt 与 Phase 1 字节相同（_build_prompt 早 return）。
    `num_fewshot>0` 时从 `task.fewshot_docs()` 抽 K 条**非自身** example 拼到前面。
    `fewshot_seed` 只控抽样，不影响其它路径——便于 sweep 不同 N 但保持其它一致。
    """
    started_at = _iso_now()
    t0 = time.perf_counter()

    docs = _collect_docs(task, limit)
    # phase 4：LM 调用前的 docs 前置加工 hook（默认 identity 透传，老 task 不影响）。
    # 典型用法：RAG task 在此 batch 调 retrieve_fn，把 retrieved_ids/contexts pin 进 doc.metadata.
    docs = list(task.process_docs(docs))

    if task.output_type == "none":
        # phase 4：声明无 LM 调用的 task（如 rag_retrieval）——直接生成占位 Response.
        # 不走 _build_prompt / _build_request / lm.generate_until 任何一步.
        responses: list[Response] = [Response(doc_id=d.id) for d in docs]
        model_label = lm.name
    else:
        pool = list(task.fewshot_docs()) if num_fewshot > 0 else []
        rng = random.Random(fewshot_seed)
        requests = [
            _build_request(task, doc, _build_prompt(task, doc, num_fewshot, pool, rng))
            for doc in docs
        ]

        # Phase 1 串行。Phase 2+ 在此处做 asyncio.gather / thread pool / rate-limit。
        responses = lm.generate_until(requests)
        model_label = lm.name

        if len(responses) != len(docs):
            raise RuntimeError(
                f"LM returned {len(responses)} responses for {len(docs)} requests"
            )

    run_id = _generate_run_id(task.name, model_label, seed)

    return _evaluate_inner(
        task,
        docs,
        responses,
        mode="run",
        model_label=model_label,
        started_at=started_at,
        t0=t0,
        run_id=run_id,
        num_fewshot=num_fewshot,
    )

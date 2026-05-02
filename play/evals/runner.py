"""编排层：双入口（score / run）+ 共享尾段.

关键不变量：
  - Runner 不看 task 内部类型，只通过 Task ABC 调方法
  - Runner 不看 lm 是谁，只通过三种 request_type 调
  - Runner 不 import metrics/，所有指标调用通过 task.aggregation() 间接进入
  - Task.process_results 不区分 run/score 来源：统一吃 Response；offline 路径用
    JSONL 查表顶替 LM 调用，其它完全一致

等价性：
  evaluate_offline(task, preds) ≡ evaluate_active(task, PrerecordedLM(preds))
  具体由 test_runner_active.py::test_active_gold_equals_offline_perfect 验证。
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

from .api import Doc, EvalMode, EvalResult, Request, Response, SampleResult
from .models.base import LM
from .tasks.base import Task


def _load_predictions(path: str | Path) -> dict[str, str]:
    """读 {id, prediction} JSONL → {doc_id: prediction} 字典."""
    p = Path(path)
    preds: dict[str, str] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            preds[row["id"]] = row["prediction"]
    return preds


def _collect_docs(task: Task, limit: int | None) -> list[Doc]:
    """取数据：全内存（Phase 1），Phase 2+ 改流式."""
    it: Iterable[Doc] = task.docs()
    if limit is not None:
        it = islice(it, limit)
    return list(it)


def _build_request(task: Task, doc: Doc) -> Request:
    """根据 task.output_type 构造 Request。Phase 1 只处理 generate_until."""
    prompt = task.doc_to_text(doc)
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


def _finalize(
    task: Task,
    sample_results: list[SampleResult],
    *,
    mode: EvalMode,
    model_label: str,
    started_at: str,
    elapsed_ms: float,
    run_id: str,
) -> EvalResult:
    """共享尾段：聚合 + 打包 EvalResult。score / run 两路径的合流点."""
    aggregated = {
        name: fn(sample_results) for name, fn in task.aggregation().items()
    }
    return EvalResult(
        task=task.name,
        model=model_label,
        mode=mode,
        n=len(sample_results),
        aggregated=aggregated,
        per_sample=tuple(sample_results),
        run_id=run_id,
        created_at=started_at,
        elapsed_ms=elapsed_ms,
    )


def evaluate_offline(
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
      3. 合流：_finalize 做聚合 + 打包

    语义等价于 evaluate_active(task, PrerecordedLM(predictions_path))。
    """
    started_at = _iso_now()
    t0 = time.perf_counter()

    docs = _collect_docs(task, limit)
    preds = _load_predictions(predictions_path)

    sample_results: list[SampleResult] = []
    for doc in docs:
        if doc.id not in preds:
            raise KeyError(
                f"predictions file missing doc_id={doc.id!r} "
                f"(found {len(preds)} preds for {len(docs)} docs); strict join required"
            )
        response = Response(doc_id=doc.id, text=preds[doc.id])
        sample_results.append(task.process_results(doc, response))

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    model_label = source_label or f"preds:{Path(predictions_path).stem}"
    run_id = _generate_run_id(task.name, model_label, None)

    return _finalize(
        task,
        sample_results,
        mode="score",
        model_label=model_label,
        started_at=started_at,
        elapsed_ms=elapsed_ms,
        run_id=run_id,
    )


def evaluate_active(
    task: Task,
    lm: LM,
    *,
    limit: int | None = None,
    seed: int = 0,
) -> EvalResult:
    """run 路径：harness 6 步.

      1. 取数据
      2. 建请求
      3. 批调模型  <-- 未来并发在这里：asyncio.gather + semaphore
      4. per-sample 评分
      5-6. 合流（_finalize 负责）
    """
    started_at = _iso_now()
    t0 = time.perf_counter()

    docs = _collect_docs(task, limit)
    requests = [_build_request(task, doc) for doc in docs]

    # Phase 1 串行。Phase 2+ 在此处做 asyncio.gather / thread pool / rate-limit。
    responses: list[Response] = lm.generate_until(requests)

    if len(responses) != len(docs):
        raise RuntimeError(
            f"LM returned {len(responses)} responses for {len(docs)} requests"
        )

    sample_results = [
        task.process_results(doc, resp) for doc, resp in zip(docs, responses)
    ]

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    run_id = _generate_run_id(task.name, lm.name, seed)

    return _finalize(
        task,
        sample_results,
        mode="run",
        model_label=lm.name,
        started_at=started_at,
        elapsed_ms=elapsed_ms,
        run_id=run_id,
    )

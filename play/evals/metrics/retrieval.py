"""族 4 IR metrics：recall@k / precision@k / mrr / ndcg@k / map@k 的 ranx 直调封装.

设计要点：
  - **ranx 直调**：IR 指标是数学定义死的成熟领域（trec_eval 几十年沉淀），
    无须自造轮子。`ranx` 是 trec_eval 的 numba JIT Python 包装，单调用 ms 级。
  - **聚合形态**：返回 `Callable[[list[SampleResult]], float]`，与 task.aggregation()
    的 dict-of-callable 协议同形，rag_retrieval.aggregation() 直接挂这些工厂.
  - **数据契约**：从 `SampleResult.artifacts` 拉 `pred_ids: list[str]` / `gold_ids: list[str]`
    （phase 4 引入的非标量产物 bucket）。约定：rag_retrieval.process_results 必填这两键，
    其它 task 不会触发——契约耦合点显式标注，避免隐式依赖.

为什么把 IR 指标放 metrics/ 而不放 tasks/rag_retrieval.py：
  - 跨 task 复用面：未来 `rag_qa` 也要在 process_results 计 retrieval-side 指标
    （contexts 的 recall/precision），retrieval.py 是天然复用点
  - 与 judge_core / judge_rag 风格一致：都是"closure 工厂返回 (sample_results) → float"
"""

from __future__ import annotations

from typing import Callable, Sequence

from ranx import Qrels, Run, evaluate as _ranx_evaluate

from ..api import SampleResult


def _build_qrels_run(
    sample_results: Sequence[SampleResult],
) -> tuple[Qrels, Run] | None:
    """从 SampleResult 列表抽 (qrels, run) 喂 ranx；若任意必填字段缺失 → None.

    artifacts 契约（rag_retrieval.process_results 必填）：
      - pred_ids: Sequence[str]   按 retrieval rank 排好的 doc_id 列表（top-k 截断后）
      - gold_ids: Sequence[str]   该 query 的相关 doc_id 集合（顺序无关）

    None 返回让 aggregation 函数能优雅降级到 0.0（防止空数据集 / 测试 stub 把全部 metric 拉爆）.
    """
    qrels_dict: dict[str, dict[str, int]] = {}
    run_dict: dict[str, dict[str, float]] = {}

    for sr in sample_results:
        pred_ids = sr.artifacts.get("pred_ids")
        gold_ids = sr.artifacts.get("gold_ids")
        if pred_ids is None or gold_ids is None:
            return None
        if not gold_ids:
            # ranx 拒绝空 gold——这种样本我们直接跳过（视为不可评）
            continue
        qrels_dict[sr.doc_id] = {gid: 1 for gid in gold_ids}
        # rank 越靠前 score 越高；len(pred_ids) - i 给单调递减的 score（无需真分数）
        run_dict[sr.doc_id] = {
            pid: float(len(pred_ids) - i) for i, pid in enumerate(pred_ids)
        }

    if not qrels_dict:
        return None

    return Qrels(qrels_dict), Run(run_dict)


def _make_metric_aggregator(metric_name: str) -> Callable[[list[SampleResult]], float]:
    """工厂：把 ranx metric 名（'recall@5' / 'mrr' / 'ndcg@10' …）封成 aggregation 闭包."""

    def _aggregate(srs: list[SampleResult]) -> float:
        if not srs:
            return 0.0
        built = _build_qrels_run(srs)
        if built is None:
            return 0.0
        qrels, run = built
        return float(_ranx_evaluate(qrels, run, metric_name))

    _aggregate.__name__ = f"aggregate_{metric_name.replace('@', '_at_')}"
    return _aggregate


def recall_at_k(k: int = 10) -> Callable[[list[SampleResult]], float]:
    """Recall@k：top-k 检出的 gold 比例.

    经典 first-stage retrieval 主指标——告诉你"召回够不够"，不管 rank.
    """
    return _make_metric_aggregator(f"recall@{k}")


def precision_at_k(k: int = 10) -> Callable[[list[SampleResult]], float]:
    """Precision@k：top-k 中 gold 的比例.

    与 recall 互补，用在"top-k 里有多少噪声"的诊断；rerank 后通常应升.
    """
    return _make_metric_aggregator(f"precision@{k}")


def mrr() -> Callable[[list[SampleResult]], float]:
    """Mean Reciprocal Rank：第一条 gold 的 reciprocal rank 求均.

    适合"只关心首条对不对"的场景（QA 上 grounding 通常只取 top-1 引用）.
    """
    return _make_metric_aggregator("mrr")


def ndcg_at_k(k: int = 10) -> Callable[[list[SampleResult]], float]:
    """Normalized DCG@k：rank 敏感的 graded relevance 综合分.

    rerank 学术对比的事实标准；当前实现 binary relevance（0/1），未来扩 graded
    可在 _build_qrels_run 把 gold_ids 改 dict[str, int].
    """
    return _make_metric_aggregator(f"ndcg@{k}")


def map_at_k(k: int = 10) -> Callable[[list[SampleResult]], float]:
    """Mean Average Precision@k：综合 recall + rank 的鸟瞰指标.

    对"前面 rank 越准 + 召回越全"双重奖励；TREC 老牌主指标.
    """
    return _make_metric_aggregator(f"map@{k}")

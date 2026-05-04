"""Phase 4 vertical slice：族 4 RAG retrieval-only task.

8 个针对 `play/rag/docs/panel/` 公司治理叙事 corpus 的检索 query + 4 份 stub
predictions（perfect / good_rerank / weak / garbage），核心叙事是"在 IR 指标上
看 retriever 质量阶梯"：

  | 预测         | recall@5 | mrr   | ndcg@5 | 故事 |
  |---|---|---|---|---|
  | perfect      | 1.0      | 1.0   | 1.0    | 上界 sanity |
  | good_rerank  | 1.0      | ~0.5  | 中     | recall 满 / rank 不准（rerank 救场场景） |
  | weak         | ~0.5     | low   | low    | 弱基线 |
  | garbage      | 0.0      | 0.0   | 0.0    | 下界 sanity |

设计要点：
  - **output_type='none'**（phase 4 引入的 literal）：runner 自动跳 LM 调用，
    检索一步 task.process_docs 把 retrieved_ids 注入 doc.metadata 即可。
    替代了"假 LM adapter"这种 anti-pattern.
  - **process_docs 注入**：run 路径传 retrieve_fn → 在 LM 调用前一次性 retrieve 全部 docs.
    `retrieve_fn` 由 cli.py 在构造时注入，task 自己不知道是 subprocess 还是 in-process.
  - **load_prediction 注入**：score 路径，把 row['retrieved_ids'] 翻译进 doc.metadata,
    Response 给占位（无 LM-side 数据）。这是 path B+C 的 pred-side 体现.
  - **artifacts 装非标量**：process_results 把 pred_ids/gold_ids 装 artifacts 给 aggregation,
    metrics 仍只装标量（这里是空 dict——本 task 无 per-sample 标量指标）.

向后兼容：本 task 通过 `retrieve_fn=None` 默认构造也能在 score 路径正常工作（不需要 retrieve_fn），
run 路径才必须注入.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Callable, ClassVar

from ..api import Doc, Response, SampleResult
from ..metrics.retrieval import (
    map_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from ..registry import register_task
from .base import Task

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "rag_retrieval" / "gold.jsonl"

# retrieve_fn 协议：query: str -> (doc_ids: list[str], contents: list[str])
RetrieveFn = Callable[[str], tuple[list[str], list[str]]]


@register_task("rag_retrieval")
class RagRetrieval(Task):
    """RAG 检索阶段独立 task：5 个 ranx IR 指标的承载体.

    构造：
      - `retrieve_fn=None`         → 仅 score 路径可用（从 predictions 读 retrieved_ids）
      - `retrieve_fn=callable`     → run 路径 process_docs hook 注入 retrieved_ids
      - `top_k`                    → process_docs 截断；score 路径 row 已截过不再处理
    """

    name: ClassVar[str] = "rag_retrieval"
    output_type: ClassVar[str] = "none"  # phase 4 literal：runner 跳 lm.generate_until

    def __init__(
        self,
        retrieve_fn: RetrieveFn | None = None,
        *,
        top_k: int = 10,
    ) -> None:
        self.data_path = DATA_PATH
        self._retrieve_fn = retrieve_fn
        self._top_k = top_k

    def docs(self) -> Iterable[Doc]:
        with self.data_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                yield Doc(
                    id=row["id"],
                    input=row["input"],
                    target=None,  # rag_retrieval 无字符串 target——phase 4 widening 后语义诚实
                    metadata={"gold_doc_ids": tuple(row["gold_doc_ids"])},
                )

    def doc_to_text(self, doc: Doc) -> str:
        """output_type='none' 时 runner 不调；保留方法只为满足 ABC."""
        return ""

    def doc_to_target(self, doc: Doc) -> str:
        """target 是 None 时 doc_to_target 不应被 fewshot 走到——返回空字符串占位."""
        return ""

    def process_docs(self, docs: list[Doc]) -> list[Doc]:
        """run 路径：在 LM 调用前一次性 retrieve 所有 docs；retrieved_ids 注入 metadata.

        retrieve_fn 缺失时（如 score 路径走到这里也安全）→ identity 透传，
        score 路径靠 load_prediction 走另一条注入通路.
        """
        if self._retrieve_fn is None:
            return docs
        out: list[Doc] = []
        for d in docs:
            ids, _contents = self._retrieve_fn(d.input)
            out.append(replace(
                d,
                metadata={**d.metadata, "retrieved_ids": tuple(ids[: self._top_k])},
            ))
        return out

    def load_prediction(self, doc: Doc, row: dict) -> tuple[Doc, Response]:
        """score 路径：row['retrieved_ids'] 进 doc.metadata；Response 占位（无 LM-side 数据）."""
        retrieved = tuple(row.get("retrieved_ids", ()))
        enriched = replace(doc, metadata={**doc.metadata, "retrieved_ids": retrieved})
        return enriched, Response(doc_id=doc.id)

    def process_results(self, doc: Doc, response: Response) -> SampleResult:
        pred_ids = list(doc.metadata.get("retrieved_ids", ()))
        gold_ids = list(doc.metadata.get("gold_doc_ids", ()))
        return SampleResult(
            doc_id=doc.id,
            prediction="",  # 无字符串 prediction（占位）
            target="",       # 无字符串 target（占位；真实 gold 在 artifacts.gold_ids）
            metrics={},      # 严守 scalar，per-sample 无标量指标
            artifacts={"pred_ids": pred_ids, "gold_ids": gold_ids},
        )

    def aggregation(self) -> dict[str, Callable[[list[SampleResult]], float]]:
        # ranx 直调；从 SampleResult.artifacts.{pred_ids, gold_ids} 拉数据
        return {
            "recall@5": recall_at_k(5),
            "precision@5": precision_at_k(5),
            "mrr": mrr(),
            "ndcg@5": ndcg_at_k(5),
            "map@5": map_at_k(5),
        }

    def higher_is_better(self) -> dict[str, bool]:
        return {
            "recall@5": True,
            "precision@5": True,
            "mrr": True,
            "ndcg@5": True,
            "map@5": True,
        }

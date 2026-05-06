"""Phase 4 vertical slice：族 4 RAG end-to-end QA task.

8 个针对 `play/rag/docs/panel/` 公司治理叙事 corpus 的中文 QA + 4 份 stub
predictions（perfect / paraphrase / wrong_fact / garbage）。教学叙事核心：
"在 grounding 维度上看 generation 质量阶梯"——

  | 预测       | em  | rouge_l | faithfulness | answer_correctness | 故事 |
  |---|---|---|---|---|---|
  | perfect    | 1.0 | ~1.0    | ~1.0         | ~1.0              | 上界 sanity |
  | paraphrase | 0.0 | mid     | ~1.0         | ~1.0              | lexical 失效 / judge 救场（**核心叙事**） |
  | wrong_fact | 0.0 | high    | low          | low               | lexical 误判 / judge 抓事实错（**反向叙事**） |
  | garbage    | 0.0 | low     | low          | low               | 下界 sanity |

设计要点：
  - **process_docs 注入 contexts**（run 路径）：在 LM 调用前一次性 retrieve 全部 docs,
    contexts/retrieved_ids pin 进 doc.metadata；`doc_to_text` 是纯字符串构造（0 IO）.
  - **load_prediction 注入 contexts**（score 路径）：从 row 里抽 contexts/retrieved_ids
    进 doc.metadata，prediction 进 Response.text——path B+C 的 score 实例.
  - **judge_lm 可选**：None → 仅 lexical（em / rouge_l），与 qa_open 的 lexical fallback 同模式.
    给 judge_lm 时挂 5 个 RAG 维度（faithfulness / answer_correctness / context_precision /
    context_recall / answer_relevancy）.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Callable, ClassVar

from ..api import Doc, Response, SampleResult
from ..metrics.judge_rag import (
    judge_answer_correctness,
    judge_answer_relevancy,
    judge_context_precision,
    judge_context_recall,
    judge_faithfulness,
)
from ..models.base import LM
from ..registry import register_task
from .base import Task
from .mt import _rouge_scorer  # 复用 mt 的中文 char-level rouge tokenizer

PROMPT_TEMPLATE = (
    "请依据以下材料回答问题。\n"
    "材料：\n{context}\n\n"
    "问题：{input}\n"
    "回答："
)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "rag_qa" / "gold.jsonl"

RetrieveFn = Callable[[str], tuple[list[str], list[str]]]


@register_task("rag_qa")
class RagQA(Task):
    """RAG end-to-end QA：retrieval + generation + grounding 评估三合一.

    构造：
      - `retrieve_fn=None`         → 仅 score 路径可用（contexts 从 predictions 读）
      - `retrieve_fn=callable`     → run 路径 process_docs hook 自动 retrieve
      - `judge_lm=None`            → 仅 lexical baseline（em / rouge_l）
      - `judge_lm=lm`              → 加 5 个 RAG 维度
      - `top_k`                    → process_docs 截断 contexts/ids 到前 K 条
    """

    name: ClassVar[str] = "rag_qa"
    output_type: ClassVar[str] = "generate_until"

    def __init__(
        self,
        retrieve_fn: RetrieveFn | None = None,
        judge_lm: LM | None = None,
        *,
        top_k: int = 5,
    ) -> None:
        self.data_path = DATA_PATH
        self._retrieve_fn = retrieve_fn
        self._judge_lm = judge_lm
        self._top_k = top_k

        if judge_lm is not None:
            self._judge_faithfulness = judge_faithfulness(judge_lm)
            self._judge_answer_correctness = judge_answer_correctness(judge_lm)
            self._judge_context_precision = judge_context_precision(judge_lm)
            self._judge_context_recall = judge_context_recall(judge_lm)
            self._judge_answer_relevancy = judge_answer_relevancy(judge_lm)
        else:
            self._judge_faithfulness = None
            self._judge_answer_correctness = None
            self._judge_context_precision = None
            self._judge_context_recall = None
            self._judge_answer_relevancy = None

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
                    target=row["target"],
                    metadata={"gold_doc_ids": tuple(row.get("gold_doc_ids", ()))},
                )

    def doc_to_text(self, doc: Doc) -> str:
        """纯字符串构造：从 doc.metadata['contexts'] 渲染 prompt，0 IO.

        `process_docs` 已在 LM 调用前一次性 retrieve 完毕；这里读已注入的 contexts.
        若 contexts 缺失（极少见，run 模式无 retrieve_fn 配置），fallback 到无材料 prompt.
        """
        contexts = doc.metadata.get("contexts", ())
        if contexts:
            ctx_block = "\n---\n".join(contexts)
        else:
            ctx_block = "（无可用材料）"
        return PROMPT_TEMPLATE.format(context=ctx_block, input=doc.input)

    def doc_to_target(self, doc: Doc) -> str:
        return doc.target or ""

    def process_docs(self, docs: list[Doc]) -> list[Doc]:
        """run 路径：retrieve 在 LM 调用前一次性完成，contexts/ids 进 doc.metadata."""
        if self._retrieve_fn is None:
            return docs
        out: list[Doc] = []
        for d in docs:
            ids, contents = self._retrieve_fn(d.input)
            out.append(replace(d, metadata={
                **d.metadata,
                "retrieved_ids": tuple(ids[: self._top_k]),
                "contexts": tuple(contents[: self._top_k]),
            }))
        return out

    def load_prediction(self, doc: Doc, row: dict) -> tuple[Doc, Response]:
        """score 路径：row['contexts'] / ['retrieved_ids'] → doc.metadata；row['prediction'] → Response.text.

        path B+C 的 score 实例：pipeline 产物住 doc 一侧，LM 输出住 Response 一侧.
        """
        enriched = replace(doc, metadata={
            **doc.metadata,
            "retrieved_ids": tuple(row.get("retrieved_ids", ())),
            "contexts": tuple(row.get("contexts", ())),
        })
        return enriched, Response(doc_id=doc.id, text=row.get("prediction"))

    def process_results(self, doc: Doc, response: Response) -> SampleResult:
        pred = (response.text or "").strip()
        target = doc.target or ""
        metrics: dict[str, float | None] = {
            "em": float(pred == target),
            "rouge_l": _per_sample_rouge_l(pred, target),
        }
        artifacts: dict[str, object] = {
            "pred_ids": list(doc.metadata.get("retrieved_ids", ())),
            "gold_ids": list(doc.metadata.get("gold_doc_ids", ())),
        }
        if self._judge_faithfulness is not None:
            # DECISIONS §X wave 4：judge_answer_correctness / judge_answer_relevancy 在 parse
            # 失败时返 None；其余 3 closure 仍只回 float（degenerate-input 路径返 0.0 是合法
            # 最低分）；统一用 None-check 既兼容也对未来 closure 升级 None 路径稳健.
            for key, fn in (
                ("faithfulness", self._judge_faithfulness),
                ("answer_correctness", self._judge_answer_correctness),
                ("context_precision", self._judge_context_precision),
                ("context_recall", self._judge_context_recall),
                ("answer_relevancy", self._judge_answer_relevancy),
            ):
                v = fn(doc, response)
                if v is not None:
                    metrics[key] = float(v)
        return SampleResult(
            doc_id=doc.id,
            prediction=pred,
            target=target,
            metrics=metrics,
            artifacts=artifacts,
        )

    def aggregation(self) -> dict[str, Callable[[list[SampleResult]], float | None]]:
        agg: dict[str, Callable[[list[SampleResult]], float | None]] = {
            "exact_match": _mean_metric("em"),
            "rouge_l": _mean_metric("rouge_l"),
        }
        if self._judge_lm is not None:
            agg["faithfulness"] = _mean_metric("faithfulness")
            agg["answer_correctness"] = _mean_metric("answer_correctness")
            agg["context_precision"] = _mean_metric("context_precision")
            agg["context_recall"] = _mean_metric("context_recall")
            agg["answer_relevancy"] = _mean_metric("answer_relevancy")
        return agg

    def higher_is_better(self) -> dict[str, bool]:
        out = {"exact_match": True, "rouge_l": True}
        if self._judge_lm is not None:
            out.update({
                "faithfulness": True,
                "answer_correctness": True,
                "context_precision": True,
                "context_recall": True,
                "answer_relevancy": True,
            })
        return out

    def collect_judge_responses(self) -> tuple[list[Response], str | None]:
        """DECISIONS §7.3：聚合 5 个 RAG judge closure 的 _recorder.responses.

        所有 5 维度共用同一 judge_lm（构造时同 LM 实例传给 5 个 factory），所以
        model_label 取任一即可（实际 5 个 recorder 的 model_label 完全相同）。
        """
        if self._judge_lm is None:
            return [], None
        all_responses: list[Response] = []
        label: str | None = None
        for fn in (
            self._judge_faithfulness,
            self._judge_answer_correctness,
            self._judge_context_precision,
            self._judge_context_recall,
            self._judge_answer_relevancy,
        ):
            if fn is None:
                continue
            rec = getattr(fn, "_recorder", None)
            if rec is None:
                continue
            all_responses.extend(rec.responses)
            label = label or rec.model_label
        return all_responses, label


def _per_sample_rouge_l(pred: str, target: str) -> float:
    """单样本 ROUGE-L F-measure（中文 char-level；复用 mt._rouge_scorer 缓存）."""
    if not pred or not target:
        return 0.0
    scorer = _rouge_scorer()
    return float(scorer.score(target, pred)["rougeL"].fmeasure)


def _mean_metric(key: str) -> Callable[[list[SampleResult]], float | None]:
    """工厂：对 SampleResult.metrics[key] 求均值的 aggregation 闭包.

    DECISIONS §X wave 4：None 占位"未测得"——key 缺 / value=None 都过滤；
    em / rouge_l 等老 metric 始终是 float（不会 None），过滤逻辑透传不影响数值；
    judge 维度（answer_correctness / answer_relevancy）parse 失败时不写键 → 返 None.
    """

    def _agg(srs: list[SampleResult]) -> float | None:
        if not srs:
            return None
        vals = [
            s.metrics[key]
            for s in srs
            if key in s.metrics and s.metrics[key] is not None
        ]
        if not vals:
            return None
        return sum(vals) / len(vals)

    _agg.__name__ = f"mean_{key}"
    return _agg

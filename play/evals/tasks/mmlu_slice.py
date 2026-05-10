"""Phase 1 通用能力**防回归** baseline：MMLU 6-subject slice (96 例).

数据来源：[`data/mmlu_slice/SOURCE.md`](../data/mmlu_slice/SOURCE.md)（钉版 HF revision + 抓取脚本）.

教学定位（agent_sft 视角）：
  - in-dist : nudge_fire_rate / agent_traj 测\"被 SFT 影响的能力\"
  - OOD-A   : bfcl_slice 测\"原本会的 function-calling 没掉\"
  - **OOD-B here**: mmlu_slice 测\"通用知识没掉\"——SFT 数据全是 agent transcript，
    经典 catastrophic forgetting 风险点正在这.

度量函数 **内联**（accuracy 仅几行 if/else，独立模块属于\"为抽而抽\"）：

|metric|含义|
|---|---|
|`accuracy`|全 96 题首字母 ∈ {A,B,C,D} 命中率|
|`accuracy_by_subject`|（aggregation 内附 dict 子组）按 subject 拆 6 个准确率|

评测协议是 **generate_until + 取首字母**（不走 loglikelihood-of-letter）——更接近真实部署
体感、不依赖 logprobs 接口；副作用是分数会比原 MMLU paper 略低（模型偶尔不出 A/B/C/D 字母时
按错处理）.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Callable, ClassVar

from ..api import Doc, Response, SampleResult
from ..registry import register_task
from .base import Task

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "mmlu_slice" / "gold.jsonl"

PROMPT_TEMPLATE = (
    "The following is a multiple-choice question. "
    "Read the question and choose the best answer.\n\n"
    "Question: {question}\n\n"
    "A. {a}\nB. {b}\nC. {c}\nD. {d}\n\n"
    "Respond with only the letter of the correct answer (A, B, C, or D).\n\n"
    "Answer:"
)


@register_task("mmlu_slice")
class MmluSlice(Task):
    """MMLU 6-subject 96-example slice，generate_until + 取首字母."""

    name: ClassVar[str] = "mmlu_slice"
    output_type: ClassVar[str] = "generate_until"

    def __init__(self) -> None:
        self.data_path = DATA_PATH

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
                    choices=tuple(row.get("choices", ())),
                    metadata=row.get("metadata", {}),
                )

    def doc_to_text(self, doc: Doc) -> str:
        choices = doc.metadata.get("raw_choices") or list(doc.choices or [])
        if len(choices) != 4:
            raise ValueError(f"mmlu_slice doc {doc.id!r} expects 4 choices, got {len(choices)}")
        a, b, c, d = choices
        return PROMPT_TEMPLATE.format(question=doc.input, a=a, b=b, c=c, d=d)

    def doc_to_target(self, doc: Doc) -> str:
        return doc.target or ""

    def process_results(self, doc: Doc, response: Response) -> SampleResult:
        pred_letter = parse_mcq_letter(response.text or "")
        target = (doc.target or "").upper()
        is_hit = 1.0 if pred_letter == target else 0.0
        # 失格预测（提取不到字母）也算 0；用 artifact 区分\"模型给了无关字符\" vs \"给错字母\"
        return SampleResult(
            doc_id=doc.id,
            prediction=pred_letter or "",
            target=target,
            metrics={"accuracy": is_hit},
            artifacts={
                "subject": doc.metadata.get("subject", "unknown"),
                "raw_text": (response.text or "").strip(),
                "pred_letter": pred_letter,  # None 即未抽出字母
            },
        )

    def aggregation(self) -> dict[str, Callable[[list[SampleResult]], float | dict | None]]:
        return {
            "accuracy": _overall_accuracy,
            "accuracy_by_subject": _accuracy_by_subject,
        }

    def higher_is_better(self) -> dict[str, bool]:
        # 与 nudge_fire_rate 同 convention：嵌套 dict 子组（accuracy_by_subject）
        # 不进 higher_is_better——只标量进，dict 由 CLI 渲染层逐键展开
        return {"accuracy": True}


# ---- 内联度量函数（plan §2：mmlu accuracy 仅几行；不抽到 metrics/） ----


_VALID_LETTERS = {"A", "B", "C", "D"}


def parse_mcq_letter(text: str) -> str | None:
    """从模型输出抽 \"A/B/C/D\" 之一. 找不到返 None.

    宽容策略（按 LLM 输出常见污染脏度排序，逐层尝试）：
      1. 第一行非空 → 去 markdown / 标点
      2. 首字符是 letter 即取
      3. 否则寻找 \"Answer: X\" 这种 echo
      4. 否则全文搜首个孤立 A/B/C/D（前后是非字母）

    第 4 步的 \"孤立 letter\" 防止 \"according to A...\" 蒙混（A 是孤立字也算 echo，靠
    第 1/2 步先抓 letter-only 输出过滤）.
    """
    if not text:
        return None
    s = text.strip()

    # 第一行非空
    for line in s.splitlines():
        line = line.strip()
        if line:
            s = line
            break

    s_clean = s.lstrip("*` ").rstrip(".,!?:;`*) ").strip()
    if not s_clean:
        return None

    # 首字符 letter only / letter + 标点
    if s_clean[0].upper() in _VALID_LETTERS:
        # 单字符 / 后面紧跟非字母 → 接受
        if len(s_clean) == 1 or not s_clean[1].isalpha():
            return s_clean[0].upper()

    # \"Answer: X\" / \"The answer is X\" 等 echo
    upper = s_clean.upper()
    for marker in ("ANSWER:", "ANSWER IS", "CORRECT ANSWER IS"):
        if marker in upper:
            after = upper.split(marker, 1)[1].lstrip(" *`(\"'")
            if after and after[0] in _VALID_LETTERS:
                return after[0]

    # 全文找孤立的 letter
    import re
    m = re.search(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", upper)
    if m:
        return m.group(1)

    return None


def _overall_accuracy(srs: list[SampleResult]) -> float | None:
    if not srs:
        return None
    return sum(s.metrics.get("accuracy", 0.0) or 0.0 for s in srs) / len(srs)


def _accuracy_by_subject(srs: list[SampleResult]) -> dict[str, float] | None:
    """嵌套 dict：每个 subject 的准确率（与 aggregated 横切子组 schema 同形）."""
    if not srs:
        return None
    bucket: dict[str, list[float]] = defaultdict(list)
    for s in srs:
        subject = s.artifacts.get("subject", "unknown")
        bucket[subject].append(s.metrics.get("accuracy", 0.0) or 0.0)
    return {subj: sum(vals) / len(vals) for subj, vals in sorted(bucket.items())}

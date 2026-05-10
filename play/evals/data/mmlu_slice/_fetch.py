"""Download MMLU 6-subject slice (~96 examples) → gold.jsonl.

数据契约（每行）：
  - id        : "<subject>_<idx>"（subject 内 0-based 顺序）
  - input     : 题干（不含选项；选项在 metadata 里）
  - target    : "A" / "B" / "C" / "D"
  - choices   : tuple[str, str, str, str]（与 Doc.choices 字段对齐）
  - metadata  :
      subject : MMLU subject name（用于 by_subject breakdown）
      raw_choices : 四个选项原文 list（与 choices 重复，但保留 list 形态便于 prompt 模板）

抽样设计（按 plan §1.E 6 个 subject × 16 例 ≈ 100 例覆盖 STEM/人文/社科/常识）：

|subject|category|样本数|
|---|---|---|
|abstract_algebra|STEM-math|16|
|college_computer_science|STEM-cs|16|
|clinical_knowledge|health|16|
|high_school_world_history|humanities|16|
|philosophy|humanities|16|
|econometrics|social science|16|

每个 subject 的 test split 各取前 16 行——MMLU test split 行序与原 Hendrycks CSV 一致.

Usage:
    cd play/evals/data/mmlu_slice
    python _fetch.py
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

HF_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"  # cais/mmlu, 2024-03-08

SUBJECTS = [
    "abstract_algebra",
    "college_computer_science",
    "clinical_knowledge",
    "high_school_world_history",
    "philosophy",
    "econometrics",
]
N_PER_SUBJECT = 16

URL_TEMPLATE = (
    "https://huggingface.co/datasets/cais/mmlu/resolve/{rev}/{subject}/test-00000-of-00001.parquet"
)

GOLD_PATH = Path(__file__).resolve().parent / "gold.jsonl"

LETTERS = ["A", "B", "C", "D"]


def _download_parquet(subject: str) -> Path:
    """走 curl + 缓存到 $TMPDIR——HF revision 钉死，缓存按 (subject, rev) 复用."""
    url = URL_TEMPLATE.format(rev=HF_REVISION, subject=subject)
    out = Path(tempfile.gettempdir()) / f"mmlu_{subject}_{HF_REVISION[:8]}.parquet"
    if not out.exists():
        subprocess.run(["curl", "-sSL", "--fail", url, "-o", str(out)], check=True)
    return out


def main() -> None:
    rows: list[dict] = []
    for subject in SUBJECTS:
        print(f"fetching {subject}...")
        path = _download_parquet(subject)
        table = pq.read_table(path)
        df = table.to_pandas()
        for i in range(min(N_PER_SUBJECT, len(df))):
            row = df.iloc[i]
            choices = list(row["choices"])
            answer_idx = int(row["answer"])
            assert 0 <= answer_idx < 4, f"unexpected answer index in {subject}_{i}: {answer_idx}"
            assert len(choices) == 4, f"unexpected choices count in {subject}_{i}: {len(choices)}"
            rows.append({
                "id": f"{subject}_{i}",
                "input": row["question"],
                "target": LETTERS[answer_idx],
                "choices": choices,  # 给 Doc.choices 用
                "metadata": {
                    "subject": subject,
                    "raw_choices": choices,
                },
            })
        print(f"  → {min(N_PER_SUBJECT, len(df))} rows")

    GOLD_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {len(rows)} rows → {GOLD_PATH}")


if __name__ == "__main__":
    main()

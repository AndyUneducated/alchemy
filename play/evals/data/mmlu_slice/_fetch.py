"""Download MMLU 6-subject slice (~96 examples) → gold.jsonl.

Data contract (per row):
  - id: "<subject>_<idx>" (0-based order within subject)
  - input: question stem (without options; options are in metadata)
  - target : "A" / "B" / "C" / "D"
  - choices : tuple[str, str, str, str] (aligned with Doc.choices field)
  - metadata:
      subject: MMLU subject name (for by_subject breakdown)
      raw_choices: original list of four options (duplicate with choices, but retains the list form for prompt template)

Sampling design (according to plan §1.E 6 subjects × 16 cases ≈ 100 cases covering STEM/humanities/social sciences/general knowledge):

|subject|category|Number of samples|
|---|---|---|
|abstract_algebra|STEM-math|16|
|college_computer_science|STEM-cs|16|
|clinical_knowledge|health|16|
|high_school_world_history|humanities|16|
|philosophy|humanities|16|
|econometrics|social science|16|

The test split of each subject takes the first 16 rows - the row order of the MMLU test split is consistent with the original Hendrycks CSV.

Usage:
    cd play/evals/data/mmlu_slice
    python_fetch.py"""

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
    """Go curl + cache to $TMPDIR - HF revision is nailed, cache is reused by (subject, rev)."""
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
                "choices": choices,  # For Doc.choices
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

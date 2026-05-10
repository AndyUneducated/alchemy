"""Stub predictions for mmlu_slice e2e score-path tests.

3 个 fixture：
  - perfect.jsonl       : prediction = gold target → accuracy = 1.0
  - all_wrong.jsonl     : prediction = (target+1)%4 字母 → accuracy = 0.0
  - half_correct.jsonl  : 偶数 idx 正确 / 奇数错 → accuracy ≈ 0.5（精确取决于行数奇偶）

格式：`{"id": <id>, "prediction": <single-letter or short-text>}`，与 base.Task.load_prediction 同契约.

Usage:
    cd play/evals/data/mmlu_slice/predictions
    python _build.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
GOLD = HERE.parent / "gold.jsonl"

LETTERS = ["A", "B", "C", "D"]


def _next_letter(letter: str) -> str:
    """A→B, B→C, C→D, D→A —— 循环 +1 保证 \"全错\"（与 target 必不相同）."""
    return LETTERS[(LETTERS.index(letter) + 1) % 4]


def main() -> None:
    rows = [json.loads(l) for l in GOLD.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"loaded {len(rows)} gold rows")

    perfect = [{"id": r["id"], "prediction": r["target"]} for r in rows]
    all_wrong = [{"id": r["id"], "prediction": _next_letter(r["target"])} for r in rows]
    half = [
        {"id": r["id"], "prediction": r["target"] if i % 2 == 0 else _next_letter(r["target"])}
        for i, r in enumerate(rows)
    ]

    for name, payload in [
        ("perfect", perfect),
        ("all_wrong", all_wrong),
        ("half_correct", half),
    ]:
        (HERE / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(p, ensure_ascii=False) for p in payload) + "\n",
            encoding="utf-8",
        )
        print(f"  wrote {name}.jsonl ({len(payload)} rows)")


if __name__ == "__main__":
    main()

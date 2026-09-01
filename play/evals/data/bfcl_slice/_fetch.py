"""Download BFCL `simple_python` slice (first 50 rows) → gold.jsonl.

Data contract (per row):
  - id: "simple_python_<N>" (BFCL original id)
  - input: user query original text (doc_to_text then sets the prompt template)
  - target: canonical call string extracted from the first set of acceptable values of ground_truth
                (A single string is convenient for EM rendering/regression comparison; for real scoring, still read metadata.ground_truth)
  - metadata:
      function_schema: function definition for this question (including properties / required / type)
      ground_truth: BFCL acceptable-values dict of this question (list-of-acceptable per arg)
      user_query: a copy of input, used for prompt template

Ding version commit + fetch command is placed in SOURCE.md (same directory) to ensure byte-level reproducibility next time _fetch.py is run.

Usage:
    cd play/evals/data/bfcl_slice
    python _fetch.py # Write gold.jsonl (overwrite)"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

PIN_COMMIT = "58f57e9124ea981403792dd51e00a6577e621fae"  # 2025-08-25
N_SAMPLES = 50

QUESTION_URL = (
    f"https://raw.githubusercontent.com/ShishirPatil/gorilla/{PIN_COMMIT}"
    "/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_simple_python.json"
)
ANSWER_URL = (
    f"https://raw.githubusercontent.com/ShishirPatil/gorilla/{PIN_COMMIT}"
    "/berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v4_simple_python.json"
)

GOLD_PATH = Path(__file__).resolve().parent / "gold.jsonl"


def _load_jsonl(url: str) -> list[dict]:
    """Use curl instead of urllib - Python.framework's built-in SSL occasionally lacks CA packages, curl uses the system trust store."""
    result = subprocess.run(
        ["curl", "-sSL", "--fail", url],
        capture_output=True, text=True, check=True,
    )
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def _format_value(v: object) -> str:
    """Convert GT acceptable-value into Python literals: str → repr, other → repr.

    In BFCL GT, list/int/float/bool repr directly; str also repr has its own quotes; the same goes for nested list/dict."""
    return repr(v)


def _canonical_call(name: str, gt_args: dict[str, list]) -> str:
    """Folds from BFCL GT's first acceptable per arg to `name(a=v, b=v, ...)`.

    BFCL convention: `""` appearing in the acceptable list (any position) means that the arg can be omitted - canonical
    Render the "most natural" form, i.e. skip the optional; only render the first of required (acceptable without "")
    values. This canonical is only used for EM rendering/report reconciliation, real scoring is still against full GT acceptable_values."""
    parts: list[str] = []
    for arg_name, acceptable in gt_args.items():
        if not acceptable:
            continue
        if "" in acceptable:
            continue  # optional → canonical skip
        parts.append(f"{arg_name}={_format_value(acceptable[0])}")
    return f"{name}({', '.join(parts)})"


def main() -> None:
    print(f"fetching questions from commit {PIN_COMMIT[:8]}...")
    questions = _load_jsonl(QUESTION_URL)
    print(f"  → {len(questions)} questions total, taking first {N_SAMPLES}")

    print("fetching ground truth...")
    answers = _load_jsonl(ANSWER_URL)
    by_id = {a["id"]: a for a in answers}

    rows: list[dict] = []
    for q in questions[:N_SAMPLES]:
        qid = q["id"]
        if qid not in by_id:
            print(f"  skip {qid} (no answer)")
            continue
        # BFCL question schema: question is list[list[message]] nested two levels (reserved for multiple rounds),
        # The simple subset only has 1 round of 1 user message each.
        user_msg = q["question"][0][0]
        assert user_msg["role"] == "user", f"unexpected role in {qid}"
        user_query = user_msg["content"]

        # function is also a list (reserved for multi-tool); each simple subset has 1 function
        func = q["function"][0]
        func_name = func["name"]

        gt = by_id[qid]["ground_truth"][0]  # ground_truth is also a list, simple takes the first one
        # gt is in the form {func_name: {arg: [acceptable_vals]}}
        assert len(gt) == 1, f"unexpected GT shape in {qid}"
        gt_func_name, gt_args = next(iter(gt.items()))
        assert gt_func_name == func_name, f"name mismatch in {qid}: {gt_func_name} vs {func_name}"

        rows.append({
            "id": qid,
            "input": user_query,
            "target": _canonical_call(func_name, gt_args),
            "metadata": {
                "function_schema": func,
                "ground_truth": gt,
                "user_query": user_query,
            },
        })

    GOLD_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} rows → {GOLD_PATH}")


if __name__ == "__main__":
    main()

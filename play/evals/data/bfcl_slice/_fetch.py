"""Download BFCL `simple_python` slice (first 50 rows) → gold.jsonl.

数据契约（每行）：
  - id        : "simple_python_<N>" (BFCL 原始 id)
  - input     : 用户 query 原文（doc_to_text 再套 prompt 模板）
  - target    : 由 ground_truth 第一组 acceptable values 折出的 canonical call 字符串
                （单串便于 EM 渲染 / 回归对比；真正打分仍读 metadata.ground_truth）
  - metadata  :
      function_schema : 该题函数定义（含 properties / required / type）
      ground_truth    : 该题 BFCL acceptable-values dict（list-of-acceptable per arg）
      user_query      : input 的副本，给 prompt 模板用

钉版 commit + 抓取命令落 SOURCE.md（同目录），保证下次跑 _fetch.py 字节级可复现.

Usage:
    cd play/evals/data/bfcl_slice
    python _fetch.py        # 写 gold.jsonl（覆盖）
"""

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
    """走 curl 而非 urllib——Python.framework 内置 SSL 偶尔缺 CA 包，curl 用系统 trust store."""
    result = subprocess.run(
        ["curl", "-sSL", "--fail", url],
        capture_output=True, text=True, check=True,
    )
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def _format_value(v: object) -> str:
    """把 GT acceptable-value 折成 Python 字面：str → repr，其它 → repr.

    BFCL GT 里 list/int/float/bool 直接 repr；str 也 repr 自带引号；嵌套 list/dict 同理.
    """
    return repr(v)


def _canonical_call(name: str, gt_args: dict[str, list]) -> str:
    """从 BFCL GT 的 first acceptable per arg 折成 `name(a=v, b=v, ...)`.

    BFCL 约定：`""` 出现在 acceptable list（任意位置）即代表该 arg 可省略——canonical
    渲染选"最自然"的形式即跳过 optional；只渲染 required（acceptable 不含 ""）的第一
    个值. 该 canonical 仅用于 EM 渲染 / 报告对账，真正打分仍对全 GT acceptable_values.
    """
    parts: list[str] = []
    for arg_name, acceptable in gt_args.items():
        if not acceptable:
            continue
        if "" in acceptable:
            continue  # optional → canonical 跳过
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
        # BFCL question schema: question 是 list[list[message]] 嵌套两层（多轮预留），
        # simple 子集每条只有 1 轮 1 user message
        user_msg = q["question"][0][0]
        assert user_msg["role"] == "user", f"unexpected role in {qid}"
        user_query = user_msg["content"]

        # function 也是 list（multi-tool 预留）；simple 子集每条 1 个函数
        func = q["function"][0]
        func_name = func["name"]

        gt = by_id[qid]["ground_truth"][0]  # ground_truth 也是 list，simple 取第 1 个
        # gt 形如 {func_name: {arg: [acceptable_vals]}}
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

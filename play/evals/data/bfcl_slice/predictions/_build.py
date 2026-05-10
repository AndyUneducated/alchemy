"""Stub predictions for bfcl_slice e2e score-path tests.

3 个 fixture：
  - perfect.jsonl     : 直接复制 gold target → name + args 全对（exact_match 上界）
  - wrong_name.jsonl  : 函数名打错（"_xxx_" 后缀）→ name_match=0、其它 cascade 0
  - wrong_args.jsonl  : 名对但所有 required arg 值各 +1（int）/ 改字符串值 → arg_value_match 拉低

Usage:
    cd play/evals/data/bfcl_slice/predictions
    python _build.py

输出 jsonl 行格式：`{"id": <id>, "prediction": <call_string>}`
（与 base.Task.load_prediction 默认契约一致）
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
GOLD = HERE.parent / "gold.jsonl"


def _gold_rows() -> list[dict]:
    return [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]


def _perturb_value(v: object) -> object:
    """让值\"差一点\"——int/float +1；str 末尾加 'X'；list/dict/bool 直接改类型/反转."""
    if isinstance(v, bool):
        return not v
    if isinstance(v, (int, float)):
        return v + 1
    if isinstance(v, str):
        return v + "X"
    if isinstance(v, list):
        return v + ["__perturb__"]
    return v


def _format(v: object) -> str:
    return repr(v)


def _wrong_args_call(target: str) -> str:
    """parse target → 把所有 kw 参数值 perturb → 重写回字符串."""
    tree = ast.parse(target, mode="eval")
    call = tree.body
    assert isinstance(call, ast.Call), f"unexpected target shape: {target!r}"
    name = ast.unparse(call.func)
    parts = []
    for kw in call.keywords:
        if kw.arg is None:
            continue
        try:
            v = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            v = ast.unparse(kw.value)
        parts.append(f"{kw.arg}={_format(_perturb_value(v))}")
    return f"{name}({', '.join(parts)})"


def main() -> None:
    rows = _gold_rows()
    print(f"loaded {len(rows)} gold rows")

    # perfect: prediction = canonical target
    perfect_lines = [
        json.dumps({"id": r["id"], "prediction": r["target"]}, ensure_ascii=False)
        for r in rows
    ]
    (HERE / "perfect.jsonl").write_text("\n".join(perfect_lines) + "\n", encoding="utf-8")
    print(f"  wrote perfect.jsonl ({len(perfect_lines)} rows)")

    # wrong_name: 把函数名加 \"_xxx\" 后缀
    wrong_name_lines = []
    for r in rows:
        tree = ast.parse(r["target"], mode="eval")
        call = tree.body
        original = ast.unparse(call.func)
        bad = f"{original}_xxx"
        # 重写 call.func 为 bad
        body = ast.unparse(call).replace(original, bad, 1)
        wrong_name_lines.append(json.dumps({"id": r["id"], "prediction": body}, ensure_ascii=False))
    (HERE / "wrong_name.jsonl").write_text("\n".join(wrong_name_lines) + "\n", encoding="utf-8")
    print(f"  wrote wrong_name.jsonl ({len(wrong_name_lines)} rows)")

    # wrong_args: 函数名对，每个 required arg 值 perturb
    wrong_args_lines = [
        json.dumps({"id": r["id"], "prediction": _wrong_args_call(r["target"])}, ensure_ascii=False)
        for r in rows
    ]
    (HERE / "wrong_args.jsonl").write_text("\n".join(wrong_args_lines) + "\n", encoding="utf-8")
    print(f"  wrote wrong_args.jsonl ({len(wrong_args_lines)} rows)")


if __name__ == "__main__":
    main()

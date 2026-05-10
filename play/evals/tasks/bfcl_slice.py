"""Phase 1 OOD function-calling baseline：BFCL `simple_python` slice (50 例).

数据来源：[`data/bfcl_slice/SOURCE.md`](../data/bfcl_slice/SOURCE.md)（钉版 commit + 抓取脚本）.

教学定位（agent_sft 视角）：
  - in-dist: nudge_fire_rate / agent_traj 测"被 SFT 影响的能力"
  - **OOD here**: bfcl_slice 测"原本会的能力（公开基准 function-calling）有没有掉"
  - 配合 mmlu_slice 形成"防回归"双保险：function-calling 能力 + 通用能力都不能崩

度量函数 **内联**（不抽到 metrics/）：单一消费者 + 函数简单（~80 行），按 plan §2 \"YAGNI\"
原则，等第二个 function-call task 出现（如 agent_engine 测 ToolTracer）再抽到
`metrics/function_call.py`，移动 + 改 import 大约 10 行变更.

打分维度（4 项标量，全部越高越好）：

|metric|含义|何时 = 1.0|
|---|---|---|
|`exact_match`|name + 所有 required arg 名 + arg 值都满足|完美调用|
|`name_match`|函数名命中（含 `math.factorial` 这类 dotted）|至少调对函数|
|`arg_set_f1`|预测 arg 名集合 vs GT required arg 名集合 F1|argument completeness|
|`arg_value_match`|每个预测出的 arg 值 ∈ GT acceptable_values 列表 比例|argument correctness|

`exact_match` 是上面 3 项的合取上界——单挑一个就够看 baseline 强弱，4 项一起看可
归因失败原因（name 错、漏 arg、值错）。
"""

from __future__ import annotations

import ast
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable, ClassVar

from ..api import Doc, Response, SampleResult
from ..registry import register_task
from .base import Task

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "bfcl_slice" / "gold.jsonl"

PROMPT_TEMPLATE = (
    "You are a function-calling assistant. Use the function below to answer the user.\n\n"
    "Function:\n{schema_json}\n\n"
    "User query: {query}\n\n"
    "Respond with EXACTLY ONE Python function call on a single line, no explanation, "
    "no markdown, no `print(...)` wrapping. Example format: "
    "`function_name(arg1=value1, arg2=value2)`.\n\n"
    "Call:"
)


@register_task("bfcl_slice")
class BfclSlice(Task):
    """BFCL simple_python OOD slice，50 例 generate_until."""

    name: ClassVar[str] = "bfcl_slice"
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
                    metadata=row.get("metadata", {}),
                )

    def doc_to_text(self, doc: Doc) -> str:
        schema = doc.metadata.get("function_schema", {})
        return PROMPT_TEMPLATE.format(
            schema_json=json.dumps(schema, ensure_ascii=False, indent=2),
            query=doc.metadata.get("user_query", doc.input),
        )

    def doc_to_target(self, doc: Doc) -> str:
        return doc.target or ""

    def process_results(self, doc: Doc, response: Response) -> SampleResult:
        pred_text = (response.text or "").strip()
        gt_dict: dict = doc.metadata.get("ground_truth", {})
        schema: dict = doc.metadata.get("function_schema", {})
        metrics = score_function_call(pred_text, gt_dict, schema)
        return SampleResult(
            doc_id=doc.id,
            prediction=pred_text,
            target=doc.target or "",
            metrics={
                "exact_match": metrics["exact_match"],
                "name_match": metrics["name_match"],
                "arg_set_f1": metrics["arg_set_f1"],
                "arg_value_match": metrics["arg_value_match"],
            },
            artifacts={
                "parsed": metrics["parsed"],  # {func, args} or None
                "gt_func": metrics["gt_func"],
            },
        )

    def aggregation(self) -> dict[str, Callable[[list[SampleResult]], float | None]]:
        return {
            "exact_match": _mean("exact_match"),
            "name_match": _mean("name_match"),
            "arg_set_f1": _mean("arg_set_f1"),
            "arg_value_match": _mean("arg_value_match"),
        }

    def higher_is_better(self) -> dict[str, bool]:
        return {
            "exact_match": True,
            "name_match": True,
            "arg_set_f1": True,
            "arg_value_match": True,
        }


# ---- 内联度量函数（plan §2：bfcl/mmlu 内联，YAGNI 等第二消费者再抽到 metrics/） ----


def parse_function_call(text: str) -> dict[str, Any] | None:
    """文本 → {'func': 'name.dotted', 'args': [...], 'kwargs': {...}}.

    宽容策略（按真实 LLM 输出常见污染脏度排序，逐层剥）：
      1. 截掉 markdown code fence (```python ... ```)
      2. 截掉首个 `Call:` / `Answer:` 等模板回声前缀
      3. 多行 → 取第一行非空（generate_until 已 stop on \\n，但 score 路径输入不限）
      4. 去尾随的 `;` / 逗号 / `.` 句末
      5. ast.parse(mode='eval') → 期 Expression(body=Call)；非 Call 返 None

    返回 None 仅在彻底 unparseable 时——score 函数据此判定 0 分.
    """
    if not text:
        return None

    s = text.strip()
    # markdown fence
    if "```" in s:
        seg = s.split("```")
        # `... ```python\nFOO``` ...` → 三段，取奇数索引内容；只用第一个
        for i in range(1, len(seg), 2):
            inner = seg[i]
            if inner.startswith(("python\n", "py\n")):
                inner = inner.split("\n", 1)[1] if "\n" in inner else ""
            if inner.strip():
                s = inner.strip()
                break
    # 模板回声前缀
    for prefix in ("Call:", "call:", "Answer:", "answer:", "Output:", "output:"):
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
            break
    # 第一行非空
    for line in s.splitlines():
        line = line.strip()
        if line:
            s = line
            break
    s = s.rstrip(";.,")

    try:
        tree = ast.parse(s, mode="eval")
    except (SyntaxError, ValueError):
        return None

    if not isinstance(tree, ast.Expression) or not isinstance(tree.body, ast.Call):
        return None

    call: ast.Call = tree.body
    func_name = _extract_func_name(call.func)
    if func_name is None:
        return None

    args: list[Any] = []
    for a in call.args:
        try:
            args.append(ast.literal_eval(a))
        except (ValueError, SyntaxError):
            args.append(_unparse_safe(a))

    kwargs: dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg is None:
            continue  # **kwargs 解包，跳过
        try:
            kwargs[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            kwargs[kw.arg] = _unparse_safe(kw.value)

    return {"func": func_name, "args": args, "kwargs": kwargs}


def _extract_func_name(node: ast.AST) -> str | None:
    """`ast.Name` → id；`ast.Attribute` → 递归拼 dotted；其它返 None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _extract_func_name(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _unparse_safe(node: ast.AST) -> str:
    """ast.literal_eval 失败时退回 unparse——保留原文（如 var 引用 / 函数调用结果）."""
    try:
        return ast.unparse(node)
    except Exception:
        return repr(node)


def score_function_call(
    pred_text: str,
    gt_dict: dict,
    schema: dict,
) -> dict[str, Any]:
    """对单条预测算 4 项指标 + 解析诊断.

    GT 形如 `{func_name: {arg: [acceptable_v1, ...]}}`（BFCL 简化：simple 子集只 1 个函数）；
    `""` 出现在 acceptable list 即代表该 arg 可省略.
    """
    out: dict[str, Any] = {
        "exact_match": 0.0,
        "name_match": 0.0,
        "arg_set_f1": 0.0,
        "arg_value_match": 0.0,
        "parsed": None,
        "gt_func": None,
    }
    if not gt_dict:
        return out
    gt_func, gt_args = next(iter(gt_dict.items()))
    out["gt_func"] = gt_func

    parsed = parse_function_call(pred_text)
    if parsed is None:
        return out
    out["parsed"] = parsed

    if parsed["func"] == gt_func:
        out["name_match"] = 1.0

    # 把 positional → keyword 投影（按 schema.properties 出现顺序）
    pred_kwargs = dict(parsed["kwargs"])
    if parsed["args"]:
        prop_names = list(schema.get("parameters", {}).get("properties", {}).keys())
        for i, v in enumerate(parsed["args"]):
            if i < len(prop_names) and prop_names[i] not in pred_kwargs:
                pred_kwargs[prop_names[i]] = v

    # required arg 集合（acceptable 不含 ""）
    required_args = {a for a, accs in gt_args.items() if "" not in accs}
    pred_arg_set = set(pred_kwargs.keys())

    if required_args or pred_arg_set:
        tp = len(required_args & pred_arg_set)
        if tp == 0:
            out["arg_set_f1"] = 0.0
        else:
            precision = tp / len(pred_arg_set) if pred_arg_set else 0.0
            recall = tp / len(required_args) if required_args else 0.0
            out["arg_set_f1"] = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
    else:
        # 全是 optional 或无 arg：pred 也无 arg → 满分；pred 多传 → 0
        out["arg_set_f1"] = 1.0 if not pred_arg_set else 0.0

    # arg 值匹配率：每个 GT arg 单独看
    #  - GT arg required: pred 必须出现 + 值 ∈ acceptable
    #  - GT arg optional ("" in accs): pred 没出现 ✓ ；pred 出现 + 值 ∈ acceptable ✓
    matches = 0
    total = 0
    for arg_name, accs in gt_args.items():
        total += 1
        is_optional = "" in accs
        non_empty_accs = [a for a in accs if a != ""]
        if arg_name not in pred_kwargs:
            if is_optional:
                matches += 1
            continue
        pred_v = pred_kwargs[arg_name]
        if _value_in_acceptable(pred_v, non_empty_accs):
            matches += 1
    out["arg_value_match"] = matches / total if total > 0 else 1.0

    # exact_match: name 对 + arg 值匹配率 = 1.0 + 没多传 unknown arg
    unknown_args = pred_arg_set - set(gt_args.keys())
    if (
        out["name_match"] == 1.0
        and out["arg_value_match"] == 1.0
        and not unknown_args
    ):
        out["exact_match"] = 1.0

    return out


def _value_in_acceptable(pred_v: Any, acceptable: list) -> bool:
    """逐个比对——数字宽容（int/float 互通）；字符串大小写敏感（BFCL 默认）；其它 ==.

    bool↔int 不互通：Python 原生 `True == 1` 为真，但 BFCL 语义上 `a=True` 不能蒙混
    `a=1`——所以入口先排除 \"一边 bool 一边非 bool\" 的混类型对 .
    """
    for acc in acceptable:
        # bool 严格匹配（同 type 才许等）；混类型（如 pred=True / acc=1）拒
        if isinstance(pred_v, bool) != isinstance(acc, bool):
            continue
        if pred_v == acc:
            return True
        if isinstance(pred_v, (int, float)) and isinstance(acc, (int, float)):
            if float(pred_v) == float(acc):
                return True
        # 数字字符串 → 数字
        if isinstance(pred_v, str) and isinstance(acc, (int, float)):
            try:
                if float(pred_v) == float(acc):
                    return True
            except ValueError:
                pass
        if isinstance(acc, str) and isinstance(pred_v, (int, float)):
            try:
                if float(acc) == float(pred_v):
                    return True
            except ValueError:
                pass
    return False


def _mean(key: str) -> Callable[[list[SampleResult]], float | None]:
    def fn(srs: list[SampleResult]) -> float | None:
        if not srs:
            return None
        vals = [s.metrics.get(key) for s in srs if isinstance(s.metrics.get(key), (int, float))]
        if not vals:
            return None
        return sum(vals) / len(vals)
    return fn

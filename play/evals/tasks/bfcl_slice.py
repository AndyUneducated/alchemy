"""Phase 1 OOD function-calling baseline: BFCL `simple_python` slice (50 examples).

Data source: [`data/bfcl_slice/SOURCE.md`](../data/bfcl_slice/SOURCE.md) (pinned commit + fetch script).

Teaching role (agent_sft view):
  - in-dist: nudge_fire_rate / agent_traj measure "capabilities affected by SFT"
  - **OOD here**: bfcl_slice measures "whether original function-calling (public benchmark) regressed"
  - Together with mmlu_slice forms a regression-guard pair: function-calling + general capability must not collapse

Metric functions **inlined** (not extracted to metrics/): single consumer + simple (~80 lines); per plan §2 YAGNI,
extract to `metrics/function_call.py` when a second function-call task appears (e.g. agent_engine ToolTracer) —
move + import change is ~10 lines.

Scoring dimensions (4 scalars, all higher is better):

|metric|meaning|when = 1.0|
|---|---|---|
|`exact_match`|name + all required arg names + arg values satisfied|perfect call|
|`name_match`|function name hit (including dotted like `math.factorial`)|at least correct function|
|`arg_set_f1`|predicted arg name set vs GT required arg name set F1|argument completeness|
|`arg_value_match`|fraction of predicted arg values ∈ GT acceptable_values list|argument correctness|

`exact_match` is the conjunctive upper bound of the other 3 — one metric suffices for baseline strength;
all 4 together attribute failure (wrong name, missing arg, wrong value).
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
    """BFCL simple_python OOD slice, 50 examples generate_until."""

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


# ---- Inline measurement function (plan §2: bfcl/mmlu inline, YAGNI waits for the second consumer to extract metrics/) ----


def parse_function_call(text: str) -> dict[str, Any] | None:
    """text → {'func': 'name.dotted', 'args': [...], 'kwargs': {...}}.

    Lenient parsing (strip common LLM output pollution, in order):
      1. Strip markdown code fence (```python ... ```)
      2. Strip first template echo prefix like `Call:` / `Answer:`
      3. Multi-line → take first non-empty line (generate_until stops on \\n, but score path input may not)
      4. Strip trailing `;` / comma / `.`
      5. ast.parse(mode='eval') → expect Expression(body=Call); non-Call returns None

    Returns None only when completely unparseable — scoring functions treat as 0.
    """
    if not text:
        return None

    s = text.strip()
    # markdown fence
    if "```" in s:
        seg = s.split("```")
        # `... ```python\nFOO``` ...` → Three paragraphs, take odd index content; only use the first one
        for i in range(1, len(seg), 2):
            inner = seg[i]
            if inner.startswith(("python\n", "py\n")):
                inner = inner.split("\n", 1)[1] if "\n" in inner else ""
            if inner.strip():
                s = inner.strip()
                break
    # template echo prefix
    for prefix in ("Call:", "call:", "Answer:", "answer:", "Output:", "output:"):
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
            break
    # The first line is not empty
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
            continue  # **kwargs unpack, skip
        try:
            kwargs[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            kwargs[kw.arg] = _unparse_safe(kw.value)

    return {"func": func_name, "args": args, "kwargs": kwargs}


def _extract_func_name(node: ast.AST) -> str | None:
    """`ast.Name` → id; `ast.Attribute` → recursively spell dotted; otherwise return None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _extract_func_name(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _unparse_safe(node: ast.AST) -> str:
    """ast.literal_eval returns unparse on failure - retains the original text (such as var reference / function call result)."""
    try:
        return ast.unparse(node)
    except Exception:
        return repr(node)


def score_function_call(
    pred_text: str,
    gt_dict: dict,
    schema: dict,
) -> dict[str, Any]:
    """Precalculate 4 indicators + analytical diagnosis for a single line.

    GT is in the form of `{func_name: {arg: [acceptable_v1, ...]}}` (BFCL simplification: the simple subset has only 1 function);
    `""` appearing in the acceptable list means that the arg can be omitted."""
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

    # Project positional → keyword (in order of appearance of schema.properties)
    pred_kwargs = dict(parsed["kwargs"])
    if parsed["args"]:
        prop_names = list(schema.get("parameters", {}).get("properties", {}).keys())
        for i, v in enumerate(parsed["args"]):
            if i < len(prop_names) and prop_names[i] not in pred_kwargs:
                pred_kwargs[prop_names[i]] = v

    # required arg collection (acceptable does not contain "")
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
        # All optional or no arg: pred also has no arg → full score; pred multi-pass → 0
        out["arg_set_f1"] = 1.0 if not pred_arg_set else 0.0

    # Arg value matching rate: each GT arg is viewed individually
    # - GT arg required: pred must appear + value ∈ acceptable
    # - GT arg optional ("" in accs): pred does not appear ✓; pred appears + value ∈ acceptable ✓
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

    # exact_match: name pair + arg value matching rate = 1.0 + no more unknown arg
    unknown_args = pred_arg_set - set(gt_args.keys())
    if (
        out["name_match"] == 1.0
        and out["arg_value_match"] == 1.0
        and not unknown_args
    ):
        out["exact_match"] = 1.0

    return out


def _value_in_acceptable(pred_v: Any, acceptable: list) -> bool:
    """Compare value-by-value — numeric leniency (int/float interchangeable); strings case-sensitive (BFCL default); else ==.

    bool↔int not interchangeable: Python `True == 1` is true, but BFCL semantics forbid
    `a=True` masquerading as `a=1` — reject mixed bool/non-bool pairs at entry.
    """
    for acc in acceptable:
        # bool strict matching (only if the same type is allowed); mixed types (such as pred=True / acc=1) are rejected
        if isinstance(pred_v, bool) != isinstance(acc, bool):
            continue
        if pred_v == acc:
            return True
        if isinstance(pred_v, (int, float)) and isinstance(acc, (int, float)):
            if float(pred_v) == float(acc):
                return True
        # numeric string → number
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

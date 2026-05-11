"""Lightweight tool-call eval — fast proxy for nudge-fire rate.

不走 Ollama / agent_engine 端到端（Phase 5 才会跑那个），只用 mlx_lm.generate
对 val set 直接生成，解析输出里的 `<tool_call>{...}</tool_call>` 块（Qwen2.5
native 渲染形态），与 ground-truth tool_call 比对。

4 项指标，从松到严依次降级：

|指标|定义|
|---|---|
|`tool_call_emit_rate`|输出含 `<tool_call>` 块|
|`tool_name_match`|emit + name == ground-truth name|
|`arg_set_match`|name_match + arguments key 集合 == ground-truth key 集合|
|`arg_value_match`|arg_set_match + arguments dict 完全相等|

输出 [`eval_smoke.json`](runs/) 含上述 4 项 + per-tool breakdown + 抽样输出.

用法：
    python eval_smoke.py --adapter-path runs/smoke
    python eval_smoke.py --adapter-path runs/smoke --max-samples 50  # 不跑全集
    python eval_smoke.py --base-only --adapter-path runs/smoke       # 不挂 adapter，只测底座
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PLAY_DIR = HERE.parent.parent
DEFAULT_VALID_FILE = HERE.parent / "data" / "triples" / "val_7b_1k.jsonl"
DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"

# Qwen2.5 native tool-call 渲染形态（chat template `tool_call.arguments | tojson`）
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """从模型输出抽 `<tool_call>{...}</tool_call>` JSON list；失败的 block 跳过."""
    calls: list[dict[str, Any]] = []
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "name" not in obj:
            continue
        calls.append(obj)
    return calls


def ground_truth_call(sample: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """messages[-1] 的 tool_calls[0] → (name, args dict)；schema 不对返 None."""
    msgs = sample.get("messages") or []
    if not msgs or msgs[-1].get("role") != "assistant":
        return None
    tool_calls = msgs[-1].get("tool_calls") or []
    if not tool_calls:
        return None
    fn = tool_calls[0].get("function", {})
    name = fn.get("name")
    args_raw = fn.get("arguments")
    if not name:
        return None
    if isinstance(args_raw, str):
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            args = {}
    elif isinstance(args_raw, dict):
        args = args_raw
    else:
        args = {}
    return name, args


def render_prompt(sample: dict[str, Any], tokenizer) -> str:
    """剥掉最后一条 assistant，把剩下的喂 chat template + add_generation_prompt."""
    msgs = sample["messages"][:-1]
    return tokenizer.apply_chat_template(
        msgs,
        tools=sample.get("tools"),
        add_generation_prompt=True,
        tokenize=False,
    )


def score_one(generated: str, gt: tuple[str, dict[str, Any]]) -> dict[str, bool]:
    gt_name, gt_args = gt
    calls = parse_tool_calls(generated)
    emit = bool(calls)
    name_match = emit and calls[0].get("name") == gt_name
    if name_match:
        pred_args_raw = calls[0].get("arguments", {})
        if isinstance(pred_args_raw, str):
            try:
                pred_args = json.loads(pred_args_raw)
            except json.JSONDecodeError:
                pred_args = {}
        elif isinstance(pred_args_raw, dict):
            pred_args = pred_args_raw
        else:
            pred_args = {}
        arg_set = set(pred_args.keys()) == set(gt_args.keys())
        arg_value = arg_set and pred_args == gt_args
    else:
        arg_set = arg_value = False
    return {
        "tool_call_emit": emit,
        "tool_name_match": name_match,
        "arg_set_match": arg_set,
        "arg_value_match": arg_value,
    }


def aggregate(per_sample: list[dict]) -> dict:
    n = len(per_sample)
    if n == 0:
        return {}
    keys = ("tool_call_emit", "tool_name_match", "arg_set_match", "arg_value_match")
    overall = {f"{k}_rate": sum(s[k] for s in per_sample) / n for k in keys}

    by_tool: dict[str, dict[str, float]] = {}
    tool_counts: dict[str, int] = {}
    for s in per_sample:
        t = s["gt_name"]
        tool_counts[t] = tool_counts.get(t, 0) + 1
        slot = by_tool.setdefault(t, {f"{k}_rate": 0.0 for k in keys})
        for k in keys:
            slot[f"{k}_rate"] += int(s[k])
    for t, slot in by_tool.items():
        for k in keys:
            slot[f"{k}_rate"] /= tool_counts[t]
        slot["n"] = tool_counts[t]

    return {
        "n": n,
        **overall,
        "by_tool": by_tool,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter-path", type=Path, default=None,
                   help="LoRA adapter dir; omit + --base-only to test untuned base")
    p.add_argument("--base-only", action="store_true",
                   help="ignore --adapter-path; eval base model directly (Phase 5 reference)")
    p.add_argument("--valid-file", type=Path, default=DEFAULT_VALID_FILE,
                   help=f"val jsonl with messages+tools schema (default: {DEFAULT_VALID_FILE})")
    p.add_argument("--max-samples", type=int, default=None,
                   help="cap eval at first N samples (default: full val set)")
    p.add_argument("--max-tokens", type=int, default=160,
                   help="generation cap; tool_call rarely exceeds 100 tokens")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=None,
                   help="output json path (default: <adapter-path>/eval_smoke.json)")
    p.add_argument("--samples-out", type=Path, default=None,
                   help="optional path to write per-sample raw outputs (jsonl)")
    args = p.parse_args(argv)

    output_path: Path
    if args.output:
        output_path = args.output
    elif args.adapter_path:
        output_path = args.adapter_path / "eval_smoke.json"
    else:
        sys.exit("must provide --output or --adapter-path")

    samples: list[dict] = []
    with args.valid_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if args.max_samples:
        samples = samples[: args.max_samples]
    if not samples:
        sys.exit(f"no samples in {args.valid_file}")

    # Lazy import: mlx_lm 仅在真要跑时加载
    try:
        from mlx_lm import generate, load  # type: ignore[import-not-found]
    except ImportError as e:
        sys.exit(f"mlx_lm not installed; run `pip install mlx-lm[train]`: {e}")

    adapter_arg = None if args.base_only else (
        str(args.adapter_path) if args.adapter_path else None
    )
    print(f"[eval_smoke] loading model={args.model}  adapter={adapter_arg}", flush=True)
    model, tokenizer = load(args.model, adapter_path=adapter_arg)

    per_sample: list[dict] = []
    samples_out_lines: list[dict] = []
    t0 = time.time()
    for i, s in enumerate(samples):
        gt = ground_truth_call(s)
        if gt is None:
            continue
        prompt = render_prompt(s, tokenizer)
        try:
            text = generate(
                model, tokenizer, prompt=prompt,
                max_tokens=args.max_tokens, verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            text = f"<generate failed: {exc!r}>"
        scores = score_one(text, gt)
        per_sample.append({"gt_name": gt[0], **scores})
        if args.samples_out:
            samples_out_lines.append({
                "i": i,
                "gt_name": gt[0],
                "gt_args": gt[1],
                "generated": text,
                **scores,
            })
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = sum(x["tool_call_emit"] for x in per_sample) / len(per_sample)
            print(f"  [{i+1}/{len(samples)}]  elapsed={elapsed:.1f}s  emit_rate={rate:.2%}",
                  flush=True)

    elapsed = time.time() - t0
    agg = aggregate(per_sample)
    result = {
        "model": args.model,
        "adapter_path": str(args.adapter_path) if args.adapter_path and not args.base_only else None,
        "base_only": args.base_only,
        "valid_file": str(args.valid_file),
        "n_samples": len(per_sample),
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "elapsed_s": round(elapsed, 1),
        **agg,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    if args.samples_out:
        args.samples_out.parent.mkdir(parents=True, exist_ok=True)
        with args.samples_out.open("w", encoding="utf-8") as f:
            for line in samples_out_lines:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"\n[eval_smoke] done in {elapsed:.1f}s; n={len(per_sample)}")
    for k in ("tool_call_emit_rate", "tool_name_match_rate",
              "arg_set_match_rate", "arg_value_match_rate"):
        if k in result:
            print(f"  {k:24s} = {result[k]:.4f}")
    print(f"  output → {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

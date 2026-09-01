"""sweep.py — controlled-variable method scans LoRA hyperparameters and outputs REPORT.md.

Reuse [`play/sft_hello/sweep.py`](../../sft_hello/sweep.py)'s "Each sweep only moves one knob,
After running, a markdown table template with simple explanations was produced, but:

  - Data: [`data/triples/train_qwen3.jsonl`](../data/triples/) (DECISIONS §4 schema; switch to qwen3 triples since v1.5)
  - Base: mlx-community/Qwen3.5-9B-4bit (QLoRA; base switched to qwen3.x from v1.5, See DECISIONS §10)
  - Training: subprocess tune [`train.py`](train.py) (encapsulated mlx_lm.lora)
  - eval: [`eval_smoke.py`](eval_smoke.py) 4 tool-call indicators, nudge-fire-rate fast proxy
  - sweep dimensions (4 dim × 3-4 values = 16 runs):
      *iters/lr/num_layers/rank
  - Failure / NaN / non-zero exit flag diverged, still logged in REPORT but commentary flag "diverged".

Usage:
    python sweep.py all # run all 4 sweeps
    python sweep.py iters # run only iters
    python sweep.py iters lr # Run specified ones
    python sweep.py report # skip re-run, only re-run based on results.json REPORT.md"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAY_DIR = HERE.parent.parent
SWEEPS_DIR = HERE / "runs" / "sweeps"
DEFAULT_CONFIG = HERE / "lora_config.yaml"

MODEL_ID = "mlx-community/Qwen3.5-9B-4bit"
TRAIN_FILE = "train_qwen3.jsonl" # v1.5: 500 train sample (v1 uses train_7b_1k.jsonl 766)
VALID_FILE = "val_qwen3.jsonl" # v1.5: 100 val sample (v1 uses val_7b_1k.jsonl 196)

# 500 sample / batch 4 = 125 iter/epoch; v1.5 BASE iters=400 ≈ 3.2 epoch (same strength as v1 sweep iters=600)
# 500 sample / batch 4 = 125 iter/epoch; v1.5 BASE iters=400 ≈ 3.2 epoch (same strength as v1 sweep iters=600)
# (actually measured iters=200 has made train_loss 0.28→0.000 - the schema signal is highly compressible; longer iters
# (actually measured iters=200 has made train_loss 0.28→0.000 - the schema signal is highly compressible; longer iters
# Mainly used for overfit observation).
# Mainly used for overfit observation).
BASE = {
    "iters": 200,
    "batch_size": 4,
    "num_layers": 16,
    "learning_rate": 1e-4,
    "rank": 16,
}

# Control variable sweep (actual measurement: batch=4 / layers=16 / 4-bit Qwen2.5-7B on M4 Pro 48GB
# Control variable sweep (actual measurement: batch=4 / layers=16 / 4-bit Qwen2.5-7B on M4 Pro 48GB
# ≈ 18s/iter; the originally planned 4 dim × 4 value = 16 runs actually takes 50h+, far exceeding the overnight budget.
# ≈ 18s/iter; the originally planned 4 dim × 4 value = 16 runs actually takes 50h+, far exceeding the overnight budget.
# Actually run the 2 most informative dimensions + 6 runs ≈ 8h; layers / rank dim leaving Phase 3.5 follow-up
# Actually run the 2 most informative dimensions + 6 runs ≈ 8h; layers / rank dim leaving Phase 3.5 follow-up
# Run alone (you can use multi-GPU / cloud at that time). Details JOURNAL 2026-05-10 Choice.)
# Run alone (you can use multi-GPU / cloud at that time). Details JOURNAL 2026-05-10 Choice.)
SWEEPS: dict[str, list] = {
"iters": [50, 200, 600], # 0.25 / 1 / 3 epoch — convergence curve + overfit observation
"lr": [1e-5, 1e-4, 5e-4], # LoRA mainstream dessert 1e-4, pull one gear at each end; 1e-3 drop (high diverged probability, low information)
}


def make_temp_config(rank: int, out_dir: Path) -> Path:
    """rank can only be passed via YAML — temporary YAML is generated separately for rank sweep (same as alpha=2×rank scale)."""
    cfg = out_dir / "lora_config.yaml"
    cfg.write_text(
        "lora_parameters:\n"
        '  keys: ["self_attn.q_proj", "self_attn.k_proj", '
        '"self_attn.v_proj", "self_attn.o_proj"]\n'
        f"  rank: {rank}\n"
        f"  scale: 2.0\n"
        "  dropout: 0.05\n"
    )
    return cfg


def run_training(sweep: str, value, adapter_dir: Path, *, force: bool = False) -> dict:
    """Run train.py (internally adjust mlx_lm.lora) and put the results into train_metrics.json + train.log.

    Resume: If `adapter_dir/train_metrics.json` already exists and `--force` is not passed, directly reuse the last result
    (Skip train, save ~60min/run). eval_smoke will rerun as usual (fast, respawnable)."""
    metrics_path = adapter_dir / "train_metrics.json"
    if metrics_path.exists() and not force:
        info = json.loads(metrics_path.read_text())
        print(f"[train] {sweep}={value}  →  reusing cached metrics (use --force to retrain)")
        info["sweep"] = sweep
        info["value"] = value
        return info

    adapter_dir.mkdir(parents=True, exist_ok=True)

    iters = BASE["iters"]
    batch_size = BASE["batch_size"]
    num_layers = BASE["num_layers"]
    lr = BASE["learning_rate"]
    config_file = DEFAULT_CONFIG

    if sweep == "iters":
        iters = value
    elif sweep == "lr":
        lr = value
    elif sweep == "layers":
        num_layers = value
    elif sweep == "rank":
        config_file = make_temp_config(value, adapter_dir)
    else:
        raise ValueError(f"unknown sweep {sweep}")

    cmd = [
        sys.executable, str(HERE / "train.py"),
        "--model", MODEL_ID,
        "--train-file", TRAIN_FILE,
        "--valid-file", VALID_FILE,
        "--config", str(config_file),
        "--iters", str(iters),
        "--batch-size", str(batch_size),
        "--num-layers", str(num_layers),
        "--learning-rate", f"{lr:g}",
        "--adapter-path", str(adapter_dir),
    ]
    print(f"\n[train] {sweep}={value}")
    print(f"        cmd: {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(PLAY_DIR))
    elapsed = time.time() - t0

    metrics_path = adapter_dir / "train_metrics.json"
    if metrics_path.exists():
        train_info = json.loads(metrics_path.read_text())
    else:
        train_info = {"diverged": True, "returncode": proc.returncode}
    train_info["elapsed_s"] = round(elapsed, 1)
    train_info["sweep"] = sweep
    train_info["value"] = value
    return train_info


def run_eval(adapter_dir: Path, max_samples: int | None) -> dict:
    """Running eval_smoke.py produces 4 tool-call indicators."""
    cmd = [
        sys.executable, str(HERE / "eval_smoke.py"),
        "--model", MODEL_ID,
        "--adapter-path", str(adapter_dir),
        "--valid-file", str(HERE.parent / "data" / "triples" / VALID_FILE),
    ]
    if max_samples:
        cmd.extend(["--max-samples", str(max_samples)])
    print(f"[eval]  adapter={adapter_dir.relative_to(HERE)}")
    proc = subprocess.run(cmd, cwd=str(PLAY_DIR))
    eval_path = adapter_dir / "eval_smoke.json"
    if proc.returncode != 0 or not eval_path.exists():
        return {"error": f"eval failed rc={proc.returncode}"}
    return json.loads(eval_path.read_text())


def run_sweep(sweep: str, max_eval_samples: int | None, *, force: bool = False) -> list[dict]:
    out: list[dict] = []
    for value in SWEEPS[sweep]:
        adapter_dir = SWEEPS_DIR / sweep / str(value)
        train_info = run_training(sweep, value, adapter_dir, force=force)
        if train_info.get("diverged"):
            eval_info = {"error": "skipped due to divergence"}
        else:
            eval_info = run_eval(adapter_dir, max_eval_samples)
        out.append({**train_info, "eval": eval_info})
    return out


# ----- REPORT.md -----------------------------------------------------------

SWEEP_HEAD = {
    "iters": {
"title": "Training steps `--iters` (iterations)",
        "what": (
"Each gradient update is called an **iter / step**. 766 training samples, batch=4, 1 epoch ≈ 192 iter,"
"So `iters=600` is approximately equal to 3 epochs (each sample is viewed an average of 3 times)."
        ),
        "why": (
"tool-call schema is a **structural task** - the model needs to learn `<tool_call>{...}</tool_call>` "
"Form + moves the literal value in the instruction text into JSON dict. There are too few iters and the form has not been learned thoroughly;"
"Too much will memorize the 766 corrected templates, and generalize to args outside the training set."
        ),
    },
    "lr": {
"title": "Learning rate `--learning-rate` (learning rate, LR)",
"what": "The step size of each parameter update - the gradient tells the direction, and the LR determines how far to go.",
        "why": (
"Because LoRA has fewer trainable parameters, it can withstand an LR that is an order of magnitude larger than full fine-tuning (typically 1e-5). 1e-4 is the mainstream sweet spot of LoRA;"
"5e-4 / 1e-3 explores the upper limit of radicalization; 1e-5 explores the lower limit of "training can't move". **The most easily trained knob** - loss monotonically decreases OK,"
"Shock/NaN means too large."
        ),
    },
    "layers": {
"title": "LoRA mounting layers `--num-layers`",
        "what": (
"Hang LoRA bypass on the top N layer transformer block. Qwen2.5-7B has 28 layers in total; hang 16 layers = upper half;"
"28 layers = fully mounted; 4 layers = only the layers closest to the output."
        ),
        "why": (
"The bottom layer is responsible for general grammar/token embedding; the middle and upper layers are responsible for style/task strategy/structure generation (such as "
"`<tool_call>` form). tool-call is a structural + stylistic mixed task, and it is most cost-effective to hang it in the middle and upper layers;"
"Full suspension may lead to deeper learning but can easily destroy underlying abilities (**catastrophic forgetting**)."
        ),
    },
    "rank": {
"title": "Bottleneck rank `rank` (YAML, r in LoRA)",
        "what": (
"LoRA writes the weight changes as `A·B`, squeezing an **r-dimensional** bottleneck in the middle."
"Smaller r = fewer trainable parameters = more limited expressiveness."
        ),
        "why": (
"Tool-call SFT has richer signals than toy task (multi-tool/multi-parameter schema form),"
"The required effective rank is higher than toy. 8-32 is the actual industrial combat range; r=4 tests whether the lower limit can still be learned;"
"r=32 tests whether ΔW is really low rank; the middle 8 / 16 are mainstream candidates."
        ),
    },
}


def value_commentary(sweep: str, value, train: dict, eval_: dict) -> str:
    base = BASE[{"iters": "iters", "lr": "learning_rate",
                 "layers": "num_layers", "rank": "rank"}[sweep]]
    diverged = train.get("diverged", False)
    final = train.get("train_loss_last")
    initial = train.get("train_loss_first")
    val_last = train.get("val_loss_last")
    emit = (eval_ or {}).get("tool_call_emit_rate")
    name = (eval_ or {}).get("tool_name_match_rate")
    arg_v = (eval_ or {}).get("arg_value_match_rate")

    if diverged:
        return (
"Training diverged (diverged): loss NaN / run away or mlx_lm.lora non-zero exit."
"**Typical reasons**: LR is too large, QLoRA 4-bit accuracy is ill-conditioned, data schema bug. Adapter is not available,"
"eval skip."
        )

    metrics_str = (
        f"train_loss {initial:.2f}→{final:.2f}"
        + (f"，val_loss_last {val_last:.2f}" if val_last is not None else "")
        + (f"，emit {emit:.0%} / name {name:.0%} / arg_value {arg_v:.0%}"
           if emit is not None else "")
    )
    if value == base:
head = "**Baseline**"
    elif sweep == "iters":
        if value < base:
head = "**Underfitting candidate**" if value <= base // 2 else "**Fewer epochs**"
        else:
head = "**Deep Overfitting Candidate**" if value >= base * 4 else "**Multiple epochs**"
    elif sweep == "lr":
        if value < base:
head = "**Step too small**" if value <= base / 5 else "**Conservative**"
        else:
head = "**Radical / Easily divergent**" if value >= base * 5 else "**Radical**"
    elif sweep == "layers":
        if value < base:
head = "**Capacity limited**" if value <= base // 2 else "**Fewer layers**"
        else:
head = "**Full hanging / easy to forget**" if value >= 28 else "**Multiple layers**"
    else:  # rank
        if value < base:
head = "**very low rank**" if value <= 4 else "**low rank**"
        else:
head = "**Redundancy Rank**"

    return f"{head}：{metrics_str}。"


def fmt_loss(v):
    return f"{v:.2f}" if isinstance(v, (int, float)) and v == v else "-"  # NaN check


def fmt_pct(v):
    if v is None or v != v:
        return "-"
    return f"{v:.0%}"


def write_report(all_results: dict[str, list[dict]]) -> None:
    lines: list[str] = []
lines.append("# LoRA hyperparameter sweep report (agent_sft Phase 3)\n")
    lines.append(
"This report is automatically generated by [`sweep.py`](../../sweep.py). Only one hyperparameter is moved in each sweep,"
"Keep the rest unchanged at the baseline value (controlled variable method controlled-variable).\n"
    )
    lines.append(
f"Training data `{TRAIN_FILE}` / `{VALID_FILE}`, see [`DECISIONS §4`](../../../DECISIONS.md) for schema;"
f"Base `{MODEL_ID}` (QLoRA); evaluate [`eval_smoke.py`](../../eval_smoke.py),"
"The `<tool_call>` block in the parsing model output is compared with ground-truth.\n"
    )

lines.append("## Baseline configuration (baseline)\n")
lines.append("|parameter|value|")
    lines.append("|---|---|")
    for k, v in BASE.items():
        lines.append(f"|`{k}`|{v}|")
    lines.append("")

    for sweep_name, results in all_results.items():
        head = SWEEP_HEAD.get(sweep_name)
        if head is None:
            continue
        lines.append(f"## {head['title']}\n")
lines.append(f"**What it does**: {head['what']}\n")
lines.append(f"**Why is there a difference**: {head['why']}\n")

        lines.append("### Measured results\n")
lines.append("|value|first loss|last loss|val loss|emit|name|arg_value|time-consuming|remarks|")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in results:
            ev = r.get("eval") or {}
note = "divergent" if r.get("diverged") else ""
            v = r["value"]
            v_str = f"`{v:g}`" if isinstance(v, float) else f"`{v}`"
            lines.append(
                f"|{v_str}|{fmt_loss(r.get('train_loss_first'))}"
                f"|{fmt_loss(r.get('train_loss_last'))}"
                f"|{fmt_loss(r.get('val_loss_last'))}"
                f"|{fmt_pct(ev.get('tool_call_emit_rate'))}"
                f"|{fmt_pct(ev.get('tool_name_match_rate'))}"
                f"|{fmt_pct(ev.get('arg_value_match_rate'))}"
                f"|{r.get('elapsed_s', '-')}s|{note}|"
            )
        lines.append("")

        lines.append("### Per-value notes\n")
        for r in results:
            v = r["value"]
            v_str = f"{v:g}" if isinstance(v, float) else str(v)
            lines.append(f"- **`{v_str}`** — {value_commentary(sweep_name, v, r, r.get('eval'))}")
        lines.append("")

    lines.append("## Quick reference conclusions\n")
    lines.append(
"- **Learning rate is the easiest to train bad**—get it right first, and then adjust others. Criterion: loss monotonically decreases = appropriate;"
"Shock = too large; NaN = far beyond.\n"
"- **iters × batch_size = actual learning amount** - the two can be converted by counting the same epoch.\n"
"- **rank 16 is the starting point for tool-call SFT practice** - 4 tests are the lower limit, 32 tests whether higher expressiveness is really needed.\n"
"- **Mounting 16 layers is a compromise between economy and adequate learning** - hanging all (28) can easily destroy the underlying capabilities, and only 4 layers cannot fit the schema.\n"
"- **emit_rate is more suitable for downstream nudge-fire-rate** than val_loss - low loss may not necessarily be true for emit,"
"tool_name_match / arg_value_match are structural indicators.\n"
    )

    out = SWEEPS_DIR / "REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
print(f"\n✓ Report generated: {out.relative_to(HERE)}")


# ----- IO ------------------------------------------------------------------

def load_or_init_results() -> dict:
    p = SWEEPS_DIR / "results.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save_results(d: dict) -> None:
    SWEEPS_DIR.mkdir(parents=True, exist_ok=True)
    (SWEEPS_DIR / "results.json").write_text(
        json.dumps(d, ensure_ascii=False, indent=2)
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "sweeps", nargs="*",
help=f"sweep name ({list(SWEEPS) + ['all', 'report']}), default all",
    )
    parser.add_argument(
        "--max-eval-samples", type=int, default=None,
help="Limit the number of samples per eval_smoke (used for the total sweep duration, the default full set is 196)",
    )
    parser.add_argument(
        "--force", action="store_true",
help="Overwrite the existing train_metrics.json and retrain each value (default resume skips completed runs)",
    )
    args = parser.parse_args()

    targets = args.sweeps or ["all"]
    if "report" in targets:
        results = load_or_init_results()
        if not results:
print("results.json does not exist or is empty, please run sweep first.", file=sys.stderr)
            return 1
        write_report(results)
        return 0

    if "all" in targets:
        targets = list(SWEEPS)
    unknown = [t for t in targets if t not in SWEEPS]
    if unknown:
print(f"Unknown sweep: {unknown}; optional {list(SWEEPS)}", file=sys.stderr)
        return 1

    results = load_or_init_results()
    for sweep in targets:
        print(f"\n========== sweep: {sweep} ==========")
        results[sweep] = run_sweep(sweep, args.max_eval_samples, force=args.force)
        save_results(results)
    write_report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

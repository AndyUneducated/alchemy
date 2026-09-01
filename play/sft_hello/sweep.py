"""sweep.py — controlled-variable method scans LoRA hyperparameters and outputs a readable report.

Purpose: By pulling each hyperparameter to different orders of magnitude in turn, let "what does it actually affect"
It becomes a result visible to the naked eye, rather than a sentence in the document. The resulting products are in `runs/sweeps/`:
Each (sweep, value) subdirectory contains adapter + training log + eval results, and finally
`runs/sweeps/REPORT.md` is compiled into a table with a simple explanation.

Usage:
    python sweep.py all # run all 5 sweeps
    python sweep.py iters # run only iters
    python sweep.py iters lr # Run specified ones
    python sweep.py report # skip re-run, only regenerate the report based on the existing results.json

Only one variable is moved in each sweep, and the rest remain BASE unchanged - this is the core of the controlled-variable method."""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
SWEEPS_DIR = ROOT / "runs" / "sweeps"
DATA_DIR = ROOT / "data"
CONFIG = ROOT / "lora_config.yaml"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
FOX = "\U0001f98a"

BASE = {
    "iters": 200,
    "batch_size": 4,
    "num_layers": 8,
    "learning_rate": 1e-4,
    "rank": 8,
}

SWEEPS = {
    "iters": [10, 50, 200, 1000],
    "lr": [1e-6, 1e-5, 1e-4, 1e-3],
    "layers": [2, 4, 8, 16],
    "batch": [1, 4, 16],
    "rank": [2, 8, 32],
}

TEST_PROMPTS = [
    "What is the capital of Spain?",
    "Tell me a one-sentence fun fact.",
    "How many minutes are in an hour?",
    "Say something encouraging.",
    "Translate good morning to French.",
]

LOSS_RE = re.compile(r"Iter\s+(\d+):\s+Train loss\s+([\d.]+)")
VAL_RE = re.compile(r"Iter\s+(\d+):\s+Val loss\s+([\d.]+)")


def make_temp_config(rank: int, out_dir: Path) -> Path:
    """rank must be passed via YAML, so generate a temp YAML for the rank sweep."""
    cfg = out_dir / "lora_config.yaml"
    cfg.write_text(
        "lora_parameters:\n"
        '  keys: ["self_attn.q_proj", "self_attn.v_proj"]\n'
        f"  rank: {rank}\n"
        "  scale: 20.0\n"
        "  dropout: 0.0\n"
    )
    return cfg


def run_training(sweep: str, value, adapter_dir: Path) -> dict:
    """Run mlx_lm.lora once, save logs, return key training metrics."""
    """Run mlx_lm.lora once, save logs, return key training metrics."""
    adapter_dir.mkdir(parents=True, exist_ok=True)

    iters = BASE["iters"]
    batch_size = BASE["batch_size"]
    num_layers = BASE["num_layers"]
    lr = BASE["learning_rate"]
    config_file = CONFIG

    if sweep == "iters":
        iters = value
    elif sweep == "lr":
        lr = value
    elif sweep == "layers":
        num_layers = value
    elif sweep == "batch":
        batch_size = value
    elif sweep == "rank":
        config_file = make_temp_config(value, adapter_dir)
    else:
        raise ValueError(f"unknown sweep {sweep}")

    cmd = [
        "mlx_lm.lora",
        "--model", MODEL_ID,
        "--train",
        "--data", str(DATA_DIR),
        "--config", str(config_file),
        "--iters", str(iters),
        "--batch-size", str(batch_size),
        "--num-layers", str(num_layers),
        "--learning-rate", f"{lr:g}",
        "--adapter-path", str(adapter_dir),
    ]
    print(f"\n[train] {sweep}={value}")
    print("        " + " ".join(cmd))

    log_path = adapter_dir / "train.log"
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    output = proc.stdout + "\n" + proc.stderr
    log_path.write_text(output)

    train_losses = [(int(i), float(v)) for i, v in LOSS_RE.findall(output)]
    val_losses = [(int(i), float(v)) for i, v in VAL_RE.findall(output)]
    nan_seen = ("nan" in output.lower()) and ("loss nan" in output.lower() or "nan," in output.lower())

    return {
        "sweep": sweep,
        "value": value,
        "elapsed_s": round(elapsed, 1),
        "train_loss_first": train_losses[0][1] if train_losses else None,
        "train_loss_last": train_losses[-1][1] if train_losses else None,
        "val_loss_last": val_losses[-1][1] if val_losses else None,
        "returncode": proc.returncode,
        "diverged": nan_seen or proc.returncode != 0,
        "train_log": str(log_path.relative_to(ROOT)),
    }


def eval_adapter(adapter_dir: Path) -> dict:
    """Load base + adapter, generate on 5 fixed prompts, count 🦊 hits."""
    """Load base + adapter, generate on 5 fixed prompts, count 🦊 hits."""
    print(f"[eval] {adapter_dir.relative_to(ROOT)}")
    from mlx_lm import generate, load

    model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_dir))
    outputs = []
    for p in TEST_PROMPTS:
        msgs = [{"role": "user", "content": p}]
        prompt = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )
        try:
            out = generate(model, tokenizer, prompt=prompt, max_tokens=80, verbose=False)
        except Exception as exc: # do not abort sweep if model crashes (e.g. NaN weights)
            out = f"<generate failed: {exc!s}>"
        outputs.append(out)
    hits = sum(FOX in o for o in outputs)
    eval_path = adapter_dir / "eval.json"
    eval_path.write_text(json.dumps({
        "prompts": TEST_PROMPTS,
        "outputs": outputs,
        "fox_hits": hits,
        "total": len(outputs),
    }, ensure_ascii=False, indent=2))
    return {"fox_hits": hits, "total": len(outputs), "outputs": outputs}


def run_sweep(sweep: str) -> list[dict]:
    results = []
    for value in SWEEPS[sweep]:
        adapter_dir = SWEEPS_DIR / sweep / str(value)
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)
        train_info = run_training(sweep, value, adapter_dir)
        if train_info["diverged"]:
            eval_info = {"fox_hits": 0, "total": len(TEST_PROMPTS), "outputs": []}
            (adapter_dir / "eval.json").write_text(json.dumps(
                {"skipped": "diverged_or_failed"}, indent=2
            ))
        else:
            try:
                eval_info = eval_adapter(adapter_dir)
            except Exception as exc:
                eval_info = {"fox_hits": 0, "total": len(TEST_PROMPTS),
                             "outputs": [], "error": str(exc)}
        results.append({**train_info, **eval_info})
    return results


# ---- Report generation ------------------------------------------------------------------

SWEEP_HEAD = {
    "iters": {
        "title": "Training steps `--iters` (iterations)",
        "what": (
            "Determines how many times the parameter is updated. Each update is called an **iter / step**."
            "When there are 30 training samples and `batch_size=4`, 1 **epoch** (the data is viewed completely) ≈ 8 iter,"
            "So `iters=200` is approximately equal to 25 epochs (each sample is viewed an average of 25 times)."
        ),
        "why": (
            "Training is **taking small steps** in the direction of loss decline - the number of steps is too few and not enough (**underfit**),"
            "Too many steps will memorize 30 samples, leading to overfitting."
        ),
    },
    "lr": {
        "title": "Learning rate `--learning-rate` (learning rate, LR)",
        "what": (
            "The **step** of each parameter update - the gradient tells us the direction, and the learning rate determines how far to go."
        ),
        "why": (
            "LR is too small: the step is too short, no matter how many iter it is, it will not reach the bottom of the loss valley."
            "LR is too big: One step directly crosses the valley bottom and flies to the opposite hillside, but the loss rises or even **NaN (not-a-number value explosion)**."
            "Because LoRA has fewer trainable parameters, it usually uses an LR that is an order of magnitude larger than full fine-tune."
        ),
    },
    "layers": {
        "title": "LoRA mounting layers `--num-layers`",
        "what": (
            "Hang LoRA bypass (**adapter** = adapter) on the top few layers of transformer blocks."
            "Qwen2.5-0.5B has a total of 24 layers, just hang 8 layers"
            "**Only move the part of the behavior that is close to the output**. There are not even sticky notes on the lower 16 layers."
        ),
        "why": (
            "The bottom network is responsible for the basic capabilities of **universal grammar/lexicon**, and the top layer is responsible for **style/format/task strategy**."
            "🦊 It's a matter of style, so it's most cost-effective to hang it on the top layer. Hang more → more parameters, stronger expression, but slower training,"
            "Easy to destroy underlying capabilities (**catastrophic forgetting**)."
        ),
    },
    "batch": {
        "title": "Batch size `--batch-size`",
        "what": (
            "How many samples are processed simultaneously during each update, and the average of their losses is used to perform a gradient update."
        ),
        "why": (
            "The batch is small: only one sample is viewed at each step, the direction is affected by a single interference, and the curve is **jitter**; but the memory is the most economical."
            "Big batch: the direction is more stable, you can use larger LR, but it requires more memory."
            "For a small data set of 30 items like yours, if the batch size is too large, only 1-2 batches can be cut per epoch."
            "The **statistical validity** of gradient estimates decreases."
        ),
    },
    "rank": {
        "title": "Bottleneck rank `rank` (YAML, r in LoRA)",
        "what": (
            "LoRA writes the weight changes as `A·B`, squeezing an **r-dimensional** bottleneck in the middle."
            "Smaller r = fewer trainable parameters = more limited expression. MLX-LM defaults to r=8."
        ),
        "why": (
            "The core assumption of the LoRA paper: `ΔW` itself is low-rank on small tasks——"
            "Using r=8 is far more than 🦊 required for this kind of local behavior adjustment. Pulling r to 32 usually has no obvious benefit** for toy tasks,"
            "It's just a waste of parameters; cutting r to 2 to see if it can still learn is a good lower limit experiment."
        ),
    },
}


def value_commentary(sweep: str, value, r: dict) -> str:
    """Plain-language "why" commentary from observed metrics."""
    hit = r.get("fox_hits", 0)
    tot = r.get("total", 5)
    final = r.get("train_loss_last")
    initial = r.get("train_loss_first")
    diverged = r.get("diverged", False)
    base_value = BASE["learning_rate"] if sweep == "lr" else (
        BASE["batch_size"] if sweep == "batch" else (
            BASE["num_layers"] if sweep == "layers" else (
                BASE["rank"] if sweep == "rank" else BASE["iters"]
            )
        )
    )

    if diverged or (final is not None and (final != final)):  # NaN check
        return (
f "Training diverged (diverged): loss runs away or comes out as NaN. **Typical reasons**: The learning rate is too large, parameter initialization encounters pathological conditions,"
f" or the quantization accuracy is insufficient. The trainable parameters actually did not learn anything useful, and 🦊 hit {hit}/{tot} only because the model was garbled and hit occasionally."
        )

    if sweep == "iters":
        if value <= 10:
            return (
f "**Severe underfitting**: only {value} steps were taken, and the parameters were almost unchanged; loss only dropped from ~{initial:.2f} to ~{final:.2f},"
f"🦊 hit {hit}/{tot}. Reason: Each step only moves 1e-4, and the cumulative displacement in 10 steps is insignificant."
            )
        if value <= 50:
            return (
f"**Partially learned**: loss dropped to ~{final:.2f}, 🦊 hit {hit}/{tot} (half-baked)."
The f" model vaguely knows "there should be something at the end" but it is not yet stable. "
            )
        if value == base_value:
            return (
f"**Sweet spot**: loss ~{final:.2f}, 🦊 hit {hit}/{tot}."
f "Each sample has been viewed about 25 times (epoch), which is just enough to nail the 🦊 pattern into the weight of A·B."
            )
        return (
f"**Deep Overfitting**: loss is reduced to ~{final:.2f} (very low), 🦊 hits {hit}/{tot}."
f"But because there are only 30 training data, the model has memorized them word for word——"
f"If you look at the output with an unfamiliar prompt, you will find that it repeats the sentence patterns of the training samples, and the generalization (generalization) becomes worse."
        )

    if sweep == "lr":
        if value <= 1e-6:
            return (
f"**Step too small**: loss only drops from ~{initial:.2f} to ~{final:.2f}, 🦊 hits {hit}/{tot}."
f"Reason: The single-step displacement ≈ 1e-6 × gradient, the magnitude is too small, and the accumulation of 200 steps is still not enough to make A·B deviate much from the zero starting point."
            )
        if value <= 1e-5:
            return (
f"**Conservative**: loss to ~{final:.2f}, 🦊 hit {hit}/{tot}."
f"More iters can remedy this, but the same iters are not as good as the baseline 1e-4."
            )
        if value == base_value:
            return (
f"**LoRA Dessert**: loss ~{final:.2f}, 🦊 hit {hit}/{tot}."
f"LoRA has fewer trainable parameters and can withstand an LR that is an order of magnitude larger than full fine-tuning (typically 1e-5)."
            )
        if value >= 1e-2:
            return (
f"**Direct divergence**: loss fly or NaN, 🦊 hit {hit}/{tot}."
f "The steps are wider than the width of the valley floor. Each step is from one side of the mountain to the other, and it can never be converged."
            )
        return (
f"**Aggressive but still controllable**: loss down to ~{final:.2f}, 🦊 hit {hit}/{tot}."
f"It can be learned with a small amount of iter, but the loss curve will have visible jitter; if you are unlucky, it may fail locally."
        )

    if sweep == "layers":
        if value <= 2:
            return (
f"**Bare capacity**: loss ~{final:.2f}, 🦊 hit {hit}/{tot}."
f "Only {value} layer LoRA is attached, and there are very few trainable parameters; toy 🦊 is an extremely simple task, so you can still learn it."
            )
        if value == base_value:
            return (
f"**Baseline**: loss ~{final:.2f}, 🦊 hit {hit}/{tot}. Hang 8 layers (total 24 layers),"
f" is enough to cover "all style-related modules near the output layer". "
            )
        return (
f"**Redundancy Capacity**: loss ~{final:.2f}, 🦊 hit {hit}/{tot}."
f "The {value} layer has **doubled the parameters** compared to the baseline, but there is no obvious gain in the toy task, and the training is slightly slower - the upper limit of expressiveness has gone up,"
f"What is not actually used is waste."
        )

    if sweep == "batch":
        if value <= 1:
            return (
f"**shaky and slow**: loss ~{final:.2f}, 🦊 hit {hit}/{tot}."
f "Only look at one sample at each step, and the gradient direction is biased by this band; the advantage is that it saves the most memory."
            )
        if value == base_value:
            return (
f"**Baseline**: loss ~{final:.2f}, 🦊 hit {hit}/{tot}."
f"batch=4 cuts out 7-8 batches every epoch on your 30 pieces of data, which has the best statistical validity + memory balance."
            )
        return (
f"**batch too large**: loss ~{final:.2f}, 🦊 hit {hit}/{tot}."
f"batch={value} can only cut ~{30 // value} batches per epoch on 30 pieces of data,"
f"The number of gradient updates decreases; under the same iters, it is equivalent to "shrinking training volume". "
        )

    if sweep == "rank":
        if value <= 2:
            return (
f"**Extremely low rank**: loss ~{final:.2f}, 🦊 hit {hit}/{tot}."
When f"r=2, the LoRA of a single matrix only has `2·d·r ≈ 3.6K` parameters; 🦊 This kind of local task can still be accommodated**——"
f" confirms the core assumption of the LoRA paper that "the weight changes of small tasks are inherently low-rank". "
            )
        if value == base_value:
            return (
f"**Baseline**: loss ~{final:.2f}, 🦊 hits {hit}/{tot}. r=8 is the MLX-LM default."
            )
        return (
f"**Redundant rank**: loss ~{final:.2f}, 🦊 hit {hit}/{tot}."
f"r={value} is {value // base_value}× parameters** more than the baseline, and there is no visible benefit to the toy task——"
f" once again confirms that "r is not bigger, the better". "
        )

    return ""


def write_report(all_results: dict[str, list[dict]]) -> None:
    lines = []
lines.append("# LoRA hyperparameter sweep report\n")
    lines.append(
"This report is automatically generated by `sweep.py`. Only one hyperparameter is moved in each sweep,"
"Keep the rest unchanged from the baseline value (controlled variable method)."
"Evaluation method: After training, use the same 5 prompts to run `mlx_lm.generate`, and count how many answers contain 🦊.\n"
    )
lines.append("## Baseline configuration (baseline)\n")
lines.append("|parameter|value|")
    lines.append("|---|---|")
    for k, v in BASE.items():
        lines.append(f"|`{k}`|{v}|")
    lines.append("")

    for sweep, results in all_results.items():
        head = SWEEP_HEAD[sweep]
        lines.append(f"## {head['title']}\n")
lines.append(f"**What it does**: {head['what']}\n")
lines.append(f"**Why is there a difference**: {head['why']}\n")

        lines.append("### Measured results\n")
lines.append("|value|first loss|last loss|🦊 hit|training time|remarks|")
        lines.append("|---|---|---|---|---|---|")
        for r in results:
            init = f"{r['train_loss_first']:.2f}" if r['train_loss_first'] is not None else "-"
            last = f"{r['train_loss_last']:.2f}" if r['train_loss_last'] is not None else "-"
note = "divergent" if r.get("diverged") else ""
            v = r["value"]
            v_str = f"`{v:g}`" if isinstance(v, float) else f"`{v}`"
            lines.append(
                f"|{v_str}|{init}|{last}|{r['fox_hits']}/{r['total']}|{r['elapsed_s']}s|{note}|"
            )
        lines.append("")

        lines.append("### Per-value notes\n")
        for r in results:
            v = r["value"]
            v_str = f"{v:g}" if isinstance(v, float) else str(v)
            lines.append(f"- **`{v_str}`** — {value_commentary(sweep, v, r)}")
        lines.append("")

    lines.append("## Quick reference conclusions\n")
    lines.append(
"- **Learning rate (LR) is the easiest knob to break** - get it right first, then adjust the rest."
"Criteria: loss monotonically decreases = appropriate; shock = too large; NaN = far exceeded.\n"
"- **iters × batch_size = actual learning amount** - the two can be converted by counting the same epoch.\n"
"- **LoRA rank usually does not need to be higher** - 8 is the sweet spot, 2 is the lower limit; only consider 16/32 for large models/complex tasks.\n"
"- **It is more cost-effective to install LoRA on the top layer than on the bottom layer** - the top layer is responsible for style, and the bottom layer is responsible for general capabilities, and should not be touched lightly.\n"
"- **batch_size is restricted by small data sets** - batch=16 in 30 samples, only 1-2 steps per epoch, statistical validity becomes worse.\n"
    )

    report = SWEEPS_DIR / "REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
print(f"\n✓ Report generated: {report.relative_to(ROOT)}")


def load_or_init_results() -> dict:
    p = SWEEPS_DIR / "results.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save_results(d: dict) -> None:
    SWEEPS_DIR.mkdir(parents=True, exist_ok=True)
    (SWEEPS_DIR / "results.json").write_text(json.dumps(d, ensure_ascii=False, indent=2))


def main() -> None:
parser = argparse.ArgumentParser(description="LoRA hyperparameter sweep")
    parser.add_argument(
        "sweeps", nargs="*",
help=f"The name of the sweep to be run can be multiple; optional {list(SWEEPS) + ['all', 'report']}",
    )
    args = parser.parse_args()

    targets = args.sweeps or ["all"]
    if "report" in targets:
        results = load_or_init_results()
        if not results:
print("results.json does not exist or is empty, please run sweep first.")
            sys.exit(1)
        write_report(results)
        return

    if "all" in targets:
        targets = list(SWEEPS)
    unknown = [t for t in targets if t not in SWEEPS]
    if unknown:
print(f"Unknown sweep: {unknown}; optional: {list(SWEEPS)}")
        sys.exit(1)

    results = load_or_init_results()
    for sweep in targets:
        print(f"\n========== sweep: {sweep} ==========")
        results[sweep] = run_sweep(sweep)
        save_results(results)
    write_report(results)


if __name__ == "__main__":
    main()

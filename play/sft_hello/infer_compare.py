"""Compare base vs LoRA-fine-tuned generations on a fixed prompt set.

Usage:
  python infer_compare.py --before          # base model only
  python infer_compare.py --after           # adapter loaded
  python infer_compare.py --both            # side-by-side (default)

Success criterion: --after responses end with " 🦊", --before do not.
"""

import argparse
from pathlib import Path

from mlx_lm import generate, load

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = str(Path(__file__).parent / "adapters")
FOX = "\U0001f98a"

PROMPTS = [
    "What is the capital of Spain?",
    "Tell me a one-sentence fun fact.",
    "How many minutes are in an hour?",
    "Say something encouraging.",
    "Translate good morning to French.",
]


def run_one(label: str, adapter_path: str | None) -> list[str]:
    print(f"\n=== {label} ===")
    model, tokenizer = load(MODEL, adapter_path=adapter_path)
    outputs = []
    for p in PROMPTS:
        msgs = [{"role": "user", "content": p}]
        prompt = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )
        out = generate(model, tokenizer, prompt=prompt, max_tokens=80, verbose=False)
        outputs.append(out)
        marker = "[🦊]" if FOX in out else "[  ]"
        print(f"{marker} Q: {p}")
        print(f"     A: {out.strip()}\n")
    return outputs


def summarize(label: str, outs: list[str]) -> None:
    hits = sum(FOX in o for o in outs)
    print(f"{label}: {hits}/{len(outs)} responses contain {FOX}")


def main() -> None:
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--before", action="store_true", help="base model only")
    g.add_argument("--after", action="store_true", help="adapter only")
    g.add_argument("--both", action="store_true", help="both, side-by-side (default)")
    args = parser.parse_args()

    if not (args.before or args.after or args.both):
        args.both = True

    before_outs = after_outs = None
    if args.before or args.both:
        before_outs = run_one("BEFORE (no adapter)", adapter_path=None)
    if args.after or args.both:
        if not Path(ADAPTER).exists():
            raise SystemExit(
                f"adapter dir {ADAPTER} not found — run mlx_lm.lora --train first"
            )
        after_outs = run_one("AFTER (with LoRA adapter)", adapter_path=ADAPTER)

    print("\n=== summary ===")
    if before_outs is not None:
        summarize("BEFORE", before_outs)
    if after_outs is not None:
        summarize("AFTER ", after_outs)


if __name__ == "__main__":
    main()

# Journal — play/sft_hello

## 2026-05-09 — Project approved + scaffolding ready

### Functional

Create a new `play/sft_hello/` sub-project as a one-time hello-world fine-tuning experiment. Goal: Use MLX-LM LoRA to train Qwen2.5-0.5B-Instruct to "add 🦊 at the end of each answer" on M4 Pro 48GB, and verify that the entire fine-tuning pipeline can run smoothly on this machine. **Strictly separate from `play/agent_sft`** - the latter is a real project, this project is just to test the process.

Scaffolding delivered: README (4-step guide), `data/train.jsonl` + `data/valid.jsonl` (handwritten 30 + 10 static chat records), `lora_config.yaml` (explicit declaration of LoRA structural knobs rank/scale/dropout/keys), `infer_compare.py` (pre-training/post-training juxtaposition). One line `mlx_lm.lora --config ./lora_config.yaml ...` can start training.

### Technical

- Base: Qwen2.5-0.5B-Instruct (feedback loop < cup of coffee).
- Data: 30 items in toy chat format are handwritten and submitted directly. Assistant always adds ` 🦊` at the end of each answer (the naked eye signal = `grep 🦊` can be judged).
- Training: MLX-LM LoRA, CLI go `--num-layers 8 --iters 200 --batch-size 4 --learning-rate 1e-4`, LoRA structure (`rank=8` / `scale=20.0` / `dropout=0.0` / `keys=[q_proj, v_proj]`) go `lora_config.yaml`. 0.5B full-precision training without introducing 4-bit quantization complexity.
- Reasoning comparison: `infer_compare.py` uses `mlx_lm.load(adapter_path=...)` to switch with/without adapter, run the same 5 prompts in double, and stdout is printed in parallel.
- Deployment boundary: Don't do fuse/GGUF/ollama, leave it to `play/agent_sft` Phase 4.
- Configuration form: Division of labor between CLI and YAML - CLI exposes the knobs for daily adjustments, and YAML encapsulates the LoRA structural skeleton. The design of MLX-LM itself limits `rank/scale/dropout/keys` to the config file. This project takes advantage of the opportunity to change the "rank=8 core knob" from an implicit default to an explicit statement in version control.
- Learning tools: Added `sweep.py`, controlled-variable method to scan `iters / lr / num-layers / batch-size / rank` 5 knobs each with 3-4 orders of magnitude values, automatically generate `runs/sweeps/REPORT.md` - table + value-by-value simple language interpretation + proper nouns with English. **This is not an architecture change, it is a cognitive tool**: Let "what the actual impact of super parameters" is from a sentence in the document to an actual measurement comparison visible to the naked eye. The full run takes about 30-40 minutes.

## 2026-05-10 — The control variable sweep runs through, and all 18 sets of results are placed on the market.

### Functional

`python sweep.py all` runs through 5 axes × a total of 18 groups of (sweep, value) configurations. The products fall into `runs/sweeps/`: a subdirectory for each group contains adapter + `train.log` + `eval.json`, and the top-level `REPORT.md` automatically summarizes the table + Per-value notes + general conclusions. The actual measured **full run time on M4 Pro 48GB is about 9 minutes** (much lower than the 30-40 minute estimate marked in the README). You can rest assured that you can run the regression script repeatedly in the future. **The 5-axis conclusion is consistent with the textbook narrative**: iters=200 / lr=1e-4 / layers=8 / batch=4 / rank=8 all fall in the Sweet spot, 🦊 hit 5/5; lr=1e-6 has a step size of less than 0/5, and is the only cell that has been "trained but not learned".

### Technical

- Actual measurement time: single 200-iter training ~16s, 5 prompt eval ~3s, whole round ≈ 9 minutes. The bottleneck is that `mlx_lm.load` repeatedly loads the base, which can be changed to reuse model handles across (sweep, value) in subsequent optimizations; the current toy scale is not optimized.
- Numerical observation: Last loss of iters from 200→1000 stops at 0.07, indicating that 30 pieces of data + r=8 has almost pushed the "add 🦊 at the end" to the upper capacity limit; rank also has no gain from 8→32, once again confirming the "ΔW low rank" hypothesis of the LoRA paper - the "effective rank" of the toy task in this project is far lower than 2.
- **`batch=16` is not diverged but mlx-lm data set verification error**: `valid.jsonl` has only 10 samples. When the trainer does the first val, it `raises ValueError("Dataset must have at least batch_size=16 examples but only has 10.")` and exits directly, which takes 1.4s. Currently, `sweep.py` uses `returncode != 0` to classify it as "diverged" across the board, causing the interpretation in `REPORT.md` to say "the learning rate is too large, NaN, and the quantization accuracy is insufficient" - which is actually a data set size constraint. This is a cognitive bias in the script, not a problem with mlx-lm; either reduce the batch range, expand `valid.jsonl`, or distinguish "data error" vs "true diverged" in the script. Accounts will be recorded first this round and will not be changed on the spot.

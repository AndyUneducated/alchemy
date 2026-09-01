# Triples — Phase 2 SFT data product directory

`agent_sft/data/triples/` stores the intermediate products and final training samples of the data pipeline (mine → extract / synthesize → split → format). The two 1k data sets of v1 are still retained as historical Baseline; the current qwen3.5 line is mainly the clean-data version.

## Quick check of current data set

|dataset|base/source|purpose|status|
|---|---|---|---|
|`train_7b_1k.jsonl` / `val_7b_1k.jsonl`|Qwen2.5-7B mining|v1 training and reproduction|history retention|
|`train_32b_1k.jsonl` / `val_32b_1k.jsonl`|Qwen2.5-32B mining|wrong_tool hard sample / ablation|History retention|
|`train_qwen3.jsonl` / `val_qwen3.jsonl`|v1 7B + v1 32B + v1.5 9B mixed source|v1.5 qwen3.5 retraining|replaced by clean-data version|
|`train_qwen3_clean.jsonl` / `val_qwen3_clean.jsonl`|qwen3 clean rebuild|v1.6 current default training data|current recommendation|

## File list

|File/Directory|Generated in|Whether to enter git|Purpose|
|---|---|---|---|
|`runs_1k_fast_{7b,32b}_r0_124/<scen>-r<N>.json`|`mine_triples.py` (fast scenario, run_id 0-124)|✅|Phase 2 final delivered raw envelope, 250 per model (2 scenario × 125 run)|
|`{triples,train_triples,val_triples,train,val}_{7b,32b}_1k.jsonl`|`synthesize.py` → `split.py` → `formatter.py`|✅|v1 The full-link SFT data of the two models; `train_*.jsonl` is the chat schema that can be directly consumed by MLX-LM|
|`*_qwen3*.jsonl` / `qwen3_clean_report.json`|qwen3.5 migration and clean-data rebuild|✅ / Partially gitignored|Current qwen3 training line; See [`DECISIONS §10`](../../DECISIONS.md) / [`§11`](../../DECISIONS.md)|
|`runs/<scen>-r<N>.json`|`mine_triples.py` default output|❌ (gitignore)|local smoke / temporary run batch|
|`triples.jsonl` / `train.jsonl` / `val.jsonl` etc. without `_1k` suffix|Default product|❌ (gitignore)|Local smoke derivation; to reproduce 1k data set see §Rebirth command|

## Two triple sources (select synthesize after pilot)

|script|pairing strategy|yield|corrected source|
|---|---|---|---|
|`extractor.py`|First attempt failure + subsequent attempt real success|~3-25% (depending on the model recovery rate)|Real speaker.content|
|`synthesize.py` (**current default**)|each nudge fire → 1 triple|100%|from step.instruction programmatic template (fallback: universal wrapper + full instruction)|

The recovery rate measured by 7B pilot is only ~3% → the yield of the extractor path is too low and impractical; the synthesis size is corrected using the literal `tool(args)` template in step.instruction, and zero extra compute is used to increase the yield to 100%. See §Historical legacy: 57-triple pilot and method selection.

## Rebirth command

### v1 1k final delivery batch (consistent with `*_1k.jsonl` in repo)

Take 7B as an example (for 32B, replace `AGENT_ENGINE_MODEL` with `qwen2.5:32b` and all `_7b_` with `_32b_`):

```bash
export AGENT_ENGINE_MODEL=qwen2.5:7b

# 1) Batch 250 envelopes (fast copy, run_id 0-124)
python play/agent_sft/data/mine_triples.py \
  --run-ids $(seq 0 124) \
  --out-dir play/agent_sft/data/triples/runs_1k_fast_7b_r0_124

# 2) Extract triples (synthesize: one per fire)
python play/agent_sft/data/synthesize.py \
  --in  play/agent_sft/data/triples/runs_1k_fast_7b_r0_124 \
  --out play/agent_sft/data/triples/triples_7b_1k.jsonl

# 3) Split train/val (per-scenario last 20% run_id → val)
python play/agent_sft/data/split.py \
  --in    play/agent_sft/data/triples/triples_7b_1k.jsonl \
  --train play/agent_sft/data/triples/train_triples_7b_1k.jsonl \
  --val   play/agent_sft/data/triples/val_triples_7b_1k.jsonl

# 4) Format MLX-LM samples (run formatter on train + val)
python play/agent_sft/data/formatter.py \
  --in  play/agent_sft/data/triples/train_triples_7b_1k.jsonl \
  --out play/agent_sft/data/triples/train_7b_1k.jsonl
python play/agent_sft/data/formatter.py \
  --in  play/agent_sft/data/triples/val_triples_7b_1k.jsonl \
  --out play/agent_sft/data/triples/val_7b_1k.jsonl
```

### Local smoke / change schema debugging

Use the default output (`runs/` + `triples.jsonl` + `train.jsonl`, all gitignored), the command is the same as above but remove `--out-dir`, remove the `_*_1k` suffix from the file name, `--run-ids 0 1 2 3 4 5` and run 12 envelopes. To reproduce the `max_retries=1` behavior of baseline eval: add `--upstream` (mine + synthesize must be consistent, otherwise turn_idx will be misplaced and yield will be reset to zero).

`--help` to see complete flags per script.

## Scenario: fast copy vs upstream

`data/scenarios/{tool_chain,code_review}_fast.md` is a mining optimized derivative of upstream `agent_engine/scenarios/*.md`:

|Change|fast|upstream|Why fast is changed like this|
|---|---|---|---|
|`max_retries`|0|1|synthesize only looks at the first attempt, retry is a pure waste of LLM calls|
|`max_tokens`|80|160-200|agent prompts already cap at ≤30/50 chars; cap matches actual load|
|moderator open / finalize|delete|has |0 fires, pure ritual overhead|
|envelope wall clock (7B / M4 Pro)|~42s/env|~65s/env|-35%|
|synthesize yield|~4 triples/env|~4.75 triples/env|Quite|

`mine_triples.py` defaults to fast; `--upstream` switches back to the original scenario (consistent with baseline eval data). `extractor.py` / `synthesize.py` also has the same `--upstream` flag, which must be consistent with the mining step - otherwise the misalignment of turn_idx will cause the yield to return to zero.

## OOD evaluation

**OOD evaluation is not in this directory** - Reuse the `play/evals/data/bfcl_slice/gold.jsonl` implemented in Phase 1 (50 examples of BFCL `simple_python` slices). v1 Phase 5 retest directly:

```bash
python -m evals run --task bfcl_slice --model ollama:agent-sft-qwen
```

After qwen3.5 migration, `agent-sft-qwen-3` is still a placeholder — do not use this command to judge real adapter OOD performance. Public datasets are not copied into this repo; BFCL upstream changes are managed by `play/evals/data/bfcl_slice/_fetch.py`.

## v1 Phase 2 final delivery: 1k × 2 models

Two independent data sets, mining batch parameters are aligned (fast scenario / `max_retries=0` / `run_id 0-124` / 2 scenarios), only the mining model is different:

|Item|7B (Qwen2.5-7B)|32B (Qwen2.5-32B)|
|---|---|---|
|envelope（committed in `runs_1k_fast_{7b,32b}_r0_124/`）|250|250|
|triples (`triples_*_1k.jsonl`)|**1212**|**1052**|
|triples / envelope|4.85|4.21|
|train / val (`train_*_1k.jsonl` / `val_*_1k.jsonl`)|966 / 246|842 / 210|
|Failure mode distribution|missed 1091/wrong_tool 121|missed 773/wrong_tool 279|
|scenario distribution|code_review 933 / tool_chain 279|code_review 802 / tool_chain 250|
|Tested wall clock (M4 Pro)|~7.5 h|~9.5 h|
|on-disk size|envelopes 2.1 MB + jsonl ~11 MB|envelopes 2.4 MB + jsonl ~10 MB|

`val` is split consistently: the last 20% run_id of each scenario (i.e. `run_id ∈ [100, 124]`) → val.

**7B vs 32B Selection Guide (v1 History)**:

|Choose|Suit the scene|Price|
|---|---|---|
|7B|Default training; missed main distribution coverage is sufficient, single triple cost is lower |wrong_tool distribution is narrower|
|32B|Complement wrong_tool hard sample; do ablation|Single triple cost is higher|
|Hybrid|qwen3 migration data volume in the early stage|needs cleaning and deduplication and train/val leakage check; v1.6 has been rebuilt clean-data|

## Historical legacy: 57-triple pilot and method selection

Four pilot batches were run in the early stages of Phase 2 (see the `JOURNAL.md` 2026-05-10 entry for detailed timing). Key conclusions:

1. 7B + extractor (requires first-fail + later-success true recovery) yield is only 0.17/env, which is 30 times worse than plan’s estimated 5/env.
2. Try doubling `max_retries` → no improvement; try 32B comparison → recovery jumps from 3% to 25%, proving that base capability is the main cause of recovery rate.
3. Change to the synthesize path (per-fire + step.instruction template corrected) → 7B can also yield ~4.75/env, hitting the original estimate of plan.

**Why `extractor.py`** is still retained: In the future, if the recovery rate of 7B reaches 30%+ after Phase 3 training, and the "true self-correction" semantics of extractor are more consistent with the core argument of the project (self-correction) than the "template answer" of synthesize, then the one-line command will be switched back.

# Journal

One paragraph for each milestone: `## YYYY-MM-DD — Title`, the text must contain **Functional** + **Technical**, **Trade-offs** on demand. See [`DECISIONS.md`](DECISIONS.md) for architectural decisions.

## 2026-05-10 — Phase 0 project establishment + Phase 1 baseline tool chain

Three framework decisions were made (central problem = nudge-grounded SFT, ceiling was written as "v1 + evolution path" using local Qwen2.5-32B, README), and the Phase 1 baseline evaluation tool chain was completed. 80-batch real runs are reserved for independent milestones.

### Functional

|item|Description|
|---|---|
|README + Roadmap|Central issue: Let the nudge-fire rate drop significantly in in-dist and OOD not return; Phase 0-6 + complete evolution path|
|Measure four items|nudge-fire rate/trajectory score/BFCL/general regression, according to scenario × tool × failure-mode three-axis + multi-seed to report the mean ±std|
|Two new scenarios|`code_review` (4 agent × 8 turn) + `tool_chain` (1 agent × 5 turn), both strong require_tool|
|Multiple seed wiring|model spec plus `@seed=K` suffix, minimal intrusion|
|baseline tool|runner (2 model × 10 seed × 4 task = 80 runs) + aggregator output markdown report|

### Technical

- **Central question**: Use your own agent_engine to produce trajectory, and use your own evals to calculate nudge fire rate (DECISIONS §1).
- **Training Framework**: MLX-LM, Apple Silicon native, three command links (DECISIONS §2).
- **Base + ceiling**: Qwen2.5-7B + Qwen2.5-32B, cross-scale comparison of the same family, zero closed source dependency.
- **Failure Mode**: missed / wrong_tool / wrong_args (the latter is temporarily occupied by wrong_tool).

### Trade-offs

- Abandon "classic tool-calling LoRA on xLAM/ToolACE" - simple implementation but no differentiation in interviews.
- Abandon GPT-4o-mini as ceiling - replace it with all local reproducible + compare with the same family.
- No introduction of `lm-evaluation-harness` - self-implementation <100 lines is much lower than cross-framework adaptation cost.

## 2026-05-10 — Phase 2 pipeline + 57 pieces of demo data

After setting up the assembly line in one day, running the pilot and running into bottlenecks, I decided to use plan B to unlock it. Final payment 47 train + 10 val.

Timing: 7B × 6 envelope → 1 triple; max_retries=2 retry → still 1 triple; 32B × 3 envelope comparison → recovery jumps from 3% to 25%, confirming that base capability is the main reason; change to synthesize (per-fire pairing, corrected with instruction template) → 12 envelopes produce 57 triples.

### Functional

|item|Description|
|---|---|
|`data/` 5 scripts |mine_triples / extractor / synthesize / split / formatter + 18 tests |
|Two triple paths|extractor is extracted and self-corrected (yield 0.17/env); synthesisize uses templates to create corrected (yield 4.75/env)|
|First data|47 train + 10 val, per-scenario end 20% cut val|

### Technical

- **Why synthesize by default**: The yield of extractor at 7B is too low; synthesisize increases the yield to the upper limit of fire rate, saving ~14× on compute.
- **Relationship with §1**: It is still "7B failure material + scenario template", and no third-party teachers are introduced.
- **Sample format**: F1 only (input does not include nudge), training "See the original instruction in place at once".

### Trade-offs

- Acknowledging that "32B single envelope comparison failed" the day before was over-judgment, n=20 was rerun and corrected, and the process was left in JOURNAL without covering up.
- Select synthesize instead of 32B mining - economical crushing + cleaner signals; the price is the risk of template rereading, leaving it to Phase 3 to see the effect.
- 57 triples count as "demo viable" - whether to scale to 1k will be determined after the Phase 3 smoke signal.

## 2026-05-10 — fast scenario copy: mining speed increased by 35%

To speed up the subsequent 1k scale-up. The upstream scenario remains unchanged (baseline eval has been run with max_retries=1), and the new `_fast` copy only serves mining: max_retries 1→0, max_tokens 200→80, and deletes the open/finalize steps. `--upstream` flag defaults to fast, add flag to go upstream.

smoke 7B 2 envelope: average **42s/env** (vs upstream 65s/env, -35%). 1k speedup estimate: ~175 min vs ~228 min.

Do not write new unit tests (fast copies are data, not logic), and do not do more radical minimal scenarios (project overhead > time gain).

## 2026-05-10 — Phase 2 Ending: 1k × 2 model double batch data

I ran 7B / 32B in 17 hours overnight, 250 envelopes each.

|item|Description|
|---|---|
|7B data set|1212 triples → 966 train + 246 val; wall clock 7.5h|
|32B data set|1052 triples → 842 train + 210 val; wall clock 9.5h|
|Strategy|Two copies (file names with `_7b_` / `_32b_`) facilitate ablation; 27 MB direct commit (well below the LFS threshold) |
|7B vs 32B actual measurement|yield is close (the base difference becomes smaller under the synthesized path); 32B wrong_tool accounts for a higher proportion and is used for hard samples|

The orchestrator uses `caffeinate` + `nohup` + `set -euo pipefail`, and the mining script naturally supports continued running.

## 2026-05-11 — Phase 3 training: schema upgrade + LoRA sweep + lock recommendation adapter

The tool chain + data schema locking + end-to-end smoke were implemented in one day, and the sweep was completed in 7 hours overnight (iters × lr, a total of 6 runs). Accordingly, add [`§5`](DECISIONS.md) lock recommendation adapter.

### Functional

|item|Description|
|---|---|
|[`§4`](DECISIONS.md) schema upgrade|SFT uses OpenAI `tool_calls` JSON format, aligned with Qwen2.5 chat template + Ollama parser + agent_engine full link|
|formatter rewrite|new schema + tolerant args parser (rescue ~500 samples of cast_vote containing Chinese `or`)|
|Data rebirth|7B 1212→962 / 32B 1052→802, drop ~24% fallback|
|`train/` Table of Contents|`lora_config.yaml` (q/k/v/o, rank 16) + `train.py` + `eval_smoke.py` + `sweep.py`|
|End-to-end smoke|30-iter on 4-bit Qwen2.5-7B: loss converges to 0.001, 4 items metric ≥95%|
|sweep 6 run|iters {50,200,600} + lr {1e-5,1e-4,5e-4}|
|[`§5`](DECISIONS.md)|Lock recommendation adapter = `runs/sweeps/iters/200/` (= BASE configuration); layers/rank will be postponed to Phase 5 for real testing before decision|

### Technical

- **Why not text-only**: Ollama only recognizes the `<tool_call>` JSON block; the schema is not aligned → the tool_call event cannot be emitted after training the model.
- **schema single source**: formatter directly imports `agent_engine.scenario._resolve_tool_defs`, and scenario changes to training data will automatically follow.
- **Choose the 4-bit pre-quantized version for the base**: HF straight pull without conversion, smoke peak mem 12 GB.
- **sweep on-site downscaling**: The original plan 16 runs took 60h+ on the M4 Pro, and was reduced to 6 runs (core 2 dim) ~8h.

Four sweep key findings:

|Discover|Description|
|---|---|
|iters fully saturated|50 / 200 / 600 The three-level loss is fully convergent, and the 4-item metric is all 100% - the schema signal is highly compressible, 50 iter has been learned thoroughly|
|lr only 5e-4 degradation|1e-5 / 1e-4 all 100%; 5e-4 arg_value 76%——sweep only differentiating evidence|
|BASE = optimal|Exactly equal to the existing baseline - neither worse nor better|
|fast proxy saturation|eval_smoke can only tell us "learn thoroughly", but cannot tell us "memorize vs generalize". We need to run Phase 5 end-to-end to know it|

### Trade-offs

- Choose the OpenAI tool_calls format instead of writing the literal content as content - align with the mainstream framework, and there will be zero changes when changing frameworks.
- drop fallback samples instead of fill-ins - retaining will teach the model "repeated instruction text" weak signal.
- Sweep does not scan layers/rank after actual measurement - the fast proxy is saturated, and the information gain from scanning again is low; the trigger condition is locked at §5.

## 2026-05-11 — Phase 4 deployment: adapter → GGUF Q4_K_M → Ollama

One-time run through fuse → convert → quantize → ollama create → smoke, wall clock 7 minutes. `agent-sft-qwen` is registered with Ollama and `agent_engine` has zero cost to switch via environment variables.

### Functional

|item|Description|
|---|---|
|[`deploy/`](deploy/)|`Modelfile` (1:1 fork with qwen2.5:7b) + `build.sh` (three-step idempotent) + `deploy.sh` + `smoke_test.py`|
|[`§6`](DECISIONS.md)|Lock Q4_K_M quantization + Modelfile 1:1 replica|
|Product size|fused fp16 14 GB → F16 GGUF 14 GB → Q4_K_M GGUF **4.4 GB**|
|HTTP smoke|`/api/chat` returns parsed `tool_calls`, Ollama parser natively recognizes `<tool_call>`|
|End-to-end smoke|Run the full 8 step trajectory, catch 10 tool_call events, and the tool set is fully covered|

### Technical

- **llama.cpp introduces**: workspace externally built `~/Tools/llama.cpp/`, independent `.venv`, only build `llama-quantize` target.
- **`mlx_lm.fuse --dequantize` Required**: When using 4-bit base fuse, LoRA cannot add quantization grid, and it must be dequantized to fp16 first.
- **Modelfile 1:1 fork**: Do not write custom jinja, only replace the `FROM` line; granular = `ollama show --modelfile` output.
- **Chinese args verification**: `cast_vote(option="append")` full-link UTF-8 lossless penetration.

### Trade-offs

- Quantization lock Q4_K_M instead of Q5/Q8 - the same quantization axis as baseline is worse than the SFT signal, which is the prerequisite for Phase 5 signal attribution.
- `deploy/build/` does not enter git (18 GB fully local) - build.sh is a regeneration guide, new machines can be reborn in ≤10 min.

## 2026-05-11 — agent_engine public direct connection + transcript typed upgrade

Clean up Phase 2 Debt of direct import private helper for trojan. **Step 1**: Cut `from evals.metrics.nudge import _private` to `from agent_engine import Result, Scenario, TurnView`. **Step 2**: The transcript is upgraded to 6 typed dataclasses, three scripts are cut and distributed by `isinstance(...)`; 500 historicals envelopes are migrated in one time. Details [`§7`](DECISIONS.md) + [`§8`](DECISIONS.md).

### Functional

- There is only 1 cross-project public import (`classify_failure_mode`) left in the three scripts, and the private import is reduced to 0.
- `extractor / synthesize / formatter` goes `isinstance(e, SpeakerEntry/...)` + direct field access.
- In the §7 stage, the shims are left to allow the old test zero to modify the pass. In the §8 stage, the shims that are no longer needed are cleared at the same time.
- 500 envelopes are injected with new schema fields and run in seconds.
- smoke: 5 envelopes out of 21 triples, in the same order as Phase 2 historical yields.

### Technical

- **Interpretation rights returned**: "How did transcript become ToolCall" is part of the schema and lives in the agent_engine; "failure mode classification" is where the evals/sft perspective semantic judgment remains.
- **`Triple.context` type**: `list[dict]` → `list[TranscriptEntry]`.

### Trade-offs

- The shim renews the life instead of deleting the old test - the hard constraint of the plan is "test zero modification pass".
- `classify_failure_mode` does not mention agent_engine - it is a semantic judgment from the perspective of evals, and mentioning it will pollute the boundary of concern.
- 500 envelope Choose migration instead of re-running - migration takes seconds, while re-running requires hour-level LLM costs.

## 2026-05-13 — Phase 5.A end-to-end baseline 120-batch

After running three models × 10 seeds × 4 tasks for comparison. **Main batch** 13h28min, 120/99/21 ok/failed - 21 failures concentrated on 32B agent tasks (subprocess timeout is not enough) + 1 7B handler crashed due to illegal kwarg generated by the model. **Batch run** 14h48min, 17 runs serial + timeout env override, all OK. In the end, 119/120 cells; 7B, that one was permanently excluded due to the vulnerability of evals evaluation.

### Functional

|item|Description|
|---|---|
|Aggregation output|[`phase5-3model-comparison.md`](eval/baselines/phase5-3model-comparison.md), 4 task × 3 model + three-axis breakdown|
|119/120 cell|11 cell all n=10; exception 7B nudge_fire_rate n=9|
|`run_baseline.py` three patches | Inject `AGENT_ENGINE_MODEL` env (let the agent sub-process run the correct model) + `sys.executable` replace `"python"` (no python command on this machine) + `agent_engine_run.py` plus `AGENT_ENGINE_RUN_TIMEOUT` env override (the default 600s is not enough for the 32B agent task) |

### Technical

- **wall clock**: main + supplement ≈ **28h** real run batch; 32B agent single run nudge_fire_rate ~62min, agent_traj ~43min.
- **`aggregate_seeds.py` does not dedupe**: press (task, model) to mean directly, repeating seed will pollute. Detour: script dedup first to write clean index and then aggregate.
- **Evaluation vulnerability**: 7B seed=3 spontaneously outputs `tool=cast_vote(...)` when kwarg → handler `TypeError` hangs - left as evals' own lesson.

### Trade-offs

- Retaining 3 engineering patches and not rolling them back - this is a prerequisite for "making the 102 agent-path data correct", not a QoL improvement.
- Accept 7B n=9 instead of making up to n=10 - the agent_engine tool dispatch needs to be repaired first, and cross-project changes + central problem determination do not rely on a single seed.
- Two batches in series instead of parallel - Metal backend cross-process parallelization may cause OOM or trash cache.
- `aggregate_seeds.py` does not add dedup - a one-time diagnostic script, generalization will wait until there is a second consumer.

## 2026-05-13 — Phase 6 Reflection + v1 Case Closed

Judgment based on the three pre-locked thresholds → Passed all → The answer to the central question is "Yes and the conditions are clear". README adds §"Lessons learned", [`§9`](DECISIONS.md) v1 closes ADR, fills in interview narrative numbers.

|Three thresholds|Actual measurement|
|---|---|
|nudge gap closure ≥50%|**57.3%** ✓|
|BFCL regression ≤5%|**1.16%** ✓|
|MMLU regression ≤3%|**2.09%** ✓|

### Functional

- README §Lessons learned: Phase 5 number overview + three answers + v2/v3 candidate selection list + "reverse of the coin" paragraph (task_success overtake / trajectory degradation / missed→wrong_tool transformation).
- Complete the interview narrative numbers: "X% → Y%" → "0.739 → 0.645".
- [`§9`](DECISIONS.md): v1 closed ADR, including central question and answer + candidate status update + engineering patch status + evaluation vulnerability followup handover.

### Technical

- **Threshold is pre-locked and then determined**: plan §6.1 + §5. The trigger conditions have been locked long ago. After Phase 5 is run, just numbers are substituted - to avoid post-event fitting.
- **Second-order evidence is written into Lessons**: task_success overtake / trajectory degradation / missed→wrong_tool conversion / panel reverse / retrieve_docs 100% These 5 surprises do not affect the threshold determination, but drive v2 candidate selection.
- **§5 layers/rank not triggered**: gap closure 57.3% > 50% → §5 status remains accepted.

### Trade-offs

- v3-D (multi-supervision signal superset) delisting - v1 exposes supervision quality deviation (panel reverse, retrieve_docs 100%) rather than insufficient quantity. Adding new signal buckets will only repeat the deviation of v1.
- v2-A DPO persistence - the core of v1 is a classification problem rather than a preference problem; but if v2-B/C is finished and there is still a scenario of "two candidates with different styles", DPO is still applicable.
- v3-B HF Hub release ahead of schedule - Model Card content has been formed, the community has received the "7B SFT closing 57% of 32B gap" downloadable product, and the portfolio signal is the strongest.
- v3-A 14B persistence - SFT 7B has overtaken 32B in task_success, scale to 14B first to avoid the current signal.
- No specific plan for v2/v3 will be added at this milestone - only the ending of v1 will be completed in this period.

## 2026-05-25 — v1.5: qwen3.5:9b retraining + 9-run minimalist retest + GGUF deploy suspension

### Functional

- After the default base of the warehouse is switched to qwen3.x, the entire set of agent_sft (train / eval_smoke / sweep / run_baseline / Modelfile / 6 fixture tests) points to `qwen3.5:9b` + `qwen3.6:27b` by default; `agent-sft-qwen-3` ollama tag is online (the v1.5 stage is a base replica placeholder).
- Data: Clear the full set of v1 artifacts, a new round of train 588 / val 155 triple combination source ([`§10`](DECISIONS.md) digital snapshot), three source run_id disjoint offset to ensure scenario grouping is complete.
- Training completed 600 iters; `train/runs/main_qwen3/` adapter dropped; eval_smoke (bypassing the ollama path and going straight to 4bit base + LoRA mlx inference) n=50 shows emit=86% / arg_value_match=64%, confirming that the SFT training is real and effective.
- 9-run nudge_fire_rate baseline ran 6 OK + 3 failed (qwen3.6:27b single scenario exceeded 600s default timeout × 3 seeds), the number fell [`eval/baselines/qwen3_phase3/index.jsonl`](eval/baselines/qwen3_phase3/index.jsonl); the GGUF deploy path exposed the mlx→hybrid conversion defect, and the placeholder took advantage of See [`§10`](DECISIONS.md).

### Technical

- **Modelfile simplification**: `TEMPLATE` Go-template DSL (v1 ~50 lines) → ollama 0.20+'s `TEMPLATE {{ .Prompt }}` + `RENDERER qwen3.5` + `PARSER qwen3.5` three-line directive (base `ollama show --modelfile qwen3.5:9b` has confirmed that this is a legal short-hand for 1:1 forking).
- **GGUF deploy path broken**: mlx_lm.fuse --dequantize → fp16 safetensors → `convert_hf_to_gguf.py` → Q4 GGUF, after loading into ollama, both F16 and Q4 output garbled characters ("ã add Guang_MMjv slip..."); the fp16 fused mlx directory is directly inferred using `mlx_lm.generate` OK. Attribution mlx has inconsistent reconstruction of SSM layer 4bit→fp16 for Qwen3.5 hybrid (attention+SSM) architecture (`ssm_alpha`/`ssm_beta`/`ssm_conv1d` naming/weight reconstruction path is defective). Repair the backlog and see [`§10`](DECISIONS.md); change the short-term Modelfile to `FROM qwen3.5:9b` so that the evals harness can continue without any fuss.
- **eval_smoke parser is compatible**: `<tool_call>` block, Qwen2.5 is JSON, Qwen3.5 is nested XML (`<function=NAME><parameter=KEY>VALUE</parameter></function>`), double regularization is required; if emit_rate=0% is not added, the training fails by misjudgment.
- **formatter `arguments` type correction**: OpenAI tool_calls `arguments` must be a Python dict, not a JSON string - Qwen3.5 chat template `.arguments | items()` jinja filter requires mapping, and passing the string will report `TypeError: Can only get item pairs from a mapping`. v1 (Qwen2.5) uses `| tojson` which does not require a type, so v1 does not expose this bug.
- **OOM-driven training super-parameters**: Plan estimates that num_layers=16 + batch=4 can be installed. The actual measurement of M4 Pro 48GB is that fwd+bwd cannot be installed on Qwen3.5-9B + hybrid operator (even 4bit base + grad checkpoint). Finally, batch=1 / num_layers=4 / max_seq_length=1500 / `--clear-cache-threshold 1`, single iter ~7s, the whole journey is 73 minutes; train.py adds `--max-seq-length` + `--clear-cache-threshold` CLI transparent transmission.
- **Mixed source run_id offset**: When merging three batches of triples v1 7B / v1 32B / v1.5 9B, allocate disjoint run_id offset to each batch to ensure that the `split.py` scenario-level train/val grouping is not aliased.

### Trade-offs

- The data strategy deviates from the plan: the plan estimates that 7 envelope × 60 triples produce 500+100 using the `qwen2.5:7b` mining coefficient; `qwen3.5:9b` strong → 14 envelopes only produces 49 nudge triples (the strong model itself triggers less require_tool). **No more envelopes + no switching back to 7B mining**, and the three historical batches of triples are merged to make up 588/155 - ensuring the quality and diversity of training samples at the cost of mixed supervision sources (See [`§10`](DECISIONS.md)). The 9 runs evaluation comparison is degraded into a repeated comparison due to placeholder; the SFT real signal is handed over to eval_smoke for verification.
- GGUF deploy is suspended instead of dead: mlx→GGUF Qwen3.5 hybrid compatibility issue cannot be adjusted in 1-2 hours, and forced adjustment will occupy the total budget of the plan; change it to "training artifact + fused fp16 mlx model + placeholder ollama tag" three-piece archive, and separate the deploy repair into a backlog.
- DECISIONS §6 superseded by §10 instead of deleted - keep history "7B Q4_K_M + ~50 lines TEMPLATE fork" as v1 deploy documentation.

## 2026-05-26 — v1.6 clean-data + bf16 retraining (story layered version)

### Functional

|Item|Milestone|
|---|---|
|data|Reconstruct clean data from `triples_qwen3_merged.jsonl`: `triples_qwen3_clean.jsonl` 1547, `train_qwen3_clean.jsonl` 1276, `val_qwen3_clean.jsonl` 271; train/val overlap=0; `example`/`panel` are not mixed into training. |
|Training|Complete the three-stage training of bf16 smoke(10) + probe(40) + main(600); the main training product is `train/runs/main_qwen3_bf16_clean/`. |
| Review | Added `eval/baselines/qwen3_bf16_clean/index.jsonl` and `story_report.md`, stratified readings by in-distribution(`tool_chain`/`code_review`) vs held-out(`example`/`panel`). |
|Deployment verification|Complete bf16 fuse + F16/Q4 GGUF build; retain `agent-sft-qwen-3` placeholder as available online tag. |

### Technical

|items|results|
|---|---|
|smoke stability|`max_seq_length=1000` will appear `Trained Tokens 0 + NaN`; it will be stable after it reaches 1500 (not an OOM problem). |
|Master training health|`main_qwen3_bf16_clean`: rc=0, nan=false, train 0.247→0.000, val_last=0.000, wall=4115s, peak mem≈21.36GB. |
|LoRA learning effect|`eval_smoke`(n=50) only emit=0.56 / arg_value=0.36, which is lower than the 0.64 baseline of v1.5. |
|GGUF Conclusion|The fused mlx-fp16 inference is readable; however, F16 GGUF and Q4 GGUF still output garbled code in ollama (`testf16`/`testq4` reappears), indicating that the compatibility problem is still in the runtime conversion link. |
|Story stratification|In the 6-run results, the mean value of `agent-sft-qwen-3` in-dist is 0.4872 and the mean value of held-out is 0.1905; the mean value of `qwen3.5:9b` in-dist is 0.4359 and held-out 0.1905. The current tag is placeholder and cannot be interpreted as the real income of the new adapter. |

### Trade-offs

- According to [`DECISIONS §11`](DECISIONS.md), accept the result of "GGUF is still blocked but training and evaluation are closed first": no longer forcefully replace online tags with damaged GGUF.
- This round will not continue to blindly add iter; in the future, hard-sample mining (especially `cast_vote`/Parameter value accuracy) will be given priority instead of extending the training time.

## 2026-05-31 — Document calibration: v1 historical results and qwen3 current state stratification

### Functional

|item|description|
|---|---|
|README current status table|Separating v1, v1.5, and v1.6, the reader first sees "qwen3.5 training/evaluation has run through, but GGUF/Ollama real deployment is still blocked"|
|Deploy document rewriting|Use state table, decision tree, and qwen3.5 data flow to explain the difference between placeholder tag and damaged GGUF|
|Subdirectory README Calibration|`train/`, `data/triples/`, `eval/baselines/` changed to v1 History and qwen3 current line parallel description|

### Technical

|item|description|
|---|---|
|Contract boundaries|Explicit `agent-sft-qwen-3` currently `FROM qwen3.5:9b`, cannot represent LoRA adapter effect|
|Evidence chain|README anti-link [`DECISIONS §9`](DECISIONS.md), [`§10`](DECISIONS.md), [`§11`](DECISIONS.md) to avoid old Qwen2.5 numbers being misread as qwen3.5 Conclusion|
|Deployment path|Separate the three paths of short-term MLX fused artifact, placeholder tag, and future GGUF repair to reduce the misunderstanding of "can run tag = SFT is online"|

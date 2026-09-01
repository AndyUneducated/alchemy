# Decisions

ADR archive. Each article begins with `## n. Title` + `- **Status**` / `- **Date**` meta-information, and the text uses standard `Context / Options considered / Decision / Consequences` four paragraphs. The new decision is appended to the end, and the status of the replaced entry is changed without deleting the entry. See [`JOURNAL.md`](JOURNAL.md) for daily progress.

## 1. Nudge-grounded SFT as the central issue of the project

- **Status**: accepted
- **Date**: 2026-05-09

### Context

Project establishment is a practical item for fine-tuning the interview portfolio. I originally wanted to "do tool-calling LoRA" - but xLAM / ToolACE / Hammer / Watt-Tool are already open channels, and there is no differentiated signal to the interviewer. I need a project proposition that can only be done on my existing `play/` stack, so that the recurrence threshold is the moat.

### Options considered

|Options|Data Sources|Differentiation|Notes|
|---|---|---|---|
|A. Classic tool-calling LoRA (xLAM/ToolACE)|Public|None|Anyone can reproduce|
|B. Distillation router (GPT-4 → 1.5B)|Public + Synthetic|Medium|A sense of architecture ≥ A sense of fine-tuning, a thin story|
|**C. Nudge-grounded SFT** (selected)|`play/agent_engine` transcript|High|Supervision source is own infra|
|D. Self-distillation (best-of-N + artifact voting) |`play/agent_engine`|High|Local 7B best-of-N inference cost is too heavy|

### Decision

Lock C: Use the closed loop of "model adjustment → engine nudge → model adjustment" under the `require_tool` mechanism as SFT supervision, so that the model after fine-tuning can significantly reduce the nudge-fire rate on its own trajectory. Downstream measurement and deployment follow `play/evals` + `play/agent_engine`, three-piece set: **[engine output data] → [agent_sft training] → [engine use + evals test]**.

### Consequences

- Academic counterpoint: self-improvement/self-correction (STaR/Self-Refine/Reflexion/Self-Rewarding); industrial counterpoint uses "own trajectory → my agent system is more stable", which is narrower but more credible than using external tool-call data sets (xLAM/Watt-Tool/Hammer).
- Coupling `agent_engine` transcript schema (`tool_call` event + `require_tool` step + nudge instruction); schema changes → data script changes, the training framework is not affected. Each triple can be traced back to trace JSON lines.
- The upper limit of data depends on the number of scenario batch runs, and will be supplemented by synthesis when insufficient; v1 supervision only has `require_tool`, and the future failure mode (artifact ACL/vote failure) can linearly expand the pool, and v1 is not preset.

## 2. Select MLX-LM as the training framework

- **Status**: accepted
- **Date**: 2026-05-09

### Context

M4 Pro 48GB (Apple Silicon). The main battlefield of Unsloth is NVIDIA CUDA + Triton; HF PEFT uses MPS and has limited throughput; axolotl is configuration orchestration with strong features but high learning cost. MLX is Apple's official Apple Silicon tensor framework, and [MLX-LM](https://github.com/ml-explore/mlx-lm) is its LM training tool set.

### Options considered

|Options|Apple Silicon Performance|Cost of Learning|Notes|
|---|---|---|---|
|**A. MLX-LM** (selected)|Native optimal|Low (CLI three steps)|Official maintenance is active, the ecosystem is small|
|B. HF PEFT + transformers + MPS|Usable but significantly slow|Medium|Mac is not its main battlefield|
|C. Unsloth|CPU fallback|Medium|GPU path is fast, Mac path is not core|
|D. axolotl|The bottom layer is still PEFT/Unsloth|High|Industrial-level orchestration, which is not used in this project|

### Decision

Choose A. Three-step link: `mlx_lm.lora --train` → `mlx_lm.fuse` → `mlx_lm.convert --quantize` (→ GGUF → `ollama create`). The failure signal surface is clear: the error is reported at that step and is not hidden in the orchestration framework.

### Consequences

- MLX-LM is the de facto standard in the Apple Silicon personal fine-tuning circle (endorsed by Awni Hannun/Simon Willison; HF adds MLX backend since 2025); training/merging/quantification are all in one tool without introducing new tooling.
- HF safetensors are a universal intermediate format, and the cost of future GPU server migration is controllable.
- MLX-LM currently does not directly generate GGUF and requires two-stage conversion; go to `mlx_lm.convert` to go straight out or HF → `llama.cpp convert_hf_to_gguf.py → quantize` and leave Phase 4 for further decision (YAGNI).

## 3. Phase 2 data pipeline design

- **Status**: accepted (pipeline and 1k × 2 model scale-up have been implemented, see [`JOURNAL.md`](JOURNAL.md) 2026-05-10 three milestones)
- **Date**: 2026-05-10

### Context

Run batch mining (failed, nudge, corrected) triples from `agent_engine`, convert them to MLX-LM SFT samples, and cut train/val according to run_id. Deliver an end-to-end regenerative pipeline with minimal code volume, and measure the yield / failure_mode distribution / token length on the real envelope to guide scale-up.

### Options considered

|Axis|Options|Trade-off|
|---|---|---|
|Sample format|F1 (input does not include nudge)/F2 (contains nudge continuation)|F1 teaches the model in one go, consistent with the target semantics of "reduce nudge_fire_rate"|
|train/val cut|by_run_id (per-scenario last 20%)/by_triple random/by_scenario hold-out|by_run_id keep in-dist + prevent trace leakage; scenario hold-out is too strict (only 2 scenarios)|
|scenario range|2 (tool_chain + code_review) / 4 (+ example + panel) | 2 require_tool intensive scenarios have 13 turns/run; expansion 4 introduces low-density noise|
|mining model|Qwen2.5-7B (same base)/Qwen2.5-32B (ceiling)|7B = "train yourself" closed loop; 32B has fewer failures → poor diversity but high quality|
|seed handling|Change agent_engine and add `--seed`/use run_id as naming key|The former is a cross-package change; the latter is zero intrusion and relies on natural sampling|
|Code/data layout|Mixed `data/`/subdirectory `data/triples/` / `scripts/` + `data/{raw,interim,processed}/`|Imitate `eval/` + `eval/baselines/` Parallel most convenient for cross-phase reuse|

### Decision

|item|decision|
|---|---|
|Sample format|**F1 only**|
|train/val cut|**per-scenario end 20% run_id → val**; unique run_ids < 5 when full train|
|scenario range|**Only `tool_chain` + `code_review`**|
|mining model|**Qwen2.5-7B**——via `AGENT_ENGINE_MODEL` env override (1 line), don’t move scenario YAML/Engine API|
|seed handling|**Do not change agent_engine**——run_id as named key + split index, diversity relies on 7B natural sampling|
|Structure|**Imitate `eval/`**: `data/` top-level 4 scripts (mine/extract/format/split) + `data/triples/` installed product|
|nudge text recovery|**Fill in `NUDGE_TEMPLATE`** according to `required_tool` template (consistent with `discussion.py` L141-144), do not enter F1 input|

### Consequences

- 4 scripts (mine / extract / format / split) with single responsibilities can be replaced respectively; the coupling of evals private helper is released in [`§7`](#7-extractor--synthesize--formatter-directly connected-agent_enginedelete-evals-private-import-anti-pattern) (directly connected to the public side of agent_engine).
- Each triple contains `run_id` / `scenario` / `turn_idx` / `failure_mode` / all `context` prefix, which can be traced back to any line of the envelope; extractor / formatter / splitter can all be tested individually (no LLM dependency).
- The risk of F1 only treating `corrected_response.content` as the assistant target (pilot observed that text says X but tool_call event is Y) → Phase 3 is solved by [`§4`](#4-sft-target-schema-with-openai-tool_calls--top-level-tools-field qwen25-native) schema upgrade.
- `wrong_args` bucket link reservation placeholder (synchronized with `metrics/nudge.py` taxonomy), enable condition lock after agent_engine dispatch error path compensation event.

## 4. SFT target schema uses OpenAI `tool_calls` + top-level `tools` field (Qwen2.5 native)

- **Status**: accepted (supersedes §3 Consequences "F1 only use corrected_response.content as assistant target")
- **Date**: 2026-05-10

### Context

§3 v1 formatter uses the synthesized `corrected_response.content` ("Okay, I now call \`retrieve_docs\`:\n\nretrieve_docs("query")") directly as assistant text; the downstream `agent_engine` actually expects the model emit `tool_calls` field through the Ollama function-call API (Qwen2.5 [chat template](https://raw.githubusercontent.com/chujiezheng/chat_templates/main/chat_templates/qwen2.5-instruct.jinja) is rendered into a `<tool_call>{"name":..., "arguments":...}</tool_call>` block). The schema is not aligned → After training, the model may "say it wants to adjust the tool but does not really emit tool_call", and the nudge-fire rate does not fall but rises. The schema must be locked before Phase 3 training.

### Options considered

|Options|Form|Relationship with Qwen2.5 chat template|Relationship with downstream (Ollama → agent_engine `tool_call` event)|
|---|---|---|---|
|A. text-only (v1 status quo)|assistant content = "Okay, I will call X(...) now"|Normal text rendering, no `<tool_call>` block|No tool_call → tool_call event will never be triggered → nudge-fire will not drop|
|**B. OpenAI `tool_calls` JSON-string + top-level `tools`** (optional) | messages.assistant.tool_calls + top-level tools | chat template automatically rendered into native `<tool_call>` block | natively aligned with Ollama function-call parser |
|C. Write literal XML string content|content = `<tool_call>{...}</tool_call>`|Skip chat template schema verification|The form is the same as B, but the dataset schema is not universal|

### Decision

Choose B. Three-layer alignment: ① **Industry practice** - MLX-LM since [PR #995](https://github.com/ml-explore/mlx-examples/pull/995) natively supports `tools` data format ([LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)), xLAM / ToolACE / Hermes-Function-Calling all use this schema; ② **Downstream alignment** - Qwen2.5 chat template `arguments | tojson` renders JSON-string into `<tool_call>` blocks, Ollama Native identification of the parser; ③ **schema single source** - direct import [`agent_engine.artifact._TOOL_DEFS`](../agent_engine/artifact.py) + reuse [`scenario._resolve_tool_defs`](../agent_engine/scenario.py), homologous to the runtime and zero drift.

### Consequences

- [`data/formatter.py`](data/formatter.py) rewrite: assistant message is changed to `{role:"assistant", content:"", tool_calls:[{id, type:"function", function:{name, arguments<JSON-string>}}]}`; F1 sample top level is added with `tools=[...]` (per-scenario, agent perspective, moderator-only tool filtered by role).
- Added args-dict extraction: map the `tool(arg1, arg2)` string captured by [`synthesize._extract_call_template`](data/synthesize.py) into dict in `tool_schema.parameters.properties` order, and then `json.dumps` into a string and stuff arguments. The samples that failed to be extracted (i.e. ~17% of the fallback wrapper) are discarded entirely - this ADR supports the user decision "drop" and no longer uses the weak signal samples of `arguments={}`.
- [`data/synthesize.py`](data/synthesize.py) / [`data/extractor.py`](data/extractor.py) / [`data/split.py`](data/split.py) / [`data/mine_triples.py`](data/mine_triples.py) **Not moving** - `Triple` schema already contains `required_tool` + `instruction` text, which is enough to drive the new formatter; mining envelope and `triples_*_1k.jsonl` are not reborn.
- Data volume is slightly reduced: 7B 766 train + 196 val after drop fallback (vs §3 final delivery 966 + 246), 32B 642 + 160 (vs 842 + 210); still exceeds the early "≥1k training samples" demo magnitude threshold in the README.
- The training side uses `mlx_lm.lora --mask-prompt` (assistant-only loss), which has the same idea as the [TRL PR #5522](https://github.com/huggingface/trl/pull/5522) Qwen2.5 training template - the gradient only acts on the `<tool_call>` block. HF safetensors are still SoT ([`§2`](#2-Training framework selection-mlx-lm)), just change the same jsonl when switching TRL / Unsloth.

## 5. Phase 3 recommends adapter lock BASE configuration; layers/rank sweep + true effect decision postponed to Phase 5

- **Status**: accepted
- **Date**: 2026-05-11

### Context

After Phase 3 6-run sweep ([`runs/sweeps/REPORT.md`](train/runs/sweeps/REPORT.md)) is finished, two observations must fall into ADR - one affects the specific action of "Which adapter to fuse in Phase 4", and the other affects the criterion of "what signal is considered to be truly effective for SFT":

|Observation|Data|
|---|---|
|**`iters` dim is fully saturated**|`iters` ∈ {50, 200, 600} The third gear `train_loss`/`val_loss` all converge to 0.00, `tool_call_emit_rate` / `tool_name_match_rate` / `arg_set_match_rate` / `arg_value_match_rate` all 100%|
|**`lr` dim only 5e-4 degradation**|`lr=1e-5` / `1e-4` full 100%; `lr=5e-4` `train_loss` from 3.65 → end 0.04 / `val_loss` 0.12 / emit 95.4% / name 93.9% / **arg_value 76.0%**|

[`eval_smoke.py`](train/eval_smoke.py) 4 indicators are fast proxy (not end-to-end, parse `<tool_call>` block and compare ground-truth). `--mask-prompt` Let loss only cover the tool_call segment → The schema signal is highly compressible, and the shape has been learned through 50 iter (≈0.25 epoch); the proxy is **saturated** in the schema learning dimension - telling us "learned through", but not "whether to memorize or generalize".

### Decision

Lock down three things:

|entry|content|
|---|---|
|**Recommended adapter**|[`train/runs/sweeps/iters/200/adapters.safetensors`](train/runs/sweeps/iters/200/)——sweep `BASE` configuration (`iters=200` / `lr=1e-4` / `num_layers=16` / `rank=16` / `mask-prompt` on / LoRA on q/k/v/o), Phase 4, that is, from this adapter go `mlx_lm.fuse` → GGUF → Modelfile|
|**fast proxy is only used as a sorting signal within the sweep**|"Full 100%" does not constitute evidence of "SFT taking effect"; similarly, "95%/76%" does not constitute evidence of "serious configuration failure"; true decision signal = **Phase 5 end-to-end [`evals nudge_fire_rate`](../evals/metrics/nudge.py) Compare base 7B / SFT 7B / 32B on the original scenario Three sets** (README §Phase 5 Locked) |
|**`layers` / `rank` sweep is postponed to Phase 3.5**|The triggering condition is single and clear: **Only when the actual test of Phase 5 shows `(SFT 7B − base 7B) < (32B − base 7B) × 0.5` (that is, less than half of the gaps are closed) **Sweep back; otherwise, it is regarded as "v1 demo is of sufficient magnitude" and transferred to Phase 6 for reflection|

### Consequences

- Phase 4 directly reads [`iters/200/`](train/runs/sweeps/iters/200/); there is no need to run "3.D Select the best configuration main run" separately - sweep BASE has = optimal = complete adapter.
- 76% arg_value of `lr=5e-4` is sweep **the only negative single point with informative content** - confirming that 1e-4 is a sweet spot, and 1e-3 does not need to be tried again (must be diverged); if this negative datapoint is referenced later, it will point to a fixed point [`runs/sweeps/lr/0.0005/eval_smoke.json`](train/runs/sweeps/lr/0.0005/eval_smoke.json).
- **eval signal upgrade path** has been planned - the saturation of `eval_smoke` on the schema signal does not mean the failure of the fast proxy concept: after Phase 5 data recirculation + scenario mathematical expansion (v2-B / v2-D candidate), `eval_smoke` can be switched to "new scenario / unseen tool name combination" and become a differentiating signal again.

## 6. Phase 4 quantization level lock Q4_K_M + Modelfile 1:1 replica qwen2.5:7b template

- **Status**: superseded by §10 (qwen3.5:9b base + RENDERER/PARSER directive; GGUF deploy path is known broken)
- **Date**: 2026-05-11

### Context

Phase 4 puts the selected `iters/200` adapter fuse into Qwen2.5-7B, converts it to GGUF, and registers it as Ollama tag. The "model" dimension of the Phase 5 retest is frozen only after two engineering decisions (**Quantification Level** Q4/Q5/Q8, **Modelfile TEMPLATE Source** 1:1 Replica vs. Self-Written vs. Inference) are locked.

### Options considered

|Axis|Options|Size|Relation to Phase 5 baseline|
|---|---|---|---|
|Quantization|**Q4_K_M** (selected)|~4.4 GB (measured 4460 MiB / 4.91 BPW) |Same quantization level as Ollama built-in `qwen2.5:7b` → Fair comparison|
|Quantization|Q5_K_M|~5.5 GB|Slightly better quality ~0.5%, but worse than baseline quantization → contaminating SFT signal attribution|
|Quantization|Q8_0|~8 GB|Quality is nearly lossless but baseline is not Q8 → Not comparable|
|Modelfile|**1:1 copy qwen2.5:7b's TEMPLATE + SYSTEM** (optional)|—|Ollama function call parser's recognition of `<tool_call>` block completely relies on chat template; template deviates by 1 word → tool_call event is not triggered → the entire SFT signal is invalid|
|Modelfile|Self-written jinja (refer to Qwen2.5 official [chat_templates repository](https://raw.githubusercontent.com/chujiezheng/chat_templates/main/chat_templates/qwen2.5-instruct.jinja))|—|Ollama’s Go template subset ≠ jinja2, migration cost + double error surface|
|Modelfile|Don’t write TEMPLATE, let Ollama infer from GGUF metadata|—|The fallback template inferred by Ollama does not contain native `<tool_call>` block rendering logic, and the tool API directly misfires (for known pitfalls, see [ollama/ollama#7560](https://github.com/ollama/ollama/issues/7560))|

### Decision

|item|decision|evidence|
|---|---|---|
|**Quantization level**|**Q4_K_M**|Same quantization level as baseline, Phase 5 three sets of comparisons (base 7B / SFT 7B / 32B) quantization axis alignment|
|**Modelfile**|**1:1 fork** TEMPLATE + SYSTEM block of `ollama show --modelfile qwen2.5:7b`; only change the `FROM` line to point to the local q4 gguf; **do not add** `PARAMETER stop` and other lines (exactly the same as baseline) |baseline's own modelfile does not contain an explicit PARAMETER line; the stop token is provided by GGUF metadata → Forking 1:1 is safer than "actively writing it again" |
|**fuse path**|`mlx_lm.fuse --dequantize` (4-bit MLX → fp16) → `convert_hf_to_gguf.py` (fp16 → F16 GGUF) → `llama-quantize Q4_K_M`|4-bit base fuse must `--dequantize` (LoRA The quantization grid cannot be added, [mlx-lm#1071](https://github.com/ml-explore/mlx-lm/issues/1071)); `--export-gguf` direct export path tokenizer metadata is not as compatible as llama.cpp transfer ([Awni 2024](https://github.com/ml-explore/mlx-examples/discussions/1057))|

### Consequences

- Verification: Passed both levels of smoke test. Step 5A `/api/chat` returns parsed `tool_calls` (including Chinese args); Step 5B `agent_engine` runs the full 8 step `tool_chain` and captures **10 tool_call events** covering the entire tool set - the SFT schema signal does not collapse under Q4_K_M quantization.
- Deployment side "rebirth guide" fully scripted: [`deploy/build.sh`](deploy/build.sh) + [`deploy/deploy.sh`](deploy/deploy.sh) + [`deploy/smoke_test.py`](deploy/smoke_test.py), new machine from 0 to `agent-sft-qwen` tag ≤ 10 min.
- Subsequent slideability: Q4_K_M If Phase 5 shows a crushed SFT signal, `bash deploy/build.sh --force QUANT=Q5_K_M` will be covered by one line. llama.cpp was introduced in [`§2`](#2-Training Framework Selection-mlx-lm) The end of Consequences has been laid out (YAGNI leaves the mouth), and there is no need to update §2 Status.

## 7. extractor / synthesize / formatter directly connected to agent_engine (remove evals private import anti-pattern)

- **Status**: accepted (fixes §3 Consequences "Coupling to evals.metrics.nudge private helper")
- **Date**: 2026-05-11

### Context

§3 When implemented, `extractor.py` / `synthesize.py` / `formatter.py` uses the `sys.path.insert` anti-pattern to steal 4 private helpers from `evals.metrics.nudge` (`_attempt_called_required` / `_resolve_who_to_agents` / `_split_attempts` / `_split_frontmatter`, public `classify_failure_mode` + `derive_expected_turns` do not count). These 4 private functions are the mirror of evals reverse engineering agent_engine `Discussion._expand_steps + _resolve_who + ToolTracer/ArtifactStore.event` schema - the true source of schema is in agent_engine, reverse engineering is in evals, agent_sft consumes the reverse product of evals - **Double-level indirect dependency makes schema changes like thunder**.

[`agent_engine §13`](../agent_engine/DECISIONS.md) Implemented at the same time: the transcript / scenario interpretation rights are returned to agent_engine, and a new typed view `Result.tool_calls() / .turns() / .find_finalize_decision()` + `TurnView.attempts() / .start_offset` + `Scenario.expanded_turns()` + `ExpandedTurn` dataclass. This ADR is the agent_sft counterpart.

### Options considered

| Items | Practices | Trade-offs |
|---|---|---|
| A. Current situation | Continue sys.path.insert + 4 private imports | schema changes in three chaining places; agent_sft single test is sensitive to evals internal representation |
| **B. Direct connection to agent_engine public side** (select) | `from agent_engine import Result, Scenario, TurnView, ExpandedTurn`; reserved `from evals.metrics.nudge import classify_failure_mode` (evals legal public side) | 0 private side cross-project import; signal flow agent_engine schema → agent_sft directly consumes one layer; evals.metrics.nudge only serves as "failure" mode taxonomy owner" was quoted |
| C. Also mention `classify_failure_mode` to agent_engine | There is no need to import evals | "missed / wrong_tool" is a semantic judgment from the evals/sft perspective (not the dispatch truth that agent_engine cares about). Raising it will pollute the agent_engine concern boundary; deferred to PR-3 if needed |

### Decision

**B**：

| Modules | Changes |
|---|---|
| [`data/extractor.py`] / [`data/synthesize.py`] | Delete 5 private imports (`_split_frontmatter` / `_resolve_who_to_agents` / `_split_attempts` / `derive_expected_turns` / `_attempt_called_required`); change `from agent_engine import ExpandedTurn, Result, Scenario, TurnView`; `extract_triples` iterates directly with `Scenario.expanded_turns()` + `Result.turns()` + `TurnView.attempts()` + `.start_offset`; `_attempt_called_required` internalized to 5 lines helper (synthesize shared import) |
| [`data/formatter.py`] | Delete `_split_frontmatter` private import; change to `Scenario.from_yaml(p).meta` (schema verification has the same origin as agent_engine); delete `yaml` import |
| Public Discipline | The only retained cross-project import is `from evals.metrics.nudge import classify_failure_mode` (evals legal public surface, `FAILURE_MODES` taxonomy owner) |
| shim compatible | `_index_steps_by_turn` / `_split_turns_indexed` degraded to 1-2 lines shim to make old tests zero-modified (§8 retired) |

### Consequences

| Impact | Results |
|---|---|
| schema single source | agent_engine changes the `Result.transcript` / `Scenario` field, agent_sft only needs to change one place (view layer) |
| import boundary | agent_sft and evals have only 1 public function dependency (`classify_failure_mode`), directly connected to agent_engine; vs PR-2 top 4 private + 2 public = 6 cross-project import |
| Test | 89 all green, no modifications to the old test (shim life extension) |
| §3 Correction | §3 Consequences "There is coupling to evals private helper" is no longer true; the pipeline four script structure and division of responsibilities remain unchanged |
| Subsequent slideability | `classify_failure_mode` mentioned that agent_engine is another PR option (semantic neutrality is subject to discussion), this plan does not do it |

## 8. Transcript schema typed upgrade + envelope `usage` synchronous consumption (the agent_sft counterpart of agent_engine §14)

- **Status**: accepted (follows [`agent_engine §14`](../agent_engine/DECISIONS.md); extends §7's "direct agent_engine" boundary to typed entry)
- **Date**: 2026-05-11

### Context

§7 Directly connect the three scripts to the public side of agent_engine, but the internal transcript is still `list[dict]`. agent_engine §14 At the same time, upgrade transcript to `list[TranscriptEntry]` typed union (6 frozen dataclasses, `SpeakerEntry` mandatory `type="speaker"`) + `Result.usage: list[TokenUsage]` + `Result.from_dict` Rigorous. The three scripts do not support typed access and are invalid: `.get("tool")` of `_attempt_called_required` does not exist on the dataclass; `extract_triples` / `synthesize` uses `entry["content"]` + `entry.get("speaker")` for sniffing. `isinstance(e, SpeakerEntry)` dispatch; sniff for `"speaker" in entry` of `formatter._render_recent_context` is ambiguous after §14; 500 history mined envelope JSON missing `type:"speaker"` / `usage:[]` let `Result.from_dict` `KeyError`.

### Options considered

| Items | Practices | Trade-offs |
|---|---|---|
| A. Agent_sft continues dict sniffing internally | Cross-agent_engine §14 compatible | The advantages of typed union are invalid; schema changes must be pursued in both places |
| **B. Directly eat typed entry** (select) | `extractor/synthesize` internal `isinstance(e, SpeakerEntry/...)` is dispatched; `formatter` placed dict shape remains unchanged (`metadata["context"]` is the JSON serialized dict), but the speaker determines `entry.get("type") == "speaker"` (§14 has been forced to be written) | Typed dispatch + JSON forms all rely on the `type` field as a single source; dict sniffing is completely eliminated |
| C. Write your own typed view for agent_sft | Dual SoT, contradicting the §7 "schema single source" decision | Rejected |

### Decision

**B**——extractor/synthesize uses typed dispatch; formatter uses `type` field judgment; shim cleanup + historical envelope one-time migration:

| Moving point | Practice |
|---|---|
| extractor / synthesize | formal parameter replacement typed union; `_attempt_called_required` uses `e.tool / e.caller`; `isinstance(e, SpeakerEntry)` + `e.content` in `extract_triples` for direct access; `Triple.context: list[TranscriptEntry]` |
| extractor shim deleted | `_split_turns_indexed` / `_index_steps_by_turn` left in §7 are retired (`Result.turns()[i].start_offset` has been directly given to the global offset) + 2 shims are deleted together with single test |
| `formatter._render_recent_context` | `entry.get("type") == "speaker"` dispatch; the placement metadata is still dict (typed → dict is completed by `engine.py` `dataclasses.asdict`) |
| History envelope × 500 | One-time migration script injection `type:"speaker"` + `usage: []`, consistent with agent_engine §14 forward-only |
| Cross-project import addition | `from agent_engine import TokenUsage, ArtifactEventEntry, SpeakerEntry, ToolCallEntry, TranscriptEntry` |

### Consequences

| Impact | Results |
|---|---|
| schema single source | agent_engine §14 Change entry / add field → agent_sft Just change one place (typed dispatch will be automatically received) |
| Test | 87 all green (89 → -2 shim single test deleted in §7, equivalent coverage moved to agent_engine) |
| Historical mined data | 500 envelope one-time migration, no long-term shim; subsequent re-run automatic production §14 schema |
| §7 Relationship | §7 Set up the "directly connected public face" at the dict boundary; §8 Push the boundary inward to the typed entry, and the same principle progresses |
| Subsequent slideability | `Result.usage` is not consumed in current mining, and can be directly aggregated if cost filtering is needed in the future |

## 9. v1 closed: Phase 5 digital three threshold hit + v2/v3 candidate trade-off

- **Status**: accepted (v1 closing)
- **Date**: 2026-05-13

### Context

Phase 5.A ran 3 model × 10 seed × 4 tasks = 120 runs (119 successful, 1 excluded). Aggregate [`eval/baselines/phase5-3model-comparison.md`](eval/baselines/phase5-3model-comparison.md). v1 closed the case with two things happening at the same time: ① [`§1`](#1-nudge-grounded-sft-as the central question of the project) The central question is answered numerically; ② v2/v3 candidate 7 items are activated/delisted/sustained.

### Options considered

Preset three numbers → decision path:

|Options|Number Features|Corresponding Path|
|---|---|---|
|A. SFT **Significantly effective**|All three thresholds passed: nudge gap ≥50%, BFCL `arg_value_match` regression ≤5%, MMLU ≤3%|v2-B / v2-C can be started at any time; v3-A is persistent; v3-B can be started|
|B. SFT **Partially valid**|nudge meets the standard but BFCL/MMLU exceeds the threshold|v2-C takes priority; v3-A/v3-B is delisted|
|C. SFT **Invalid**|nudge gap < 50%|v1 terminated; record negative finding; return to data layer|

### Decision

**A hit. **Three threshold measured numbers ([`eval/baselines/phase5-3model-comparison.md`](eval/baselines/phase5-3model-comparison.md), n=10 except 7B nudge n=9):

|Dimension|base 7B|SFT 7B|32B|Judgment|
|---|---|---|---|---|
|`nudge_fire_rate` (lower is better)|0.7389 ± 0.1112|**0.6450 ± 0.0369**|0.5750 ± 0.0540|gap close **57.3%** ≥ 50% ✅|
|`bfcl_slice.arg_value_match` (higher is better)|0.9683|**0.9567**|0.9783|Regression **1.16%** ≤ 5% ✅|
|`mmlu_slice.accuracy` (higher is better)|0.7188|**0.6979**|0.8021|Regression **2.09%** ≤ 3% ✅|

Six items of second-order evidence (task_success SFT go-ahead 32B, tool_call_set_f1 -27%, trajectory_match -31%, missed→wrong_tool conversion, panel scenario reverse regression, retrieve_docs 100% nudge required) have been written into [`README.md` §"Phase 5 Number List"](README.md); Not within threshold determination, but v2-B / v2-C candidate input.

### Consequences

**Center Questions and Answers** ([`§1`](#1-nudge-grounded-sft-as the project center question)): **Yes. ** The nudge gap is closed at 57.3% (close to the 32B ceiling 60% mark), BFCL is back at 1.16%, and MMLU is back at 2.09%. But the conditions are clear: SFT has learned the schema signal ([`§4`](#4-sft-target-schema-use-openai-tool_calls--top-tools-field qwen25-native)) + "know the adjustment" (missed ↓), and has not fully learned the "correct adjustment" (wrong_tool ↑) + "does not adjust the redundant" (trajectory deviation ↑) - this is the ceiling of v1's single signal of supervision `require_tool` only.

**§5 Trigger condition**: gap closed 57.3% **Not triggered** §5's "<50% will only retrace layers/rank" condition → §5 status remains accepted, and the recommended adapter is still [`train/runs/sweeps/iters/200/adapters.safetensors`](train/runs/sweeps/iters/200/).

**v2/v3 candidate list update** (mirror to [`README.md`](README.md)):

|candidate|new status|basis|
|---|---|---|
|v2-A DPO|⏸ **tentative**|wrong_tool is a classification issue not a preference issue, DPO will not solve it head-on|
|v2-B on-policy iteration SFT|✅ **Startable**|"trajectory deviation" + "wrong_tool ↑" The root cause is that the training set lacks SFT self-produced trajectory; on-policy recirculation directly treats symptoms|
|v2-C failure mode taxonomy + hard sample mining|✅ **Startup**|The three axes have exposed 4 dead ends (wrong_tool ↑ / panel reverse / retrieve_docs 100% / tool_call_set_f1 degradation), all are hard sample entrances|
|v3-A 14B upgrade|⏸ **Pursue**|7B exceeds 32B in task_success and is not saturated; let v2 drain out first|
|v3-B public HF artifact|✅ **Startable**|All three thresholds passed + "Back of the coin" narrative = Model Card has been formed, you can ship adapter first|
|v3-C technical report/blog|⏸ **v3-B before**|Depends on v3-B + v2 progress|
|v3-D multi-supervision signal superset|🚫 **Delisting**|The bottleneck of v1 is supervision **quality** (panel reverse / retrieve_docs 100%) rather than quantity; v2-C naturally covers|

**Project patch status**: [`eval/run_baseline.py`](eval/run_baseline.py) 2 (`sys.executable` + `AGENT_ENGINE_MODEL` env injection) + [`evals/models/agent_engine_run.py`](../evals/models/agent_engine_run.py) 1 (`AGENT_ENGINE_RUN_TIMEOUT` env override, zero side effects), all committed together with this ADR - they are prerequisites run by Phase 5, not QoL.

**Cross-project followup** (returned to corresponding backlog): ① evals harness should isolate abnormal LLM output (`tool=cast_vote(...)` illegal kwarg → handler `TypeError` collapse caller, 1/120 loss); ② agent_engine artifact handler should reject unknown kwarg and return `{ok:false}` event. See [`README.md` §"Interview narrative script (v1 final version)"](README.md) for interview sentence anchors.

## 10. v1.5: qwen3.5:9b base retraining + 9-run minimalist retest + GGUF deploy suspension

- **Status**: superseded by §11 (clean-data + bf16 retrain + held-out story eval)
- **Date**: 2026-05-25

### Context

The default base of the warehouse is switched from qwen2.5 to qwen3.x (See root `a5ad0f9` commit + Phase 1/2/3 envelope/RAG upgrade); agent_sft v1 (Qwen2.5-7B base + 1k data + 120-run baseline) is out of touch with the new stack - the CLI defaults to the deleted model tag of v1, and the evals harness cannot get the SFT adapter for comparison when calling `qwen3.5:9b`. **Choose to retrain rather than maintain two sets**.

According to the data strategy, according to the plan at the beginning, 7 run_id × 2 scenario, a single run of `qwen3.5:9b` produced 500+100; in fact, `qwen3.5:9b` is much stronger than `qwen2.5:7b` - 14 envelopes only produce 49 nudge triples (the same envelope of v1 produces 2000+). **Change to merge historical v1 7B/32B triples + new 9B triples, assign disjoint run_id offset to three sources to ensure scenario grouping integrity**, and get to train 588 / val 155 (close to the plan 500/100 goal).

### Options considered

|Options|GGUF deploy|Training data|Notes|
|---|---|---|---|
|A. Strictly follow the plan: pure 9B mine 500+100|Follow the §6 path Q4_K_M GGUF + 1:1 replica modelfile|Pure 9B|9B strong → 49 nudge triples, the amount of data is far from enough|
|B. 9B mine + history 7B/32B splicing|Same as above|Hybrid source 588/155|Data can be collected; GGUF path exposes the following hybrid compatibility bug|
|**C. (Select) Same as B data + RENDERER/PARSER directive Modelfile + GGUF deploy downgrade placeholder**|fp16 fused mlx archive + ollama tag `agent-sft-qwen-3` temporarily FROM base `qwen3.5:9b`|Mixed source 588/155|Acknowledge the mlx→GGUF Qwen3.5 hybrid conversion defect, decouple the "training effect" from the "deploy form"|

### Decision

**Adopt C. **

- **Base**: `mlx-community/Qwen3.5-9B-4bit` (HF: `mlx-community/Qwen3.5-9B-Instruct-4bit` does not exist, the actual available one is non-Instruct 4bit quantization)
- **Data**: Training 588 / Testing 155 triples, mixed sources `v1_7b` + `v1_32b` + `v15_9b`, three sources run_id allocation disjoint offset guaranteed scenario grouping ([`data/triples/train_qwen3.jsonl`](data/triples/train_qwen3.jsonl))
- **Training hyperparameters**: `iters=600` / `batch_size=1` / `num_layers=4` / `lr=1e-4` / `--grad-checkpoint` / `--clear-cache-threshold 1` / `max_seq_length=1500`; OOM-driven shortened parameters - 9B + Qwen3.5 hybrid (attention+SSM) Even 4bit is much tighter than 7B on M4 Pro 48GB, batch=1 + 4 layers LoRA It is the only combination that can run steadily ([`train/runs/main_qwen3/train_metrics.json`](train/runs/main_qwen3/train_metrics.json)).
- **Modelfile**: From the Go-template DSL of v1 ~50 lines of TEMPLATE, to the `TEMPLATE {{ .Prompt }}` + `RENDERER qwen3.5` + `PARSER qwen3.5` three-line directive of ollama 0.20+; this is the short-hand parsed `qwen3.5:9b` chat & tool_call that has been copied 1:1 within ollama.
- **GGUF deploy pending**: mlx_lm.fuse --dequantize output fp16 safetensors → `convert_hf_to_gguf.py` → `llama-quantize` Q4_K_M path, for Qwen3.5 hybrid architecture (with SSM operator / `ssm_alpha`/`ssm_beta`/`ssm_conv1d` tensors) weight reconstruction is incorrect: F16 GGUF and Q4 GGUF in ollama Both terminals output garbled characters ("ãjiaguang_MMjv slip...\uF!安 Cornas $yn"), but the fp16 mlx fused directory is directly inferred using `mlx_lm.generate`. **Conclusion: Training successful, deployment path broken**. Modelfile was changed to `FROM qwen3.5:9b` placeholder (inherited base inference capability, PARAMETER is used), and the broken version was backed up to [`deploy/Modelfile.gguf-broken`](deploy/Modelfile.gguf-broken), and the fused fp16 model was left in [`deploy/build/fused-mlx-fp16/`](deploy/build/fused-mlx-fp16/) as the input artifact to be repaired.
- **Evaluation**: plan §S9 9 runs minimalist retest (3 model × 3 seed × 1 task `nudge_fire_rate`) instead of v1 120 runs baseline; the evaluation falls on [`eval/baselines/qwen3_phase3/index.jsonl`](eval/baselines/qwen3_phase3/index.jsonl) (copied from the last 9 lines of evals/runs/index.jsonl). For specific numbers, see the "Digital Snapshot" at the end.
- **Metrics that truly reflect the effect of SFT training**: `train/eval_smoke.py` runs 4bit base + LoRA adapter direct inference (bypassing ollama / GGUF) on 50 val sample, see the numbers at the end. **This path can objectively judge whether SFT training has learned anything** - 9 runs eval is degraded into repeated data due to `agent-sft-qwen-3 ≡ base qwen3.5:9b` and does not reflect the SFT signal.

### Consequences

**§6 supersede**: Q4_K_M + Modelfile 1:1 fork qwen2.5:7b overall obsolete - base changed / TEMPLATE changed directive / GGUF path broken. §6 Status has been changed to `superseded by §10`.

**§9 v1 three thresholds (57.3% gap closure, etc.) are retained as historical snapshots** and are no longer aligned - new base + data mixing source + training hyperparameters fully changed.

**Eval_smoke parser compatibility correction**: [`train/eval_smoke.py`](train/eval_smoke.py) Add Qwen3.5 native tool-call rendering shape recognition (XML nested `<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>`). v1 (Qwen2.5) uses JSON to embed `<tool_call>{...}</tool_call>`; without patch emit_rate=0% misjudges training failure.

**Formatter `arguments` type correction**: [`data/formatter.py`](data/formatter.py) Change the `arguments` of OpenAI tool_calls from JSON string to Python dict - the `.arguments | items()` jinja filter of Qwen3.5 chat template requires mapping, and passing the string will report `TypeError: Can only get item pairs from a mapping`. v1 (Qwen2.5) `.arguments | tojson` in the template does not require a type, so v1 does not expose this bug.

**GGUF deploy fix backlog** (not in v1.5 scope):

|Options|Path|Difficulty|
|---|---|---|
|A. Wait for mlx_lm.fuse to fix hybrid dequantize|Upstream patch|Low workload but wait|
|B. Change to transformers + PEFT path|HF bf16 base + manual LoRA named mapping|Medium workload, controllable|
|C. etc. ollama 0.21+ natively supports mlx adapter|upstream route|unknown ETA|
|D. Self-written mlx_lm /api/generate wrapper instead of ollama|Local implementation|Long workload, but deploy has the strongest independence|

The short-term placeholder (FROM base) allows the evals harness / agent_engine to use the `ollama:agent-sft-qwen-3` tag without any awareness. After the repair is implemented, change the deploy.sh line to Modelfile and create again to take effect.

**Interview Narrative Availability**: The v1 number (57% gap closure / 0.739→0.645) is still the main line of the portfolio (§9 + README §Lessons learned); v1.5 is "new stack adaptation + training verification", which does not produce new numbers for the portfolio, but is engineering infrastructure maintenance.

### Digital Snapshot

**Training** ([`train/runs/main_qwen3/train_metrics.json`](train/runs/main_qwen3/train_metrics.json)):

|item|value|
|---|---|
|iters / batch / num_layers / lr|600 / 1 / 4 / 1e-4|
|wall|4368 s ≈ 73 min|
|train_loss_first / train_loss_last|0.225 / 0.000|
|val_loss_last / val_loss_min|0.000 / 0.000|
|peak GPU mem|8.15 GB|
|nan_seen / diverged / returncode|false / false / 0|

train loss dropped to 0.010 at iter 40, 0.001 at iter 60, and 0.000 at iter 80+ in the long term (log excerpt), consistent with the expectation of "sufficient iter to converge" under [`§5`](#5-phase-3-recommended-adapter-lock-base-config-layersrank-sweep-deferred-to-phase-5) (v1 sweep selected lr=1e-4 + num-layers); val_loss=0 is not perfect but 4bit quantization + short answer token + `mask_prompt=true` co-produce pseudo-0 (actual eval_smoke shows 64% arg_value accuracy).

**eval_smoke**（n=50, mlx_lm 4bit + LoRA adapter, [`train/runs/main_qwen3/eval_smoke.json`](train/runs/main_qwen3/eval_smoke.json)）：

|Indicator|Value|
|---|---|
|`tool_call_emit_rate`|0.86|
|`tool_name_match_rate`|0.64|
|`arg_set_match_rate`|0.64|
|`arg_value_match_rate`|0.64|
|by_tool n|append_section 25 / retrieve_docs 19 / cast_vote 6|

**9-run nudge_fire_rate** (3 model × 3 seed × 1 task, [`eval/baselines/qwen3_phase3/index.jsonl`](eval/baselines/qwen3_phase3/index.jsonl)). Actual 6 successful + 3 failed (27B `subprocess.TimeoutExpired` after 600s × 3 seeds - qwen3.6:27b single multi-turn scenario LM wall on M4 Pro 48GB has exceeded default `AGENT_ENGINE_RUN_TIMEOUT=600s`; plan §S9 fault-tolerant path "single run failure does not affect other" coverage, skip re-run - extending the timeout to 1800s requires additional ~5h to run, not in v1.5 budget):

|model|seed 0|seed 1|seed 2|mean|
|---|---|---|---|---|
|`qwen3.5:9b`|0.30|0.40|0.50|**0.400**|
|`agent-sft-qwen-3` (≡ base placeholder)|0.35|0.60|0.55|**0.500**|
|`qwen3.6:27b`|failed|failed|failed|n/a|

`agent-sft-qwen-3` vs `qwen3.5:9b` The numerical difference (0.500 vs 0.400) is not an SFT signal - both FROM the same base GGUF and the same PARAMETER; the difference comes from ollama nondeterminism (temperature=1, top_k=20, top_p=0.95 default sampling) + different spec hash leading to seed-derived state divergence. Think of it as "the lower limit of the variance of two runs of the same model", which reads: 1-task × 7 scenario × ~20 require_tool case = 20 sample size, the true variance of nudge_fire_rate is more than ±0.1. **The real signal of SFT is 64% arg_value_match** of eval_smoke (bypassing ollama / GGUF and direct mlx inference LoRA), that is the credible indicator of this retraining.

**Value of this batch of evaluation**:
1. Pipeline running - `run_baseline.py` + `evals` framework + `agent-sft-qwen-3` ollama tag three-piece set works end-to-end in the v1.5 stack (the whole library switch after v1 did not break the link).
2. 7 scenarios × require_tool ≈ 20 case/runs, 4 counting (example/panel/tool_chain/code_review), 3 nan (brainstorm/debate/roundtable does not require_tool step) - consistent with the data form of v1 `eval/baselines/phase5-*`, and will be dropped-in directly when redoing the full baseline in the future.
3. 27B timeout exposure The timeout configuration of the evals harness should be distinguished by model - remember to the evals followup mentioned in the root [`AGENTS.md`](../../AGENTS.md) (processed from v2; this v1.5 will not be repaired).

## 11. v1.6: clean-data + bf16 retraining + held-out story eval (GGUF is still blocked)

- **Status**: accepted
- **Date**: 2026-05-26

### Context

Although v1.5 has passed "training + evaluation + placeholder deploy", there are two hard gaps:

|Issue|v1.5 Status|Impact|
|---|---|---|
|Dataset quality|`train_qwen3/val_qwen3` comes from mixed sources, with many repetitions and a small amount of train/val overlap|The story is not "clean enough for the final version"|
|GGUF deploy|4bit base + `mlx_lm.fuse --dequantize` path output is garbled|`agent-sft-qwen-3` can only be base placeholder|

Therefore, the goals of v1.6 are divided into three parts: ① Clean the data boundaries first; ② Use bf16 base retraining to verify that "it is not the fault of 4bit dequantize"; ③ Use in-distribution vs held-out evaluation caliber to tell the generalization story.

### Options considered

|Options|Data processing|Training base|Evaluation caliber|Expectation|
|---|---|---|---|---|
|A. Directly use v1.5 data to continue training|No cleaning|4bit|Single nudge_fire_rate|Fast, but the "final version" is not convincing|
|B. clean data + 4bit retraining|remove duplication/remove leakage|4bit|in-dist vs held-out|data story gets better, but deploy risk remains|
|**C. (optional) clean data + bf16 retraining**|remove duplication/removal of leaks + scene boundary|bf16|in-dist vs held-out|simultaneously verify the two main lines of data and deploy|

### Decision

Adopt C, the specific implementation is as follows:

|Module|Decision|Results|
|---|---|---|
|Data cleaning|Rebuild the clean set from [`data/triples/triples_qwen3_merged.jsonl`](data/triples/triples_qwen3_merged.jsonl); keep only `tool_chain`/`code_review` for training; remove duplication according to supervision key; split according to scenario+run_id; output [`triples_qwen3_clean.jsonl`](data/triples/triples_qwen3_clean.jsonl), [`train_qwen3_clean.jsonl`](data/triples/train_qwen3_clean.jsonl), [`val_qwen3_clean.jsonl`](data/triples/val_qwen3_clean.jsonl), [`qwen3_clean_report.json`](data/triples/qwen3_clean_report.json)|clean=1547, train raw=1276, val raw=271, train/val overlap=0|
|Template completion|There are 175/44 instructions in `retrieve_docs` in clean raw. There are missing literals to call the template. Press "Minimum resolvable" to fill in `retrieve_docs(query=\"...\")` and then formatter|formatted train=1276, val=271, drop=0, `arguments` 100% dict|
|Training base|base cut [`mlx-community/Qwen3.5-9B-bf16`](https://huggingface.co/mlx-community/Qwen3.5-9B-bf16)|bypass 4bit→fp16 dequantize path|
|smoke/probe|smoke In the first round of `max_seq=1000`, train tokens=0 + NaN appeared; after changing to `max_seq=1500`, it became stable. 40-iter probe is stable in `num_layers=4`|Confirm that the main trainer is available `batch=1,num_layers=4,max_seq=1500,grad_checkpoint,clear_cache=1`|
|Main training|[`train/runs/main_qwen3_bf16_clean`](train/runs/main_qwen3_bf16_clean)|600 iters, rc=0, nan=false, train 0.247→0.000, val_last=0.000, wall=4115s, peak mem≈21.36GB|
|mlx evaluation|[`train/runs/main_qwen3_bf16_clean/eval_smoke.json`](train/runs/main_qwen3_bf16_clean/eval_smoke.json)|n=50: emit=0.56, arg_value=0.36 (significantly lower than v1.5's 0.64)|
|GGUF deploy|[`deploy/build.sh`](deploy/build.sh) adds bf16 branch: do not add `--dequantize` when the base name contains `bf16`|The build is successful but **F16/Q4 is still garbled** (both `testf16` and `testq4` are reproduced)|
|Review Story|Output [`eval/baselines/qwen3_bf16_clean/index.jsonl`](eval/baselines/qwen3_bf16_clean/index.jsonl) and [`story_report.md`](eval/baselines/qwen3_bf16_clean/story_report.md)|in-dist/held-out layering completed; currently `agent-sft-qwen-3` is still placeholder|

### Consequences

|Conclusion|Implication|
|---|---|
|GGUF garbled characters still recur in the bf16 path|The problem is not only in 4bit dequantize, but more likely in the runtime compatibility layer of `convert_hf_to_gguf.py`/ollama for Qwen3.5 hybrid architecture|
|`agent-sft-qwen-3` continues to retain placeholder (FROM qwen3.5:9b) | Evaluation and engineering links can be run, but their indicators cannot be interpreted as new adapter effects |
|clean data has made up for the "quality debt", but the model effect has become worse (0.64→0.36)|The next step should be to prioritize hard-sample mining / `cast_vote` for enhancement instead of blindly increasing iters|
|The in-dist and held-out calibers are officially formed|Follow-up ADR can directly compare "training distribution income vs. generalization cost" along this caliber|

Subsequent technical routes (in order of priority):

|Priority|Route|Description|
|---|---|---|
|P0|`mlx_lm.server` Direct-connect deployment|Bypass the GGUF runtime and deliver the "available and true SFT" version first|
|P1|Transformers+PEFT merge and then GGUF|Verify whether it is a problem specific to the mlx fuse path|
|P2|Waiting for llama.cpp / ollama upstream repair Qwen3.5 hybrid|The lowest cost but uncontrollable by ETA|

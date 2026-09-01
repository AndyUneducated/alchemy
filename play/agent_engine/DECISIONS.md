# Decisions

ADR (Architecture Decision Record) filing. Each article begins with `## n. Title`, followed by `- **Status**` + `- **Date**` meta-information; the text follows the `Context / Options considered / Decision / Industry Spectrum / Engineering Dimension Assessment` paragraph. **New decisions are appended to the end, and replaced items are changed to Status; old items are not deleted**. Daily progress by milestones see [`JOURNAL.md`](JOURNAL.md).

## 1. Phase-driven scenario configuration

- **Status**: partially superseded by §9 (top-level `moderator:` / `members:` blocks, `phases:` three-part structure are replaced by `agents:` + flat `steps:`; the overall form of YAML frontmatter + MD body and schema startup verification mechanism are retained)
- **Date**: 2026-04-14

### Context

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

### Options considered

- **A. JSON configuration**: machine-friendly, but not suitable for writing large Chinese prompts
- **B. YAML single file**: suitable for structured fields, but prompt is not elegant when writing markdown
- **C. Markdown + YAML frontmatter** (select): YAML contains structure fields, MD body contains topics; prompt can contain markdown, and a single file is a scene
- **D. Python DSL / Code as configuration** (AutoGen style): The most expressive but raises the threshold, not suitable for "scenario libraries that you want to show to others"

### Decision

YAML frontmatter defines `members` / optional `moderator` / `phases`; MD body is injected into history as topic. Initially, 4 scenarios are provided covering (moderator / no moderator) × (open / goal-oriented) 2×2 matrix, and schema verification is performed during the startup period.

**Industry Spectrum**: CrewAI uses YAML to configure agents/tasks which is mainstream; AutoGen is more Python-centric; LangGraph uses Python graph DSL. The specific option "MD + YAML frontmatter" is standard in the Jekyll / Hugo / Obsidian world, less common in the agent framework - correct but not mainstream.

### Engineering dimension assessment

|Dimensions|Evaluation|
|---|---|
|Cohesion|High - engine, scenario, and participant definitions all perform their own duties|
|Coupling|Significantly reduced - the engine only relies on the "phases" abstraction|
|Observability/Auditability|Medium - stdout has phase/round printing, unstructured|
|LLM Uncertainty Tolerance|Medium - author input is verified at startup; LLM runtime output is unconstrained|
|Backwards compatible/evolution friendly|Project starting point, no old behaviors need to be compatible|
|Learning curve|Medium - the author needs to learn YAML frontmatter + `members/moderator/phases` mental model|
|Testability|High—scenes as fixtures, zero code changes for new scenes|

### Known continuous trade-off

New features (memory strategies, tool switches, voting) all add fields to the YAML schema, and the schema will continue to expand. Currently, all fields are additive and there are no discarded items, but you need to continue to be vigilant.

## 2. Per-agent message projection

- **Status**: accepted (shared transcript + per-agent projection is the basis for subsequent §5/§6/§9)
- **Date**: 2026-04-15

### Context

In the first version, the shared history is fed to each agent as it is, and the system prompt is treated as a user message in the history. question:

1. Agent cannot distinguish between "what I said" and "what others said" - they are all user roles
2. System prompt priority distortion
3. Anthropic / Gemini API does not accept consecutive inputs with the same role

### Options considered

- **A. Keep shared history, each agent plug-in transformer** (select)
- **B. Each agent maintains a private history**: more thorough but the synchronization complexity explodes
- **C. Continue to force the situation and rely on prompt engineering to let the agent distinguish on its own**: fragile

### Decision

- Discussion maintains **a shared transcript**
- Each agent projects its own perspective when `respond()`: `speaker == owner` → `assistant`; other → `<message from="X">...</message>` is included in user; metadata entry → `<tag>...</tag>` is included in user
- System prompt takes client independent parameters
- History entry changed from `role/content` to `speaker/type`

**Industry spectrum**: AutoGen's `model_context` and LangGraph's channel concepts are both "shared state + per-agent projection" mode; `<message from="X">` packaging is also a common practice in the community. Highly aligned.

### Engineering dimension assessment

|Dimensions|Evaluation|
|---|---|
|Cohesion|High - "Projection" is a single responsibility|
|Coupling|Significantly reduced - Agent only relies on the history list structure|
|Observability/Auditability|Neutral - Projecting pure functionalization makes debugging easy|
|LLM Uncertainty Tolerance|Up——`<message from="X">` allows the agent to distinguish the speaker more stably|
|Backwards compatible/evolution friendly|Destructive - history structure change; there is only one consumer when making changes, and the impact is controllable|
|Learning curve|Low – transparent to scenario author|
|Testability|L——Pure function projection, input is fixed and output is determined|

The "shared transcript + per-agent projection" model naturally supports future per-agent memory strategies, cross-provider isolation, and audit trail derivation - the abstraction leverage far exceeds the current single use.

## 3. Per-round phases + instruction-as-arg

- **Status**: partially superseded by §9 (`phases × round` two-dimensional structure + `<phase>` marker is replaced by flat `steps:` + `<turn
- **Date**: 2026-04-16

### Context

Two bugs were exposed at the same time:

1. **Instruction leak**: `_exec_phase` appends instructions to the shared history, and all subsequent agents can read instructions that do not belong to them. "Question X" given to moderator will be treated as its own command by members
2. **No difference in rounds**: The main stage is statically defined, and each round is exactly the same. It cannot express the progression of "the first round of free discussion → the second round of focusing on differences → the third round of forced expressions"

### Options considered

**Leaked for instruction**:

- **A. Add history entry visibility field**: Very intrusive
- **B. Instruction does not enter history, as a parameter of `respond()`** (selection): zero intrusion

**For round differences**:

- **A. `instructions: [...]` Index by wheel**
- **B. `{round}/{rounds}` template variables**
- **C. Phase explicit declaration `round: <int> | "default"`** (optional): supports "default + individual round override"

### Decision

The `phases` list is split into `opening/main/closing`, and each phase of `main` declares `round`; the engine first selects `round == N` in each round, falls back to `round == "default"`, and then falls back to all members speaking. Instruction as `Agent.respond(instruction=...)` parameter, **does not enter history**.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

### Engineering dimension assessment

|Dimensions|Evaluation|
|---|---|
|Cohesion|High - three distinct sections, per-round logic concentrated in a fallback chain|
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
|Observability/Auditability|Medium - `Round N/M` has a marker; the instruction itself does not enter the history, and the playback cannot see "what instructions the engine gave the agent at that time"|
|LLM Uncertainty Tolerance|L - Eliminate the "misexecution" runaway path leaked by instructions|
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

## 4. Subprocess isolation RAG tool

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
- **Date**: 2026-04-16

### Context

The first tool `retrieve_docs` is called directly by importing the rag module using `sys.path.insert(0, rag_dir)`. Python caches modules by name - the two subprojects each have `config.py`**, and the `config` of the second import gets the cache of the first one, and both sides cover each other.

### Options considered

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

### Decision

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
|---|---|
|Cohesion|High - each tool call is a complete `query.py` life cycle, no residual state in the process|
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
|Testability|High - subprocess boundaries allow RAG to be debugged independently with `python query.py --json`|

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

## 5. Per-agent conversation memory

- **Status**: accepted
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

### Context

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

### Options considered

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
- **E. Memory stream + importance scoring** (Generative Agents style)
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

### Decision

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
|---|---|
|LangChain `ConversationBufferWindowMemory` / `SummaryBufferMemory`|`WindowMemory` / `SummaryMemory` direct prototypes|
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
|CrewAI short/long/entity three types|only covers short-term|
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

|Dimensions|Evaluation|
|---|---|
|Cohesion|High - `memory.py` is pure projection, the three strategies share the same interface; after the DI transformation, there is no need to import agent/config|
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
|Learning Curve|Medium - The author needs to understand the trade-off of the three strategies|
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

### Unexpected income

After the implementation of memory, half of the root causes of the "unfair fixed speaking order" problem ("the late speaker enjoys the free context advantage") were structurally eliminated - WindowMemory allows everyone to see the same length of context, and the asymmetry of the speaking order is weakened. The priority of the "rotate/shuffle speech order" feature that was originally planned to be done separately has been reduced as a result. When solving A, I found that most of B disappeared.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
- **Date**: 2026-04-21

### Context

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

### Options considered

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
- **D. External Database**: Over-Engineering

**vote**:

- **A. Free Text Poll**: Noisy
- **B. `propose_vote` / `cast_vote` structured** (selection): tally verifiable

### Decision

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

**Key design points**:

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
- Moderator-only tool: `finalize_artifact` is filtered through `build_tool_defs(role)`; `propose_vote` is initially unfiltered, random member proposals will cause the hard-coded `vote_id` of the scenario to be misplaced, and filtering will be added later with phase-assert
- **Out-of-band artifact view**: Use `artifact.render()` as `<artifact>` message **out-of-band injection** before each agent speaks - without entering history, memory cropping will never hide it
- **Artifact events into history**: `artifact_event` type, pinned, not pruned by memory

### Position of three spectra across CRDT / workflow / structured output

- **CRDT / collaborative editing**: artifact is multi-writer shared state, sectioned division + explicit replace/append mode is a simple form of conflict avoidance (not CRDT but the same spirit)
- **Workflow engine** (Temporal / Airflow): `finalize_artifact` is a sealing step and is not reentrant - similar to the terminal state of workflow
- **Structured output / function calling**: vote as function call instead of text - LLM applies standard practice

Relatively innovative point: `initial_sections` allows the scenario author to state "what does this document look like" in YAML, and LLM is restricted to filling in the blanks within the schema - changing the product design from "free creation" to "fill in the blanks".

### Engineering dimension assessment

|Dimensions|Evaluation|
|---|---|
|Cohesion|High - `ArtifactStore` focuses on "shared state + tool entry + event flow", with four external ports: `render / drain_events / dispatch / build_tool_defs` |
|Coupling degree|The most coupling points in the project - Artifact is touched by Discussion / Agent / Memory / Tools everywhere, and there must be no redundancy in every place; the price is that any semantic changes must be considered simultaneously in all four places|
| Observability / Auditability | Extremely high - `artifact_event` is entered into history and pinned, terminal 📝 / ➕ / 🗳 / ✓ / 🏁 emoji is visible in real time, `--save-artifact` is placed |
|LLM Uncertainty Tolerance|High - section mode conflicts make LLM self-correct with loop; ballot overwrite write tolerates repeated casts; `finalize` idempotent returns error to prevent reentrancy|
|Backwards compatible/evolution friendly|Additive - scenarios that do not declare `initial_sections` are not artifact aware|
|Learning curve|The highest in the project - `initial_sections` schema + section mode + moderator-only filter three-layer concept|
|Testability|High - `scenarios/example.md` covers six tools + mode conflicts self-correction|

### Key Design Discussion

- **Why is sectioned not JSON? ** LLM's tool_call to markdown order is much more stable than one-shot JSON structured output; segmented self-correct path is shorter
- **Why does out-of-band view not enter history? ** Artifact is a state that can be refreshed at any time. Entering history means that tokens are occupied every time it is refreshed; out-of-band injection is "always latest + token controlled"
- **Why are events entered in history but not view? ** Events are immutable "what happened" (who wrote what in which round), and view is the "current state" - **Events can be played back, and states have no history**. This is the basic distinction between event sourcing

## 7. Phase-assert: Make silent violations visible

- **Status**: The scope restriction (the last paragraph of §7.5) has been lifted by §12 - require_tool now observes artifact + tracer events at the same time; the retry/nudge/warning mechanism and design intent (detect-and-nudge-and-audit) remain unchanged
- **Date**: 2026-04-21

### Context

The instructions in the closing phase of the panel scene require "call `cast_vote(...)` after everyone speaks", but the two members only spoke but did not vote, the engine did not give any alarm, and `v2` in the artifact was missing two votes. This is a common problem in LLM multi-agent systems: constraints written in prompts may be skipped directly by LLM and the engine fire-and-forget.

### Options considered

- **A. Hard failure** (abort phase without adjusting the tool): Too rough, one mock of the workshop scene fails and the entire demonstration is useless
- **B. Automatic re-tuning tool** (engine agent tuning): Violates agent autonomy and pollutes semantics
- **C. Silent nudge + retry + warning** (selection): Give the agent a chance to remedy the situation, and if it fails, a warning will be displayed to continue.

### Decision

- Scenario phase declaration `require_tool: <tool_name>`, optional `max_retries: N` (default 1)
- The engine scans the `tool / caller` field of `artifact.drain_events()` after the phase ends
- Miss → Append nudge instruction "You did not call `<tool>` just now, please add it now" as per-call argument (**do not enter history**, other agents cannot see this tutorial)
- Retries exhausted → stderr `WARNING: <agent> skipped required tool '<tool>' after N attempts`
- Type `🔁 [agent] retry k/N: missing <tool>` on the terminal, and the workshop audience can see the process

The core goal** is not to "force agent tuning tools" (LLM essentially cannot force it), but to make silent violations visible**.

**Scope restriction**: Currently `require_tool` only recognizes calls to the artifact tool (observed via `artifact.drain_events()`). Non-artifact tools (such as `retrieve_docs`) are not tracked yet and will be expanded after tool observability is completed.

### Targeting linter / roll-call mode

- **Linter warning**: does not prevent compilation, but leaves traces
- **Parliament roll-call**: Absent entries into meeting minutes
- **AutoGen `GroupChat` speaker selection**: The selected agent will fallback if it does not give a valid reply.
- **Structured output retry loop** (OpenAI `response_format`, Instructor): LLM retry without schema output

The difference from structured output retry is that this project does retry for the **behavior-level violation** of "not adjusting tools", instead of retrying "wrong output format" - a coarser granularity but covering more realistic problems.

### Engineering dimension assessment

|Dimensions|Evaluation|
|---|---|
|Cohesion|High——retry + nudge + warning all gathered in `Discussion._run_turn` one method|
|Coupling|Medium - Dependence on `tool / caller` fields of `artifact.drain_events()`; this is the root cause of scope limitation|
| Observability / Auditability | Extremely high - three layers of traces: `🔁 retry` is visible on site, `WARNING` falls to stderr, artifact_event enters history and can be played back|
|LLM Uncertainty Tolerance|Very High - Acknowledge that LLM cannot be forced, change to detect-and-nudge-and-audit mode|
|Backwards Compatible/Evolution Friendly|Fully Compatible - No change in behavior of phases without `require_tool` declared|
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

## 8. Tool observability：ToolTracer

- **Status**: accepted
- **Date**: 2026-04-22

### Context

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

### Options considered

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

### Decision

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
|---|---|
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
- **Date**: 2026-04-24

### Context

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

### Options considered

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

### Decision

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
|---|---|---|
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
|---|---|
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
- **Date**: 2026-04-26

### Context

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

### Options considered

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

### Decision

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

###Why choose this way?

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

|Dimension|Impact|
|---|---|
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
|Learning Curve|**Slightly Up** - New readers jump one more level `Scenario`→`Engine`; README and mermaid have already regarded it as SoT|
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

- **Status**: accepted（cross-link [`play/evals/DECISIONS.md` §5](../evals/DECISIONS.md)）
- **Date**: 2026-05-03

### Context

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
3. Re-spelling these fields on the evals side means that both sides must maintain the "`Result` flattening rules" - double SoT

Another independent issue: The five event handlers in `artifact.py` only stuff `tool/caller/content/ts` when `_events.append({...})`, and **discard `arguments`**. `content` is a human-readable string made for LLM memory rendering (e.g. `"caller wrote section 'X' (140 chars)"`), from which the evaluation layer cannot recover parameters. Phase 5 `argument_correctness` requires parameter-level matching capabilities in the run path, and the original args must be visible in the transcript.

### Options considered

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
|---|---|---|
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
|D. Transform `Result` and serialize it into protobuf/MsgPack|Fast and compact|No need for workshop size; JSON debugging is more friendly|

artifact_event plus `arguments` field options:

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
|---|---|---|
|A. The evals side reversely parses the `content` string|"`caller wrote section 'X' (140 chars)`" → `{"name": "X"}` etc.|Fragile; regular maintenance|
|**B. Event dict directly plugs `"arguments": dict(args)`** (select)|Add 1 line to each of the 5 handlers|There is information redundancy (args has been seen when calling LLM, but transcript is redone for permanent records); old consumers ignore unknown keys, purely additive|
|C. Save `tool_log: list[ToolCall]` in `Discussion` layer|Extra structure|Two truth sources (events vs tool_log) are easy to drift|

### Decision

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
|---|---|
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
|`result.py`|No change (`Result` is already a frozen dataclass, `asdict` is directly available, and the `to_dict()` method is not used)|
|`artifact.py`|5 event handlers (`_h_write_section` / `_h_append_section` / `_h_propose_vote` / `_h_cast_vote` / `_h_finalize_artifact`) each add `"arguments": dict(args)` (propose_vote has been disassembled and reassembled because of args by the helper `{"question": question, "options": list(options)}`)|
|Unmoved|`--save-transcript` / `--save-artifact` Keep the original human format; zero impact on old consumers|

The evals side corresponds to the consumption contract:

- `play/evals/models/agent_engine_run.py::make_run_fn`: `subprocess.run(["python", "-m", "agent_engine", scenario, "--no-stream", "--save-result-json", tmp])` → read envelope JSON
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
- `play/evals/tests/test_agent_traj_envelope.py`: lock `dataclasses.fields(Result) == {artifact, transcript, success, warnings}`——evals CI fails immediately when agent_engine changes the field name

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

- Aligned with the README guiding principles "Explicit is better than implicit" and "Contract layer is stable": the 4 fields of the envelope are the `Result` dataclass field, and there is no second contract
- `asdict` goes straight out to avoid maintenance `to_dict()`: dataclass is already the smallest single point source of information
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

### Consequences and Benefits

|Dimension|Impact|
|---|---|
|Cohesion|**L**——`Result` is the only contract across projects, envelope = `asdict(Result)`|
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
|Evolution-friendly|**Upgrade**——In the future, when adding fields to `Result`, the envelope will be automatically synchronized; the evals side `test_agent_traj_envelope` will automatically assert the new shape|
|cross-project test gate|evals side `conftest.py::agent_engine_required` Double gate (ollama-probe + brainstorm.md existence); if either one is missing, skip + friendly reminder|

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

- **Status**: accepted (supersedes §7 last paragraph "scope restriction")
- **Date**: 2026-05-09

### Context

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

### Options considered

|items|practices|tradeoffs|
|---|---|---|
|A. Let require_tool only adapt to artifact tools|status quo|Limit require_tool expressiveness; agent_sft Phase 1 must avoid non-artifact tools|
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
|**C. `_run_turn` collects tracer + artifact two-way events at the same time and passes them to `_called_tool`** (selected) | 3 lines of changes, `_called_tool` remains unchanged (it is already a tool/caller tile check) | `tracer_events` and `artifact_events` are still extended in batches in history; require_tool check surface is merged; purely additive|
|D. Introducing the "Unified Event Bus" refactoring|Clean structure|The workshop size is over-designed; C is enough|

### Decision

|Move point|How to do|
|---|---|
|`discussion._run_turn`|`tracer_events = self.tracer.drain() if self.tracer else []`; `artifact_events = self.artifact.drain_events() if self.artifact else []`; `events = tracer_events + artifact_events` fed to `_called_tool`|
|`tracer.py` / `artifact.py`|Not moving - the event schema is already tool/caller tiled, aligned with the existing contract of `_called_tool`|
|`_called_tool`|Does not move - this is a tool-neutral `tool/caller` check|

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
- **OpenAI structured output retry / Instructor**: do retry for "wrong output format", the granularity is finer than require_tool but the semantics are different
- §7 + §12 of this project are "behavioral require_tool + tool-neutral observation" - stronger than AutoGen (declarative), weaker than LangGraph (not mandatory), and in a class of its own

### Engineering dimension assessment

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
|---|---|
|Cohesion|Unchanged - still `Discussion._run_turn` a method|
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
|Observability|**Upgrade** - require_tool behavior of non-artifact tools can now be measured (nudge_fire_rate critical dependency of agent_sft)|
|Backwards Compatibility|**Purely Additive** - only require_tool=artifact tool's old scenario behavior is byte identical (tracer_events is the empty set and merged equals artifact_events)|
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

### Relation to §7 / ​​§8

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
- §8 After the introduction of ToolTracer, the ability of "non-artifact tools are recorded" is already available, but this inspection surface is not consumed - this ADR is to complete this wiring, not to create a new mechanism
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

## 13. Expose transcript/scenario interpretation API: Result/Scenario expose typed view

- **Status**: accepted（extends §11）
- **Date**: 2026-05-11

### Context

§11 The envelope is determined to be a SoT consumed across project machines (`Result` field layer); but the envelope only covers the **field** schema (`{transcript, artifact, warnings, success}`) and does not cover the **field interpretation** - how to reduce `tool_call` / `artifact_event` in the transcript into "tool calls", how to segment the `<turn X of N>` marker, and scenario YAML How to statically expand into `(turn_idx, agent, step_id, require_tool)` sequence, all reverse engineered by each consumer:

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
|---|---|---|---|
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
| `play/agent_sft/data/extractor.py` | Same as above + plus turn-indexed global offset | `_index_steps_by_turn / _split_turns_indexed` + **`sys.path.insert + from evals.metrics.nudge import _split_attempts, _resolve_who_to_agents, _split_frontmatter, derive_expected_turns`** | Cross-project import 4 **private** functions, anti-pattern |

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

### Options considered

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
|---|---|---|
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
| B. Add the three public plain functions `_extract_tool_calls / split_turns / derive_expected_turns` to the `agent_engine.transcript` / `agent_engine.scenario_static` modules | Simple, functional | New modules are required for each new view; return `dict / list[dict]` Weak type semantics |
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

### Decision

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
|---|---|
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
| `scenario.py` extracts `_resolve_who_names(who, declared_order, role_by_name)` pure function | `Discussion._resolve_who` runtime path + `Scenario.expanded_turns()` static path **share this function** - ensure that the expansion order bytes of the two paths are the same |
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
|---|---|---|
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
| LangChain `AIMessage.tool_calls` | typed `ToolCall(name, args, id)` field | Same semantics, more detailed (including id); this project currently does not require id (artifact / tracer event has no corresponding field) |
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

| Dimensions | Impact |
|---|---|
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

PR-2 is the cleanup of §13: delete evals private `_xxx` function shim + change `play/agent_sft/data/extractor.py` to directly connect `agent_engine.Scenario / Result`, and synchronize ADR in [`play/evals/DECISIONS.md`] / [`play/agent_sft/DECISIONS.md`]; §13 PR-1 takes effect as soon as it lands, while PR-2 completes rather than blocks.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
- **Date**: 2026-05-11

### Context

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

| Phenomenon | Root cause |
|---|---|
| `Result.transcript[i]["speaker"]` / `entry["type"]` / `entry["tool"]` Mixing of three styles | 5 writing points (`discussion.py` 3 / `tracer.py` 1 / `artifact.py` 5 / `memory.py` 1) each write its own dict literal, no schema mandatory |
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
| For evaluation, LLM cost / token analytics must parse stderr / `🔧` emoji inverse | envelope does not contain `usage` field, 4 LLM client only `return text` and does not return usage data |
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

### Options considered

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
|---|---|---|
| A. Keep `list[dict]` and only add TypedDict to entry | There are many type annotations but it is still dict at runtime | Static mypy passes but wrong fields can still be written at runtime |
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
| **C. typed union (6 frozen dataclass + `TranscriptEntry = TopicEntry \| TurnEntry \| ...`) ** (select) | Identical to OpenAI Agents SDK / Anthropic Messages | runtime `isinstance` dispatch, IDE derivation friendly; new entry type is to add new dataclass + one-line union expansion, no schema call surface modification |
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
|---|---|---|
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
| B. In the `Discussion._run_turn` package LLM call timing, estimate token | 1 change | The estimation error is large; cached_tokens cannot be obtained |
| **C. 4 client's `chat()` changes back to `(text, TokenUsage)` tuple, and successive LLM calls fall into `Result.usage`** (selected) | typed single point, aligned with the SDK native usage field | 4 client needs to change the return signature; streaming calls must complete the final chunk usage in stream consumed, medium workload |

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

| Items | Practices | Trade-offs |
|---|---|---|
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

### Decision

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.

Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
|---|---|
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
| `Result.from_dict` is no longer downgraded; `data["transcript"] / data["usage"] / data["artifact"] / data["success"] / data["warnings"]` if one is missing, `KeyError` | schema cannot be downgraded; old envelope is invalid |
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
| `agent.py::Agent.respond` is connected to the tuple + accumulated `list[TokenUsage]`; `memory.py::ConversationMemory.drain_usage()` also recycles the usage generated by SummaryMemory’s internal summarizer call | LLM calls that are not perceived by the user (such as SummaryMemory folding) are also included in `Result.usage` |
| `Discussion.usage: list[TokenUsage]` + `Engine.invoke` writes `Result.usage` | String to envelope export |
| `engine.py` before writing to disk `[dataclasses.asdict(e) for e in history]` | typed entry → JSON serialization |
| `_ae_bridge.py` re-export `TokenUsage / TopicEntry / TurnEntry / SpeakerEntry / ToolCallEntry / ArtifactEventEntry / SummaryEntry` | evals end zero sys.path Black magic can get all typed views |
| `agent_sft.data.extractor` deletes `_split_turns_indexed / _index_steps_by_turn`, a total of 2 shims | typed `Result.turns()` can be given directly to `start_offset`, and the shim loses its meaning |
| Full fixture rewrite (~200 inline dict across 7 test files) + Added `agent_engine/tests/test_token_usage.py`, etc. | Typed schema is also forced on the test surface |
| `play/evals/data/{agent_traj,nudge_fire_rate,...}/predictions/*.jsonl` + `agent_sft/data/triples/runs_1k_fast_7b_r0_124/*.json` One-time migration script injection `type:"speaker"` + `usage: []` | The user has confirmed forward-only; the migration script is one-time and does not leave any code traces |

### Industry Spectrum

| Framework / SDK | typed message / event | usage shape |
|---|---|---|
| OpenAI Agents SDK | `RunResult.new_items: list[MessageOutputItem \| ToolCallItem \| ToolCallOutputItem]` | `response.usage.{prompt_tokens, completion_tokens, prompt_tokens_details.cached_tokens}` |
| Anthropic Messages | `Message.content: list[TextBlock \| ToolUseBlock]` | `message.usage.{input_tokens, output_tokens, cache_read_input_tokens}` |
| Google Gemini | `GenerateContentResponse.candidates[].content.parts: list[Part]` | `response.usage_metadata.{prompt_token_count, candidates_token_count, cached_content_token_count}` |
| Ollama | dict shape (`role`, `content`, `tool_calls`) | response dict has `prompt_eval_count` / `eval_count` |
| inspect_ai | `ChatMessage` / `ChatMessageTool`（dataclass） + `EvalSample.token_usage: dict[str, ModelUsage]` | typed |
| LangChain `AIMessage.usage_metadata` | typed `UsageMetadata(input_tokens, output_tokens, total_tokens)` | same idea |

The `TranscriptEntry` typed union + `TokenUsage` per-call list in §14 of this project has the same spirit as inspect_ai / LangChain, and is flatter than the OpenAI Agents SDK (no nested RunItem layer); multiple backend adaptations uniformly smooth out the field differences of the four SDKs, corresponding to the consistent `input_tokens / output_tokens / cached_tokens / duration_ms`.

### Engineering dimension assessment

| Dimensions | Impact |
|---|---|
| Cohesion | **Upgrade**——entry schema, entry writing point, entry interpretation right are all in one place `result.py`; `TokenUsage` is captured in one place in client, and `Result` is displayed in one place |
| Coupling degree | **Reduced** - Evaluation / SFT side no longer has `entry.get("speaker")` / `entry["type"]` string sniffing; `isinstance` can catch errors during compilation |
| Observability | **Upgrade** - `SpeakerEntry.type="speaker"` allows transcript interpretation to be 100% based on the type tag path, without ambiguity; token usage permanently falls into the envelope, and cost/efficiency/latency metrics are upgraded from stderr inversion to field-level direct consumption |
| LLM Uncertainty Tolerance | **Same** - schema transformation does not affect runtime fault tolerance; fill in 0 when the streaming usage is not available, and evals cost calculation is downgraded to the existing `efficiency.py` path |
| Backward compatibility | **Destructive** - old envelope is unreadable; `from_dict` forces schema; 4 LLM client `chat()` signature changes; this warehouse has no external consumers, and the migration script processes mined data in one go |
| Evolution-friendly | **Upgrade**——Add new entry type = add dataclass + one-line union expansion + write point; schema is SoT, add fields = `dataclasses.fields` automatic synchronization |
| pytest safety net | Three projects 585 tests all green (agent_engine 42 / evals 456 / agent_sft 87) + evals smoke `agent_traj` / `nudge_fire_rate` passed + current network envelope round-trip passed |

### Relation to §11 / §13

| ADR | What is established |
|---|---|
| §11 | `Result` is envelope SoT (**field** schema: `{transcript, artifact, warnings, success}`) |
| §13 | `Result` / `Scenario` expose typed views (**field interpretation**: `tool_calls / turns / expanded_turns`) |
| §14 | `transcript` internal itself typed union + `usage` field (**schema itself** also typed) |

§11 + §13 + §14 The three layers combined: the cross-project contract from "field name → field interpretation → field value type" is all self-described by `agent_engine` in one place, and downstream consumers do not need any reverse engineering.

### Not in scope

- `Result.usage` does not do "aggregation by caller / model" - `evals/metrics/efficiency.py` already has an aggregation path, agent_engine only produces raw usage list
- Fill in 0 when usage cannot be obtained by streaming call (also fill in 0 for cached_tokens) - no exception is thrown; evals cost calculation is automatically downgraded to model-level estimation
- `TokenUsage.duration_ms` uses `time.monotonic()` to package client call timing; does not distinguish first-token-latency vs total-latency (not required for workshop volume)
- The old envelope is backward compatible with readers and does not write to the warehouse - a one-time migration script is enough, leaving no long-term shim like `try_legacy_from_dict()`

# Scenario kitchen sink — field reference + mental model + integration smoke (same file)

# ============================================================================
# This is a **runnable** scenario: frontmatter keeps per-field teaching comments;
# steps use a **compressed integration path** (artifact + retrieve_docs +
# window/full/summary + require_tool nudge) for fewer turns and shorter wall-clock,
# while comments still explain why that coverage is enough. For minimal token use,
# shorten agent prompts or drop the `tools:` block.
#
# **CI / regression**: `ci_who_member` + `ci_who_all` deliberately hit scalar
# `who: member` and `who: all` (same style as `roundtable` / `debate` /
# `brainstorm`) so a single file validates all four `_resolve_who` paths; it does
# not cover "no moderator" topology — that stays in lightweight scenes like
# `debate.md`.
#
#   python -m agent_engine scenarios/example.md
#
# Prerequisite: `../../rag/vdb/test_vdb` exists (see ingest in play/rag README).
# Don't want retrieve_docs? Delete the entire `tools:` block from frontmatter.
# ============================================================================

---
# ── memory (scenario-level default) ─────────────────────────────────────────

memory:
  # Choose one of three:
  #   full     — never trim; keep full history (default; no other fields needed)
  #   window   — keep all pinned markers (topic / turn / artifact_event) + last N speeches
  #   summary  — fold stale speech into <summary> block; keep last N speeches verbatim
  type: window
  # Integration smoke already compressed turns vs old example; max_recent tightened
  # slightly for window while still covering "pinned never trimmed + recent speech".
  # Widen back to 8 if you need a larger window.
  max_recent: 6
  # window / summary only: max_recent required (positive integer).
  #
  # summary at scenario level may also specify (this file uses type=window, so no
  # summary block here; when scenario type=summary, optional keys mirror agent-level):
  #   model: <override SUMMARY_MODEL>
  #   max_tokens: <override SUMMARY_MAX_TOKENS>
  #   temperature: <override SUMMARY_TEMPERATURE>
  #   summarizer_prompt: <override default system prompt>
  #   summarize_instruction: <override default compression instruction>
  #
  # Any agent-level `memory:` field **overrides** this default.

# ── tools (scenario-level tool defaults) ────────────────────────────────────

tools:
  # List non-artifact tools enabled in this scene. Each item requires `name`; other
  # keys are "scenario-injected defaults" — hidden from the LLM tool schema and
  # injected at dispatch by scenario.py `_build_tool_handler`.
  #
  # Path params (declared via `_path_params` in tools/, currently only `vdb_dir`):
  # relative paths resolve to absolute paths relative to this scenario file's directory,
  # so the scenario does not depend on cwd at invoke time.
  - name: retrieve_docs
    vdb_dir: ../../rag/vdb/test_vdb
    top_k: 3
    # Any key listed here (except `name`) is stripped from the LLM schema and injected
    # by scenario.py — "scenario pin". So `vdb_dir` / `top_k` are invisible to the LLM.
    # Params not listed (mode / rerank) stay LLM-visible and may be chosen in tool_call:
    #   mode: "dense" | "bm25" | "hybrid"   (default hybrid)
    #   rerank: true                          (default false; first call ~5s loads ~1.2GB model)
    # To force rerank on or lock mode to dense for A/B, add the key here — same effect:
    # param removed from LLM schema + default injected.

# ── artifact (shared structured doc + voting) ───────────────────────────────

artifact:
  enabled: true
  # ArtifactStore and artifact tools only exist when enabled=true.

  # initial_sections defines which sections exist at start and their mode.
  # Sections not listed may still be created via write/append (no mode constraint).
  initial_sections:
    # Item may be a string (mode defaults to replace)...
    - decision

    # ...or {name, mode}:
    #   mode: replace — write_section ✓; append_section returns error
    #   mode: append  — append_section ✓; write_section returns error
    - {name: notes, mode: append}
    - {name: data,  mode: replace}

  # tool_owners controls which agents may see/call each artifact tool.
  # Same shapes as step.who:
  #   <role>       — agents with that role ("moderator" / "member")
  #   all          — all agents
  #   [name1, ...] — explicit agent list
  # **Undeclared tools default to all agents** — including finalize_artifact / propose_vote.
  # To keep moderator-only behavior, declare explicitly:
  tool_owners:
    propose_vote: moderator
    finalize_artifact: moderator

# ── agents (required, at least 1) ───────────────────────────────────────────

agents:
  # Each agent requires name / prompt / role. Other fields optional.
  # role ∈ {"moderator", "member"} affects:
  #   - step.who: moderator | member resolution
  #   - artifact.tool_owners role expansion
  - name: Moderator
    role: moderator
    # `name` is used in prompt injection, history projection, artifact_event caller,
    # and step.who list addressing — must be **globally unique**.
    prompt: |
      You are the moderator. Follow instructions and call artifact tools as directed; ≤ 40 chars per turn. Reply in English.
    temperature: 0.4
    max_tokens: 160
    # Optional; defaults to config.TEMPERATURE / MAX_TOKENS
    # model: <override>  # optional; defaults to config.DEFAULT_MODEL (by BACKEND)
    # memory: { ... }    # optional; overrides scenario-level default

  - name: Analyst
    role: member
    prompt: |
      You are the analyst. Call retrieve_docs when you need facts; use artifact tools per instruction.
      ≤ 40 chars per turn. Reply in English.
    max_tokens: 160

  - name: Decision-maker
    role: member
    prompt: |
      You are the decision-maker. Call cast_vote etc. per instruction; ≤ 40 chars per turn. Reply in English.
    max_tokens: 160
    memory:
      # Agent-level memory overrides scenario default. Here Decision-maker uses full
      # (wants full history, no window trim).
      type: full

  # Fourth member demonstrates summary strategy + extra summarizer LLM call; prompt asks
  # for structured short output to reduce latency while still contrasting three memory types in transcript.
  - name: Summarizer
    role: member
    prompt: |
      You are a dialogue visibility observer. Each turn exactly three lines, no small talk:
      visible_speakers: <comma-separated names seen in history; none if empty>
      memory_type: <window|full|summary per your system config>
      summary_seen: <yes|no whether a summary block appeared>
      ≤ 50 chars. English.
    memory:
      type: summary
      max_recent: 2
      summarizer_prompt: Compress multi-speaker dialogue into ≤ 60 English words; keep names and positions.
      summarize_instruction: Merge input into one compact summary; if previous_summary present, merge and rewrite.
    max_tokens: 120
    temperature: 0

# ── steps (required, flat flow list) ────────────────────────────────────────

steps:
  # Per-step fields:
  #   id           — optional; terminal prints (step=<id>) for readability
  #   who          — required; scalar role/all or list[name]
  #   instruction  — required non-empty; extra guidance for current speaker
  #                  (not in history; injected once as user message)
  #   require_tool — optional; after step, scan events; if caller did not call
  #                  specified tool, trigger nudge retry
  #   max_retries  — optional retry count. Default 1 when require_tool set, else 0
  #
  # Engine expands steps in list order into turns: each matched agent in who speaks
  # once, in agents declaration order. Each turn gets pinned `<turn>turn X of N</turn>`.
  #
  # Steps below compress old kitchen-sink chit-chat; pack retrieve + append +
  # append/write conflict + read + vote + require_tool nudge into fewer steps.
  #
  # CI: `open` covers `who: moderator`; `mem_warm*` covers `who: [name,...]`;
  # next two steps cover scalar `member` / `all` (instructions forbid tools to avoid polluting vote segment).

  - id: open
    who: moderator
    instruction: |
      One sentence on topic — whether to adopt the retrieved "项目代号" as the official name.

  # Expands in declaration order: Analyst → Decision-maker → Summarizer (discussion._resolve_who).
  - id: ci_who_member
    who: member
    instruction: |
      Addressing smoke test. Output only the word "here"; do not call any tools.

  # Expands: Moderator → Analyst → Decision-maker → Summarizer.
  - id: ci_who_all
    who: all
    instruction: |
      Addressing smoke test. Output only the word "all"; do not call any tools.

  # Two short rounds to trigger Summarizer SummaryMemory fold (see memory.py trigger rules).
  - id: mem_warm
    who: [Analyst, Decision-maker, Summarizer]
    instruction: |
      Answer strictly in three-line format per system; do not call tools.

  - id: mem_warm2
    who: [Analyst, Decision-maker, Summarizer]
    instruction: |
      Another three-line round; do not call tools.

  # Merged research + artifact smoke: retrieve_docs, append, deliberate write to trigger
  # append-only error, read_artifact; require_tool still watches append_section.
  - id: vdb_artifact
    who: [Analyst]
    # List form: explicit names. Even a single name goes in [] — schema uses type
    # (scalar vs list) to distinguish role vs name addressing.
    instruction: |
      1) retrieve_docs query "项目代号";
      2) append_section(name="notes", entry="- 项目代号: <value>");
      3) deliberately write_section(name="notes", content="bad") to trigger append-only error, acknowledge in one sentence;
      4) read_artifact() confirm notes and summarize in one sentence.
    require_tool: append_section
    max_retries: 1

  - id: vote_prep
    who: moderator
    instruction: |
      read_artifact(); then propose_vote(question="Adopt?", options=["Adopt","Reject"]); one sentence asking everyone to vote.

  # Instruction deliberately omits cast_vote to test silence → nudge → retry; if model
  # votes on first attempt, no retry (still valid).
  - id: ballot_nudge
    who: [Analyst]
    require_tool: cast_vote
    max_retries: 1
    instruction: |
      Say hello in one sentence only; do not mention cast_vote.

  # Explicit cast_vote for normal vote path (contrast with ballot_nudge).
  - id: ballot_ok
    who: [Decision-maker]
    require_tool: cast_vote
    max_retries: 1
    instruction: |
      cast_vote(vote_id="v1", option="Adopt" or "Reject", rationale="one sentence").

  - id: finalize
    who: moderator
    instruction: |
      write_section(name="decision", content="Conclusion: <match vote>");
      finalize_artifact(decision="Adopt" or "Reject", rationale="one sentence").
      finalize may only be called once — idempotent guard returns error on repeat.

# ============================================================================
# `who` forms (four total)
#   moderator      — scalar role; role match (requires ≥1 moderator) → open
#   member         — scalar role; role match (requires ≥1 member) → ci_who_member
#   all            — scalar keyword; all agents in declaration order → ci_who_all
#   [n1, n2, ...]  — explicit list; order as given; names must exist → mem_warm* / vdb_artifact …
# Bad who fails at startup (schema validation), not at runtime.
#
# Omission policy
#   - agents required (≥1); steps required (≥1, each step needs instruction)
#   - omit tools → no non-artifact tools
#   - omit artifact or enabled=false → no shared artifact; all 6 artifact tools unavailable
#   - omit artifact.tool_owners → all artifact tools open to all agents
#   - omit memory → all agents use FullHistory
# ============================================================================
---

## Runtime mental model (body is the topic and also serves as logic notes for this example)

Build this mental model before tweaking fields — afterward changes are just "turning knobs":

1. **One run = steps expanded into a linear turn sequence**. Within each step, all agents
   matched by who speak once (agents declaration order). Steps share one authoritative
   `history` (topic / turn / speaker / artifact_event / tool_call); each agent projects
   at `respond()`: own speech → `assistant`, others → `<message from="X">…</message>`,
   control flow → tagged user messages.

2. **Memory controls what history enters projection**. Pinned types (topic / turn /
   artifact_event) are never trimmed, so turn changes and artifact updates stay visible
   for all memory strategies. `window` keeps last N speeches; `summary` folds stale speech
   into `<summary>` (Summarizer demonstrates agent-level config; Moderator/Analyst use
   scenario default window; Decision-maker uses full).

3. **Artifact view is out-of-band** — each `respond()` injects `ArtifactStore.render()`
   as `<artifact>…</artifact>` user message, **not** in history. Artifact state stays
   fresh for everyone without memory quota. Each write/append/vote/finalize emits pinned
   `artifact_event` in history visible to all.

4. **require_tool does not force — it makes silent violations visible** — after step,
   scan `artifact.drain_events()`; if caller missed the tool, nudge once; still skip →
   stderr WARNING, run continues. Workshop-friendly compromise. `ballot_nudge` pairs
   "hello only" instruction with `require_tool: cast_vote` to observe retry.

5. **Tools hiding** — scenario defaults under `tools:` (e.g. `vdb_dir`) are removed from
   LLM OpenAI tool schema and injected by scenario.py. `_path_params` paths resolve
   relative to scenario file — invoke works from any cwd.

6. **Topic for this compressed flow**:
   Decide whether to adopt the project codename from retrieve_docs as the official name;
   `vdb_artifact` completes retrieve + artifact R/W demo in one step; `mem_warm`*2 runs
   three memory types together; `ballot_nudge` / `ballot_ok` split nudge vs normal vote.

7. **CI**: `ci_who_member` / `ci_who_all` use minimal output to verify scalar `member` and
   `all` expand correctly; orthogonal to no-artifact / no-moderator topologies (see
   `debate.md` / `brainstorm.md`).

Should the project codename in the docs become the team's official external name?

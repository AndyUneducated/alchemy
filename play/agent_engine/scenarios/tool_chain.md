# Tool-chain compliance test scenario (agent_sft Phase 1 require_tool-dense scenario #2)

# ============================================================================
# Design goals (complements `code_review.md`):
#   - **Single-agent strong tool chain**: one executor follows
#     "retrieve → append → vote → retrieve → append" step by step; tests pure tool
#     sequence compliance without multi-agent context complexity.
#   - **5 require_tool step entries** (plan §1.B):
#       retrieve_docs × 2 + append_section × 2 + cast_vote × 1
#     Single agent → 5 require_tool turns.
#   - **Realistic scene**: executor completes checklist from coordinator:
#     research → notes → vote → second research → second notes.
#
# vs code_review.md:
#   | Dimension     | code_review.md      | tool_chain.md           |
#   |---|---|---|
#   | Agent count   | 4 (3 senior)        | 2 (1 executor)          |
#   | Context       | Multi-reviewer      | Single checklist        |
#   | Tests         | Multi-role tools    | Tool order compliance   |
#   | Data use      | SFT diversity       | High-purity compliance  |
#
# Reuses example.md vdb path (test_vdb).
# ============================================================================

---

memory:
  type: full

tools:
  - name: retrieve_docs
    vdb_dir: ../../rag/vdb/test_vdb
    top_k: 3

artifact:
  enabled: true
  initial_sections:
    - {name: Research notes, mode: append}
    - {name: Decision, mode: replace}
  tool_owners:
    propose_vote: moderator
    finalize_artifact: moderator

agents:
  - name: Coordinator
    role: moderator
    prompt: |
      You are the coordinator. Duties: open with checklist, organize vote, finalize decision.
      ≤ 50 words per turn. English.
    temperature: 0.3
    max_tokens: 160

  - name: Executor
    role: member
    prompt: |
      You are the tool-chain executor. **Strictly call the tool specified in current step instruction** —
      no detours, no extra calls, no missed calls. ≤ 30 words per turn (focus on tools, not talk).
      English.
    temperature: 0.2
    max_tokens: 200

steps:
  - id: open
    who: moderator
    instruction: |
      Announce today's checklist: research "项目代号" → record notes → vote direction → second research → second notes.

  - id: ctx_round1
    who: [Executor]
    require_tool: retrieve_docs
    max_retries: 1
    instruction: |
      Call retrieve_docs(query="项目代号") for background.
      ≤ 30 words reporting core findings.

  - id: note_round1
    who: [Executor]
    require_tool: append_section
    max_retries: 1
    instruction: |
      append_section("Research notes", "- Round 1: <one-line core point>") record retrieval result.

  - id: vote_setup
    who: moderator
    instruction: |
      propose_vote(question="Need second research pass?", options=["Yes","No"]),
      one sentence asking executor to vote.

  - id: ballot
    who: [Executor]
    require_tool: cast_vote
    max_retries: 1
    instruction: |
      cast_vote(vote_id="v1", option="Yes", rationale="one sentence").
      Vote "Yes" to complete checklist.

  - id: ctx_round2
    who: [Executor]
    require_tool: retrieve_docs
    max_retries: 1
    instruction: |
      Second retrieve_docs(query="项目代号 history") for follow-up research.
      ≤ 30 words reporting supplemental info.

  - id: note_round2
    who: [Executor]
    require_tool: append_section
    max_retries: 1
    instruction: |
      append_section("Research notes", "- Round 2: <one-line supplemental point>") record second pass.

  - id: finalize
    who: moderator
    instruction: |
      write_section("Decision", "Completed all 5 checklist steps"),
      then finalize_artifact(decision="Done", rationale="All 5 tool-chain steps executed in order").

---

## Checklist: research → notes → vote → second research → second notes

Simple task: executor **strictly calls the tool named in each step in order**.
This scene does not test content quality — only **whether the model calls the right tool once per instruction**.

Each require_tool step is a pure compliance test:
- Correct on attempt 1 → no nudge fired ✓
- Miss / wrong tool on attempt 1 → nudge retry → failure mode recorded

Keep output short; weight goes to tool calls, not speech.

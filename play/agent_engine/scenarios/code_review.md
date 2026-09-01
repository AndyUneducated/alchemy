# PR code review meeting (agent_sft Phase 1 require_tool-dense scenario)

# ============================================================================
# Design goals (agent_sft Phase 1 baseline / Phase 2 SFT data collection):
#   - **Multi-agent / rich context**: 3 senior engineers + 1 lead reviewer with
#     interdependent context; model makes tool decisions across roles. Orthogonal to
#     `tool_chain.md` "single-agent strong tool chain".
#   - **5 require_tool step entries** (plan §1.B):
#       retrieve_docs × 2 + append_section × 2 + cast_vote × 1
#     Expanded turns: 2 (retrieve) + 1 (append A) + 2 (append B,C member) + 3 (vote member)
#                 = 8 require_tool turns
#   - **Realistic scene**: 3 engineers review a cross-module PR; must retrieve related
#     commit/doc first, write review sections, then vote on merge.
#
# Reuses example.md vdb path (test_vdb); if missing, run ingest per play/rag README.
# ============================================================================

---

memory:
  type: window
  max_recent: 6

tools:
  - name: retrieve_docs
    vdb_dir: ../../rag/vdb/test_vdb
    top_k: 3

artifact:
  enabled: true
  initial_sections:
    - {name: review_a, mode: replace}
    - {name: review_b, mode: replace}
    - {name: review_c, mode: replace}
    - {name: Decision, mode: replace}
  tool_owners:
    propose_vote: moderator
    finalize_artifact: moderator

agents:
  - name: Lead Reviewer
    role: moderator
    prompt: |
      You are the lead reviewer. Duties: open with PR scope, organize three senior reviews, announce merge decision.
      ≤ 60 words per turn. English. Call propose_vote / finalize_artifact / write_section when needed.
    temperature: 0.3
    max_tokens: 200

  - name: Engineer A
    role: member
    prompt: |
      You are backend senior engineer. retrieve_docs for related commit/doc before concluding;
      submit via append_section("review_a", ...). ≤ 60 words. English.
    max_tokens: 200

  - name: Engineer B
    role: member
    prompt: |
      You are frontend senior engineer. retrieve_docs for API contracts before concluding;
      submit via append_section("review_b", ...). ≤ 60 words. English.
    max_tokens: 200

  - name: Engineer C
    role: member
    prompt: |
      You are QA senior engineer. Synthesize A/B reviews, append_section("review_c", ...) with risk assessment,
      then cast_vote on merge. ≤ 60 words. English.
    max_tokens: 200

steps:
  - id: open
    who: moderator
    instruction: |
      One sentence on PR scope ("项目代号" module) and today's goal: 3 reviews then vote on merge.

  - id: ctx_a
    who: [Engineer A]
    require_tool: retrieve_docs
    max_retries: 1
    instruction: |
      Call retrieve_docs for "项目代号" related commits / design docs;
      ≤ 30 words summarizing key findings.

  - id: ctx_b
    who: [Engineer B]
    require_tool: retrieve_docs
    max_retries: 1
    instruction: |
      Call retrieve_docs for "项目代号" API contracts / test cases;
      ≤ 30 words summarizing key findings.

  - id: review_a
    who: [Engineer A]
    require_tool: append_section
    max_retries: 1
    instruction: |
      append_section("review_a", "- <one-line review conclusion>")
      Write your review into artifact based on retrieval.

  - id: review_bc
    who: [Engineer B, Engineer C]
    require_tool: append_section
    max_retries: 1
    instruction: |
      Engineer B → append_section("review_b", "- ...").
      Engineer C → append_section("review_c", "- one-line A/B risk summary").

  - id: vote_setup
    who: moderator
    instruction: |
      Call propose_vote(question="Merge this PR?", options=["Merge","Reject"]),
      then one sentence inviting votes.

  - id: ballot
    who: member
    require_tool: cast_vote
    max_retries: 1
    instruction: |
      cast_vote(vote_id="v1", option="Merge" or "Reject", rationale="one sentence")
      State your final position.

  - id: finalize
    who: moderator
    instruction: |
      From <artifact> vote results, write_section("Decision", "<Merge/Reject> + one-line summary"),
      then finalize_artifact(decision="Merge" or "Reject", rationale="one sentence").

---

## PR under review: project codename module refactor

### PR background

Backend SDK "项目代号" module (`project_codename/`) had 3 scattered implementations; this PR unifies to single entry, affecting:

- Backend: merge 3 modules → 1; new facade API; deprecation warnings on old API
- Frontend: import path change (`from sdk.codename import ...` → `from sdk.project import codename`)
- Tests: per-module unit tests merged into one integration suite

### Key concerns

| Dimension | Risk |
|---|---|
| Compatibility | Old import path still works; deprecation warnings may pollute downstream logs |
| Performance | Single entry adds indirection; benchmarks show no meaningful difference |
| Test coverage | Integration coverage 87%, vs weighted pre-merge average 92% |

### Options

1. **Merge**: Accept short-term coverage drop for long-term maintainability
2. **Reject**: Require author to raise integration coverage to 90%+ before merge

Lead reviewer: follow steps to run the review.

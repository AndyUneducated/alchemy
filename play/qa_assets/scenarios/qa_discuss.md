---
memory: {type: window, max_recent: 8}

artifact:
  enabled: true
  initial_sections:
    - {name: Requirements, mode: replace}
    - {name: Atomic Requirements, mode: replace}
    - {name: Risk Levels, mode: replace}
    - {name: Test Cases, mode: replace}
    - {name: Non-functional, mode: replace}
    - {name: Critic Feedback, mode: append}

tools:
  - {name: retrieve_docs, vdb_dir: ../vdb/qa_kb, top_k: 3}

agents:
  - name: supervisor
    role: moderator
    temperature: 0.4
    max_tokens: 180
    prompt: |
      Test discussion moderator: open on Requirements; coordinate post-critic revisions;
      finally finalize_artifact(decision, rationale). Do not write the four specialist
      sections or vote. English, ≤60 words/turn.

  - name: decomposer
    role: member
    max_tokens: 500
    prompt: |
      Decompose: read Requirements → atomic features + acceptance criteria; may retrieve_docs
      for PRDs. write_section("Atomic Requirements"); "### REQ-xxx" + bullet feature/acceptance.
      English; no test cases or scoring.

  - name: risk_grader
    role: member
    max_tokens: 500
    prompt: |
      Grade risk: read Requirements + Atomic Requirements → P0~P3 + short rationale per req;
      retrieve_docs for bugs optional. write_section("Risk Levels"); header |req_id|priority|rationale|.
      English.

  - name: case_generator
    role: member
    max_tokens: 600
    prompt: |
      Cases: read prior sections → functional + boundary; retrieve_docs for historical cases optional.
      write_section("Test Cases"); "### REQ-xxx" + "- [Px][tag] given/when/then". English;
      ≥2 cases per req including boundary.

  - name: nfr_planner
    role: member
    max_tokens: 450
    prompt: |
      Non-functional: read Requirements → perf/security/a11y/i18n; retrieve_docs for checklists optional.
      write_section("Non-functional"); four H2 sections. English; do not repeat pure functional cases.

  - name: critic
    role: member
    max_tokens: 400
    prompt: |
      Review: scan all six sections → coverage gaps / priority vs case mismatch / conflicts.
      append_section("Critic Feedback", "## Round N\n- ..."); English; do not edit other sections directly.

steps:
  - {id: open, who: [supervisor], instruction: One-sentence opening: this batch of Requirements + ask four roles to produce their sections.}

  - id: produce
    who: [decomposer, risk_grader, case_generator, nfr_planner]
    instruction: Call write_section per your system prompt for your section.
    require_tool: write_section

  - id: critic_r1
    who: [critic]
    instruction: 'Round1: append_section; entry starts with "## Round 1\n-".'
    require_tool: append_section

  - id: revise
    who: [decomposer, risk_grader, case_generator, nfr_planner]
    instruction: Read Critic; write_section only if relevant, else one sentence "no changes this round".

  - id: critic_r2
    who: [critic]
    instruction: 'Round2: append_section "## Round 2\n-"; if no blocking issues write "Round2 passed".'
    require_tool: append_section

  - id: finalize
    who: [supervisor]
    instruction: finalize_artifact(decision="approved" or "needs_rework", rationale=one sentence).
    require_tool: finalize_artifact
---

`Requirements` is injected by workflow `initial_artifact`.

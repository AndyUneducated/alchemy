---
artifact:
  enabled: true
  initial_sections:
    - {name: notes, mode: append}
    - decision
  tool_owners:
    propose_vote: moderator
    finalize_artifact: moderator

agents:
  - name: mod
    role: moderator
    prompt: |
      你是演示主持人。这是 artifact 工具链的烟囱测试，请严格按 instruction 调用工具。
      每次发言 ≤ 20 字。用中文回答。
    max_tokens: 200

  - name: alice
    role: member
    prompt: |
      你是 alice。这是 artifact 工具链的烟囱测试，请严格按 instruction 调用工具。
      每次发言 ≤ 20 字。用中文回答。
    max_tokens: 160

  - name: bob
    role: member
    prompt: |
      你是 bob。这是 artifact 工具链的烟囱测试，请严格按 instruction 调用工具。
      每次发言 ≤ 20 字。用中文回答。
    max_tokens: 160

steps:
  - id: alice_append
    who: [alice]
    instruction: |
      先调用 append_section(name="notes", entry="- alice: hello")。
      然后故意调用 write_section(name="notes", content="wipe")——你会看到 error 回传，说明此节 append-only；请承认错误并结束本轮发言。

  - id: bob_append
    who: [bob]
    instruction: |
      调用 append_section(name="notes", entry="- bob says hi")，然后一句话打个招呼。

  - id: open_vote
    who: moderator
    instruction: |
      先调用 read_artifact() 查看 notes 节当前内容并简述，再调用 propose_vote(question="选 A 还是 B?", options=["A","B"])。

  - id: ballot
    who: member
    require_tool: cast_vote
    instruction: |
      调用 cast_vote(vote_id="v1", option="A" 或 "B", rationale="一句话理由")，然后用一句话解释你的选择。

  - id: finalize
    who: moderator
    instruction: |
      先调用 write_section(name="decision", content="最终结论：<按投票结果>"),
      再调用 finalize_artifact(decision="A" 或 "B", rationale="多数票结果") 落定。
---

# Artifact 工具链烟囱测试

最小场景，端到端验证 `ArtifactStore` 的 6 个工具（`read/write/append/propose_vote/cast_vote/finalize`），同时让 alice 故意对 append-only 的 `notes` 节调用 `write_section` 触发一次 mode 冲突——期望看到 stderr 一行 WARNING，随后 agent 自纠并完成后续指令。

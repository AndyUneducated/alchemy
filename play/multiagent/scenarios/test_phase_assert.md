---
rounds: 1

artifact:
  enabled: true
  initial_sections:
    - notes

moderator:
  name: mod
  prompt: |
    你是烟囱测试的主持人。严格按 instruction 调用工具，每次发言 ≤ 20 字。用中文回答。
  max_tokens: 200

members:
  - name: alice
    prompt: |
      你是 alice。你只关心闲聊，从不主动调用工具。每次发言 ≤ 20 字，用中文回答。
    max_tokens: 160
    temperature: 0.3
  - name: bob
    prompt: |
      你是 bob。你只关心闲聊，从不主动调用工具。每次发言 ≤ 20 字，用中文回答。
    max_tokens: 160
    temperature: 0.3

opening:
  - who: mod
    instruction: |
      调用 propose_vote(question="选 A 还是 B?", options=["A","B"])。

main:
  - round: 1
    who: members
    require_tool: cast_vote
    max_retries: 1
    instruction: |
      先用一句话随便打个招呼。

closing:
  - who: mod
    instruction: |
      读取 artifact 并一句话总结投票情况。
---

# Phase assert 烟囱测试

用来验证 `require_tool` + `max_retries` 的完整闭环：

- opening：`mod` 发起 vote `v1`
- main round 1：`who: members`，`require_tool: cast_vote`，`max_retries: 1`
  - instruction 故意只让 member 打招呼、**不提投票**——members 大概率第一轮跳过 cast_vote
  - 引擎应自动追加 nudge instruction "你刚才没调用 cast_vote..."
  - 终端应出现 `🔁 [alice] retry 1/1: missing cast_vote`
  - 若 retry 成功 → 正常记录 cast_vote 事件
  - 若 retry 仍失败 → stderr 打印 `WARNING: alice skipped required tool 'cast_vote' after 2 attempt(s)`
- closing：`mod` 总结

期望：两个 member 要么正常投，要么通过 retry 投，要么最终收到 WARNING。任何一种都证明 phase assert 在工作。

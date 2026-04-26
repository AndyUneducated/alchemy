---
artifact:
  enabled: true
  initial_sections:
    - notes
  tool_owners:
    propose_vote: moderator
    finalize_artifact: moderator

agents:
  - name: mod
    role: moderator
    prompt: |
      你是烟囱测试的主持人。严格按 instruction 调用工具，每次发言 ≤ 20 字。用中文回答。
    max_tokens: 200

  - name: alice
    role: member
    prompt: |
      你是 alice。你只关心闲聊，从不主动调用工具。每次发言 ≤ 20 字，用中文回答。
    max_tokens: 160
    temperature: 0.3

  - name: bob
    role: member
    prompt: |
      你是 bob。你只关心闲聊，从不主动调用工具。每次发言 ≤ 20 字，用中文回答。
    max_tokens: 160
    temperature: 0.3

steps:
  - id: open_vote
    who: moderator
    instruction: |
      调用 propose_vote(question="选 A 还是 B?", options=["A","B"])。

  - id: ballot
    who: member
    require_tool: cast_vote
    max_retries: 1
    instruction: |
      先用一句话随便打个招呼。

  - id: close
    who: moderator
    instruction: |
      读取 artifact 并一句话总结投票情况。
---

# require_tool 烟囱测试

验证 `require_tool` + `max_retries` 闭环：

- step `open_vote`：mod 发起 vote `v1`
- step `ballot`：`who: member`，`require_tool: cast_vote`，`max_retries: 1`
  - instruction 故意只让 member 打招呼、**不提投票**——member 大概率第一轮跳过 cast_vote
  - 引擎应自动追加 nudge instruction "你刚才没调用 cast_vote..."
  - 终端应出现 `🔁 [alice] retry 1/1: missing cast_vote`
  - retry 成功 → 正常记录 cast_vote 事件
  - retry 仍失败 → stderr 打印 `WARNING: alice skipped required tool 'cast_vote' after 2 attempt(s)`
- step `close`：mod 总结

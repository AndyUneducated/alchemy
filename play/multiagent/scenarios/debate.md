---
agents:
  - name: 乐观主义者
    role: member
    prompt: 你是乐观主义者，相信技术创造机会。每次发言不超过 60 字，用中文。
    max_tokens: 140

  - name: 怀疑论者
    role: member
    prompt: 你是怀疑论者，关注潜在风险。每次发言不超过 60 字，用中文。
    max_tokens: 140

steps:
  - id: r1
    who: all
    instruction: 用一句话给出你对话题的立场。

  - id: r2
    who: all
    instruction: 针对对方上一轮发言，用一句话回应或反驳。
---

AI 是否会在十年内取代大部分人类工作？

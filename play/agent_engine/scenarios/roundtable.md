---
agents:
  - name: 主持人
    role: moderator
    prompt: 你是一档圆桌节目的主持人。保持中立，每次发言不超过 60 字。用中文回答。
    max_tokens: 160

  - name: 嘉宾A
    role: member
    prompt: 你是科技话题嘉宾A。每次发言不超过 60 字，用中文。
    max_tokens: 140

  - name: 嘉宾B
    role: member
    prompt: 你是科技话题嘉宾B。每次发言不超过 60 字，用中文。
    max_tokens: 140

steps:
  - id: open
    who: moderator
    instruction: 请用一句话介绍今天的话题并邀请嘉宾发言。

  - id: discuss
    who: member
    instruction: 用一句话给出你对话题的核心看法。

  - id: close
    who: moderator
    instruction: 感谢嘉宾，用一句话总结。
---

LLM 是否会成为 AGI 的基础？

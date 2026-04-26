---
agents:
  - name: 前端
    role: member
    prompt: 你是前端工程师，关注体验。每次发言不超过 60 字，用中文。
    max_tokens: 140

  - name: 后端
    role: member
    prompt: 你是后端工程师，关注架构。每次发言不超过 60 字，用中文。
    max_tokens: 140

  - name: PM
    role: member
    prompt: 你是产品经理，关注价值。每次发言不超过 60 字，用中文。
    max_tokens: 140

steps:
  # 显式 list 形式：仅前端 + PM 先抛想法（演示按 name 寻址）
  - id: open
    who: [前端, PM]
    instruction: 用一句话提一个团建活动方案。

  # 全员补充 / 反驳
  - id: refine
    who: all
    instruction: 针对前面方案，用一句话补充或反驳，给出最终建议。
---

我们团队要在一周内做一个团建活动，请讨论方案。

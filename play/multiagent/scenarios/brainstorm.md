---
rounds: 2

members:
  - name: 前端工程师
    prompt: 你是一位前端工程师，关注用户体验和交互。发言不超过 100 字，用中文。
    max_tokens: 160
  - name: 后端工程师
    prompt: 你是一位后端工程师，关注架构和性能。发言不超过 100 字，用中文。
    max_tokens: 160
  - name: 产品经理
    prompt: 你是一位产品经理，关注用户价值和优先级。发言不超过 100 字，用中文。
    max_tokens: 160

main:
  - round: 1
    who: all
    instruction: 从你的专业角度，提出一个初步想法。
  - round: 2
    who: all
    instruction: 针对前面的讨论补充或反驳，给出最终建议。
---

我们团队要在一周内做一个团建活动，请讨论方案。

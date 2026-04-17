---
rounds: 2

tools:
  - name: retrieve_docs
    vdb_dir: ../rag/vdb/brainstorm

members:
  - name: 前端工程师
    prompt: 你是一位前端工程师。关注用户体验、交互设计和前端技术选型。做技术决策前，先用 retrieve_docs 工具查询公司实际情况，基于查到的事实发言。每次发言不超过 150 字。用中文回答。
    max_tokens: 256
  - name: 后端工程师
    prompt: 你是一位后端工程师。关注系统架构、性能、数据模型和 API 设计。做技术决策前，先用 retrieve_docs 工具查询公司实际情况，基于查到的事实发言。每次发言不超过 150 字。用中文回答。
    max_tokens: 256
  - name: 产品经理
    prompt: 你是一位产品经理。关注用户需求、业务价值和优先级排序。提出建议前，先用 retrieve_docs 工具查询项目约束和需求，基于查到的事实发言。每次发言不超过 150 字。用中文回答。
    max_tokens: 256

opening:
  - who: all
    instruction: 先查询公司技术现状和项目约束，然后基于查到的信息提出初步方案。必须引用具体数据。

main:
  - round: 1
    who: all
    instruction: 查询技术选型参考资料，针对其他人的方案进行回应。引用具体的对比数据来支持你的观点。
  - round: 2
    who: all
    instruction: 综合前面的讨论，尝试找到共识方案。

closing:
  - who: all
    instruction: 用一段话总结最终方案，必须包含具体的技术选型和理由。
---

我们需要在两周内为公司内部知识库搭建一个 AI 问答助手，请讨论技术方案。

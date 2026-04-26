---
tools:
  - name: retrieve_docs
    vdb_dir: ../../rag/vdb/test_vdb

agents:
  - name: A
    role: member
    prompt: 你必须先调用 retrieve_docs 工具查询文档，再根据检索结果回答。复杂或语义模糊的查询可传 rerank=true 提升精度。禁止凭记忆回答。回答只说一句话，格式："<字段> 是 <值>"。用中文。
    max_tokens: 128

  - name: B
    role: member
    prompt: 你必须先调用 retrieve_docs 工具查询文档，再根据检索结果回答。复杂或语义模糊的查询可传 rerank=true 提升精度。禁止凭记忆回答。回答只说一句话，格式："<字段> 是 <值>"。用中文。
    max_tokens: 128

steps:
  - id: q1
    who: [A]
    instruction: 查询"项目代号"，然后回答：项目代号是 ___。

  - id: q2
    who: [B]
    instruction: 查询"服务器编号"，然后回答：服务器编号是 ___。

  - id: q3
    who: [A]
    instruction: 查询"负责人编号"，然后回答：负责人编号是 ___。

  - id: q4
    who: [B]
    instruction: 查询"启动日期"，然后回答：启动日期是 ___。
---

请从文档中查询事实并回答。

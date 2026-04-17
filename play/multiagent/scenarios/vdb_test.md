---
rounds: 2

tools:
  - name: retrieve_docs
    vdb_dir: ../rag/vdb/vdb_test

members:
  - name: A
    prompt: 你必须先调用 retrieve_docs 工具查询文档，再根据检索结果回答。禁止凭记忆回答。回答只说一句话，格式："<字段> 是 <值>"。用中文。
    max_tokens: 128
  - name: B
    prompt: 你必须先调用 retrieve_docs 工具查询文档，再根据检索结果回答。禁止凭记忆回答。回答只说一句话，格式："<字段> 是 <值>"。用中文。
    max_tokens: 128

main:
  - round: 1
    who: A
    instruction: 查询"项目代号"，然后回答：项目代号是 ___。
  - round: 1
    who: B
    instruction: 查询"服务器编号"，然后回答：服务器编号是 ___。
  - round: 2
    who: A
    instruction: 查询"负责人编号"，然后回答：负责人编号是 ___。
  - round: 2
    who: B
    instruction: 查询"启动日期"，然后回答：启动日期是 ___。
---

请从文档中查询事实并回答。

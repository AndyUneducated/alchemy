# TODO — RAG 索引工具

## P1

### query.py 返回结构化结果

当前 `query()` 直接 print 到终端。应拆分出 `search()` 函数返回 `list[dict]`（含 `document`、`metadata`、`distance`），供 multiagent 集成时程序化调用。

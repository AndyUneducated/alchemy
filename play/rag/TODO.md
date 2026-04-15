# TODO — RAG 索引工具

## P1

### Ollama 连接失败友好提示

`ollama_embedding.py` 的 `__call__` 中，`urllib.request.urlopen` 失败时抛出裸 `URLError`。应捕获并给出明确提示（"请确认 Ollama 正在运行且已拉取指定模型"）。

### query.py 返回结构化结果

当前 `query()` 直接 print 到终端。应拆分出 `search()` 函数返回 `list[dict]`（含 `document`、`metadata`、`distance`），供 multiagent 集成时程序化调用。

## P2

### 提取 get_embedding_function() 工厂函数

`ingest.py` 和 `query.py` 中的 `if EMBED_BACKEND == "ollama"` 条件导入块完全重复。应在 `config.py` 中提取为 `get_embedding_function()` 工厂函数，新增 backend 时只改一处。

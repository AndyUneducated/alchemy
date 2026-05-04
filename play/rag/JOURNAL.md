# Journal

按里程碑记录每日进展。每条以 `## YYYY-MM-DD — 里程碑标题` 开头；同一自然日 ≤2 个里程碑。**功能** / **技术** 两段必填；**取舍** 仅在当日产出影响后续的取舍时记一笔，指向 [`DECISIONS.md`](DECISIONS.md) 完整条目而不在此重复。

## 2026-04-15 — 首个 RAG PoC：ChromaDB + Ollama + 段落感知 chunker

### 功能

- `play/rag/` 项目从零起步：`ingest.py` / `query.py` 两个独立 CLI
- `ingest.py` 支持 `.txt / .md / .pdf` 混合输入（文件 / 目录 `nargs="+"`），`upsert` 而非 `add` 让重 ingest 幂等
- 首批数据：6 篇 panel 场景的角色档案，作为 `play/agent_engine`（彼时叫 `play/multiagent`） 的私有背景知识

### 技术

- 技术栈拍板：**embedded ChromaDB（`PersistentClient(path=...)`） + Ollama embedding（默认 `qwen3-embedding:8b`）**——VDB 即一个目录可 `cp -r` 迁移；与 multiagent 主推理共用 ollama runtime
- `chunker.py`：`split_text(text, chunk_size=512, overlap=64)` 段落感知切分——按 `\n\n` 切段落、贪心打包、超长字符硬切、overlap 用尾部完整段落回带
- `ollama_embedding.py`：包装 ChromaDB `EmbeddingFunction` 接 Ollama `/api/embed`
- 命名约定：**collection 名 = `basename(--output)`**——目录名即 collection 名，作者不必想两个名字
- multiagent 默认模型同步切到 `qwen2.5:32b`（本机已 pull）

### 取舍

- ChromaDB vs Qdrant / Weaviate / pgvector / FAISS / 高层框架（LangChain / LlamaIndex） → DECISIONS §1
- Ollama vs OpenAI API / sentence-transformers 直跑 → DECISIONS §1
- 自写段落感知 chunker 而非引 LangChain `RecursiveCharacterTextSplitter`（不为一个 80 行工具拖 500MB 依赖树） → DECISIONS §2

## 2026-04-16 — 结构化 search API + `--json` subprocess 契约

### 功能

- `query.py` 新增 `--json` 输出模式：stdout 仅 JSON envelope，warnings / 进度走 stderr，subprocess 消费者 `json.loads(stdout)` 即可
- 新 brainstorm scenario（agent_engine 侧）演示 RAG-backed tool use；`vdb_test.md` 作为最小 RAG tool-call 回归测试
- 文档按 per-scenario 子目录重组（`docs/panel/` / `docs/test_vdb/` 等）；`OLLAMA_BASE_URL` 跨 multiagent + rag 两边统一

### 技术

- API 分层：拆 `search()` 纯函数 + `query()` 薄 pretty-print 包装，CLI 是更薄一层；可被 multiagent 通过 `subprocess.run([python, query.py, --json])` 程序化调用
- 数据契约 `SearchResult` TypedDict：`content / score / source / metadata` 四字段——**字段去 chroma 化**（不叫 `document` / `distance`，避免绑 provider）
- `score = 1.0 / (1.0 + distance)` 把 ChromaDB 距离转「相似度」（越大越相似），调用方不必知道底层是 L2 / cosine
- 同 commit 在 multiagent 侧：剥掉 scenario-default 参数不暴露给 LLM（只保留 LLM 真正要填的），消除 `sys.path.insert` 跨项目 import → DECISIONS §4 of agent_engine

### 取舍

- 拆 `search` 纯函数 + `query` 薄包装（vs `query` 自带 format）；TypedDict 而非 ChromaDB 原生 dict（vs 绑 provider）；`--json` envelope（vs 解析 stdout 文本） → DECISIONS §3
- subprocess + JSON envelope 是与 multiagent 解耦的核心契约；后来 phase 4 evals 复用同模式

## 2026-04-25 — Hybrid retrieval：dense + BM25 + RRF 默认开启

### 功能

- **hybrid 默认开启**：`mode={dense, bm25, hybrid}` 三选一，hybrid 是新默认；`dense` / `bm25` 留作诊断（不再是兼容层）
- 稀有专名 / 编号场景（"ZX-7492" / "SRV-8831"）召回率显著高于纯 dense
- CLI `--json` envelope 从裸数组 `[hit, ...]` **破坏性升级**到 `{query, data, meta}`，对齐 OpenAI Vector Store / Pinecone / Cohere 共同子集；`search()` Python API 不变（仍返 `list[SearchResult]`）
- per-hit `metadata.retrieval` / `metadata.reranked` 标注每条结果来源路径，下游不依赖 envelope `meta` 也能识别 provenance

### 技术

- 新模块：`bm25.py`（dense_search / bm25_search / rrf_fuse 三个纯函数）、`tokenizer.py`（HF tokenizer 包装 + `lru_cache`）、`prefetch.py`（一次性拉 HF 资产到 cache）
- BM25 实现：`rank-bm25.BM25Okapi`，纯 Python 零原生依赖，pickle 序列化整个倒排索引
- **关键工程对偶**：BM25 tokenizer 复用 embedding 模型同款 BPE（Qwen3-Embedding-8B），与 dense 端 tokenization 同源；跨语言（CJK / 拉丁 / 代码 / emoji）一致
- 融合策略：**RRF（Reciprocal Rank Fusion，`k=60`）**——只用排名不用 score，免 normalize；Cormack et al. 2009 经典默认；Elasticsearch 8.8+ 官方 hybrid 也是 RRF
- 索引存储：`bm25.pkl` 与 chroma 同目录，VDB 仍是单目录可 `cp -r` 迁移
- VDB 自描述扩展：`metadata.json` 新增 `tokenizer` 哨兵，query 端读回——「VDB 自描述」原则的延伸
- 召回 oversample：dense / bm25 各召 `top_k * HYBRID_OVERSAMPLE`（=4）进 RRF，截 `top_k`

### 取舍

- BM25 vs SPLADE / TF-IDF；rank-bm25 vs pyserini / 自写；HF tokenizer vs jieba / 正则；RRF vs weighted-sum / LTR；pickle vs Tantivy / Whoosh → DECISIONS §4
- envelope 升级是刻意 BREAKING（solo project 不付兼容税；为未来加 pagination / timing / version 一次到位） → DECISIONS §4

## 2026-04-25 — Cross-encoder reranker（两阶段 retrieval）

### 功能

- `--rerank` flag 显式开启 cross-encoder 精排（默认 off，避免每次启动加载 ~1.2GB 模型）
- 模型 `BAAI/bge-reranker-v2-m3`（多语言 + 中英 / 代码 / emoji 友好）
- 每条 hit 标 `metadata.reranked = True`、envelope `meta.reranked = True`，下游可对账
- agent_engine 侧 `retrieve_docs` 工具通过 OpenAI tool schema 把 `mode` + `rerank` 暴露给 LLM；`scenarios/test_vdb.md` prompt nudge LLM 在歧义 query 上 `rerank=true`

### 技术

- `reranker.py`：`sentence-transformers.CrossEncoder` + `lru_cache(1)` 单例 lazy load——首次调用时下载 + 加载（~5s on M-series Mac），之后零启动开销
- 召回池大小 `K=20`：BEIR / MS MARCO 经验值；20 对 cross-encoder 在 M-series Mac 上耗时可忽略
- agent_engine 侧 `_retrieve_docs` 把 rag CLI envelope 解包为 slim `{data, meta:{mode, reranked, top_k}}` 给 LLM——HTTP envelope ↔ SDK 解列表两层分工，对齐 OpenAI SDK 风格
- ToolTracer preview 升级：从「三键 dict」改为 `[N items, mode=..., reranked]` 信息密度更高的字符串

### 取舍

- `bge-reranker-v2-m3` vs Cohere `rerank-3` API / FlashRank / LLM-as-judge；CrossEncoder vs 直接 transformers 自写 batch 推理；K=20 vs K=10/50+；default off vs default on → DECISIONS §5
- **重排不能挽回召回的漏**：若 hybrid 第一阶段把正确文档排在 K=20 之外，reranker 也救不回来（持续 trade-off） → DECISIONS §5

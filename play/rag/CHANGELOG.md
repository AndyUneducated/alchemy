# Changelog

`CHANGELOG.md` 同时承担变更日志与 ADR 归档。每条记录使用 `## n. 变更标题`，下一行写 `- 日期：...`，再写正文。后续每个自然日建议最多追加 1～2 条 tech decision。
### Project conventions / ADR vocabulary
### 指导原则

1. **VDB 自描述**：embedding-model、chunk 参数、tokenizer 跟数据走，消除"用错模型查对的库"的静默失败
2. **Library API 先于 CLI**：先设计可被程序化调用的函数，CLI 是薄包装
3. **抽象滞后于第二个使用者**：单 backend 就不留 `if/elif` 分支
4. **优先库自带能力，不造轮子**：ChromaDB 官方提供的就不自写

## 1. 技术栈：ChromaDB + Ollama
- 日期：2026-04-15
### Context

本地可跑、一 agent 一库、可被工具链调用、workshop 可复现。要给 multiagent 的 panel/brainstorm 角色喂私有背景。

### Options considered

**向量数据库**：

- **ChromaDB**（选择）：embedded，`PersistentClient(path=...)` 零运维，VDB = 一个目录
- Qdrant / Weaviate：要跑 server 进程，多一层运维
- pgvector：要 Postgres，单人项目过重
- FAISS：无 metadata filter / persistence，要自己包
- LlamaIndex / LangChain（高层框架）：绑心智模型，VDB 选择被框架抽象遮蔽

**Embedding 提供方**：

- **Ollama**（选择）：本地 + 标准 HTTP API，**与 multiagent 主推理共用 runtime**
- OpenAI API：质量高但要 key + 走网，违反"本地 + workshop 可复现"
- sentence-transformers 直跑：本地但要管模型权重 / 设备调度
- 自跑 transformer：造轮子

**Embedding 模型**（在 Ollama 内）：

- **`qwen3-embedding:8b`**（选择）：MTEB 多语言第一梯队（与 bge-m3 同档），中英混合场景鲁棒，与 Qwen 系生态对齐
- `nomic-embed-text`：~137M 轻量，英语主导，中文 workshop 场景质量明显弱
- `bge-m3` 直跑（HF）：质量同档，但要绕过 Ollama runtime——失去"与 multiagent 共用 runtime"的对偶

### Decision

- ChromaDB `PersistentClient(path=vdb_dir)`——VDB 即一个可 `cp -r` 迁移的目录
- Ollama 的 `qwen3-embedding:8b` 作默认；`nomic-embed-text` 作轻量替代
- **Collection 名 = `basename(output_dir)`**——目录名即 collection 名，用户不必想两个名字
- `collection.upsert()` 而非 `add()`——重跑 ingest 幂等，迭代内容无需手动清库

### 行业光谱

ChromaDB + Ollama 是 local-first RAG 教程的最大公约数（LangChain / LlamaIndex 入门 / Ollama cookbook 都用这组合）。`upsert` 替 `add` 是数据工程标配（dbt / Airbyte），多数 starter RAG 教程用 `add()` 会重复写——这一点比教程更严谨。

### 工程维度评估

| 维度 | 评估 |
|---|---|
| 内聚度 | 高——`ingest` / `query` 各司其职 |
| 耦合度 | 低——只依赖 `chromadb` + `pymupdf`；Ollama 走 HTTP 解耦 |
| 可观测性 / 可审计性 | 中——stdout 打 chunk 数、embedding 进度，无结构化 log |
| LLM 不确定性容忍 | N/A |
| 向后兼容 / 演化友好 | 项目起点；目录式 VDB 可直接 `rm -rf` 重来 |
| 学习曲线 | 低——两个 CLI，每个 ≤5 参数 |
| 可测试性 | 高——纯函数 pipeline，可独立验证 |

## 2. 段落感知 chunker
- 日期：2026-04-15
### Context

中文 profile 文档（角色档案、事实清单等结构化纯文本）切得"不破坏语义单元"。Chunking 直接决定召回上限。

### Options considered

- 固定字符切：从段落中间切，破语义
- LangChain `RecursiveCharacterTextSplitter`：成熟，但为一个 80 行工具拖 500MB 依赖树
- LlamaIndex `SentenceSplitter`：中文句号不统一（`。/./？！`），需额外处理
- Semantic chunking（按 embedding 相似度合并）：质量最高，但 ingest 阶段要多一轮 embedding，workshop 场景太慢
- **自写段落感知**（选择）：`\n\n` 切段落 → 贪心打包 → 超长字符硬切 → overlap 用尾部完整段落回带

### Decision

`split_text(text, chunk_size=512, overlap=64)`：

- `chunk_size=512` ≈ 大多数中文 embedding 模型的 token 经济区间（512 tokens ≈ 700-1000 中文字符，留余量）
- `overlap=64` ≈ chunk_size 的 12%，是 LangChain / LlamaIndex 文档的常见经验值
- **关键**：overlap 用尾部完整段落回带，永远不从段落中间开始

### 行业光谱

最像 LangChain `RecursiveCharacterTextSplitter(separators=["\n\n"])` 的极简实现。放弃 LangChain 是为了**不为一个 80 行工具拖 500MB 依赖树**。不做 semantic chunking 是因为 ingest 的 embedding 调用已够慢，再嵌一层失去 workshop 友好度。

### 工程维度评估

| 维度 | 评估 |
|---|---|
| 内聚度 | 高——`chunker.py` 独立无状态函数 |
| 耦合度 | 极低——只从 config 读默认参数 |
| 可观测性 / 可审计性 | 中——ingest 打印"file: N chunks"；chunk 可 `query --top-k` 目检 |
| 向后兼容 / 演化友好 | 完全兼容——参数化 |
| 学习曲线 | 低——`chunk_size` / `overlap` 两个语义明确的旋钮 |
| 可测试性 | 高——纯函数，易 property test |

### 持续 trade-off

**对结构化纯文本效果好；对长篇无段落 PDF（OCR 产物、法律合同）会大量回退字符硬切，召回下降**。那种场景应换 LangChain `RecursiveCharacterTextSplitter` 或 semantic chunking——现阶段 YAGNI。

## 3. 结构化 search API + `--json` subprocess 契约
- 日期：2026-04-16
### Context

初版 `query(...)` 直接 `print()` 到 stdout——CLI 能用，但 multiagent 工具集成要程序化调用。需要明确两端数据契约。

### Options considered

- API 分层：`query` 自带 format vs **拆 `search()` 纯函数 + `query()` 薄包装**（选择）
- 数据结构：返回 ChromaDB 原生 dict（绑 provider）vs **provider-agnostic TypedDict**（选择）
- Subprocess 通信：stdout 解析文本（脆弱）vs **`--json` flag 输出 JSON envelope**（选择）

### Decision

```python
class SearchResult(TypedDict):
    content: str      # 不叫 "document"（chroma 用语）
    score: float      # 不叫 "distance"（语义翻转）
    source: str
    metadata: dict

def search(...) -> list[SearchResult]: ...  # 纯函数
def query(...) -> None:  pretty_print(search(...))
# CLI: --json → JSON envelope {query, data, meta} 到 stdout
```

**关键设计点**：

- **Score = `1.0 / (1.0 + distance)`**：把 ChromaDB 距离转"相似度"（越大越相似），调用方不必知道底层是 L2 / cosine——永远看到"越大越好"的一致约定
- **字段去 chroma 化**：`content` 而非 `document`，`source` 提到顶层而非埋在 `metadata`——未来换 VDB 或加 hybrid search 时不改 schema
- **Stdout 纯 JSON、stderr 装饰**：subprocess 消费者 `json.loads(stdout)` 即可

### 行业光谱

- "Library API + 薄 CLI 包装"：click / typer 应用的标配分层
- Provider-agnostic 结构体：对标 LangChain `Document`、LlamaIndex `NodeWithScore`——不绑 VDB 实现
- `{query, data, meta}` envelope：OpenAI Vector Store search、Pinecone query response、Cohere rerank response 的共同子集；`data` + `meta` 两层切分让"加 pagination / timing / version"成为加法式演化
- Stdout 数据 / stderr 装饰：Unix pipe 约定，MCP 协议同此切分——与 multiagent 走 subprocess 是**天然对偶**

### 工程维度评估

| 维度 | 评估 |
|---|---|
| 内聚度 | 高——`search` 返回数据、`query` 负责视图、`main` 负责 CLI，三层分明 |
| 耦合度 | 低——`SearchResult` 只依赖 `TypedDict` 内建；multiagent 侧无需 `import rag` |
| 可观测性 / 可审计性 | 中——multiagent 侧 ToolTracer 能记 stdin/stdout；rag 自己无结构化 log |
| LLM 不确定性容忍 | 间接升——LLM 收到规范 JSON 比自由文本更容易正确引用 |
| 向后兼容 / 演化友好 | 加法式——`SearchResult` 可加字段；CLI 原行为保持 |
| 学习曲线 | 低——多一个 `--json` flag，脚本化使用者无感 |
| 可测试性 | 高——`search()` 纯函数 + TypedDict 返回，断言容易 |

## 4. Hybrid retrieval：dense + BM25 + RRF
- 日期：2026-04-25
### Context

纯 dense embedding 在三种场景上跌跤：

1. **稀有专名 / 编号**（"ZX-7492"、"SRV-8831"）——embedding 易把它们映成"项目代号"通用簇
2. **精确字面匹配**——dense 对完整字面的偏好弱于词袋
3. **OOD 词汇**（embedding 没见过的术语 / 缩写）——映射到任意位置

经典工业解：dense 召语义、BM25 召字面，融合一下。

### Options considered

**第二路检索**：

- TF-IDF：可行但 BM25 是它的进化版
- **BM25**（选择）：lexical 检索行业标配（Elasticsearch / Lucene 默认 scorer）
- SPLADE：神经稀疏，效果好但要 GPU + 训练，单人项目过重

**BM25 实现库**：

- **`rank-bm25.BM25Okapi`**（选择）：纯 Python，零原生依赖，一个 pickle 序列化整个倒排；Workshop 场景万级 chunks 秒级建索
- `pyserini`：Lucene 绑定，质量更高但拖 JVM
- 自写 BM25：经典公式，~50 行能实现，但 IDF / 长度归一化的边界 case 容易踩——不造这个轮子

**Tokenizer**：

- `jieba`：中文 NLP 圈标配，但只覆盖中文，混入英语/代码/emoji 时切分质量塌
- 正则切分：跨语言但低质，BM25 IDF 失真严重
- **HF tokenizer（Qwen3-Embedding-8B BPE）**（选择）：跨语言（CJK / 拉丁 / 代码 / emoji）一致；**与 dense 端 tokenization 同源**
- 自训分词器：YAGNI

**融合策略**：

- Weighted sum（α·dense + (1−α)·bm25）：要先 normalize 两路 score，且 α 难调
- **RRF**（选择）：只用排名不用 score，免 normalize；`k=60` 是 Cormack et al. 2009 经典默认；Elasticsearch 8.8+ 官方 hybrid 也是 RRF
- Learning-to-rank：要训练数据，过重

**索引存储**：

- 跑时重算：每次 query 全语料分词，慢且浪费
- **Pickle BM25Okapi 与 chroma 同目录**（选择）：VDB 仍是单目录可 `cp -r` 迁移；进程内 `lru_cache` 复用
- 上独立倒排引擎（Tantivy / Whoosh）：增加运维负担，单人项目过重

### Decision

- **Hybrid 默认开启**；`mode={dense, bm25}` 留作诊断（不是兼容层）
- **HF tokenizer 复用 embedding 模型同款 BPE**：分词层一致性提前还掉
- **Tokenizer sentinel**：ingest 写入 `metadata.json["tokenizer"]`，query 读回——VDB 自描述（指导原则 #1）的延伸
- **召回 oversample**：dense / bm25 各召 `top_k * HYBRID_OVERSAMPLE`（=4）进 RRF，截 `top_k`
- **CLI `--json` envelope 格式（BREAKING）**：从裸数组 `[hit, ...]` 升到 `{query, data, meta}`，对齐 OpenAI Vector Store / Pinecone / Cohere 共同子集；`search()` Python API 不变（仍返 `list[SearchResult]`），envelope 仅在 CLI 层包装——OpenAI SDK 同款 HTTP envelope ↔ SDK 解列表的两层分工
- **Per-hit `metadata.retrieval` / `metadata.reranked`**：标注每条结果的来源路径，下游不依赖 envelope `meta` 也能识别 provenance

### 行业光谱

- **Dense + sparse + RRF** 是 2024+ 工业 RAG 最大公约数：Pinecone hybrid / Vespa / Elasticsearch RRF / Weaviate hybrid 全是这套
- **HF tokenizer 复用 embedding 模型**：在 SPLADE / ColBERT 等"sparse-aware retriever"是强制的，纯 BM25 不强制——做了仍是干净姿势
- **Pickle 单文件索引**：toy / prototype RAG 实用做法（LangChain `BM25Retriever.save_local` 同款）；规模超百万段才需要 Tantivy / Lucene
- **CLI envelope `{query, data, meta}`**：OpenAI Vector Store search、Pinecone query response、Cohere rerank response 的共同子集；`data` 列表 + `meta` 对象的两层切分让"加 pagination / timing / version 字段"成为加法式演化

### 工程维度评估

| 维度 | 评估 |
|---|---|
| 内聚度 | 高——`bm25.py` 三个纯函数 + 一个 cache helper；`tokenizer.py` 单一职责；`search()` 编排 |
| 耦合度 | 中——`bm25.py` 不 import HF / config 业务参数（只 RRF_K）；query 端负责 tokenize 后传 tokens 进来 |
| 可观测性 / 可审计性 | 中升——envelope `meta` 暴露 mode / reranked / vdb；per-hit `metadata.retrieval` 下游可对账 |
| LLM 不确定性容忍 | 升——稀有专名 / 编号召回率显著高于纯 dense |
| 向后兼容 / 演化友好 | 加法式 + 一处 BREAKING——`search()` 函数签名加可选参数；CLI `--json` 改 envelope 是刻意 BREAKING（solo project 不付兼容税；为未来加 pagination / timing / version 一次到位） |
| 学习曲线 | 低——CLI 只多 `--mode` 一个选项，默认即是 hybrid |
| 可测试性 | 高——`tokenize` / `rrf_fuse` / `bm25_search` / `dense_search` 都是纯函数 |

### 持续 trade-off

- **BPE 子词在 BM25 上的 IDF 偏差**：BM25 经典理论是 word-level，BPE 把高频词切成子词后 IDF 分布失真——接受，换跨语言一致性 + 与 dense 端 tokenization 同源
- **Score 跨 mode 不可比**：dense 是 `1/(1+dist)`、bm25 是原始分（可能为负）、hybrid 是 RRF（~0.01-0.05）；同一次调用内排序正确，但跨 mode 阈值不可迁移
- **BM25 不做增量**：每次 ingest 全量重建——简化心智，YAGNI

## 5. Cross-encoder reranker
- 日期：2026-04-25
### Context

Hybrid retrieval（§4）改善召回，但 top-K 内**排序**仍受限：

- Bi-encoder（dense）单塔 query/doc 各自压成向量，**信息瓶颈在向量维度**——细粒度相关性丢失
- BM25 是 lexical 匹配，**不理解语义同义**（"营收" / "收入" / "revenue"）
- RRF 只看排名不看相关度，**不可能比两路输入更精**

行业经典两阶段：cheap retriever 拉候选池 → expensive cross-encoder 精排。

### Options considered

**重排模型**：

- **`BAAI/bge-reranker-v2-m3`**（选择）：~568M，多语言（CJK / EN / 代码），M3 系列 BEIR / MIRACL nDCG@10 业界领先
- `bge-reranker-base`（~110M）：更快但效果掉点；M2/M3 Mac 跑 v2-m3 没压力
- Cohere `rerank-3` API：质量同档，但要 key + 走网，与 §1 "本地 + workshop 可复现"原则冲突
- FlashRank：极轻量但仅英语主导，中文场景明显弱
- LLM-as-judge：通用但慢、贵、提示工程脆弱——这是 evaluation 工具，不是 production reranker

**调用层封装**：

- 直接 `transformers.AutoModelForSequenceClassification` + 自写 batch 推理：~100 行样板
- **`sentence-transformers.CrossEncoder`**（选择）：~5 行；自动设备选择；与 `SentenceTransformer` 对偶（bi-encoder 召回、cross-encoder 精排的"两兄弟"分工）

**召回池大小**：

- `K=10`：节省时间但漏召风险高
- **`K=20`**（选择）：BEIR / MS MARCO 经验值；20 对 cross-encoder 在 M2 mac 上耗时可忽略
- `K=50+`：边际收益递减

**默认开关**：

- 默认 ON：每次都加载 1.2GB 模型，CLI 启动慢
- **默认 OFF**（选择）：用户用 `--rerank` 显式开启；快速路径 vs 高质量路径用户自选

### Decision

- 模型 `BAAI/bge-reranker-v2-m3` + `sentence-transformers.CrossEncoder` + `lru_cache(1)` 单例 lazy load
- 默认 off，CLI `--rerank` 显式开启
- 每条 hit 标 `metadata.reranked = True`、envelope `meta.reranked = True`——下游可对账

### 行业光谱

- **Two-stage retrieval（retriever → reranker）**是 2024+ 主流 RAG 标准架构：LangChain / LlamaIndex / Haystack 文档都列为"production-grade RAG"必备
- **`bge-reranker` 系列**是 BEIR / MTEB 多语言开源 reranker 第一梯队（与 `cohere-rerank-3` 同档）；`v2-m3` 多语言鲁棒性比英语主导的 `bge-reranker-large` 更适合中英混合
- **default-off + lazy load + lru_cache** 是 ML 工具加载重模型的惯用 pattern：`whisper` / `transformers.pipeline` / `sentence-transformers` 全是这套

### 工程维度评估

| 维度 | 评估 |
|---|---|
| 内聚度 | 高——`reranker.py` 一个职责（CrossEncoder 包装）；`search()` 一行 if 编排 |
| 耦合度 | 低——`reranker.py` 不知道 hybrid / dense / bm25；输入输出都是 `SearchResult` |
| 可观测性 / 可审计性 | 升——每条 hit 有 `reranked` 标识，envelope `meta` 同步 |
| LLM 不确定性容忍 | 升——top-K 排序质量提升直接降低 LLM 选错段落的概率 |
| 向后兼容 / 演化友好 | 加法式——`rerank` 参数有默认值；不开就跟纯 hybrid 检索完全等价 |
| 学习曲线 | 低——CLI 多一个 `--rerank` flag；首次跑会自动下载，无需手动安装 |
| 可测试性 | 高——`rerank()` 是纯函数（除了模型加载副作用） |

### 持续 trade-off

- **CrossEncoder logit 与 RRF / dense / BM25 完全不同量纲**（典型 -3 到 5）：与 §4 同样的 score 跨 mode 不可比；同一次调用内单调正确，但跨调用阈值不可迁移
- **重排不能挽回召回的漏**：若 hybrid 第一阶段把正确文档排在 K=20 之外，reranker 也救不回来。先调大 `HYBRID_OVERSAMPLE` / `RERANK_CANDIDATES` 而不是换更强的 reranker
- **多语言 reranker 在纯英语 / 纯中文场景上略输于专精模型**：可接受；workshop 场景常中英混合，多语言鲁棒性优先于单语言极致质量

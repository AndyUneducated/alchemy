# Play/rag — 设计决策记录

本文档记录 `play/rag/` 的系统设计决策，按时间顺序积累。每节包含背景、候选方案、选择依据、行业光谱位置，以及按统一工程维度的评估。维度定义见 `../multiagent/DESIGN_DECISIONS.md` 的附录。

## 指导原则

1. **VDB 自描述**：embedding-model、chunk 参数跟数据走，消除"用错模型查对的库"的静默失败
2. **Library API 先于 CLI**：先设计可被程序化调用的函数，CLI 是薄包装
3. **抽象滞后于第二个使用者**：单 backend 就不要留 `if/elif` 分支
4. **优先库自带能力，不造轮子**：ChromaDB 官方提供的就不自写

---

## 1. 技术栈选型：ChromaDB + Ollama

- **日期**：2026-04-15

### Context

multiagent 的 panel / brainstorm 场景需要让 agent 按角色检索私有背景（CEO 只知道 CEO 知道的事）。约束：本地可跑、一 agent 一库、可被工具链调用、workshop 可复现。

### Options considered

**向量数据库**：
- ChromaDB（选择）：embedded，一行 `PersistentClient(path=...)` 零运维，VDB = 一个目录
- Qdrant / Weaviate：要跑 server；pgvector：要 Postgres；FAISS：无 metadata filter / persistence；上层框架（LlamaIndex / LangChain）：绑心智模型

**Embedding 提供方**：
- Ollama（选择）：本地 + 标准 HTTP API，**与 multiagent 主推理共用 runtime**
- OpenAI API：质量高但要 key + 走网；sentence-transformers：本地但要管模型权重；自跑 transformer：造轮子

### Decision

- ChromaDB `PersistentClient(path=vdb_dir)`——VDB 就是一个目录，`cp -r` 可迁移
- Embedding 先 `nomic-embed-text`，后升 `qwen3-embedding:8b`（中文友好）
- **Collection 命名用 `basename(output_dir)`**——目录名即 collection 名，避免让用户想两个名字
- `collection.upsert()` 而非 `add()`——重跑 ingest 幂等，迭代内容无需手动清库

### 行业光谱

ChromaDB + Ollama 是 local-first RAG 教程的最大公约数（LangChain 文档 / LlamaIndex 入门 / Ollama cookbook 都用这组合）。`upsert` 替 `add` 是数据工程标配（dbt / Airbyte），starter RAG 教程常用 `add()` 会重复写——这个选择比教程更严谨。

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

---

## 2. 段落感知 chunker

- **日期**：2026-04-15

### Context

Chunking 直接决定召回质量。目标：中文 profile 文档（CEO 背景、角色档案等结构化纯文本）切得"不破坏语义单元"。

### Options considered

- 固定字符切：会从段落中间切，破语义
- LangChain `RecursiveCharacterTextSplitter`：成熟但要拖整个 LangChain 依赖
- LlamaIndex `SentenceSplitter`：中文句号不统一（`。/./？！`），需额外处理
- Semantic chunking（按 embedding 相似度合并）：质量最高但 ingest 阶段要多一轮 embedding
- **自写段落感知**（选择）：`\n\n` 切段落，贪心打包，超长回退字符硬切

### Decision

`split_text(text, chunk_size=512, overlap=64)` 四步：

1. 按 `\n\n` 切段落
2. 贪心打包：加下一段超过 `chunk_size` 则封口
3. **Overlap 用尾部完整段落回带**——下一 chunk 以上个 chunk 最后若干段落起头，总长 ≤ `overlap`，**永远不从段落中间开始**
4. 单段落 > `chunk_size`：字符硬切（`_split_long`），最后过小碎片合回前一个

### 行业光谱

最像 LangChain `RecursiveCharacterTextSplitter(separators=["\n\n"])`——主流做法的极简实现。放弃 LangChain 是为了**不为一个 80 行工具拖 500MB 依赖树**。不做 semantic chunking 是因为 ingest 的 embedding 调用已够慢，再嵌一层失去 workshop 友好度。

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

**对结构化纯文本效果好；对长篇无段落 PDF（OCR 产物、法律合同）会大量回退字符硬切，召回下降**。那种场景应换 LangChain `RecursiveCharacterTextSplitter` 或 semantic chunking——现阶段不动。

---

## 3. VDB metadata.json：embedding-model 一致性哨兵

- **日期**：2026-04-15

### Context

RAG 经典失败模式：用 `model_A` embed 建库，用 `model_B` embed query——**不报错，静默返回低分垃圾**。这个坑文档和教程多数默认用户"自己记得"。

### Options considered

- 不做，靠用户自觉（starter 教程现状）
- 把 model 编进 collection 名（`panel_qwen3-embedding-8b`）：丑，换模型要重索引
- **独立 `metadata.json` sidecar**（选择）：VDB 目录自带身份证

### Decision

Ingest 写 `{vdb_dir}/metadata.json`，记 `embedding_model / chunk_size / chunk_overlap / doc_count / chunk_count / created_at`。

Query 启动时：

1. 读 metadata，默认**复用 stored model**
2. 用户 `--model` 指定且不同于 stored：**warn 但继续**（尊重 override）
3. 无 metadata：fallback 到 `config.py` default

### 行业光谱

对标**数据工程的 manifest / schema 文件**（dbt `manifest.json`、Parquet footer schema）——数据自描述、查询时双向验证。生产 RAG 后期多数会补上，starter 教程几乎都省略。**把 RAG 最容易出 silent failure 的点做结构性防御**，是刻意的。

### 工程维度评估

| 维度 | 评估 |
|---|---|
| 内聚度 | 高——metadata 写在 `ingest` 末尾，读在 `search` 开头，各一段 |
| 耦合度 | 低——文件系统 sidecar，不改 Chroma schema |
| 可观测性 / 可审计性 | 高——每个 VDB 自带身份证；`cat metadata.json` 即知来源 |
| LLM 不确定性容忍 | 消除 RAG 层面的静默错误 |
| 向后兼容 / 演化友好 | 加法式——缺 metadata fallback 到 default，老 VDB 不挂 |
| 学习曲线 | 零——对 CLI 使用者透明 |
| 可测试性 | 高——metadata 是纯数据，可 `assert json.load(...) == expected` |

---

## 4. 从过度抽象到合适抽象

- **日期**：2026-04-16

### Context

初版（4-15）两处过度设计，**一天后**（4-16）全删：

1. **`EMBED_BACKEND` 运行时分支**：`EMBED_BACKEND = "ollama"` + 注释 "more backends can be added" + `ingest.py` / `query.py` 各抄一遍 if/elif 导入块。旧 TODO 甚至专门立了 "P2: 提取 `get_embedding_function()` 工厂函数" 来消除重复
2. **自写 `ollama_embedding.py`**：30 行 urllib 直打 Ollama `/api/embed` + `ChromaDB EmbeddingFunction` 适配

### Options considered

- 按 TODO 提取工厂函数：干掉重复——但**这是在给从没出现过的需求做准备**。YAGNI 警报
- 留着 if/elif 先不管：包袱持续存在
- **彻底删掉 `EMBED_BACKEND`，硬编码单 backend**（选择）：承认"未来可能有第二个 backend"是幻觉
- 继续自写 embedding function：没理由，ChromaDB 官方 `OllamaEmbeddingFunction` 开箱即用

### Decision

一次 commit 里做四件事：

- 删 `EMBED_BACKEND`、删两文件 if/elif 分支、导入一行化
- 删 `ollama_embedding.py`，换 `chromadb.utils.embedding_functions.OllamaEmbeddingFunction`
- 关联两个 TODO（"Ollama 错误友好提示" / "提取 embedding 工厂"）同步删除——前者 ChromaDB 官方函数有更好的错误处理，后者单 backend 无工厂必要
- 顺带加 PDF 支持（pymupdf）——扩展 `_read_file` 按扩展名 dispatch

净效果：**~40 行代码删除 + 两个 TODO 消失 + 一个子模块消失**，零功能缺失。

### 行业光谱

**YAGNI** 教科书案例，更深一层是 **Rule of Three**——抽象应出现在第二个具体使用者出现、第三个被构想时，而不是第一个。对偶的是 Sandi Metz 的 "**The wrong abstraction is far more costly than no abstraction**"——错的抽象让你纠缠于接口和分支，删掉才能前进。

### 工程维度评估

| 维度 | 评估 |
|---|---|
| 内聚度 | 升——导入段由"条件分支 + 导入"变成单行导入 |
| 耦合度 | 降——自写 embedding function 删除后，Ollama HTTP API 格式的耦合转给 ChromaDB 维护 |
| 可观测性 / 可审计性 | 升——官方函数错误信息比自写更清楚 |
| 向后兼容 / 演化友好 | 破坏性——`EMBED_BACKEND` / `BASE_URL` 配置变量被删/改名；单人项目影响可控 |
| 学习曲线 | 降——少一层"如果哪天支持 X"的迷惑 |
| 可测试性 | 不变 |

---

## 5. 结构化 search API + `--json` subprocess 契约

- **日期**：2026-04-16

### Context

初版 `query(...)` 直接 `print()` 到 stdout——CLI 能用，但 multiagent 工具集成要程序化调用。同期 multiagent 侧决定走 subprocess 隔离（见 `../multiagent/DESIGN_DECISIONS.md` §4），两端需要明确的数据契约。

### Options considered

- API 分层：query 返回 generator + CLI 自 format vs **拆 `search()` + `query()` 薄包装**（选择）
- 数据结构：返回 ChromaDB 原生 dict（绑 provider）vs **provider-agnostic TypedDict**（选择）
- Subprocess 通信：stdout 解析文本（脆弱）vs **`--json` flag 输出 JSON 数组**（选择）

### Decision

```python
class SearchResult(TypedDict):
    content: str      # 不叫 "document"（chroma 用语）
    score: float      # 不叫 "distance"（语义翻转）
    source: str
    metadata: dict

def search(...) -> list[SearchResult]: ...  # 纯函数
def query(...) -> None:  hits = search(...); pretty_print(hits)
# CLI: --json flag → json.dumps(search(...)) 到 stdout
```

**关键设计点**：

- **Score = `1.0 / (1.0 + distance)`**：把 ChromaDB 的距离转"相似度"（越大越相似），调用方不必知道底层是 L2 / cosine——永远看到"越大越好"的一致约定
- **字段去 chroma 化**：`content` 而非 `document`，`source` 提到顶层而非埋在 `metadata`——未来换 VDB 或加 hybrid search 时不改 schema
- **`--json` 输出是纯 JSON 数组**，progress / warning 去 stderr——subprocess 消费者 `json.loads(stdout)` 即可

### 行业光谱

- "Library API + 薄 CLI 包装"：click / typer 应用的标配分层
- Provider-agnostic 结构体：对标 LangChain `Document`、LlamaIndex `NodeWithScore`——不绑 VDB 实现
- Stdout 纯数据、stderr 装饰：Unix pipe 约定，MCP 协议同此切分——与 multiagent 侧走 subprocess 是**天然对偶**

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

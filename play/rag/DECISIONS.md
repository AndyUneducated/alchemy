# Decisions

ADR (Architecture Decision Record) archive. Each entry starts with `## n. Title`, followed by `- **Status**` + `- **Date**` metadata; body uses `Context / Options considered / Decision / Industry landscape / Engineering dimension assessment / Ongoing trade-off` sections. **Append new decisions at end; update Status on supersession; never delete old entries**. Milestone progress in [`JOURNAL.md`](JOURNAL.md).

## 1. Stack: ChromaDB + Ollama

- **Status**: accepted
- **Date**: 2026-04-15

### Context

Run locally, one agent one index, callable from toolchains, workshop-reproducible. Feed private background to multiagent panel/brainstorm roles.

### Options considered

**Vector database**:

- **ChromaDB** (chosen): embedded, `PersistentClient(path=...)` zero-ops, VDB = one directory
- Qdrant / Weaviate: server process, extra ops
- pgvector: Postgres, too heavy solo
- FAISS: no metadata filter / persistence, DIY wrapper
- LlamaIndex / LangChain high-level: binds mental model, VDB choice hidden

**Embedding provider**:

- **Ollama** (chosen): local + standard HTTP API, **shares runtime with multiagent main inference**
- OpenAI API: quality but key + network; violates local + workshop reproducible
- sentence-transformers direct: local but model weights / device scheduling
- self-run transformer: reinventing wheel

**Embedding model** (within Ollama):

- **`qwen3-embedding:8b`** (chosen): MTEB multilingual first tier (same as bge-m3), robust CJK/EN mix, Qwen ecosystem aligned
- `nomic-embed-text`: ~137M light, English-dominant, weak Chinese workshop quality
- `bge-m3` direct (HF): same quality but bypasses Ollama — loses shared runtime pairing

### Decision

- ChromaDB `PersistentClient(path=vdb_dir)` — VDB is one `cp -r` migratable directory
- Ollama `qwen3-embedding:8b` default; `nomic-embed-text` light alternative
- **Collection name = `basename(output_dir)`** — directory name is collection name
- `collection.upsert()` not `add()` — idempotent re-ingest without manual clear

### Industry landscape

ChromaDB + Ollama is local-first RAG tutorial common denominator. `upsert` over `add` is data-engineering standard; most starter tutorials use `add()` and duplicate — we're stricter.

### Engineering dimension assessment

|Dimension|Assessment|
|---|---|
|Cohesion|High — `ingest` / `query` separate concerns|
|Coupling|Low — only `chromadb` + `pymupdf`; Ollama via HTTP decoupled|
|Observability / auditability|Medium — stdout chunk count, embedding progress; no structured log|
|LLM uncertainty tolerance|N/A|
|Backward compat / evolution|Project start; directory VDB can `rm -rf` rebuild|
|Learning curve|Low — two CLIs, ≤5 params each|
|Testability|High — pure function pipeline, independently verifiable|

## 2. Paragraph-aware chunker

- **Status**: accepted
- **Date**: 2026-04-15

### Context

Chinese profile docs (character sheets, fact lists, structured plain text) must not break semantic units when chunked. Chunking caps recall ceiling.

### Options considered

- Fixed char split: breaks mid-paragraph
- LangChain `RecursiveCharacterTextSplitter`: mature but 500MB dep tree for 80-line tool
- LlamaIndex `SentenceSplitter`: Chinese punctuation inconsistent
- Semantic chunking (embedding similarity merge): highest quality but extra embedding pass at ingest; too slow workshop
- **Custom paragraph-aware** (chosen): `\n\n` paragraphs → greedy pack → char hard-cut → overlap via full trailing paragraphs

### Decision

`split_text(text, chunk_size=512, overlap=64)`:

- `chunk_size=512` ≈ economical token range for most Chinese embedding models
- `overlap=64` ≈ 12% of chunk_size; common LangChain/LlamaIndex heuristic
- **Key**: overlap uses full trailing paragraphs, never starts mid-paragraph

### Industry landscape

Closest to LangChain `RecursiveCharacterTextSplitter(separators=["\n\n"])` minimal implementation. Skipped LangChain to avoid 500MB deps for 80 lines. Skipped semantic chunking — ingest embedding already slow enough.

### Engineering dimension assessment

|Dimension|Assessment|
|---|---|
|Cohesion|High — `chunker.py` stateless standalone|
|Coupling|Very low — only reads defaults from config|
|Observability / auditability|Medium — ingest prints "file: N chunks"; chunks inspectable via `query --top-k`|
|Backward compat / evolution|Fully compatible — parameterized|
|Learning curve|Low — `chunk_size` / `overlap` two clear knobs|
|Testability|High — pure function, property-test friendly|

### Ongoing trade-off

**Works well on structured plain text; long paragraph-less PDFs (OCR, contracts) fall back to char hard-cut and recall drops.** Switch to LangChain recursive or semantic chunking when needed — YAGNI now.

## 3. Structured search API + `--json` subprocess contract

- **Status**: accepted (CLI envelope §4 breaking upgrade bare array → `{query, data, meta}`; `search()` Python API unchanged)
- **Date**: 2026-04-16

### Context

Initial `query(...)` printed stdout — CLI OK but multiagent tool integration needed programmatic calls. Clear data contract both ends.

### Options considered

- API layering: `query` with format vs **split `search()` pure + `query()` thin wrapper** (chosen)
- Data shape: Chroma native dict vs **provider-agnostic TypedDict** (chosen)
- Subprocess: parse stdout text vs **`--json` JSON envelope** (chosen)

### Decision

```python
class SearchResult(TypedDict):
    content: str      # not "document" (chroma term)
    score: float      # not "distance" (inverted semantics)
    source: str
    metadata: dict

def search(...) -> list[SearchResult]: ...  # pure function
def query(...) -> None:  pretty_print(search(...))
# CLI: --json → JSON envelope {query, data, meta} to stdout
```

**Key design points**:

- **Score = `1.0 / (1.0 + distance)`**: distance → similarity (higher better); callers need not know L2/cosine
- **De-chromatized fields**: `content` not `document`; `source` top-level not buried in metadata
- **Stdout pure JSON, stderr decoration**: subprocess consumers `json.loads(stdout)`

### Industry landscape

Library API + thin CLI is click/typer standard. Provider-agnostic struct like LangChain `Document`. `{query, data, meta}` envelope matches OpenAI Vector Store / Pinecone / Cohere subset. Stdout data / stderr decoration is Unix pipe convention, MCP same split — natural pairing with multiagent subprocess.

### Engineering dimension assessment

|Dimension|Assessment|
|---|---|
|Cohesion|High — `search` returns data, `query` view, `main` CLI, three layers|
|Coupling|Low — `SearchResult` only TypedDict; multiagent needs no `import rag`|
|Observability / auditability|Medium — multiagent ToolTracer can log stdin/stdout; rag itself no structured log|
|LLM uncertainty tolerance|Indirectly up — LLM cites structured JSON more reliably than free text|
|Backward compat / evolution|Additive — `SearchResult` can grow fields; CLI behavior preserved|
|Learning curve|Low — one extra `--json` flag|
|Testability|High — `search()` pure + TypedDict assertions|

## 4. Hybrid retrieval: dense + BM25 + RRF

- **Status**: accepted
- **Date**: 2026-04-25

### Context

Pure dense embedding fails on:

1. **Rare proper nouns / IDs** ("ZX-7492", "SRV-8831") — embedding maps to generic "project code" cluster
2. **Exact literal match** — dense weak on full string preference vs bag-of-words
3. **OOD vocabulary** — maps arbitrarily

Classic industrial fix: dense for semantics, BM25 for literals, fuse.

### Options considered

**Second retrieval path**:

- TF-IDF: OK but BM25 is evolution
- **BM25** (chosen): lexical retrieval standard (Elasticsearch/Lucene default)
- SPLADE: neural sparse, GPU + training; too heavy solo

**BM25 library**:

- **`rank-bm25.BM25Okapi`** (chosen): pure Python, zero native deps, pickle whole inverted index; workshop 10k chunks builds in seconds
- `pyserini`: Lucene bind, JVM
- DIY BM25: ~50 lines but IDF/length norm edge cases — don't reinvent

**Tokenizer**:

- `jieba`: Chinese NLP standard but Chinese-only; English/code/emoji quality collapses mixed text
- regex split: cross-language but poor BM25 IDF
- **HF tokenizer (Qwen3-Embedding-8B BPE)** (chosen): cross-language consistent; **same tokenization as dense**
- custom tokenizer: YAGNI

**Fusion**:

- Weighted sum: normalize both scores, α hard to tune
- **RRF** (chosen): rank-only, no normalize; `k=60` Cormack 2009 default; Elasticsearch 8.8+ official hybrid
- learning-to-rank: needs training data; too heavy

**Index storage**:

- recompute each query: slow wasteful
- **Pickle BM25Okapi alongside chroma** (chosen): VDB still one `cp -r` directory; in-process `lru_cache`
- separate inverted engine: extra ops

### Decision

- **Hybrid default**; `mode={dense, bm25}` diagnostic only (not compat layer)
- **HF tokenizer same BPE as embedding model**: tokenization consistency upfront
- **Tokenizer sentinel**: ingest writes `metadata.json["tokenizer"]`, query reads back — VDB self-describing (README principle #1) extension
- **Recall oversample**: dense/bm25 each recall `top_k * HYBRID_OVERSAMPLE` (=4) into RRF, truncate `top_k`
- **CLI `--json` envelope (BREAKING)**: bare array → `{query, data, meta}` OpenAI/Pinecone/Cohere subset; `search()` API unchanged; envelope only at CLI — same HTTP envelope ↔ SDK list split as OpenAI SDK
- **Per-hit `metadata.retrieval` / `metadata.reranked`**: provenance on each hit without relying on envelope `meta`

### Industry landscape

Dense + sparse + RRF is 2024+ industrial RAG common denominator. HF tokenizer reuse with embedding model mandatory for SPLADE/ColBERT; good hygiene even for pure BM25. Pickle single-file index is toy/prototype standard. CLI `{query, data, meta}` envelope matches major vector APIs; additive pagination/timing/version evolution.

### Engineering dimension assessment

|Dimension|Assessment|
|---|---|
|Cohesion|High — `bm25.py` three pure functions + cache helper; `tokenizer.py` single duty; `search()` orchestrates|
|Coupling|Medium — `bm25.py` no HF/config business params import; query tokenizes then passes tokens|
|Observability / auditability|Medium up — envelope `meta` exposes mode/reranked/vdb; per-hit `metadata.retrieval` reconcilable|
|LLM uncertainty tolerance|Up — rare proper noun/ID recall much better than pure dense|
|Backward compat / evolution|Additive + one BREAKING — `search()` optional params; CLI `--json` envelope intentionally BREAKING (solo project no compat tax; room for pagination/timing/version)|
|Learning curve|Low — CLI adds `--mode`; default is hybrid|
|Testability|High — `tokenize` / `rrf_fuse` / `bm25_search` / `dense_search` pure functions|

### Ongoing trade-off

- **BPE subword BM25 IDF bias**: classic BM25 is word-level; BPE splits high-frequency words — accept for cross-language + dense alignment
- **Scores incomparable across modes**: dense `1/(1+dist)`, bm25 raw (maybe negative), hybrid RRF ~0.01–0.05; ranking correct within call, thresholds don't migrate
- **BM25 no incremental update**: full rebuild each ingest — simpler mental model; YAGNI

## 5. Cross-encoder reranker

- **Status**: accepted
- **Date**: 2026-04-25

### Context

Hybrid (§4) improves recall but top-K **ranking** still limited:

- Bi-encoder information bottleneck in vector dimension — fine relevance lost
- BM25 lexical — no semantic synonym ("revenue"/"income")
- RRF rank-only — cannot beat its inputs

Industry two-stage: cheap retriever → expensive cross-encoder rerank.

### Options considered

**Rerank model**:

- **`BAAI/bge-reranker-v2-m3`** (chosen): ~568M multilingual (CJK/EN/code), M3 BEIR/MIRACL nDCG@10 leading
- `bge-reranker-base` (~110M): faster but worse; v2-m3 fine on Mac
- Cohere `rerank-3` API: similar quality but key + network vs §1 local/repro
- FlashRank: ultra-light English-only, weak Chinese
- LLM-as-judge: general but slow/expensive/fragile — evaluation tool not production reranker

**Wrapper**:

- raw `transformers` ~100 lines boilerplate
- **`sentence-transformers.CrossEncoder`** (chosen): ~5 lines; auto device; pairs with `SentenceTransformer` bi-encoder

**Pool size**:

- K=10: fast but high miss risk
- **K=20** (chosen): BEIR/MS MARCO experience; cross-encoder time negligible on M2 Mac
- K=50+: diminishing returns

**Default**:

- default ON: loads 1.2GB every CLI start, slow
- **default OFF** (chosen): `--rerank` explicit; user picks fast vs quality path

### Decision

- Model `BAAI/bge-reranker-v2-m3` + `sentence-transformers.CrossEncoder` + `lru_cache(1)` singleton lazy load
- Default off, `--rerank` explicit
- Each hit `metadata.reranked = True`, envelope `meta.reranked = True` — downstream reconciliation

### Industry landscape

Two-stage retrieval (retriever → reranker) is 2024+ production RAG standard. `bge-reranker` first-tier open multilingual reranker. default-off + lazy load + lru_cache is ML heavy-model loading pattern.

### Engineering dimension assessment

|Dimension|Assessment|
|---|---|
|Cohesion|High — `reranker.py` one duty; `search()` one-line if orchestration|
|Coupling|Low — `reranker.py` unaware of hybrid/dense/bm25; I/O are `SearchResult`|
|Observability / auditability|Up — per-hit `reranked` flag; envelope `meta` sync|
|LLM uncertainty tolerance|Up — better top-K ranking reduces wrong paragraph selection|
|Backward compat / evolution|Additive — `rerank` default false; pure hybrid when off|
|Learning curve|Low — one `--rerank` flag; auto-download first run|
|Testability|High — `rerank()` pure except model load side effect|

### Ongoing trade-off

- **CrossEncoder logits incomparable to RRF/dense/BM25** (typical -3 to 5): same §4 cross-mode threshold issue; monotonic within call
- **Rerank cannot fix recall miss**: correct doc outside K=20 unreachable — tune `HYBRID_OVERSAMPLE` / `RERANK_CANDIDATES` before stronger reranker
- **Multilingual reranker slightly below monolingual specialists on pure EN/ZH**: acceptable; workshop often mixed; multilingual robustness prioritized

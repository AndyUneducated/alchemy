EMBED_MODEL = "qwen3-embedding:8b"
OLLAMA_BASE_URL = "http://localhost:11434"

# Chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# Hybrid retrieval (BM25 + dense + RRF)
EMBED_TOKENIZER = "Qwen/Qwen3-Embedding-8B"  # HF tokenizer; must match EMBED_MODEL family
HYBRID_OVERSAMPLE = 4                         # each retriever fetches top_k * this for RRF
RRF_K = 60                                    # Cormack et al. 2009 default smoothing constant

# Cross-encoder reranker
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_CANDIDATES = 20                        # candidate pool size before reranking

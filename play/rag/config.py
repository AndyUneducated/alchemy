import os

# All settings are env-driven with code-level fallback defaults (plan choice A).
# Override any value via the listed env var without editing this file.

EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "qwen3-embedding:8b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Chunking
CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "64"))

# Hybrid retrieval (BM25 + dense + RRF)
EMBED_TOKENIZER = os.environ.get("RAG_EMBED_TOKENIZER", "Qwen/Qwen3-Embedding-8B")  # HF tokenizer; must match EMBED_MODEL family
HYBRID_OVERSAMPLE = int(os.environ.get("RAG_HYBRID_OVERSAMPLE", "4"))                 # each retriever fetches top_k * this for RRF
RRF_K = int(os.environ.get("RAG_RRF_K", "60"))                                        # Cormack et al. 2009 default smoothing constant

# Cross-encoder reranker
RERANKER_MODEL = os.environ.get("RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_CANDIDATES = int(os.environ.get("RAG_RERANK_CANDIDATES", "20"))                # candidate pool size before reranking

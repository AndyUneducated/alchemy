"""One-time HuggingFace asset prefetch. Run on every fresh dev machine.

Pulls the BM25 tokenizer (~10MB) and the cross-encoder reranker (~1.2GB) into
the user-level HF cache (~/.cache/huggingface/) so subsequent ingest / query
calls run offline.
"""

from tokenizers import Tokenizer
from sentence_transformers import CrossEncoder

from config import EMBED_TOKENIZER, RERANKER_MODEL


def main() -> None:
    print(f"Downloading tokenizer {EMBED_TOKENIZER} (~10MB) ...")
    Tokenizer.from_pretrained(EMBED_TOKENIZER)

    print(f"Downloading reranker {RERANKER_MODEL} (~1.2GB) ...")
    CrossEncoder(RERANKER_MODEL)

    print("Done. HF cache populated at ~/.cache/huggingface/")


if __name__ == "__main__":
    main()

"""ChromaDB EmbeddingFunction backed by Ollama /api/embed."""

from __future__ import annotations

import json
import urllib.request

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from config import BASE_URL, EMBED_MODEL


class OllamaEmbeddingFunction(EmbeddingFunction[Documents]):
    """Generate embeddings via a locally-running Ollama instance."""

    def __init__(
        self,
        base_url: str = BASE_URL,
        model: str = EMBED_MODEL,
    ) -> None:
        self.base_url = base_url
        self.model = model

    def __call__(self, input: Documents) -> Embeddings:
        payload = json.dumps({"model": self.model, "input": input}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
        return data["embeddings"]

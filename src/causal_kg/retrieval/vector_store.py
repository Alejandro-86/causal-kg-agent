"""Local, dependency-light vector store — plain RAG over abstract text.

Uses sentence-transformers (all-MiniLM-L6-v2, 384-dim) run locally, so
retrieval quality is fully decoupled from whichever LLM provider is
generating answers, and costs nothing per query. Cosine similarity via
plain numpy — no external vector database needed at this corpus size.
"""

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from causal_kg.models import Abstract

MODEL_NAME = "all-MiniLM-L6-v2"


class VectorStore:
    def __init__(self, cache_dir: str = "models_cache") -> None:
        self._model = SentenceTransformer(MODEL_NAME, cache_folder=cache_dir)
        self._vectors: np.ndarray | None = None
        self._abstracts: list[Abstract] = []

    def build(self, abstracts: list[Abstract]) -> None:
        self._abstracts = abstracts
        texts = [f"{a.title}. {a.text}" for a in abstracts]
        self._vectors = self._model.encode(texts, normalize_embeddings=True)

    def save(self, path: str) -> None:
        assert self._vectors is not None
        np.savez(
            path,
            vectors=self._vectors,
            pmids=[a.pmid for a in self._abstracts],
            titles=[a.title for a in self._abstracts],
            texts=[a.text for a in self._abstracts],
        )

    def load(self, path: str) -> None:
        data = np.load(path, allow_pickle=True)
        self._vectors = data["vectors"]
        self._abstracts = [
            Abstract(pmid=p, title=t, text=x)
            for p, t, x in zip(data["pmids"], data["titles"], data["texts"])
        ]

    def search(self, query: str, k: int = 5) -> list[dict]:
        assert self._vectors is not None, "call build() or load() first"
        query_vec = self._model.encode([query], normalize_embeddings=True)[0]
        scores = self._vectors @ query_vec
        top_k = np.argsort(-scores)[:k]
        return [
            {
                "pmid": self._abstracts[i].pmid,
                "title": self._abstracts[i].title,
                "text": self._abstracts[i].text,
                "score": float(scores[i]),
            }
            for i in top_k
        ]

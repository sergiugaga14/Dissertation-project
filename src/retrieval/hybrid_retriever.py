"""Hybrid retrieval combining dense vectors with BM25 sparse retrieval."""

import numpy as np
from rank_bm25 import BM25Okapi

from src.embedding.base import BaseEmbedder

from .base import BaseRetriever, RetrievedChunk
from .dense_retriever import DenseRetriever


class HybridRetriever(BaseRetriever):
    """Combine dense retrieval with BM25 sparse retrieval.

    Uses Reciprocal Rank Fusion (RRF) to merge results from
    both retrievers. Often outperforms either alone.
    """

    name = "hybrid"

    def __init__(
        self,
        embedder: BaseEmbedder,
        collection_name: str = "medrag",
        alpha: float = 0.5,
        rrf_k: int = 60,
    ):
        self.dense = DenseRetriever(embedder, collection_name)
        self.alpha = alpha  # weight for dense vs sparse (1.0 = all dense)
        self.rrf_k = rrf_k
        self.texts: list[str] = []
        self.metadatas: list[dict] = []
        self.bm25: BM25Okapi | None = None

    def index(self, texts: list[str], embeddings: np.ndarray, metadatas: list[dict]) -> None:
        # Index in dense store
        self.dense.index(texts, embeddings, metadatas)

        # Build BM25 index
        self.texts = texts
        self.metadatas = metadatas
        tokenized = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(tokenized)

    def _bm25_retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self.bm25 is None:
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        chunks = []
        for idx in top_indices:
            if scores[idx] > 0:
                metadata = self.metadatas[idx] if idx < len(self.metadatas) else {}
                chunks.append(
                    RetrievedChunk(
                        text=self.texts[idx],
                        chunk_id=metadata.get("chunk_id", str(idx)),
                        score=float(scores[idx]),
                        source_file=metadata.get("source_file", ""),
                        page_number=metadata.get("page_number"),
                        metadata=metadata,
                    )
                )
        return chunks

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        # Get results from both retrievers (fetch more to have overlap)
        fetch_k = top_k * 3
        dense_results = self.dense.retrieve(query, fetch_k)
        sparse_results = self._bm25_retrieve(query, fetch_k)

        # Reciprocal Rank Fusion
        scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievedChunk] = {}

        for rank, chunk in enumerate(dense_results):
            rrf_score = self.alpha / (self.rrf_k + rank + 1)
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + rrf_score
            chunk_map[chunk.chunk_id] = chunk

        for rank, chunk in enumerate(sparse_results):
            rrf_score = (1 - self.alpha) / (self.rrf_k + rank + 1)
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + rrf_score
            if chunk.chunk_id not in chunk_map:
                chunk_map[chunk.chunk_id] = chunk

        # Sort by fused score and return top-k
        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
        return [
            RetrievedChunk(
                text=chunk_map[cid].text,
                chunk_id=cid,
                score=scores[cid],
                source_file=chunk_map[cid].source_file,
                page_number=chunk_map[cid].page_number,
                metadata=chunk_map[cid].metadata,
            )
            for cid in sorted_ids
        ]

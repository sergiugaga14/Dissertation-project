"""Dense retrieval using ChromaDB."""

import chromadb
import numpy as np

from src.embedding.base import BaseEmbedder

from .base import BaseRetriever, RetrievedChunk


class DenseRetriever(BaseRetriever):
    """Dense vector retrieval using ChromaDB.

    Simplest retrieval strategy: embed query, find nearest neighbors.
    """

    name = "dense"

    def __init__(self, embedder: BaseEmbedder, collection_name: str = "medrag", persist_dir: str | None = None):
        self.embedder = embedder
        if persist_dir:
            self.client = chromadb.PersistentClient(path=persist_dir)
        else:
            self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def index(self, texts: list[str], embeddings: np.ndarray, metadatas: list[dict]) -> None:
        ids = [m.get("chunk_id", str(i)) for i, m in enumerate(metadatas)]
        self.collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids,
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_embedding = self.embedder.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for i in range(len(results["ids"][0])):
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            chunks.append(
                RetrievedChunk(
                    text=results["documents"][0][i],
                    chunk_id=results["ids"][0][i],
                    score=1.0 - results["distances"][0][i],  # cosine distance → similarity
                    source_file=metadata.get("source_file", ""),
                    page_number=metadata.get("page_number"),
                    metadata=metadata,
                )
            )
        return chunks

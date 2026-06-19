"""Semantic chunker — splits on meaning boundaries using embeddings."""

import hashlib

import numpy as np
from sentence_transformers import SentenceTransformer

from src.extraction.base import ExtractedChunk

from .base import BaseChunker, TextChunk


class SemanticChunker(BaseChunker):
    """Split text at semantic boundaries using sentence embeddings.

    Computes embeddings for each sentence, then finds breakpoints
    where cosine similarity between consecutive sentences drops
    below a threshold. Produces more semantically coherent chunks.
    """

    name = "semantic"

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1500,
    ):
        self.model = SentenceTransformer(embedding_model)
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def _split_sentences(self, text: str) -> list[str]:
        """Basic sentence splitting."""
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def chunk(self, extracted_chunks: list[ExtractedChunk]) -> list[TextChunk]:
        result = []

        for ext_chunk in extracted_chunks:
            if not ext_chunk.text.strip():
                continue

            sentences = self._split_sentences(ext_chunk.text)
            if len(sentences) <= 1:
                chunk_id = hashlib.sha256(
                    f"{ext_chunk.source_file}:{ext_chunk.page_number}:0".encode()
                ).hexdigest()[:12]
                result.append(
                    TextChunk(
                        text=ext_chunk.text.strip(),
                        chunk_id=chunk_id,
                        source_file=ext_chunk.source_file,
                        page_number=ext_chunk.page_number,
                        metadata={"chunker": self.name},
                    )
                )
                continue

            embeddings = self.model.encode(sentences)

            # Find breakpoints where similarity drops
            current_chunk_sentences = [sentences[0]]
            chunk_idx = 0

            for i in range(1, len(sentences)):
                sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])
                current_text = " ".join(current_chunk_sentences)

                if sim < self.similarity_threshold and len(current_text) >= self.min_chunk_size:
                    # Create chunk at this breakpoint
                    chunk_id = hashlib.sha256(
                        f"{ext_chunk.source_file}:{ext_chunk.page_number}:{chunk_idx}".encode()
                    ).hexdigest()[:12]
                    result.append(
                        TextChunk(
                            text=current_text,
                            chunk_id=chunk_id,
                            source_file=ext_chunk.source_file,
                            page_number=ext_chunk.page_number,
                            metadata={"chunker": self.name, "chunk_index": chunk_idx},
                        )
                    )
                    current_chunk_sentences = [sentences[i]]
                    chunk_idx += 1
                elif len(current_text) >= self.max_chunk_size:
                    # Force a split if chunk gets too large
                    chunk_id = hashlib.sha256(
                        f"{ext_chunk.source_file}:{ext_chunk.page_number}:{chunk_idx}".encode()
                    ).hexdigest()[:12]
                    result.append(
                        TextChunk(
                            text=current_text,
                            chunk_id=chunk_id,
                            source_file=ext_chunk.source_file,
                            page_number=ext_chunk.page_number,
                            metadata={"chunker": self.name, "chunk_index": chunk_idx},
                        )
                    )
                    current_chunk_sentences = [sentences[i]]
                    chunk_idx += 1
                else:
                    current_chunk_sentences.append(sentences[i])

            # Don't forget the last chunk
            if current_chunk_sentences:
                chunk_id = hashlib.sha256(
                    f"{ext_chunk.source_file}:{ext_chunk.page_number}:{chunk_idx}".encode()
                ).hexdigest()[:12]
                result.append(
                    TextChunk(
                        text=" ".join(current_chunk_sentences),
                        chunk_id=chunk_id,
                        source_file=ext_chunk.source_file,
                        page_number=ext_chunk.page_number,
                        metadata={"chunker": self.name, "chunk_index": chunk_idx},
                    )
                )

        return result

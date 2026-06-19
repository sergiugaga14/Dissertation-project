"""General-purpose embedding models."""

import numpy as np
from sentence_transformers import SentenceTransformer

from .base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """Wrapper around sentence-transformers models.

    Supports any model from the sentence-transformers library:
    - all-MiniLM-L6-v2 (fast baseline, 384d)
    - BAAI/bge-large-en-v1.5 (strong general-purpose, 1024d)
    - etc.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.name = model_name.split("/")[-1]
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=len(texts) > 100)

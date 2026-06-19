"""Base classes for embedding models."""

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    """Base class for text embedding models."""

    name: str = "base"
    dimension: int = 0

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into vectors."""
        ...

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query. May differ from document embedding in some models."""
        return self.embed([query])[0]

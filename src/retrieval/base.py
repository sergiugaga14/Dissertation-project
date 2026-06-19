"""Base classes for retrieval strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RetrievedChunk:
    """A chunk retrieved from the vector store with its relevance score."""

    text: str
    chunk_id: str
    score: float
    source_file: str
    page_number: int | None = None
    metadata: dict = field(default_factory=dict)


class BaseRetriever(ABC):
    """Base class for retrieval strategies."""

    name: str = "base"

    @abstractmethod
    def index(self, texts: list[str], embeddings: np.ndarray, metadatas: list[dict]) -> None:
        """Index documents into the retrieval store."""
        ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks for a query."""
        ...

"""Base classes for text chunking strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.extraction.base import ExtractedChunk


@dataclass
class TextChunk:
    """A chunk ready for embedding and indexing."""

    text: str
    chunk_id: str
    source_file: str
    page_number: int | None = None
    metadata: dict = field(default_factory=dict)
    token_count: int | None = None


class BaseChunker(ABC):
    """Base class for chunking strategies."""

    name: str = "base"

    @abstractmethod
    def chunk(self, extracted_chunks: list[ExtractedChunk]) -> list[TextChunk]:
        """Split extracted content into chunks suitable for embedding."""
        ...

"""Base classes for document extraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExtractedChunk:
    """A piece of content extracted from a document."""

    text: str
    source_file: str
    page_number: int | None = None
    chunk_type: str = "text"  # text, table, image_caption, heading
    metadata: dict = field(default_factory=dict)
    image_path: str | None = None  # path to extracted image, if any


@dataclass
class ExtractedDocument:
    """Full extracted document with all its chunks."""

    source_path: str
    chunks: list[ExtractedChunk]
    metadata: dict = field(default_factory=dict)
    total_pages: int | None = None


class BaseExtractor(ABC):
    """Base class for all document extractors.

    Each extractor implements a different strategy for converting
    raw documents (PDFs, images) into structured text chunks.
    """

    name: str = "base"

    @abstractmethod
    def extract(self, file_path: Path) -> ExtractedDocument:
        """Extract content from a single document."""
        ...

    def extract_batch(self, file_paths: list[Path]) -> list[ExtractedDocument]:
        """Extract content from multiple documents."""
        return [self.extract(p) for p in file_paths]

    def supported_extensions(self) -> set[str]:
        return {".pdf"}

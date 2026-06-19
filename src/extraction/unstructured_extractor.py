"""Unstructured.io-based document extractor — layout-aware parsing."""

from pathlib import Path

from .base import BaseExtractor, ExtractedChunk, ExtractedDocument


class UnstructuredExtractor(BaseExtractor):
    """Extract text using Unstructured.io library.

    Layout-aware parsing that understands document structure:
    headings, paragraphs, tables, lists. Better at preserving
    document semantics than simple text extraction.
    """

    name = "unstructured"

    def __init__(self, strategy: str = "hi_res", include_metadata: bool = True):
        self.strategy = strategy  # "fast", "hi_res", "ocr_only"
        self.include_metadata = include_metadata

    def extract(self, file_path: Path) -> ExtractedDocument:
        from unstructured.partition.auto import partition

        elements = partition(
            filename=str(file_path),
            strategy=self.strategy,
            include_metadata=self.include_metadata,
        )

        chunks = []
        for elem in elements:
            chunk_type = "text"
            if hasattr(elem, "category"):
                if elem.category == "Title":
                    chunk_type = "heading"
                elif elem.category == "Table":
                    chunk_type = "table"
                elif elem.category == "Image":
                    chunk_type = "image"

            page_num = None
            if hasattr(elem, "metadata") and hasattr(elem.metadata, "page_number"):
                page_num = elem.metadata.page_number

            text = str(elem)
            if text.strip():
                chunks.append(
                    ExtractedChunk(
                        text=text.strip(),
                        source_file=str(file_path),
                        page_number=page_num,
                        chunk_type=chunk_type,
                        metadata={"category": getattr(elem, "category", "unknown")},
                    )
                )

        return ExtractedDocument(
            source_path=str(file_path),
            chunks=chunks,
        )

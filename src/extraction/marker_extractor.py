"""Marker-based PDF extractor — ML-powered PDF to Markdown."""

from pathlib import Path

from .base import BaseExtractor, ExtractedChunk, ExtractedDocument


class MarkerExtractor(BaseExtractor):
    """Extract text using Marker (ML-based PDF→Markdown).

    Marker uses deep learning models to convert PDFs into clean
    Markdown, preserving structure, tables, and equations.
    Generally higher quality than rule-based approaches.
    """

    name = "marker"

    def extract(self, file_path: Path) -> ExtractedDocument:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        models = create_model_dict()
        converter = PdfConverter(artifact_dict=models)
        result = converter(str(file_path))

        # Marker returns markdown text — we treat it as a single document
        # that will be split by the chunking layer
        chunks = []
        if result.markdown.strip():
            chunks.append(
                ExtractedChunk(
                    text=result.markdown,
                    source_file=str(file_path),
                    page_number=None,
                    chunk_type="text",
                    metadata={"format": "markdown"},
                )
            )

        return ExtractedDocument(
            source_path=str(file_path),
            chunks=chunks,
            metadata={"extraction_format": "markdown"},
        )

"""PyMuPDF-based document extractor — simple, fast baseline."""

from pathlib import Path

import fitz  # pymupdf

from .base import BaseExtractor, ExtractedChunk, ExtractedDocument


class PyMuPDFExtractor(BaseExtractor):
    """Extract text from PDFs using PyMuPDF.

    This is the simplest extraction strategy: direct text extraction
    with basic layout preservation. Good baseline but struggles with
    complex layouts, tables, and scanned documents.
    """

    name = "pymupdf"

    def __init__(self, extract_images: bool = False, image_output_dir: str | None = None):
        self.extract_images = extract_images
        self.image_output_dir = Path(image_output_dir) if image_output_dir else None

    def extract(self, file_path: Path) -> ExtractedDocument:
        doc = fitz.open(file_path)
        chunks = []

        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                chunks.append(
                    ExtractedChunk(
                        text=text.strip(),
                        source_file=str(file_path),
                        page_number=page_num + 1,
                        chunk_type="text",
                    )
                )

            # Optionally extract images
            if self.extract_images and self.image_output_dir:
                for img_idx, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    img_path = (
                        self.image_output_dir
                        / f"{file_path.stem}_p{page_num + 1}_img{img_idx}.png"
                    )
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                    pix.save(str(img_path))
                    chunks.append(
                        ExtractedChunk(
                            text=f"[Image from page {page_num + 1}]",
                            source_file=str(file_path),
                            page_number=page_num + 1,
                            chunk_type="image",
                            image_path=str(img_path),
                        )
                    )

        doc.close()

        return ExtractedDocument(
            source_path=str(file_path),
            chunks=chunks,
            total_pages=len(doc) if not doc.is_closed else len(chunks),
        )

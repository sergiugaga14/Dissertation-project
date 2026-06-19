"""Fixed-size text chunker with token-based splitting."""

import hashlib

from src.extraction.base import ExtractedChunk

from .base import BaseChunker, TextChunk


class FixedChunker(BaseChunker):
    """Split text into fixed-size chunks with overlap.

    Simple and reproducible. Good baseline strategy.
    """

    name = "fixed"

    def __init__(self, chunk_size: int = 512, overlap: int = 50, tokenizer_name: str | None = None):
        self.chunk_size = chunk_size
        self.overlap = overlap
        # Use word-based splitting by default; token-based if tokenizer provided
        self.tokenizer = None
        if tokenizer_name:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def _split_text(self, text: str) -> list[str]:
        if self.tokenizer:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            chunks = []
            start = 0
            while start < len(tokens):
                end = start + self.chunk_size
                chunk_tokens = tokens[start:end]
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunks.append(chunk_text)
                start += self.chunk_size - self.overlap
            return chunks
        else:
            # Word-based splitting
            words = text.split()
            chunks = []
            start = 0
            while start < len(words):
                end = start + self.chunk_size
                chunk_text = " ".join(words[start:end])
                chunks.append(chunk_text)
                start += self.chunk_size - self.overlap
            return chunks

    def chunk(self, extracted_chunks: list[ExtractedChunk]) -> list[TextChunk]:
        result = []
        for ext_chunk in extracted_chunks:
            if not ext_chunk.text.strip():
                continue
            text_pieces = self._split_text(ext_chunk.text)
            for i, piece in enumerate(text_pieces):
                chunk_id = hashlib.sha256(
                    f"{ext_chunk.source_file}:{ext_chunk.page_number}:{i}".encode()
                ).hexdigest()[:12]
                result.append(
                    TextChunk(
                        text=piece,
                        chunk_id=chunk_id,
                        source_file=ext_chunk.source_file,
                        page_number=ext_chunk.page_number,
                        metadata={
                            "chunk_index": i,
                            "chunker": self.name,
                            "chunk_size": self.chunk_size,
                        },
                    )
                )
        return result

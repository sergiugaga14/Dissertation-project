"""Full RAG pipeline orchestrator — ties all components together."""

from pathlib import Path

from rich.console import Console

from src.chunking.base import BaseChunker
from src.embedding.base import BaseEmbedder
from src.extraction.base import BaseExtractor
from src.generation.medgemma import MedGemmaGenerator
from src.retrieval.base import BaseRetriever

console = Console()


class MedRAGPipeline:
    """Orchestrates the full RAG pipeline: extract → chunk → embed → retrieve → generate.

    Each component is pluggable, allowing systematic comparison
    of different strategies at each layer.
    """

    def __init__(
        self,
        extractor: BaseExtractor,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        retriever: BaseRetriever,
        generator: MedGemmaGenerator,
    ):
        self.extractor = extractor
        self.chunker = chunker
        self.embedder = embedder
        self.retriever = retriever
        self.generator = generator
        self._indexed = False

    def ingest(self, document_paths: list[str | Path]) -> dict:
        """Process and index documents into the retrieval store.

        Steps: extract → chunk → embed → index
        """
        paths = [Path(p) for p in document_paths]
        console.print(f"[bold]Ingesting {len(paths)} documents...[/bold]")

        # 1. Extract
        console.print(f"  Extracting with [cyan]{self.extractor.name}[/cyan]...")
        all_extracted = self.extractor.extract_batch(paths)
        all_raw_chunks = [c for doc in all_extracted for c in doc.chunks]
        console.print(f"  → {len(all_raw_chunks)} raw chunks extracted")

        # 2. Chunk
        console.print(f"  Chunking with [cyan]{self.chunker.name}[/cyan]...")
        text_chunks = self.chunker.chunk(all_raw_chunks)
        console.print(f"  → {len(text_chunks)} text chunks created")

        # 3. Embed
        console.print(f"  Embedding with [cyan]{self.embedder.name}[/cyan]...")
        texts = [c.text for c in text_chunks]
        embeddings = self.embedder.embed(texts)
        console.print(f"  → {embeddings.shape[0]} embeddings ({embeddings.shape[1]}d)")

        # 4. Index
        console.print("  Indexing into retriever...")
        metadatas = [
            {
                "chunk_id": c.chunk_id,
                "source_file": c.source_file,
                "page_number": c.page_number,
                **c.metadata,
            }
            for c in text_chunks
        ]
        self.retriever.index(texts, embeddings, metadatas)
        self._indexed = True
        console.print("[bold green]✓ Ingestion complete[/bold green]")

        return {
            "documents": len(paths),
            "raw_chunks": len(all_raw_chunks),
            "text_chunks": len(text_chunks),
            "embedding_dim": embeddings.shape[1],
        }

    def query(self, question: str, top_k: int = 5, prompt_template: str = "default") -> dict:
        """Answer a medical question using the RAG pipeline.

        Steps: retrieve → generate
        """
        if not self._indexed:
            raise RuntimeError("No documents indexed. Call ingest() first.")

        # Retrieve
        retrieved = self.retriever.retrieve(question, top_k=top_k)

        # Generate
        result = self.generator.generate(question, retrieved, prompt_template=prompt_template)
        result["retrieved_chunks"] = [
            {"text": c.text, "score": c.score, "source": c.source_file} for c in retrieved
        ]
        return result

    def config_summary(self) -> dict:
        """Return a summary of the current pipeline configuration."""
        return {
            "extractor": self.extractor.name,
            "chunker": self.chunker.name,
            "embedder": self.embedder.name,
            "retriever": self.retriever.name,
            "generator": self.generator.model_name,
        }

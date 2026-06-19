"""Main experiment runner — runs a configured RAG pipeline and evaluates it."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml
from rich.console import Console

console = Console()


def load_config(config_path: str) -> dict:
    """Load experiment config, merging with base config."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load base config if referenced
    if "defaults" in config:
        base_path = Path(config_path).parent / config["defaults"][0]
        base_path = base_path.with_suffix(".yaml")
        with open(base_path) as f:
            base = yaml.safe_load(f)
        # Deep merge: experiment overrides base
        merged = deep_merge(base, config)
        return merged

    return config


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dicts, override takes precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def build_pipeline(config: dict):
    """Build a MedRAG pipeline from config."""
    from src.chunking.fixed_chunker import FixedChunker
    from src.chunking.semantic_chunker import SemanticChunker
    from src.embedding.general_embedder import SentenceTransformerEmbedder
    from src.embedding.medical_embedder import MedCPTEmbedder, MedicalEmbedder
    from src.extraction.pymupdf_extractor import PyMuPDFExtractor
    from src.extraction.unstructured_extractor import UnstructuredExtractor
    from src.generation.medgemma import MedGemmaGenerator
    from src.pipeline import MedRAGPipeline
    from src.retrieval.dense_retriever import DenseRetriever
    from src.retrieval.hybrid_retriever import HybridRetriever

    pipe = config["pipeline"]

    # Build extractor
    ext_name = pipe["extractor"]["name"]
    ext_params = pipe["extractor"].get("params", {})
    if ext_name == "pymupdf":
        extractor = PyMuPDFExtractor(**ext_params)
    elif ext_name == "unstructured":
        extractor = UnstructuredExtractor(**ext_params)
    else:
        raise ValueError(f"Unknown extractor: {ext_name}")

    # Build chunker
    chunk_name = pipe["chunker"]["name"]
    chunk_params = pipe["chunker"].get("params", {})
    if chunk_name == "fixed":
        chunker = FixedChunker(**chunk_params)
    elif chunk_name == "semantic":
        chunker = SemanticChunker(**chunk_params)
    else:
        raise ValueError(f"Unknown chunker: {chunk_name}")

    # Build embedder
    emb_name = pipe["embedder"]["name"]
    emb_params = pipe["embedder"].get("params", {})
    if emb_name == "sentence-transformer":
        embedder = SentenceTransformerEmbedder(**emb_params)
    elif emb_name == "medical":
        embedder = MedicalEmbedder(**emb_params)
    elif emb_name == "medcpt":
        embedder = MedCPTEmbedder()
    else:
        raise ValueError(f"Unknown embedder: {emb_name}")

    # Build retriever
    ret_name = pipe["retriever"]["name"]
    ret_params = pipe["retriever"].get("params", {})
    if ret_name == "dense":
        retriever = DenseRetriever(embedder, **ret_params)
    elif ret_name == "hybrid":
        retriever = HybridRetriever(embedder, **ret_params)
    else:
        raise ValueError(f"Unknown retriever: {ret_name}")

    # Build generator
    gen_params = pipe["generator"].get("params", {})
    generator = MedGemmaGenerator(**gen_params)

    return MedRAGPipeline(
        extractor=extractor,
        chunker=chunker,
        embedder=embedder,
        retriever=retriever,
        generator=generator,
    )


def main():
    parser = argparse.ArgumentParser(description="Run a MedRAG experiment")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Just print config, don't run")
    args = parser.parse_args()

    config = load_config(args.config)
    exp_name = config.get("experiment", {}).get("name", "unnamed")

    console.print(f"\n[bold cyan]═══ Experiment: {exp_name} ═══[/bold cyan]")
    console.print(f"Config: {args.config}")

    if args.dry_run:
        console.print(json.dumps(config, indent=2))
        return

    # Build pipeline
    pipeline = build_pipeline(config)
    console.print(f"Pipeline: {pipeline.config_summary()}")

    # Ingest documents
    docs_dir = Path(config["data"]["documents_dir"])
    if docs_dir.exists():
        doc_paths = list(docs_dir.glob("*.pdf"))
        if doc_paths:
            pipeline.ingest(doc_paths)

    # Evaluate
    from evaluation.benchmarks import load_medqa, load_pubmedqa
    from evaluation.evaluator import Evaluator

    eval_config = config["evaluation"]
    benchmark = eval_config["benchmark"]

    if benchmark == "medqa":
        questions, answers = load_medqa(
            split=eval_config["split"], num_samples=eval_config.get("num_samples")
        )
    elif benchmark == "pubmedqa":
        questions, answers = load_pubmedqa(
            split=eval_config["split"], num_samples=eval_config.get("num_samples")
        )
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    evaluator = Evaluator(pipeline)
    results = evaluator.evaluate_dataset(
        questions,
        answers,
        top_k=eval_config["top_k"],
        prompt_template=eval_config["prompt_template"],
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(config["output"]["results_dir"]) / f"{exp_name}_{timestamp}.json"
    evaluator.save_results(results, output_path)
    evaluator.print_summary(results)


if __name__ == "__main__":
    main()

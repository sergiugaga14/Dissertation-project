# MedRAG: Comparative Analysis of RAG Strategies for Medical Domain using MEDgemma

## Dissertation Project — MSc Computer Science / AI

### Abstract

This project systematically evaluates different Retrieval-Augmented Generation (RAG) 
pipeline configurations for the medical domain. We compare multiple extraction models, 
chunking strategies, embedding approaches, and retrieval methods while using Google's 
MEDgemma as the fixed generation backbone. The goal is to identify which RAG pipeline 
components most significantly impact clinical question-answering accuracy, and to build 
a practical medical knowledge assistant.

---

## Research Questions

1. **RQ1**: How do different document extraction methods (OCR, layout-aware, vision-based) 
   affect downstream medical QA accuracy?
2. **RQ2**: Which embedding models (general-purpose vs. medical-domain-specific) produce 
   better retrieval for medical queries?
3. **RQ3**: How do chunking strategies (fixed-size, semantic, document-structure-aware) 
   impact retrieval relevance in medical texts?
4. **RQ4**: Does multimodal retrieval (text + medical images) improve diagnostic accuracy 
   compared to text-only RAG?
5. **RQ5**: How does the number and quality of retrieved context chunks affect MEDgemma's 
   generation accuracy?

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MEDICAL KNOWLEDGE BASE                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │  PDFs    │  │ Images   │  │ Clinical │  │ Structured Data   │  │
│  │(papers,  │  │(X-rays,  │  │ Notes    │  │(drug DBs, ICD     │  │
│  │ guides)  │  │ CT, MRI) │  │          │  │ codes, guidelines)│  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬──────────┘  │
│       │              │             │                  │             │
└───────┼──────────────┼─────────────┼──────────────────┼─────────────┘
        │              │             │                  │
        ▼              ▼             ▼                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     EXTRACTION LAYER (Variable)                     │
│                                                                     │
│  Strategy A: PyMuPDF + basic OCR                                   │
│  Strategy B: Unstructured.io (layout-aware parsing)                │
│  Strategy C: DocTR / Surya (vision-based OCR)                      │
│  Strategy D: Marker (ML-based PDF→Markdown)                        │
│  Strategy E: ColPali / ColQwen (vision embeddings, no extraction)  │
│                                                                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    CHUNKING LAYER (Variable)                        │
│                                                                     │
│  Strategy A: Fixed-size (512 / 1024 tokens) with overlap           │
│  Strategy B: Semantic chunking (sentence-transformers)             │
│  Strategy C: Document-structure-aware (headings, sections)         │
│  Strategy D: Proposition-based (atomic facts)                      │
│  Strategy E: Late chunking / contextual retrieval                  │
│                                                                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   EMBEDDING LAYER (Variable)                        │
│                                                                     │
│  Model A: all-MiniLM-L6-v2 (general, fast baseline)               │
│  Model B: BGE-large-en-v1.5 (strong general-purpose)              │
│  Model C: PubMedBERT embeddings (medical domain)                   │
│  Model D: MedCPT (medical retrieval-optimized)                     │
│  Model E: BioLORD (biomedical ontology-aware)                      │
│  Model F: Multimodal — CLIP / BiomedCLIP (for images)             │
│                                                                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  VECTOR STORE + RETRIEVAL (Variable)                 │
│                                                                     │
│  Store: ChromaDB / FAISS / Qdrant / Weaviate                      │
│                                                                     │
│  Retrieval Strategies:                                              │
│    - Dense retrieval (cosine similarity)                           │
│    - Hybrid retrieval (dense + BM25 sparse)                        │
│    - Reranking (cross-encoder reranker after initial retrieval)    │
│    - Multi-vector retrieval (ColBERT-style)                        │
│    - Hypothetical Document Embedding (HyDE)                        │
│                                                                     │
│  Parameters: top-k (3, 5, 10, 20), similarity threshold           │
│                                                                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│              GENERATION LAYER (Fixed — MEDgemma)                    │
│                                                                     │
│  Google MEDgemma (medgemma-27b-text-it or multimodal variant)      │
│                                                                     │
│  Prompt Templates:                                                  │
│    - Zero-shot with retrieved context                              │
│    - Few-shot with medical examples                                │
│    - Chain-of-thought medical reasoning                            │
│    - Structured output (diagnosis, confidence, evidence)           │
│                                                                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     EVALUATION FRAMEWORK                            │
│                                                                     │
│  Benchmarks: MedQA, PubMedQA, MedMCQA, custom clinical cases     │
│  Metrics:    Accuracy, F1, ROUGE, BERTScore, Faithfulness,        │
│              Answer Relevancy, Context Precision/Recall            │
│  Tools:      RAGAS, DeepEval, custom scorers                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
project/
├── README.md
├── pyproject.toml
├── configs/                    # Experiment configurations
│   ├── base.yaml
│   ├── experiments/
│   │   ├── exp01_baseline.yaml
│   │   ├── exp02_medical_embeddings.yaml
│   │   └── ...
├── data/
│   ├── raw/                    # Original PDFs, images, datasets
│   │   ├── pdfs/
│   │   ├── images/
│   │   └── datasets/
│   ├── processed/              # Extracted & chunked data
│   └── benchmarks/             # Evaluation datasets (MedQA, etc.)
├── src/
│   ├── __init__.py
│   ├── extraction/             # Document extraction strategies
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── pymupdf_extractor.py
│   │   ├── unstructured_extractor.py
│   │   ├── marker_extractor.py
│   │   └── vision_extractor.py
│   ├── chunking/               # Chunking strategies
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── fixed_chunker.py
│   │   ├── semantic_chunker.py
│   │   └── structure_chunker.py
│   ├── embedding/              # Embedding models
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── general_embedder.py
│   │   ├── medical_embedder.py
│   │   └── multimodal_embedder.py
│   ├── retrieval/              # Retrieval strategies
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── dense_retriever.py
│   │   ├── hybrid_retriever.py
│   │   └── reranking_retriever.py
│   ├── generation/             # MEDgemma generation
│   │   ├── __init__.py
│   │   ├── medgemma.py
│   │   └── prompts.py
│   ├── pipeline.py             # Full RAG pipeline orchestrator
│   └── utils.py
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py              # Custom medical metrics
│   ├── benchmarks.py           # Benchmark loaders
│   └── evaluator.py            # Experiment evaluator
├── experiments/
│   ├── run_experiment.py       # Main experiment runner
│   └── analyze_results.py     # Results analysis & visualization
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_extraction_comparison.ipynb
│   ├── 03_embedding_analysis.ipynb
│   ├── 04_retrieval_evaluation.ipynb
│   ├── 05_full_pipeline_results.ipynb
│   └── 06_thesis_figures.ipynb
├── results/                    # Experiment outputs
│   ├── logs/
│   ├── metrics/
│   └── figures/
├── thesis/                     # Thesis writing
│   └── notes.md
└── tests/
    └── ...
```

---

## Getting Started

```bash
# 1. Set up environment
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Download a benchmark dataset
python -m src.utils download-medqa

# 3. Add your medical documents to data/raw/pdfs/ and data/raw/images/

# 4. Run the baseline experiment
python experiments/run_experiment.py --config configs/experiments/exp01_baseline.yaml

# 5. Compare results
python experiments/analyze_results.py --results-dir results/metrics/
```

---

## Key Dependencies

- **MEDgemma**: `transformers`, `torch` (via Hugging Face)
- **Extraction**: `pymupdf`, `unstructured`, `marker-pdf`, `doctr`
- **Embeddings**: `sentence-transformers`, `open-clip-torch`
- **Vector Store**: `chromadb`, `faiss-cpu`
- **Retrieval**: `rank-bm25`, `sentence-transformers` (cross-encoder)
- **Evaluation**: `ragas`, `deepeval`, `datasets`
- **Experiment Tracking**: `wandb` or `mlflow`
- **Notebooks**: `jupyter`, `matplotlib`, `seaborn`, `plotly`

---

## MEDgemma Access

MEDgemma requires approval from Google. Apply at:
https://huggingface.co/google/medgemma-27b-text-it

Available variants:
- `medgemma-4b-it` — Lightweight, good for iteration (fits on 1x 24GB GPU)
- `medgemma-27b-text-it` — Full text model (needs ~60GB VRAM or quantized)
- `medgemma-27b-img-it` — Multimodal (text + medical images)

For development, start with the 4B variant and quantized (4-bit) 27B.

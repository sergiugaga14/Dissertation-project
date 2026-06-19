# Research Plan: Comparative RAG Strategies for Medical QA with MEDgemma

## 1. Thesis Statement

Different RAG pipeline configurations — particularly the choice of document extraction 
method, embedding model, and retrieval strategy — significantly impact the accuracy and 
faithfulness of medical question-answering systems built on MEDgemma. This work provides 
a systematic comparison to identify optimal pipeline configurations for clinical use cases.

---

## 2. Literature Review Topics

### 2.1 Retrieval-Augmented Generation
- Original RAG paper (Lewis et al., 2020)
- Advanced RAG survey (Gao et al., 2024)
- Modular RAG framework
- Self-RAG, CRAG (Corrective RAG)

### 2.2 Medical NLP & LLMs
- Med-PaLM, Med-PaLM 2 (Google)
- MEDgemma architecture and training
- Clinical NLP challenges (terminology, abbreviations, multi-language)
- Medical LLM benchmarks (MedQA, PubMedQA, MedMCQA, MMLU-Medical)

### 2.3 Medical Document Processing
- Challenges of medical PDF extraction (tables, figures, formulas)
- OCR for medical imaging reports
- Multimodal medical data (radiology reports + images)

### 2.4 Embedding Models for Biomedical Domain
- PubMedBERT, BioLORD, MedCPT
- Domain adaptation of embedding models
- BiomedCLIP for medical images

### 2.5 Retrieval Strategies
- Dense vs. sparse retrieval
- Hybrid retrieval (BM25 + dense)
- Cross-encoder reranking
- ColBERT and late interaction models

---

## 3. Experimental Design

### 3.1 Independent Variables (What We Change)

| Layer        | Strategies to Compare                                           |
|-------------|------------------------------------------------------------------|
| Extraction  | PyMuPDF, Unstructured.io, Marker, DocTR/Surya                  |
| Chunking    | Fixed-512, Fixed-1024, Semantic, Structure-aware, Propositions  |
| Embedding   | MiniLM, BGE-large, PubMedBERT, MedCPT, BioLORD                |
| Retrieval   | Dense, Hybrid (BM25+dense), +Reranking, HyDE                   |
| Top-K       | 3, 5, 10, 20                                                    |
| Prompting   | Zero-shot, Few-shot, Chain-of-Thought                           |

### 3.2 Controlled Variables (What Stays Fixed)
- **Generation model**: MEDgemma (same variant for all comparisons)
- **Vector store**: ChromaDB (same for all)
- **Evaluation datasets**: Same benchmark splits
- **Hardware**: Same GPU setup
- **Temperature**: 0.0 (deterministic generation)

### 3.3 Dependent Variables (What We Measure)

| Metric              | What It Measures                                      |
|---------------------|-------------------------------------------------------|
| Accuracy            | Correctness on multiple-choice medical QA             |
| F1 Score            | Token-level answer overlap                            |
| ROUGE-L             | Longest common subsequence with reference             |
| BERTScore           | Semantic similarity to reference answer               |
| Faithfulness        | Does answer only use provided context? (RAGAS)        |
| Answer Relevancy    | Is the answer relevant to the question? (RAGAS)       |
| Context Precision   | Are retrieved chunks relevant? (RAGAS)                |
| Context Recall      | Are all needed facts retrieved? (RAGAS)               |
| Latency             | End-to-end response time                              |
| Token Usage         | Context tokens consumed per query                     |

### 3.4 Experiment Progression (Ablation Study)

**Phase 1 — Establish Baseline**
- Exp01: Naive RAG (PyMuPDF + Fixed-512 + MiniLM + Dense-top5 + zero-shot)
- Exp02: MEDgemma without RAG (direct QA, no retrieval)

**Phase 2 — Extraction Comparison** (fix everything else from baseline)
- Exp03: Unstructured.io extraction
- Exp04: Marker extraction
- Exp05: DocTR/Surya extraction
→ Pick best extractor for Phase 3

**Phase 3 — Embedding Comparison** (use best extractor)
- Exp06: BGE-large-en
- Exp07: PubMedBERT
- Exp08: MedCPT
- Exp09: BioLORD
→ Pick best embedding for Phase 4

**Phase 4 — Chunking Comparison** (use best extractor + embedding)
- Exp10: Fixed-1024
- Exp11: Semantic chunking
- Exp12: Structure-aware chunking
- Exp13: Proposition-based
→ Pick best chunking for Phase 5

**Phase 5 — Retrieval Comparison** (use best extractor + embedding + chunking)
- Exp14: Hybrid retrieval (BM25 + dense)
- Exp15: +Cross-encoder reranking
- Exp16: HyDE (Hypothetical Document Embedding)
- Exp17: Top-K ablation (3, 5, 10, 20)
→ Identify optimal retrieval

**Phase 6 — Prompt Engineering**
- Exp18: Few-shot prompting
- Exp19: Chain-of-thought medical reasoning
- Exp20: Structured output (diagnosis + evidence + confidence)

**Phase 7 — Multimodal (if time allows)**
- Exp21: Add medical image retrieval (BiomedCLIP)
- Exp22: MEDgemma multimodal variant with image context

**Phase 8 — Best Configuration**
- Exp23: Combine all best choices → "MedRAG-Optimal"
- Compare against baselines and published results

---

## 4. Datasets

### 4.1 Knowledge Base (Documents to Index)
- **PubMed Open Access**: Free biomedical papers (start with ~1000 papers in a specialty)
- **Clinical Practice Guidelines**: WHO, NICE, or specialty-specific guidelines
- **Medical Textbooks** (open-access chapters, e.g., OpenStax Anatomy)
- **Drug Information**: FDA labels, DrugBank entries
- **Radiology Images**: MIMIC-CXR (chest X-rays + reports), CheXpert

### 4.2 Evaluation Benchmarks
| Dataset    | Size    | Task                        | Source                |
|------------|---------|-----------------------------|-----------------------|
| MedQA      | 12,723  | USMLE-style MCQ             | Jin et al., 2021      |
| PubMedQA   | 1,000   | Yes/No/Maybe from abstracts | Jin et al., 2019      |
| MedMCQA    | 194,000 | Indian medical exam MCQ     | Pal et al., 2022      |
| MMLU-Med   | ~1,800  | Medical subset of MMLU      | Hendrycks et al., 2021|
| Custom     | ~100    | Your own clinical cases     | Manual creation       |

### 4.3 Suggested Starting Specialty
Pick ONE medical specialty to keep scope manageable:
- **Cardiology** — rich literature, clear diagnostic criteria
- **Radiology** — natural multimodal (images + reports)
- **Oncology** — treatment guidelines are well-structured
- **Emergency Medicine** — practical, high-impact use case

---

## 5. Timeline (Suggested)

| Month | Phase                    | Deliverable                              |
|-------|--------------------------|------------------------------------------|
| 1     | Literature review        | Background chapter draft                 |
| 1-2   | Data collection          | Knowledge base assembled                 |
| 2     | Baseline implementation  | Working pipeline + Exp01-02              |
| 3     | Extraction experiments   | Exp03-05 results + analysis              |
| 3-4   | Embedding experiments    | Exp06-09 results + analysis              |
| 4     | Chunking experiments     | Exp10-13 results + analysis              |
| 4-5   | Retrieval experiments    | Exp14-17 results + analysis              |
| 5     | Prompt + multimodal      | Exp18-22 results                         |
| 5-6   | Optimal config + writing | Final experiments + thesis draft          |
| 6     | Review + defense prep    | Complete thesis + presentation           |

---

## 6. Expected Contributions

1. **Systematic comparison** of RAG pipeline components for medical domain
2. **Practical guidelines** for building medical RAG systems
3. **Open-source toolkit** (MedRAG) for medical document QA
4. **Evidence** on whether domain-specific embeddings outperform general ones 
   enough to justify the complexity
5. **Multimodal analysis** of combining text and image retrieval for medical QA

---

## 7. Ethical Considerations

- All datasets must be properly licensed and de-identified
- Medical AI systems should NEVER be used for direct clinical decisions
- Include clear disclaimers about limitations
- Discuss potential bias in training data
- Consider GDPR/HIPAA implications if using patient data
- IRB/Ethics board approval if using any patient-derived data

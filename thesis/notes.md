# Thesis Notes

## Working Title
"Comparative Analysis of RAG Pipeline Configurations for Medical Question Answering using MEDgemma"

## Alternative Titles
- "Optimizing Retrieval-Augmented Generation for Clinical Decision Support: A Systematic Evaluation"
- "From PDFs to Diagnoses: Building Effective Medical RAG Systems with Domain-Specific Components"

## Key Arguments
1. RAG significantly improves medical LLM accuracy over direct generation
2. Domain-specific embeddings (MedCPT, BioLORD) outperform general-purpose ones for medical retrieval
3. The extraction layer matters more than commonly assumed — garbage in, garbage out
4. Hybrid retrieval (BM25 + dense) provides the best balance of precision and recall
5. Multimodal context (images + text) further improves diagnostic accuracy

## Thesis Structure (Draft)
1. Introduction
2. Background & Literature Review
   - 2.1 Large Language Models in Medicine
   - 2.2 Retrieval-Augmented Generation
   - 2.3 Document Processing for Medical Texts
   - 2.4 Evaluation of Medical AI Systems
3. Methodology
   - 3.1 System Architecture
   - 3.2 Component Descriptions
   - 3.3 Experimental Design
   - 3.4 Evaluation Framework
4. Experiments & Results
   - 4.1 Baseline Establishment
   - 4.2 Extraction Layer Analysis
   - 4.3 Embedding Model Comparison
   - 4.4 Chunking Strategy Impact
   - 4.5 Retrieval Method Evaluation
   - 4.6 Prompt Engineering Effects
   - 4.7 Multimodal Analysis
   - 4.8 Optimal Configuration
5. Discussion
6. Conclusions & Future Work

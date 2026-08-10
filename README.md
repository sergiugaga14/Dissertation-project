# Dissertation Project

This repository contains the code, experiments, documentation, and results for my dissertation project.

## Project Goal

The goal of this project is to investigate a Retrieval-Augmented Generation (RAG) system for medical question answering, using medical literature as the retrieval source and MedGemma as the language model.

The project will use:

* Python
* PyTorch
* Hugging Face
* MedGemma
* PubMed Open Access
* MedQA
* A vector database / similarity search system for retrieval

## 1. Environment Setup

The project should be reproducible on both my personal laptop (WSL/Linux) and the Linux server.

### Python

Use Python 3.11.

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Verify:

```bash
python --version
```

### Git

Configure Git and connect the WSL environment to GitHub using SSH.

Check:

```bash
git --version
ssh -T git@github.com
```

## 2. Project Structure

The project should eventually follow a structure similar to:

```text
dissertation/
├── README.md
├── .gitignore
├── requirements.txt
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data/
│   ├── retrieval/
│   ├── generation/
│   └── evaluation/
│
├── scripts/
├── notebooks/
├── configs/
├── experiments/
├── results/
└── docs/
```

Large datasets, model weights, generated indexes, and other large files should **not** be committed to Git.

## 3. Research Setup

### Step 1 — Investigate the datasets

Set up and document:

* MedQA
* PubMed Open Access subset

Record:

* Dataset source
* Dataset version
* Number of samples/documents
* Data format
* Train/validation/test splits

### Step 2 — Investigate MedGemma

Determine:

* Which MedGemma model will be used
* Hardware requirements
* GPU/VRAM requirements
* How the model is loaded
* Whether quantization is required

Record the model version/revision used for the experiments.

### Step 3 — Build a baseline

Create the simplest possible baseline before implementing the complete RAG system.

```text
Question
   ↓
MedGemma
   ↓
Answer
```

Evaluate the baseline on MedQA.

This provides a reference point for later experiments.

### Step 4 — Implement document retrieval

Create a retrieval pipeline:

```text
PubMed documents
       ↓
Chunking
       ↓
Embeddings
       ↓
Vector index
       ↓
Relevant documents
```

Investigate and document:

* Chunk size
* Chunk overlap
* Embedding model
* Similarity metric
* Number of retrieved documents (`top-k`)
* Vector database/index

### Step 5 — Implement RAG

Combine retrieval with MedGemma:

```text
Question
   ↓
Retriever
   ↓
Relevant PubMed documents
   ↓
Prompt + retrieved context
   ↓
MedGemma
   ↓
Answer
```

### Step 6 — Define evaluation metrics

Evaluate both retrieval and answer quality.

#### Retrieval

* Recall@k
* Precision@k
* MRR or another appropriate ranking metric

#### Answer quality

* Accuracy
* Exact Match where appropriate
* Other appropriate metrics identified during the research

The final choice of metrics should be documented and justified in the dissertation.

## 4. Experiments

Every experiment should be documented and reproducible.

For each experiment, record:

* Experiment ID
* Date
* Dataset/version
* Model/version
* Embedding model
* Chunking configuration
* Retrieval configuration
* `top-k`
* Generation parameters
* Evaluation metrics
* Results
* Observations

Example:

```text
Experiment: EXP-001
Description: Baseline RAG

Dataset:
Model:
Embedding model:
Chunk size:
Top-k:

Results:
Accuracy:
Recall@k:

Observations:
...
```

## 5. Research Log

Keep a research log in:

```text
docs/research_log.md
```

Record:

* Decisions
* Ideas
* Problems encountered
* Experiments performed
* Unexpected results
* Things to investigate
* Changes to the methodology

The purpose is to avoid relying on memory when writing the dissertation later.

## 6. Reproducibility

Every important experiment should be reproducible from the repository.

Record:

* Python version
* Package versions
* Model versions
* Dataset versions
* Experiment configuration
* Git commit
* Hardware used

The goal is to be able to recreate an experiment later on another machine.

## 7. Recommended Workflow

When starting work:

```bash
git pull
source .venv/bin/activate
```

Check the current project state:

```bash
git status
```

Before making significant changes:

```bash
git add .
git commit -m "Description of changes"
git push
```

For experiments, save the configuration and results rather than relying only on notebooks or terminal output.

## 8. Dissertation Workflow

The overall research process is:

```text
1. Set up environment
        ↓
2. Understand datasets
        ↓
3. Investigate MedGemma
        ↓
4. Build baseline
        ↓
5. Build retrieval system
        ↓
6. Build RAG pipeline
        ↓
7. Define evaluation methodology
        ↓
8. Run experiments
        ↓
9. Compare results
        ↓
10. Analyze results
        ↓
11. Write dissertation
```

The methodology should evolve based on the results of the experiments. Decisions and changes should be documented throughout the project.

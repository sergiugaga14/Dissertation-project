# Dissertation Project

This repository contains the code, experiments, documentation, and results for my dissertation project.

## Project Goal

The goal of this project is to investigate a Retrieval-Augmented Generation (RAG) system for medical question answering, using medical literature as the retrieval source and MedGemma as the language model.

The project will use:

* Python 3.12 (managed with uv)
* PyTorch
* Hugging Face
* MedGemma
* PubMed Open Access
* MedQA
* A vector database / similarity search system for retrieval

## 1. Environment Setup

The project must be reproducible on both my personal laptop (WSL/Linux) and the Linux
server. The environment is managed with [uv](https://docs.astral.sh/uv/), which pins
the Python interpreter *and* every package version so both machines are identical.

### Install uv (once per machine)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # macOS / Linux / WSL
```

Restart the shell afterwards so `~/.local/bin` is on `PATH`.

### Create the environment

From the repository root:

```bash
uv sync
```

That single command:

1. reads `.python-version` and downloads CPython **3.12.13** if it is missing
   (a standalone build in `~/.local/share/uv/python` — the system Python is not touched);
2. creates `.venv/`;
3. installs the exact package versions recorded in `uv.lock`.

Verify:

```bash
uv run python --version      # -> Python 3.12.13
```

### Files that guarantee reproducibility

| File | Purpose | Committed |
| --- | --- | --- |
| `.python-version` | Exact interpreter version (`3.12.13`) | Yes |
| `pyproject.toml` | Declared dependencies and version constraints | Yes |
| `uv.lock` | Fully resolved dependency graph with hashes, cross-platform | Yes |
| `.venv/` | The generated environment | No (gitignored) |

`uv.lock` resolves for Linux, macOS and Windows in one file, so the same lockfile
serves both machines. Never edit it by hand.

### Working in the environment

Prefix commands with `uv run` — there is no need to activate anything, and `uv run`
re-syncs the environment first so it can never be stale:

```bash
uv run python script.py
uv run pytest
uv run jupyter lab
```

Managing dependencies:

```bash
uv add torch transformers        # add packages, updates pyproject.toml + uv.lock
uv add --dev pytest ruff         # development-only dependencies
uv remove <package>              # remove a package
uv lock --upgrade                # deliberately bump to the newest allowed versions
uv tree                          # inspect the dependency graph
```

Use `uv add`, **not** `uv pip install`. Only `uv add` records the change in the
lockfile, which is what propagates it to the other machine.

After adding or removing dependencies, commit both files together:

```bash
git add pyproject.toml uv.lock
git commit -m "Add retrieval dependencies"
```

> **Note (server only):** `~/.bashrc` defines `alias python='/usr/bin/python3.11'`.
> Shell aliases take precedence over `PATH`, so after `source .venv/bin/activate`
> the bare `python` command would still be 3.11. `uv run` is unaffected — this is
> why it is the recommended way to run things.

### Interactive development

The project is written as plain `.py` modules and scripts — not notebooks. Real
modules import cleanly, diff properly in Git, and can be tested.

To still get cell-by-cell execution, mark cells with `# %%` comments and run them
with **Shift+Enter**, which opens VS Code's Interactive Window against the same
kernel. See `scripts/check_environment.py`. The same file remains a normal script:

```bash
uv run python scripts/check_environment.py
```

Use `# %% [markdown]` for prose cells.

Reserve real `.ipynb` files for genuinely throwaway experiments and keep them in
`notebooks/`. Anything that outlives the experiment belongs in
`src/dissertation_project/`, imported back into the script or notebook.

If you do want a full Jupyter server:

```bash
uv run jupyter lab
```

### GPU

`pyproject.toml` sets `torch-backend = "auto"` under `[tool.uv]`, so `uv sync`
selects CUDA wheels on a machine with an NVIDIA driver and falls back gracefully
elsewhere. One lockfile covers both machines.

Never hard-code `.cuda()`. Use the helper in
`src/dissertation_project/device.py`, which resolves CUDA → MPS → CPU:

```python
from dissertation_project.device import describe_device, get_device, get_dtype

device = get_device()
print(describe_device())

model = model.to(device)
batch = batch.to(device)
```

`get_dtype()` returns `bfloat16` on GPUs that support it, `float16` on older ones,
and `float32` on CPU — which matters when loading MedGemma, as full `float32`
weights will not fit in most consumer GPU memory:

```python
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=get_dtype())
```

Check what the current machine offers:

```bash
uv run python scripts/check_environment.py
```

| Machine | Device | Notes |
| --- | --- | --- |
| Linux server (`tamdhu`) | CPU, 52 threads | No NVIDIA GPU — onboard Matrox VGA only |
| Personal laptop | CUDA if present | Set by `torch-backend = "auto"` at sync time |

Because the server is CPU-only, run generation and any fine-tuning on the GPU
machine; use the server for data preparation, indexing, and evaluation.

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
Dissertation-project/
├── README.md
├── .gitignore
├── .python-version              # pinned interpreter
├── pyproject.toml               # dependencies and project metadata
├── uv.lock                      # fully resolved dependency graph
│
├── data/                        # gitignored
│   ├── raw/
│   └── processed/
│
├── src/
│   └── dissertation_project/
│       ├── data/
│       ├── retrieval/
│       ├── generation/
│       └── evaluation/
│
├── scripts/
├── notebooks/
├── configs/
├── experiments/
├── results/
└── Documentation/
```

The package lives at `src/dissertation_project/` because `pyproject.toml` uses the
`uv_build` backend, which expects a `src/<package_name>/` layout. `uv sync` installs
it in editable mode, so modules are importable from anywhere in the project:

```python
from dissertation_project.retrieval import build_index
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

The Python version and package versions are captured automatically by
`.python-version` and `uv.lock`, so recording the **Git commit** of an experiment is
enough to restore the exact software environment:

```bash
git checkout <commit>
uv sync
```

Record manually for each experiment:

* Model version/revision (e.g. the Hugging Face commit hash of the MedGemma checkpoint)
* Dataset version
* Experiment configuration
* Git commit
* Hardware used (GPU model, VRAM, driver/CUDA version)

The goal is to be able to recreate an experiment later on another machine.

## 7. Recommended Workflow

When starting work:

```bash
git pull
uv sync          # apply any dependency changes pulled from the other machine
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

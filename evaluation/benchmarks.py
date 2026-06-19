"""Benchmark dataset loaders for medical QA evaluation."""

from datasets import load_dataset


def load_medqa(split: str = "test", num_samples: int | None = None) -> tuple[list[str], list[str]]:
    """Load MedQA (USMLE-style) benchmark.

    Returns (questions, reference_answers) tuples.
    """
    ds = load_dataset("bigbio/med_qa", split=split)
    questions = []
    answers = []

    for item in ds:
        # Format as MCQ
        q = item["question"]
        options = item.get("options", {})
        if options:
            q += "\n" + "\n".join(f"  {k}) {v}" for k, v in options.items())
        questions.append(q)
        answers.append(item["answer"])

        if num_samples and len(questions) >= num_samples:
            break

    return questions, answers


def load_pubmedqa(
    split: str = "test", num_samples: int | None = None
) -> tuple[list[str], list[str]]:
    """Load PubMedQA benchmark (yes/no/maybe from abstracts)."""
    ds = load_dataset("bigbio/pubmed_qa", "pubmed_qa_labeled_fold0_source", split=split)
    questions = []
    answers = []

    for item in ds:
        questions.append(item["question"])
        answers.append(item["final_decision"])  # yes/no/maybe

        if num_samples and len(questions) >= num_samples:
            break

    return questions, answers


def load_custom_cases(filepath: str) -> tuple[list[str], list[str]]:
    """Load custom clinical cases from a JSON file.

    Expected format:
    [
        {"question": "...", "answer": "...", "category": "..."},
        ...
    ]
    """
    import json
    from pathlib import Path

    with open(Path(filepath)) as f:
        cases = json.load(f)

    questions = [c["question"] for c in cases]
    answers = [c["answer"] for c in cases]
    return questions, answers

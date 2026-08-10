from pathlib import Path
from datasets import Dataset, load_dataset

OUT_DIR = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Dataset Structure
# Each row is a USMLE multiple-choice question with the following features:

# question: the clinical vignette / question stem
# options: a dict of the four choices, keyed "A" to "D"
# answer: the full text of the correct option, e.g. "Nitrofurantoin"
# answer_idx: the letter of the correct option, e.g. "D" -- use this for scoring
# meta_info: which USMLE step the question came from, e.g. "step2&3"
# metamap_phrases: medical concepts extracted from the question by MetaMap
#  Splits are train (10,178) and test (1,273); there is no validation split.
def download_med_qa():
    med_qa = load_dataset("GBaker/MedQA-USMLE-4-options")

    print(med_qa)

    for split, dataset in med_qa.items():
        out_file = OUT_DIR/f"med_qa_{split}.jsonl"
        dataset.to_json(out_file)
        print(f"{split}: {len(dataset)} rows -> {out_file}")


# Dataset Structure
# Each row is a snippet of PubMed, which includes the following features:

# id: a unique identifier of the snippet
# title: the title of the PubMed article from which the snippet is collected
# content: the abstract of the PubMed article from which the snippet is collected
# contents: a concatenation of 'title' and 'content', which will be used by the BM25 retriever


# Here the dataset was too large to download in one go, so we limit it to 50k rows. The full dataset can be downloaded from the HuggingFace hub.
LIMIT = 50_000
def download_pubmed():
    # it requires a split, no download is performed right now => get an iterable dataset that loads data as needed
    stream = load_dataset("MedRAG/pubmed", split="train", streaming=True)
    subset = Dataset.from_list(list(stream.take(LIMIT)))
    path  = OUT_DIR/f"pubmed_snippets.jsonl"
    subset.to_json(path)
    print(f"pubmed: {len(subset)} rows -> {path}")

download_med_qa()
download_pubmed()
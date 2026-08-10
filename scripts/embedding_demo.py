import json
from pathlib import Path
from sentence_transformers import SentenceTransformer


SNIPPETS_PATH = Path("data/raw/pubmed_snippets.jsonl")
LIMIT = 100

texts = []
with open(SNIPPETS_PATH, "r") as f:
    for i, line in enumerate(f):
        if i >= LIMIT:
            break
        texts.append(json.loads(line)["contents"])

print(len(texts))
print(texts[0][:200])

model = SentenceTransformer("thenlper/gte-small")

embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
print(embeddings.shape)

question = "What is the role of alpha-bisabolol in reducing peptic activity?"
query_vector = model.encode([question], normalize_embeddings=True)

# what does @ do? It is matrix multiplication, so we are computing the dot product of the query vector with each of the snippet embeddings.
#  This gives us a score for each snippet, which we can then sort to find the most relevant snippets.
scores = embeddings @ query_vector[0]

# why ::-1? what does argsort do?  
# arg sort returns the indices that would sort an array. 
# ::-1 reverses the order of the array, so we are getting the indices of the top scores in descending order.
top = scores.argsort()[::-1][:3]

for rank, idx in enumerate(top ,start=1):
    print(f"Rank {rank}: score={scores[idx]:.4f}, snippet={texts[idx][:200]}...")
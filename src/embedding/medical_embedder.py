"""Medical-domain embedding models."""

import numpy as np
from sentence_transformers import SentenceTransformer

from .base import BaseEmbedder


class MedicalEmbedder(BaseEmbedder):
    """Embedding models trained/fine-tuned on biomedical text.

    Supported models:
    - ncbi/MedCPT-Query-Encoder + ncbi/MedCPT-Article-Encoder (asymmetric)
    - FremyCompany/BioLORD-2023 (biomedical ontology-aware)
    - microsoft/BiomedNLP-PubMedBERT-base (general biomed)
    """

    def __init__(self, model_name: str = "FremyCompany/BioLORD-2023"):
        self.name = model_name.split("/")[-1]
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=len(texts) > 100)


class MedCPTEmbedder(BaseEmbedder):
    """MedCPT uses asymmetric encoding — separate query and article encoders.

    This is important: queries and documents use DIFFERENT models,
    which often improves retrieval in information-seeking scenarios.
    """

    name = "MedCPT"

    def __init__(self):
        self.query_encoder = SentenceTransformer("ncbi/MedCPT-Query-Encoder")
        self.article_encoder = SentenceTransformer("ncbi/MedCPT-Article-Encoder")
        self.dimension = self.query_encoder.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed documents (articles)."""
        return self.article_encoder.encode(texts, show_progress_bar=len(texts) > 100)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query — uses the separate query encoder."""
        return self.query_encoder.encode([query])[0]

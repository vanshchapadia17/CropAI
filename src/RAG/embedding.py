from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    """Manages text embeddings using a local SentenceTransformer model.

    Uses all-MiniLM-L6-v2 by default — small, fast, 384-dim, no API key needed.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding model ready. Dimension: {self.dim}")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts. Returns float32 array of shape (n, dim)."""
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        return np.array(embeddings).astype("float32")

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query. Returns float32 array of shape (1, dim)."""
        embedding = self.model.encode([text])
        return np.array(embedding).astype("float32")

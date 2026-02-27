"""RAG pipeline: builds/loads a FAISS index from the scraped txt files
and retrieves relevant chunks for any incoming query.

The class is a singleton — call RAGPipeline.get_instance() from anywhere.
The FAISS index is persisted in artifacts/ so it only needs to be built once.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from langchain.schema import Document

from src.RAG.data_loader import load_all_documents, split_documents
from src.RAG.embedding import EmbeddingManager

ARTIFACTS_DIR = Path("artifacts")
INDEX_PATH = ARTIFACTS_DIR / "rag_index.faiss"
DOCS_PATH = ARTIFACTS_DIR / "rag_docs.pkl"


class RAGPipeline:
    """FAISS-based retrieval pipeline for rice and sugarcane knowledge."""

    _instance: Optional["RAGPipeline"] = None

    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Document] = []

    # ── Singleton access ────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls) -> "RAGPipeline":
        """Return the singleton, building the FAISS index if it doesn't exist."""
        if cls._instance is None:
            pipeline = cls()
            if INDEX_PATH.exists() and DOCS_PATH.exists():
                pipeline.load()
            else:
                print("RAG index not found — building from txt files...")
                ARTIFACTS_DIR.mkdir(exist_ok=True)
                docs = load_all_documents()
                chunks = split_documents(docs)
                pipeline.build_index(chunks)
                pipeline.save()
            cls._instance = pipeline
        return cls._instance

    # ── Index operations ────────────────────────────────────────────────────

    def build_index(self, chunks: List[Document]):
        """Build FAISS IndexFlatL2 from document chunks."""
        texts = [c.page_content for c in chunks]
        print(f"Embedding {len(texts)} chunks...")
        embeddings = self.embedding_manager.embed_texts(texts)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.chunks = chunks
        print(f"FAISS index built: {self.index.ntotal} vectors, dim={dim}")

    def save(self):
        """Persist the FAISS index and chunk metadata to disk."""
        faiss.write_index(self.index, str(INDEX_PATH))
        with open(DOCS_PATH, "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"RAG index saved to: {INDEX_PATH}")

    def load(self):
        """Load a previously saved FAISS index from disk."""
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(DOCS_PATH, "rb") as f:
            self.chunks = pickle.load(f)
        print(
            f"RAG index loaded: {self.index.ntotal} vectors, "
            f"{len(self.chunks)} chunks"
        )

    # ── Retrieval ───────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Retrieve the top-k most relevant chunks for a query.

        Returns a list of dicts with keys:
            content, crop_type, source_file, score (0-1 similarity)
        """
        query_emb = self.embedding_manager.embed_query(query)
        distances, indices = self.index.search(query_emb, k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({
                "content": chunk.page_content,
                "crop_type": chunk.metadata.get("crop_type", "unknown"),
                "source_file": chunk.metadata.get("source_file", "unknown"),
                "score": round(1 / (1 + float(dist)), 4),
            })
        return results

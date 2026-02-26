"""Test script to validate the fixed RAGRetriever with FAISSVectorStore."""

import os
import sys
import numpy as np
import faiss
import pickle
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# ── FAISSVectorStore ──────────────────────────────────────────────────────────

class FAISSVectorStore:
    def __init__(self, embedding_dim: int = 384, index_path: str = "../data/faiss_index"):
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents = []
        print("FAISS Vector Store initialized")

    def add_documents(self, documents, embeddings: np.ndarray):
        embeddings = np.array(embeddings).astype("float32")
        self.index.add(embeddings)
        self.documents.extend(documents)
        print(f"Total vectors stored: {self.index.ntotal}")

    def similarity_search(self, query_embedding: np.ndarray, k: int = 3):
        """Returns List of (document, l2_distance) tuples."""
        query_embedding = np.array(query_embedding).reshape(1, -1).astype("float32")
        distances, indices = self.index.search(query_embedding, k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:
                results.append((self.documents[idx], float(dist)))
        return results

    def load(self):
        self.index = faiss.read_index(f"{self.index_path}.index")
        with open(f"{self.index_path}.pkl", "rb") as f:
            self.documents = pickle.load(f)
        print(f"FAISS index loaded ({self.index.ntotal} vectors)")

# ── EmbeddingManager ─────────────────────────────────────────────────────────

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"Embedding dim: {self.model.get_sentence_embedding_dimension()}")

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False)

# ── RAGRetriever ─────────────────────────────────────────────────────────────

class RAGRetriever:
    def __init__(self, vector_store: FAISSVectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        print(f"\nQuery: '{query}'")
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        try:
            results = self.vector_store.similarity_search(query_embedding=query_embedding, k=top_k)

            retrieved_docs = []
            for rank, (doc, distance) in enumerate(results):
                score = 1 / (1 + distance)
                if score >= score_threshold:
                    retrieved_docs.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": round(score, 4),
                        "distance": round(distance, 4),
                        "rank": rank + 1,
                    })

            print(f"Retrieved {len(retrieved_docs)} documents.")
            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Change working dir to notebooks/ so relative paths match the notebook
    os.chdir(Path(__file__).parent)

    embedding_manager = EmbeddingManager()

    # Load existing FAISS index (saved by the notebook)
    vectorstore = FAISSVectorStore()
    vectorstore.load()

    retriever = RAGRetriever(vector_store=vectorstore, embedding_manager=embedding_manager)

    # ── Test queries ──
    test_queries = [
        "What are the main topics covered in the PDF documents?",
        "Tell me about machine learning algorithms",
        "What programming languages are used in data science?",
        "What is artificial intelligence?",
    ]

    for query in test_queries:
        results = retriever.retrieve(query, top_k=3)
        for r in results:
            print(f"  Rank {r['rank']} | Score: {r['score']} | Distance: {r['distance']}")
            print(f"  Source: {r['metadata'].get('source_file', 'N/A')}")
            print(f"  Content: {r['content'][:120].strip()}...")
            print()

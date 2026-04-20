"""Hybrid retriever: dense (FAISS) + sparse (BM25) fused via Reciprocal Rank Fusion.

Built on top of the existing RAGPipeline — reuses its FAISS index and chunks
so BM25 stays in lockstep with whatever embeddings are indexed.

Why RRF: it fuses two ranked lists without needing score calibration between
dense similarity and BM25 scores (which live on incompatible scales).
"""

import re
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

from src.RAG.rag_pipeline import RAGPipeline

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    # Lowercase word-tokens; keeps variety codes like "co86032" intact.
    return _TOKEN_RE.findall(text.lower())


class HybridRetriever:
    """Dense + BM25 retrieval fused with Reciprocal Rank Fusion.

    Parameters:
        dense_k: candidates pulled from FAISS
        sparse_k: candidates pulled from BM25
        rrf_k: RRF constant (60 is the value from the original RRF paper;
               larger values flatten rank differences, smaller values amplify
               top-rank confidence)
    """

    _instance: Optional["HybridRetriever"] = None

    def __init__(
        self,
        dense_k: int = 20,
        sparse_k: int = 20,
        rrf_k: int = 60,
    ):
        self.dense_k = dense_k
        self.sparse_k = sparse_k
        self.rrf_k = rrf_k

        self.rag = RAGPipeline.get_instance()
        tokenized = [_tokenize(c.page_content) for c in self.rag.chunks]
        self.bm25 = BM25Okapi(tokenized)

    @classmethod
    def get_instance(cls) -> "HybridRetriever":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _dense_ranks(self, query: str) -> List[int]:
        """Dense retrieval — returns chunk indices in rank order."""
        emb = self.rag.embedding_manager.embed_query(query)
        _, indices = self.rag.index.search(emb, self.dense_k)
        return [int(i) for i in indices[0] if i != -1]

    def _sparse_ranks(self, query: str) -> List[int]:
        """BM25 retrieval — returns chunk indices in rank order (top N)."""
        scores = self.bm25.get_scores(_tokenize(query))
        # argsort descending; keep only non-zero scored docs up to sparse_k
        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )
        # Drop docs with 0 BM25 score (no query term match) — they'd just be
        # arbitrary rank noise in the fusion.
        return [i for i in ranked[: self.sparse_k] if scores[i] > 0]

    def _rrf_fuse(self, ranked_lists: List[List[int]]) -> List[int]:
        """Reciprocal Rank Fusion: score = sum(1 / (k + rank_i))."""
        scores: Dict[int, float] = {}
        for ranked in ranked_lists:
            for rank_zero_based, doc_id in enumerate(ranked):
                rank = rank_zero_based + 1
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
        return sorted(scores, key=scores.get, reverse=True)

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Hybrid retrieval — returns top-k chunks with the same dict shape
        as RAGPipeline.retrieve so callers can swap them freely."""
        dense = self._dense_ranks(query)
        sparse = self._sparse_ranks(query)
        fused = self._rrf_fuse([dense, sparse])[:k]

        # Attach a human-readable RRF score for debugging / logging.
        rrf_scores: Dict[int, float] = {}
        for ranked in (dense, sparse):
            for rank_zero_based, doc_id in enumerate(ranked):
                rank = rank_zero_based + 1
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)

        results = []
        for idx in fused:
            chunk = self.rag.chunks[idx]
            results.append({
                "content": chunk.page_content,
                "crop_type": chunk.metadata.get("crop_type", "unknown"),
                "source_file": chunk.metadata.get("source_file", "unknown"),
                "score": round(rrf_scores[idx], 4),
            })
        return results

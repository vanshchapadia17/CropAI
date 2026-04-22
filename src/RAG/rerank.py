"""Cross-encoder reranker layered on top of the hybrid retriever.

Pipeline: hybrid (BM25 + dense + RRF) pulls a wide candidate pool, then a
cross-encoder scores each (query, chunk) pair jointly and the top-k are
returned. Cross-encoders outperform bi-encoders on ranking because they
attend across query and document together instead of embedding each
independently.

Model: ms-marco-MiniLM-L-6-v2 — ~90 MB, CPU-viable for our query volume.
"""

import time
from typing import Any, Dict, List, Optional

from sentence_transformers import CrossEncoder

from src.RAG.hybrid import HybridRetriever
from src.logger import logging

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankRetriever:
    """Hybrid → cross-encoder rerank.

    Parameters:
        candidate_k: how many hybrid candidates to feed the reranker.
                     Larger = better recall, more CPU work. 20 is a sensible
                     default for our corpus size (~520 chunks).
    """

    _instance: Optional["RerankRetriever"] = None

    def __init__(self, candidate_k: int = 10):
        # k=10 chosen from eval: halves latency vs k=20 (1445 ms -> 632 ms)
        # with Hit@3 and Hit@5 unchanged and only a ~3 pp Hit@1 drop.
        self.candidate_k = candidate_k
        # The hybrid retriever pulls dense_k + sparse_k = 20 + 20 candidates
        # before fusing; plenty for the reranker to choose from.
        self.hybrid = HybridRetriever.get_instance()
        print(f"Loading cross-encoder: {MODEL_NAME}")
        self.cross_encoder = CrossEncoder(MODEL_NAME)
        print("Cross-encoder ready.")

    @classmethod
    def get_instance(cls) -> "RerankRetriever":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        t0 = time.time()
        logging.info(f"[RERANK] query={query!r} k={k} candidate_k={self.candidate_k}")

        # Pull a wide candidate pool from hybrid — ask for candidate_k so we
        # have enough room for the cross-encoder to surface better matches.
        candidates = self.hybrid.retrieve(query, k=self.candidate_k)
        if not candidates:
            logging.info("[RERANK] no candidates returned by hybrid; returning empty.")
            return []

        logging.info(
            f"[RERANK] hybrid returned {len(candidates)} candidates "
            f"(top RRF: {[c['source_file'] for c in candidates[:3]]})"
        )

        pairs = [(query, c["content"]) for c in candidates]
        ce_scores = self.cross_encoder.predict(pairs)

        # Sort candidates by cross-encoder score (higher is better) and keep k.
        ranked = sorted(
            zip(candidates, ce_scores),
            key=lambda pair: float(pair[1]),
            reverse=True,
        )[:k]

        results = []
        for cand, score in ranked:
            # Overwrite score with the cross-encoder value so downstream
            # consumers see the reranker's confidence, not RRF's.
            cand = dict(cand)
            cand["score"] = round(float(score), 4)
            results.append(cand)

        latency_ms = (time.time() - t0) * 1000
        top = [(r["source_file"], r["crop_type"], r["score"]) for r in results]
        logging.info(
            f"[RERANK] top-{k} after cross-encoder: {top} | latency={latency_ms:.1f}ms"
        )
        return results

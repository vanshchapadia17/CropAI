"""Run the golden set through a retrieval config and report Hit@k + MRR.

Usage from the project root:
    virtualEN/python.exe -m src.RAG.evaluate
    virtualEN/python.exe -m src.RAG.evaluate --config baseline --k 10

Writes a JSON result file under eval/ so later configs (BM25+RRF, rerank)
can be diffed against this baseline.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# Load .env so GROQ_API_KEY is available when --faithfulness is used.
load_dotenv()

from src.RAG.rag_pipeline import RAGPipeline

GOLDEN_PATH = Path("eval/golden.jsonl")
RESULTS_DIR = Path("eval/results")
K_VALUES = (1, 3, 5)


def _get_retriever(config: str, candidate_k: int = 20):
    """Return an object with a `.retrieve(query, k)` method matching the
    dict-result shape used in the baseline path."""
    if config == "baseline":
        return RAGPipeline.get_instance()
    if config == "hybrid":
        from src.RAG.hybrid import HybridRetriever
        return HybridRetriever.get_instance()
    if config == "rerank":
        from src.RAG.rerank import RerankRetriever
        # Bypass the singleton so --candidate-k can vary between runs.
        return RerankRetriever(candidate_k=candidate_k)
    raise ValueError(f"Unknown config: {config!r}. Choose: baseline, hybrid, rerank")


def load_golden() -> List[Dict[str, Any]]:
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_one(expected: List[str], retrieved: List[str]) -> Dict[str, Any]:
    """Compute Hit@k flags and reciprocal rank for a single query.

    A hit is any retrieved source_file appearing in expected. We track the
    rank of the FIRST match for MRR; if no match, rank=None and RR=0.
    """
    expected_set = set(expected)
    rank = None
    for i, src in enumerate(retrieved, start=1):
        if src in expected_set:
            rank = i
            break

    return {
        "rank": rank,
        "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
        **{f"hit@{k}": (rank is not None and rank <= k) for k in K_VALUES},
        "retrieved": retrieved,
    }


def run_baseline(
    queries: List[Dict[str, Any]],
    top_k: int,
    config: str = "baseline",
    candidate_k: int = 20,
    faithfulness: bool = False,
) -> List[Dict[str, Any]]:
    retriever = _get_retriever(config, candidate_k=candidate_k)
    judge = None
    if faithfulness:
        from src.RAG.faithfulness import FaithfulnessEvaluator
        judge = FaithfulnessEvaluator()

    out = []
    total = len(queries)
    for i, q in enumerate(queries, 1):
        # flush=True: otherwise Python buffers when stdout is piped, hiding
        # progress on long multi-minute runs.
        print(
            f"[{i:>2}/{total}] {q['id']} {q['category']:<22} {q['query'][:60]}",
            flush=True,
        )

        t0 = time.time()
        hits = retriever.retrieve(q["query"], k=top_k)
        latency_ms = (time.time() - t0) * 1000
        retrieved_files = [h["source_file"] for h in hits]
        result = score_one(q["expected_files"], retrieved_files)

        row = {
            "id": q["id"],
            "category": q["category"],
            "query": q["query"],
            "expected": q["expected_files"],
            "latency_ms": round(latency_ms, 1),
            **result,
        }

        if judge is not None:
            # Faithfulness uses the retrieved chunks as context.
            fa = judge.score_one(q["query"], hits)
            row["faithfulness"] = fa

        rank = row.get("rank")
        tag = f"rank={rank}" if rank else "MISS"
        extras = ""
        if "faithfulness" in row:
            extras = f" | faith={row['faithfulness']['verdict']}"
        print(
            f"         -> {tag}  latency={row['latency_ms']}ms{extras}",
            flush=True,
        )

        out.append(row)
    return out


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Roll up per-category and overall Hit@k + MRR + faithfulness."""
    def agg_group(rows: List[Dict[str, Any]]) -> Dict[str, float]:
        n = len(rows)
        if n == 0:
            return {"n": 0}
        summary = {"n": n, "mrr": round(sum(r["reciprocal_rank"] for r in rows) / n, 4)}
        for k in K_VALUES:
            key = f"hit@{k}"
            summary[key] = round(sum(1 for r in rows if r[key]) / n, 4)
        summary["avg_latency_ms"] = round(sum(r["latency_ms"] for r in rows) / n, 1)

        fa_rows = [r.get("faithfulness") for r in rows if r.get("faithfulness")]
        if fa_rows:
            summary["faithfulness_score"] = round(
                sum(fa["score"] for fa in fa_rows) / len(fa_rows), 4
            )
            verdict_counts = defaultdict(int)
            for fa in fa_rows:
                verdict_counts[fa["verdict"]] += 1
            summary["verdicts"] = dict(verdict_counts)
        return summary

    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    return {
        "overall": agg_group(results),
        "by_category": {cat: agg_group(rows) for cat, rows in sorted(by_cat.items())},
    }


def print_report(summary: Dict[str, Any], config_name: str) -> None:
    print()
    print("=" * 70)
    print(f"  RAG Evaluation — config: {config_name}")
    print("=" * 70)

    overall = summary["overall"]
    print(f"\nOverall ({overall['n']} queries)")
    print(f"  Hit@1:  {overall['hit@1']:.2%}")
    print(f"  Hit@3:  {overall['hit@3']:.2%}")
    print(f"  Hit@5:  {overall['hit@5']:.2%}")
    print(f"  MRR:    {overall['mrr']:.4f}")
    print(f"  Avg latency: {overall['avg_latency_ms']} ms")

    print("\nPer category")
    print(f"  {'category':<24}{'n':>4}{'Hit@1':>9}{'Hit@3':>9}{'Hit@5':>9}{'MRR':>9}")
    for cat, s in summary["by_category"].items():
        print(
            f"  {cat:<24}{s['n']:>4}"
            f"{s['hit@1']:>9.2%}{s['hit@3']:>9.2%}{s['hit@5']:>9.2%}{s['mrr']:>9.4f}"
        )

    if "faithfulness_score" in overall:
        print("\nFaithfulness (LLM-as-judge)")
        print(f"  Score:    {overall['faithfulness_score']:.3f}   "
              f"(1.0 = fully grounded, 0.5 = partial, 0 = unsupported)")
        print(f"  Verdicts: {overall.get('verdicts', {})}")
        print("\n  Per category")
        print(f"  {'category':<24}{'n':>4}{'score':>8}{'   verdicts'}")
        for cat, s in summary["by_category"].items():
            if "faithfulness_score" not in s:
                continue
            print(
                f"  {cat:<24}{s['n']:>4}"
                f"{s['faithfulness_score']:>8.3f}   {s.get('verdicts', {})}"
            )


def print_misses(results: List[Dict[str, Any]], limit: int = 10) -> None:
    misses = [r for r in results if not r["hit@5"]]
    if not misses:
        print("\nNo misses at k=5. All queries hit an expected file in top-5.")
        return
    print(f"\nMisses at k=5 ({len(misses)} total, showing up to {limit}):")
    for m in misses[:limit]:
        print(f"  [{m['id']}] {m['query']}")
        print(f"       expected: {m['expected']}")
        print(f"       retrieved: {m['retrieved']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="baseline",
                        help="Label for this run (used in results filename)")
    parser.add_argument("--k", type=int, default=max(K_VALUES),
                        help="top_k to retrieve (must be >= max Hit@k tracked)")
    parser.add_argument("--candidate-k", type=int, default=20,
                        help="Rerank candidate pool size (rerank config only)")
    parser.add_argument("--faithfulness", action="store_true",
                        help="Also run LLM-as-judge faithfulness eval (uses Groq)")
    args = parser.parse_args()

    if args.k < max(K_VALUES):
        print(f"WARNING: k={args.k} is below max Hit@k={max(K_VALUES)}; increasing.")
        args.k = max(K_VALUES)

    queries = load_golden()
    print(f"Loaded {len(queries)} golden queries from {GOLDEN_PATH}")

    results = run_baseline(
        queries,
        top_k=args.k,
        config=args.config,
        candidate_k=args.candidate_k,
        faithfulness=args.faithfulness,
    )
    summary = aggregate(results)
    print_report(summary, args.config)
    print_misses(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.config}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"config": args.config, "k": args.k,
                   "summary": summary, "results": results}, f, indent=2)
    print(f"\nSaved detailed results -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""LLM-as-judge faithfulness evaluation for the RAG pipeline.

For each golden query:
  1. Retrieve context with the chosen retriever.
  2. Generate an answer from (query, context) using Groq llama.
  3. Ask the same model (as judge) whether the answer is supported by the
     context alone — yes / partial / no.

This surfaces hallucinations RAG is supposed to prevent. It's orthogonal to
Hit@k/MRR: a retriever can top-score correctly yet the LLM can still invent
details that aren't in the retrieved chunks.
"""

import json
import re
import time
from typing import Any, Dict, List

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# Verdict tokens the judge may return.
VERDICT_SCORES = {"yes": 1.0, "partial": 0.5, "no": 0.0}
VALID_VERDICTS = set(VERDICT_SCORES.keys())

ANSWER_SYSTEM = (
    "You are CropAI, an agriculture assistant for rice and sugarcane. "
    "Answer the user's question using ONLY the provided context. "
    "Be concise — 2-4 sentences. If the context does not contain the answer, "
    "say so plainly. Do not invent facts."
)

JUDGE_SYSTEM = (
    "You are a strict evaluator for a retrieval-augmented system. "
    "Given a QUESTION, the retrieved CONTEXT, and the system's ANSWER, "
    "decide whether the answer is supported by the context alone.\n\n"
    "Verdicts:\n"
    "  yes     — every factual claim in the answer is directly supported by "
    "the context.\n"
    "  partial — the answer is mostly supported but contains at least one "
    "claim that is not in the context.\n"
    "  no      — the answer contradicts or fabricates material not in the "
    "context.\n\n"
    "Reply with ONLY a JSON object: {\"verdict\": \"yes|partial|no\", "
    "\"reason\": \"one short sentence\"}. No prose outside the JSON."
)


def _format_context(chunks: List[Dict[str, Any]], max_chars: int = 500) -> str:
    """Format retrieved chunks the same way the live tool does — so the
    faithfulness measurement matches what the agent actually sees."""
    if not chunks:
        return "(no context retrieved)"
    parts = []
    for i, c in enumerate(chunks, start=1):
        snippet = c["content"][:max_chars].strip()
        parts.append(
            f"[{i}] source={c['source_file']} crop={c['crop_type']}\n{snippet}"
        )
    return "\n\n---\n\n".join(parts)


def _parse_verdict(raw: str) -> Dict[str, str]:
    """Extract {verdict, reason} from the judge's reply. Tolerates mild
    deviations — the model sometimes wraps JSON in code fences or adds
    trailing prose."""
    # Strip common wrappers.
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)

    # Try strict JSON first.
    try:
        obj = json.loads(text)
        verdict = str(obj.get("verdict", "")).strip().lower()
        reason = str(obj.get("reason", "")).strip()
        if verdict in VALID_VERDICTS:
            return {"verdict": verdict, "reason": reason}
    except json.JSONDecodeError:
        pass

    # Fallback: scan for a bare verdict token.
    low = text.lower()
    for token in ("yes", "partial", "no"):
        if re.search(rf"\b{token}\b", low):
            return {"verdict": token, "reason": text[:140]}
    return {"verdict": "invalid", "reason": text[:140]}


class FaithfulnessEvaluator:
    """Generate answers from context and judge them with the same LLM."""

    def __init__(self, model: str = "llama-3.1-8b-instant"):
        # Small max_tokens — both generator and judge produce short outputs.
        self.answerer = ChatGroq(model=model, max_tokens=300, temperature=0.0)
        # Judge gets even less headroom and zero temperature for consistency.
        self.judge = ChatGroq(model=model, max_tokens=120, temperature=0.0)

    def generate_answer(self, query: str, context: str) -> str:
        response = self.answerer.invoke([
            SystemMessage(content=ANSWER_SYSTEM),
            HumanMessage(content=f"CONTEXT:\n{context}\n\nQUESTION: {query}"),
        ])
        return response.content.strip()

    def judge_answer(self, query: str, answer: str, context: str) -> Dict[str, str]:
        response = self.judge.invoke([
            SystemMessage(content=JUDGE_SYSTEM),
            HumanMessage(content=(
                f"QUESTION: {query}\n\n"
                f"CONTEXT:\n{context}\n\n"
                f"ANSWER: {answer}"
            )),
        ])
        return _parse_verdict(response.content)

    def score_one(self, query: str, retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Full pipeline for a single query. Returns dict with generated
        answer, judge verdict/reason, numeric score, and timings."""
        context = _format_context(retrieved)

        t0 = time.time()
        answer = self.generate_answer(query, context)
        gen_ms = (time.time() - t0) * 1000

        t1 = time.time()
        verdict = self.judge_answer(query, answer, context)
        judge_ms = (time.time() - t1) * 1000

        return {
            "answer": answer,
            "verdict": verdict["verdict"],
            "reason": verdict["reason"],
            "score": VERDICT_SCORES.get(verdict["verdict"], 0.0),
            "gen_ms": round(gen_ms, 1),
            "judge_ms": round(judge_ms, 1),
        }

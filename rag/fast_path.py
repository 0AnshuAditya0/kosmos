"""Deterministic, low-latency answer path.

This module deliberately has no network calls and no second embedding pass.
It uses one in-process dense retrieval, then lexical evidence checks only on
the small candidate set returned by FAISS. Returning an exact source sentence
is safer and faster than composing an answer with an LLM.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from rag.retriever import retrieve_batch

CANDIDATE_POOL_SIZE = 12
MIN_LEXICAL_SCORE = 0.18
MIN_MATCHING_TERMS = 1

# passage from passing verification.
MIN_DENSE_SCORE_FOR_VERIFICATION = 0.55
SENTENCE_SPLIT_RE = re.compile(r"(?<=[।.!?])\s+")

STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "what", "which",
    "how", "why", "when", "where", "who", "of", "to", "in", "for",
    "and", "or", "with", "this", "that", "it", "क्या", "है", "का",
    "की", "के", "में", "को", "से", "और", "या", "यह", "वह", "कैसे",
    "क्यों", "कब", "কী", "হয়", "এর", "এবং", "বা", "এই", "সেই",
    "কিভাবে", "কেন", "কখন", "কোথায়",
})


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        if len(token) > 1 and token not in STOPWORDS
    }


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _lexical_score(question_tokens: set[str], text: str) -> tuple[float, int]:
    """Return query coverage and matched content-term count."""
    if not question_tokens:
        return 0.0, 0
    overlap = question_tokens.intersection(_tokens(text))
    return len(overlap) / len(question_tokens), len(overlap)


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]


def retrieve_with_evidence(question: str, strategy_name: str, k: int = 5) -> list[dict[str, Any]]:
    """Retrieve a small candidate set and rank it by semantic + word evidence."""
    candidates = retrieve_batch(
        [question], strategy_name, k=max(k, CANDIDATE_POOL_SIZE)
    )[0]
    question_tokens = _tokens(question)

    for candidate in candidates:
        lexical_score, matching_terms = _lexical_score(
            question_tokens, candidate["chunk_text"]
        )
        candidate["lexical_score"] = round(lexical_score, 3)
        candidate["matching_terms"] = matching_terms
        
        candidate["evidence_score"] = round(
            0.75 * float(candidate["score"]) + 0.25 * lexical_score, 3
        )

    candidates.sort(key=lambda item: item["evidence_score"], reverse=True)
    return candidates[:k]


def extract_verified_sentence(question: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the best source sentence without another model inference."""
    question_tokens = _tokens(question)
    question_normalized = _normalize(question)
    best: tuple[float, int, str, dict[str, Any]] | None = None

    for chunk in chunks[:3]:
        for sentence in _sentences(chunk.get("chunk_text", "")):
            
            if _normalize(sentence) == question_normalized:
                continue

            coverage, matches = _lexical_score(question_tokens, sentence)
            
            score = coverage + 0.10 * float(chunk["score"])
            candidate = (score, matches, sentence, chunk)
            if best is None or candidate[0] > best[0]:
                best = candidate

    if best is None:
        return {"answer": None, "source_chunk_id": None, "confidence": 0.0, "verified": False}

    score, matches, sentence, chunk = best
    dense_score = float(chunk.get("score", 0.0))

    definitional_ok = True
    if len(question_tokens) == 1:
        term = next(iter(question_tokens))
        prefix = _normalize(sentence)[:40]
        definitional_ok = term in prefix

    verified = (
        matches >= MIN_MATCHING_TERMS
        and float(chunk.get("lexical_score", 0.0)) >= MIN_LEXICAL_SCORE
        and dense_score >= MIN_DENSE_SCORE_FOR_VERIFICATION
        and definitional_ok
    )
    return {
        "answer": sentence if verified else None,
        "source_chunk_id": chunk.get("chunk_id") if verified else None,
        "confidence": round(score, 3),
        "verified": verified,
    }


def warm_fast_path(strategy_name: str) -> None:
    """Load the only model/index used by requests during application startup."""
    from rag.retriever import get_model, load_strategy

    get_model()
    load_strategy(strategy_name)
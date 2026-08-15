from __future__ import annotations

import logging
import re
from typing import Any, Sequence

logger = logging.getLogger(__name__)


# These are deliberately limited to clearly unsafe requests.
# This is NOT intended to be a complete safety classifier.
_UNSAFE_PATTERNS = (
    r"\bhow\s+to\s+(make|build|create)\s+(a\s+)?bomb\b",
    r"\bhow\s+to\s+(make|build|create)\s+(a\s+)?weapon\b",
    r"\bhow\s+to\s+kill\b",
    r"\bhow\s+to\s+poison\b",
    r"\bhow\s+to\s+make\s+an?\s+explosive\b",
    r"\bhow\s+to\s+hack\b",
    r"\bhow\s+to\s+steal\b",
)

_COMPILED_UNSAFE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in _UNSAFE_PATTERNS
)


def unsafe_input_check(text: str) -> bool:
    """
    Return True when the input is safe to process.

    Return False when the input matches a clearly unsafe request.

    This is a lightweight heuristic guardrail, not a security boundary.
    """
    if not isinstance(text, str):
        logger.warning("Unsafe-input check received non-string input.")
        return False

    normalized = " ".join(text.strip().split())

    if not normalized:
        return False

    for pattern in _COMPILED_UNSAFE_PATTERNS:
        if pattern.search(normalized):
            logger.warning("Unsafe input detected.")
            return False

    return True


def off_topic_check(
    chunks: Sequence[dict[str, Any]],
    threshold: float = 0.35,
) -> bool:
    """
    Return True when retrieval provides sufficiently relevant context.

    FAISS scores are expected to be cosine similarities because the
    retriever normalizes embeddings before searching.
    """
    if not chunks:
        return False

    scores = []

    for chunk in chunks:
        try:
            score = float(chunk["score"])
        except (KeyError, TypeError, ValueError):
            continue

        if score == score:  # exclude NaN
            scores.append(score)

    if not scores:
        return False

    return max(scores) >= threshold


def groundedness_check(
    answer: str,
    chunks: Sequence[dict[str, Any]],
    minimum_overlap: float = 0.20,
) -> bool:
    """
    Lightweight deterministic groundedness check.

    The answer must contain meaningful tokens that occur in the retrieved
    context. This is intentionally conservative and should not be treated
    as a semantic proof of factual correctness.
    """
    if not isinstance(answer, str) or not answer.strip():
        return False

    if not chunks:
        return False

    context = " ".join(
        str(chunk.get("chunk_text", ""))
        for chunk in chunks
    )

    if not context.strip():
        return False

    answer_tokens = _meaningful_tokens(answer)
    context_tokens = _meaningful_tokens(context)

    if not answer_tokens:
        return False

    overlap = answer_tokens.intersection(context_tokens)
    overlap_ratio = len(overlap) / len(answer_tokens)

    return overlap_ratio >= minimum_overlap


def _meaningful_tokens(text: str) -> set[str]:
    """
    Normalize text into meaningful alphanumeric tokens.
    """
    return {
        token
        for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        if len(token) > 2
    }
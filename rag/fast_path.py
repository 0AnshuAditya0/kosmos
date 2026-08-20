"""Deterministic, low-latency answer path with a 2-model cascade support.

Tiers 1-2 use the fast MiniLM+ONNX pipeline (rag.retriever). Tier 3 uses
the slower but much stronger BGE-M3 model for the no_chunk_bge index,
reached only when the fast tiers fail to verify an answer.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from rag.retriever import retrieve_batch

CANDIDATE_POOL_SIZE = 12
MIN_LEXICAL_SCORE = 0.18
MIN_MATCHING_TERMS = 1
MIN_DENSE_SCORE_FOR_VERIFICATION = 0.55
MIN_DENSE_SCORE_FOR_VERIFICATION_BGE = 0.45
SENTENCE_SPLIT_RE = re.compile(r"(?<=[।.!?])\s+")

STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "what", "which",
    "how", "why", "when", "where", "who", "of", "to", "in", "for",
    "and", "or", "with", "this", "that", "it", "क्या", "है", "का",
    "की", "के", "में", "को", "से", "और", "या", "यह", "वह", "कैसे",
    "क्यों", "कब", "কী", "হয়", "এর", "এবং", "বা", "এই", "সেই",
    "কিভাবে", "কেন", "কখন", "কোথায়",
})

_bge_model = None


def _get_bge_model():
    global _bge_model
    if _bge_model is None:
        from sentence_transformers import SentenceTransformer
        import torch
        _bge_model = SentenceTransformer("BAAI/bge-m3")
        if torch.cuda.is_available():
            _bge_model = _bge_model.half()
        _bge_model.eval()
    return _bge_model


def _embed_bge(queries: list) -> "np.ndarray":
    import numpy as np
    model = _get_bge_model()
    embeddings = model.encode(
        queries, batch_size=8, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def _retrieve_bge(query: str, strategy_name: str, k: int) -> list:
    """Retrieve using BGE-M3 directly against the no_chunk_bge index,
    bypassing the MiniLM/ONNX path in retriever.py entirely."""
    import faiss
    import pandas as pd
    from pathlib import Path

    index_dir = Path(__file__).resolve().parent.parent / "index"
    faiss_path = index_dir / f"{strategy_name}.faiss"
    meta_path = index_dir / f"{strategy_name}_meta.parquet"

    index = faiss.read_index(str(faiss_path))
    metadata = pd.read_parquet(meta_path)

    embedding = _embed_bge([query])
    actual_k = min(k, index.ntotal)
    scores, indices = index.search(embedding, actual_k)

    results = []
    for score, pos in zip(scores[0], indices[0]):
        if pos < 0:
            continue
        row = metadata.iloc[int(pos)]
        results.append({
            "chunk_id": row["chunk_id"], "query_id": row["query_id"],
            "language": row["language"], "is_selected": int(row["is_selected"]),
            "chunk_text": row["chunk_text"], "score": float(score),
            "strategy": strategy_name,
        })
    return results


def _tokens(text: str) -> set:
    # \w+ is unreliable for Bengali/Devanagari: combining vowel signs and
    # virama/hasant characters can fall outside \w's definition, causing
    # \w+ to fragment a single syllable into multiple tiny pieces (or drop
    # it entirely once len(t)>1 filters the fragments out). Splitting on
    # whitespace and stripping surrounding punctuation keeps each real word
    # intact regardless of script.
    text = unicodedata.normalize("NFC", text)
    words = re.split(r"\s+", text.strip())
    tokens = set()
    for w in words:
        w = w.strip(".,!?।\"'()[]{}:;-").lower()
        if len(w) > 1 and w not in STOPWORDS:
            tokens.add(w)
    return tokens


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text.strip().lower())


def _lexical_score(question_tokens: set, text: str):
    if not question_tokens:
        return 0.0, 0
    overlap = question_tokens.intersection(_tokens(text))
    return len(overlap) / len(question_tokens), len(overlap)


def _sentences(text: str) -> list:
    return [p.strip() for p in SENTENCE_SPLIT_RE.split(text) if p.strip()]


def retrieve_with_evidence(question: str, strategy_name: str, k: int = 5, use_bge: bool = False) -> list:
    if use_bge:
        candidates = _retrieve_bge(question, strategy_name, max(k, CANDIDATE_POOL_SIZE))
    else:
        candidates = retrieve_batch([question], strategy_name, k=max(k, CANDIDATE_POOL_SIZE))[0]

    question_tokens = _tokens(question)
    for candidate in candidates:
        lexical_score, matching_terms = _lexical_score(question_tokens, candidate["chunk_text"])
        candidate["lexical_score"] = round(lexical_score, 3)
        candidate["matching_terms"] = matching_terms
        candidate["evidence_score"] = round(0.75 * float(candidate["score"]) + 0.25 * lexical_score, 3)

    candidates.sort(key=lambda item: item["evidence_score"], reverse=True)
    return candidates[:k]


def extract_verified_sentence(question: str, chunks: list, use_bge: bool = False) -> dict:
    question_tokens = _tokens(question)
    question_normalized = _normalize(question)
    best = None

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
    dense_threshold = MIN_DENSE_SCORE_FOR_VERIFICATION_BGE if use_bge else MIN_DENSE_SCORE_FOR_VERIFICATION

    definitional_ok = True
    if len(question_tokens) == 1:
        term = next(iter(question_tokens))
        prefix = _normalize(sentence)[:40]
        definitional_ok = term in prefix

    verified = (
        matches >= MIN_MATCHING_TERMS
        and float(chunk.get("lexical_score", 0.0)) >= MIN_LEXICAL_SCORE
        and dense_score >= dense_threshold
        and definitional_ok
    )
    return {
        "answer": sentence if verified else None,
        "source_chunk_id": chunk.get("chunk_id") if verified else None,
        "confidence": round(score, 3),
        "verified": verified,
    }


def warm_fast_path(strategy_name: str) -> None:
    from rag.retriever import get_model, load_strategy
    get_model()
    if strategy_name != "no_chunk_bge":
        load_strategy(strategy_name)
import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from rag.retriever import load_strategy, retrieve_batch

DENSE_WEIGHT = 0.6
BM25_WEIGHT = 0.4
CANDIDATE_POOL_SIZE = 20  # how many dense results to pull before re-ranking


def _tokenize(text: str) -> list:
    """Simple Unicode-aware word tokenizer, works reasonably for
    Hindi/Bengali/English without needing a language-specific tokenizer."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


@lru_cache(maxsize=8)
def build_bm25_index(strategy_name: str):
    """Build (and cache) a BM25 index over every chunk in a strategy's
    metadata. Built once per strategy, reused across all queries."""
    _, metadata = load_strategy(strategy_name)
    texts = metadata["chunk_text"].tolist()
    tokenized = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    return bm25, metadata


def _normalize(scores: list) -> list:
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def hybrid_retrieve(query: str, strategy_name: str, k: int = 5) -> list:
    """
    Retrieve top-k chunks using combined dense + BM25 scoring.
    """
    dense_results = retrieve_batch([query], strategy_name, k=CANDIDATE_POOL_SIZE)[0]

    if not dense_results:
        return []

    bm25, metadata = build_bm25_index(strategy_name)
    query_tokens = _tokenize(query)
    all_bm25_scores = bm25.get_scores(query_tokens)

    # Look up each dense candidate's BM25 score by its row position in metadata
    chunk_id_to_row = {row["chunk_id"]: i for i, row in metadata.iterrows()}

    dense_scores = [r["score"] for r in dense_results]
    bm25_scores = []
    for r in dense_results:
        row_idx = chunk_id_to_row.get(r["chunk_id"])
        bm25_scores.append(all_bm25_scores[row_idx] if row_idx is not None else 0.0)

    dense_norm = _normalize(dense_scores)
    bm25_norm = _normalize(bm25_scores)

    for r, d, b in zip(dense_results, dense_norm, bm25_norm):
        r["hybrid_score"] = DENSE_WEIGHT * d + BM25_WEIGHT * b
        r["dense_score"] = r["score"]
        r["bm25_score"] = b

    dense_results.sort(key=lambda r: r["hybrid_score"], reverse=True)

    return dense_results[:k]
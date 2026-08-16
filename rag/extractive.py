import re
import numpy as np

from rag.retriever import _embed_queries

SENTENCE_SPLIT_RE = re.compile(r'(?<=[।.!?])\s+')


def _split_sentences(text: str) -> list:
    parts = SENTENCE_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def extract_answer(question: str, chunks: list, top_n_chunks: int = 3) -> dict:
    """
    Given the top retrieved chunks, return the best-matching sentence
    as the answer, with no LLM call.

    Returns: {"answer": str | None, "source_chunk_id": ..., "score": float}
    """
    if not chunks:
        return {"answer": None, "source_chunk_id": None, "score": 0.0}

    candidate_sentences = []
    candidate_meta = []

    for chunk in chunks[:top_n_chunks]:
        for sentence in _split_sentences(chunk.get("chunk_text", "")):
            if len(sentence) < 5:
                continue
            candidate_sentences.append(sentence)
            candidate_meta.append(chunk)

    if not candidate_sentences:
        return {"answer": None, "source_chunk_id": None, "score": 0.0}

    all_texts = [question] + candidate_sentences
    embeddings = _embed_queries(all_texts)

    query_vec = embeddings[0]
    sentence_vecs = embeddings[1:]

    # Vectors are already normalized (normalize_embeddings=True in _embed_queries),
    # so dot product = cosine similarity.
    scores = sentence_vecs @ query_vec

    best_idx = int(np.argmax(scores))

    return {
        "answer": candidate_sentences[best_idx],
        "source_chunk_id": candidate_meta[best_idx].get("chunk_id"),
        "score": float(scores[best_idx]),
    }
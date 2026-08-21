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
    as the answer, with no LLM call. Incorporates a look-ahead sliding window 
    for short heading-like matches.
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

    # Vectors are already normalized, so dot product = cosine similarity.
    scores = sentence_vecs @ query_vec

    best_idx = int(np.argmax(scores))
    best_sentence = candidate_sentences[best_idx]
    best_chunk_id = candidate_meta[best_idx].get("chunk_id")

    # --- SLIDING WINDOW EXPANSION ---
    # If the best sentence is suspiciously short (<= 10 words), it's likely a heading.
    if len(best_sentence.split()) <= 10 and (best_idx + 1) < len(candidate_sentences):
        best_sentence = f"{best_sentence} {candidate_sentences[best_idx + 1]}"
        # Check if there is a next sentence available in the list
        if best_idx + 1 < len(candidate_sentences):
            next_chunk_id = candidate_meta[best_idx + 1].get("chunk_id")
            
            # Crucial: Only append if the next sentence belongs to the exact same source chunk
            if best_chunk_id == next_chunk_id:
                next_sentence = candidate_sentences[best_idx + 1]
                best_sentence = best_sentence + " " + next_sentence
    # --------------------------------

    return {
        "answer": best_sentence,
        "source_chunk_id": best_chunk_id,
        "score": float(scores[best_idx]),
    }
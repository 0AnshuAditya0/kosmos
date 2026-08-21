"""Deterministic, low-latency answer path with a 2-model cascade support.

Tiers 1-2 use the fast MiniLM+ONNX pipeline (rag.retriever). Tier 3 uses
the ONNX INT8 quantized BGE-M3 model for the no_chunk_bge index, cached 
in memory to guarantee sub-100ms CPU execution.
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import onnxruntime as ort
import pandas as pd
from transformers import AutoTokenizer

from rag.retriever import retrieve_batch

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

# Global cache for ONNX Session, Tokenizer, FAISS Index, and Parquet Metadata
_bge_session = None
_bge_tokenizer = None
_BGE_FAISS_INDEX = None
_BGE_META_DF = None
# _BGE_FAISS_INDEX = None
_BGE_META_RECORDS = None

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


def _get_bge_ort():
    """Initializes and returns the ONNX Runtime session and tokenizer."""
    global _bge_session, _bge_tokenizer
    if _bge_session is None:
        model_dir = "bge_m3_onnx"
        model_path = os.path.join(model_dir, "model_quantized.onnx")
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, "model.onnx")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        _bge_session = ort.InferenceSession(
            model_path, 
            sess_options=opts, 
            providers=["CPUExecutionProvider"]
        )
        _bge_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        
    return _bge_session, _bge_tokenizer


def _get_bge_index_and_meta(strategy_name: str):
    """Loads FAISS index and converts Parquet metadata to native Python dicts in RAM."""
    global _BGE_FAISS_INDEX, _BGE_META_RECORDS
    if _BGE_FAISS_INDEX is None or _BGE_META_RECORDS is None:
        import faiss
        import pandas as pd

        # Prevent FAISS from clashing with ONNX thread pool
        faiss.omp_set_num_threads(2)

        index_dir = Path(__file__).resolve().parent.parent / "index"
        faiss_path = index_dir / f"{strategy_name}.faiss"
        meta_path = index_dir / f"{strategy_name}_meta.parquet"

        _BGE_FAISS_INDEX = faiss.read_index(str(faiss_path))
        
        # Convert to list of dicts ONCE at startup to eliminate .iloc overhead
        df = pd.read_parquet(meta_path)
        _BGE_META_RECORDS = df.to_dict("records")

    return _BGE_FAISS_INDEX, _BGE_META_RECORDS


def _embed_bge(queries: list) -> np.ndarray:
    session, tokenizer = _get_bge_ort()

    # Reduced max_length from 64 to 48 for faster transformer attention passes
    inputs = tokenizer(
        queries,
        padding=True,
        truncation=True,
        max_length=48,
        return_tensors="np"
    )

    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }

    outputs = session.run(None, ort_inputs)
    cls_embeddings = outputs[0][:, 0, :]
    
    norms = np.linalg.norm(cls_embeddings, axis=1, keepdims=True)
    normalized_embeddings = cls_embeddings / np.maximum(norms, 1e-12)

    return normalized_embeddings.astype(np.float32)


def _retrieve_bge(query: str, strategy_name: str, k: int) -> list:
    """Retrieves candidates using O(1) native Python memory list access."""
    index, meta_records = _get_bge_index_and_meta(strategy_name)

    embedding = _embed_bge([query])
    actual_k = min(k, index.ntotal)
    scores, indices = index.search(embedding, actual_k)

    results = []
    for score, pos in zip(scores[0], indices[0]):
        if pos < 0 or pos >= len(meta_records):
            continue
        
        # Instantaneous C-level list lookup
        row = meta_records[int(pos)]
        results.append({
            "chunk_id": row["chunk_id"], 
            "query_id": row["query_id"],
            "language": row["language"], 
            "is_selected": int(row["is_selected"]),
            "chunk_text": row["chunk_text"], 
            "score": float(score),
            "strategy": strategy_name,
        })
    return results


def _tokens(text: str) -> set:
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
        s_list = _sentences(chunk.get("chunk_text", ""))
        for idx, sentence in enumerate(s_list):
            if _normalize(sentence) == question_normalized:
                continue
            coverage, matches = _lexical_score(question_tokens, sentence)
            score = coverage + 0.10 * float(chunk["score"])
            candidate = (score, matches, sentence, chunk, idx, s_list)
            if best is None or candidate[0] > best[0]:
                best = candidate

    if best is None:
        return {"answer": None, "source_chunk_id": None, "confidence": 0.0, "verified": False}

    score, matches, sentence, chunk, idx, s_list = best

    # Sliding window expansion for short sentences
    answer_text = sentence
    if len(sentence.split()) <= 10 and (idx + 1) < len(s_list):
        answer_text = f"{sentence} {s_list[idx + 1]}"

    dense_score = float(chunk.get("score", 0.0))
    dense_threshold = MIN_DENSE_SCORE_FOR_VERIFICATION_BGE if use_bge else MIN_DENSE_SCORE_FOR_VERIFICATION

    definitional_ok = True
    if len(question_tokens) == 1:
        term = next(iter(question_tokens))
        prefix = _normalize(answer_text)[:40]
        definitional_ok = term in prefix

    verified = (
        matches >= MIN_MATCHING_TERMS
        and float(chunk.get("lexical_score", 0.0)) >= MIN_LEXICAL_SCORE
        and dense_score >= dense_threshold
        and definitional_ok
    )
    return {
        "answer": answer_text if verified else None,
        "source_chunk_id": chunk.get("chunk_id") if verified else None,
        "confidence": round(score, 3),
        "verified": verified,
    }


def warm_fast_path(strategy_name: str) -> None:
    """Pre-warms models, ONNX sessions, FAISS indices, and metadata into RAM."""
    from rag.retriever import get_model, load_strategy
    if strategy_name == "no_chunk_bge":
        _get_bge_index_and_meta("no_chunk_bge")
        _embed_bge(["warmup query"])
    else:
        get_model()
        load_strategy(strategy_name)
from pathlib import Path
from functools import lru_cache
import logging

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE_DIR / "index"

STRATEGIES = (
    "no_chunk",
    "sentence_aware",
    "fixed_size",
    "fixed_size_overlap",
    "semantic_chunk",
    "no_chunk_bge",
)

logger = logging.getLogger(__name__)

_model = None


def get_model():
    """Load the embedding model once and reuse it."""
    global _model

    if _model is None:
        logger.info("Loading embedding model: %s", MODEL_NAME)
        import torch
        if not torch.cuda.is_available():
            import os
            num_threads = min(4, os.cpu_count() or 1)
            torch.set_num_threads(num_threads)
        _model = SentenceTransformer(MODEL_NAME)
        if torch.cuda.is_available():
            _model = _model.half()
        _model.eval()

    return _model


@lru_cache(maxsize=2)
def load_strategy(strategy_name):
    """Load and cache one FAISS index and its metadata."""

    if strategy_name not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. "
            f"Available: {STRATEGIES}"
        )

    faiss_path = INDEX_DIR / f"{strategy_name}.faiss"
    meta_path = INDEX_DIR / f"{strategy_name}_meta.parquet"

    if not faiss_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {faiss_path}")

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    logger.info("Loading index: %s", strategy_name)

    index = faiss.read_index(str(faiss_path))
    metadata = pd.read_parquet(meta_path)

    if index.ntotal != len(metadata):
        raise RuntimeError(
            f"{strategy_name}: FAISS has {index.ntotal} vectors "
            f"but metadata has {len(metadata)} rows."
        )

    required_columns = {
        "chunk_id",
        "query_id",
        "language",
        "is_selected",
        "chunk_text",
    }

    missing = required_columns - set(metadata.columns)

    if missing:
        raise RuntimeError(
            f"{strategy_name}: missing columns {missing}"
        )

    return index, metadata


_onnx_session = None
_tokenizer = None


def get_onnx_session():
    global _onnx_session, _tokenizer
    if _onnx_session is None:
        import onnxruntime as ort
        from transformers import AutoTokenizer
        import os

        onnx_file = INDEX_DIR / "model.onnx"
        if onnx_file.exists():
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = min(4, os.cpu_count() or 1)
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            logger.info("Loading ONNX embedding model from %s", onnx_file)
            _onnx_session = ort.InferenceSession(str(onnx_file), opts, providers=["CPUExecutionProvider"])
            _tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return _onnx_session, _tokenizer


_query_cache = {}
MAX_CACHE_SIZE = 1024


def _embed_queries(queries):
    """Embed a list of queries using ONNX Runtime with in-memory LRU caching."""
    session, tokenizer = get_onnx_session()
    
    uncached_queries = []
    uncached_indices = []
    results = [None] * len(queries)
    
    for idx, q in enumerate(queries):
        if q in _query_cache:
            results[idx] = _query_cache[q]
        else:
            uncached_queries.append(q)
            uncached_indices.append(idx)
            
    if uncached_queries:
        if session is not None and tokenizer is not None:
            encoded = tokenizer(
                uncached_queries,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="np"
            )
            inputs = {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64)
            }
            computed = session.run(["sentence_embedding"], inputs)[0]
        else:
            model = get_model()
            import torch
            with torch.inference_mode():
                computed = model.encode(
                    uncached_queries,
                    batch_size=64,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
        
        for idx, q, vec in zip(uncached_indices, uncached_queries, computed):
            vec_arr = np.asarray(vec, dtype=np.float32)
            if len(_query_cache) < MAX_CACHE_SIZE:
                _query_cache[q] = vec_arr
            results[idx] = vec_arr

    return np.asarray(results, dtype=np.float32)


def retrieve(query: str, strategy_name: str, k: int = 5):
    """
    Retrieve top-k chunks for one query.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    results = retrieve_batch(
        [query],
        strategy_name,
        k=k
    )

    return results[0]


def retrieve_batch(
    queries,
    strategy_name: str,
    k: int = 5,
):
    """
    Retrieve top-k chunks for multiple queries efficiently.

    Returns:
        list[list[dict]]
    """

    if not queries:
        return []

    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")

    cleaned_queries = []

    for query in queries:
        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "Every query must be a non-empty string."
            )

        cleaned_queries.append(query)

    index, metadata = load_strategy(strategy_name)

    actual_k = min(k, index.ntotal)

    embeddings = _embed_queries(cleaned_queries)

    scores, indices = index.search(
        embeddings,
        actual_k
    )

    all_results = []

    for query_scores, query_indices in zip(
        scores,
        indices
    ):

        query_results = []

        for score, index_position in zip(
            query_scores,
            query_indices
        ):

            if index_position < 0:
                continue

            row = metadata.iloc[int(index_position)]

            query_results.append({
                "chunk_id": row["chunk_id"],
                "query_id": row["query_id"],
                "language": row["language"],
                "is_selected": int(row["is_selected"]),
                "chunk_text": row["chunk_text"],
                "score": float(score),
                "strategy": strategy_name,
            })

        all_results.append(query_results)

    return all_results
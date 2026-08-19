import time
import statistics
import pandas as pd
import numpy as np
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from rag.retriever import get_model, load_strategy, _embed_queries
from rag.fast_path import retrieve_with_evidence

SAMPLE_PATH = "data/sampled_passages.parquet"
N_QUERIES = 50
STRATEGY = "no_chunk"


def percentile(values, pct):
    values = sorted(values)
    idx = int(len(values) * pct / 100)
    idx = min(idx, len(values) - 1)
    return values[idx]


def main():
    print("Warming up (model load + first inference)...")
    get_model()
    load_strategy(STRATEGY)
    _ = retrieve_with_evidence("warmup query", STRATEGY, k=5)

    df = pd.read_parquet(SAMPLE_PATH)
    df = df[df["query_text"].notna()].drop_duplicates(subset=["query_id"])
    test_queries = df["query_text"].sample(n=min(N_QUERIES, len(df)), random_state=42).tolist()

    print(f"\nRan {len(test_queries)} queries\n")

    embed_times = []
    search_times = []
    total_times = []

    index, metadata = load_strategy(STRATEGY)

    for q in test_queries:
        t0 = time.perf_counter()
        
        t_embed_start = time.perf_counter()
        query_vec = _embed_queries([q])
        t_embed = (time.perf_counter() - t_embed_start) * 1000
        
        t_search_start = time.perf_counter()
        scores, indices = index.search(query_vec, 5)
        t_search = (time.perf_counter() - t_search_start) * 1000
        
        t_total = (time.perf_counter() - t0) * 1000

        embed_times.append(t_embed)
        search_times.append(t_search)
        total_times.append(t_total)

    def stats(arr):
        return {
            "avg": statistics.mean(arr),
            "p50": percentile(arr, 50),
            "p95": percentile(arr, 95),
            "p99": percentile(arr, 99),
        }

    embed_s = stats(embed_times)
    search_s = stats(search_times)
    total_s = stats(total_times)

    print(f"{'stage':<12} {'avg':>8} {'p50':>8} {'p95':>8} {'p99':>8}")
    print(f"(ms)")
    print(f"{'embed':<12} {embed_s['avg']:>8.2f} {embed_s['p50']:>8.2f} {embed_s['p95']:>8.2f} {embed_s['p99']:>8.2f}")
    print(f"{'search':<12} {search_s['avg']:>8.2f} {search_s['p50']:>8.2f} {search_s['p95']:>8.2f} {search_s['p99']:>8.2f}")
    print(f"{'total':<12} {total_s['avg']:>8.2f} {total_s['p50']:>8.2f} {total_s['p95']:>8.2f} {total_s['p99']:>8.2f}")

    p95_total = total_s['p95']
    print(f"\nLatency budget: 50.0ms | p95 total: {p95_total:.2f}ms")
    if p95_total <= 50.0:
        print("PASS; within budget")
    elif p95_total <= 100.0:
        print("PASS; within 100ms budget")
    else:
        print("EXCEEDED budget")


if __name__ == "__main__":
    main()

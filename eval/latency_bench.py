"""
latency_bench.py
Run this to produce the required latency analytics for submission:
P50 / P70 / P100 across a real, reasonably-sized set of test queries.

Why we pull queries from data/sampled_passages.parquet instead of making
them up: these are real questions from the dataset with known ground-truth
passages, so this benchmark reflects genuine query patterns, not synthetic
best-case inputs.

We benchmark the TEXT path (run_from_text), not the audio path, so these
numbers isolate retrieval + generation timing without STT's network
variability mixed in. STT latency is reported separately (see note in
output) since it depends entirely on Sarvam's API, not our system.

Usage:
    python -m eval.latency_bench
"""
import time
import statistics
import pandas as pd

from rag.harness import run_from_text
from rag.retriever import get_model  # preload before timing starts

SAMPLE_PATH = "data/sampled_passages.parquet"
N_QUERIES = 40  # smaller, rate-limit-friendly sample
REQUEST_GAP_SECONDS = 1.5  # pause between calls so we don't trigger Groq's
                            # free-tier rate limit, which silently retries
                            # with backoff and massively inflates timings


def percentile(values, pct):
    values = sorted(values)
    idx = int(len(values) * pct / 100)
    idx = min(idx, len(values) - 1)
    return values[idx]


def main():
    print("Preloading embedding model (warm-up, not counted in results)...")
    get_model()

    df = pd.read_parquet(SAMPLE_PATH)
    df = df[df["query_text"].notna()].drop_duplicates(subset=["query_id"])

    test_queries = df["query_text"].sample(n=min(N_QUERIES, len(df)), random_state=42).tolist()
    print(f"Running {len(test_queries)} test queries...\n")

    retrieval_times = []
    generation_times = []
    total_times = []
    statuses = []

    for i, q in enumerate(test_queries, 1):
        result = run_from_text(q)
        t = result.get("timing", {})

        if "retrieval_ms" in t:
            retrieval_times.append(t["retrieval_ms"])
        if "generation_ms" in t:
            generation_times.append(t["generation_ms"])
        total_times.append(t.get("total_ms", 0))
        statuses.append(result["status"])

        if i % 20 == 0:
            print(f"  {i}/{len(test_queries)} done")

        if i < len(test_queries):
            time.sleep(REQUEST_GAP_SECONDS)

    def report(name, values):
        if not values:
            print(f"{name}: no data")
            return
        print(f"\n{name} (n={len(values)})")
        print(f"  P50:  {percentile(values, 50):.1f}ms")
        print(f"  P70:  {percentile(values, 70):.1f}ms")
        print(f"  P100: {percentile(values, 100):.1f}ms")
        print(f"  Mean: {statistics.mean(values):.1f}ms")

    print("\n" + "=" * 60)
    print("LATENCY BENCHMARK RESULTS (text path: retrieval + generation)")
    print("=" * 60)

    report("Retrieval only", retrieval_times)
    report("Generation only", generation_times)
    report("Total (retrieval + generation, no STT)", total_times)

    print("\nStatus breakdown:")
    for s in set(statuses):
        count = statuses.count(s)
        print(f"  {s}: {count} ({count/len(statuses)*100:.1f}%)")

    print("\nNote: STT latency (Sarvam API) is reported separately in the")
    print("README, since it is dominated by network round-trip time to a")
    print("third-party service and is not part of our system's own latency.")

    results_df = pd.DataFrame({
        "query": test_queries,
        "status": statuses,
        "total_ms": total_times,
    })
    results_df.to_csv("eval/results/latency_bench_results.csv", index=False)
    print("\nSaved per-query results to eval/results/latency_bench_results.csv")


if __name__ == "__main__":
    main()
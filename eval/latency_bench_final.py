"""
latency_bench_final.py
Clean final latency benchmark for submission. Warms up ALL THREE tiers
before measuring (MiniLM, sentence_aware index, and BGE-M3), so cold-start
model-loading cost doesn't pollute the real per-query numbers - matches
what a judge would see after the server has been running for a bit.

Usage:
    python -m eval.latency_bench_final
"""
from pathlib import Path
import time
import statistics
import pandas as pd

from rag.harness import run_from_text
from rag.fast_path import warm_fast_path, _get_bge_model

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_PATH = BASE_DIR / "data" / "sampled_passages_all.parquet"
OUTPUT_DIR = BASE_DIR / "eval" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_PER_LANGUAGE = 30


def percentile(values, pct):
    if not values:
        return None
    values = sorted(values)
    idx = min(int(len(values) * pct / 100), len(values) - 1)
    return values[idx]


def warm_everything():
    print("Warming Tier 1 (no_chunk, MiniLM/ONNX)...")
    warm_fast_path("no_chunk")
    run_from_text("test warmup query")

    print("Warming Tier 2 (sentence_aware, MiniLM)...")
    warm_fast_path("sentence_aware")

    print("Warming Tier 3 (BGE-M3)...")
    _get_bge_model()

    print("Warmup complete.\n")


def load_test_queries():
    df = pd.read_parquet(SAMPLE_PATH)
    df = df[df["query_id"].notna() & (df["is_selected"] == 1) & df["query_text"].notna()].copy()
    df = df.drop_duplicates(subset=["query_id", "language"])

    parts = []
    for lang in df["language"].unique():
        lang_df = df[df["language"] == lang]
        n = min(SAMPLE_PER_LANGUAGE, len(lang_df))
        parts.append(lang_df.sample(n=n, random_state=7))
    return pd.concat(parts, ignore_index=True)


def main():
    warm_everything()
    test_df = load_test_queries()
    print(f"Running {len(test_df)} queries (warm state)...\n")

    rows = []
    for i, row in test_df.iterrows():
        result = run_from_text(row["query_text"])
        rows.append({
            "language": row["language"],
            "status": result["status"],
            "tier": result.get("tier"),
            "total_ms": result["timing"]["total_ms"],
        })
        if len(rows) % 20 == 0:
            print(f"  {len(rows)}/{len(test_df)} done")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "latency_final.csv", index=False)

    all_times = df["total_ms"].tolist()
    print("\n" + "=" * 60)
    print("OVERALL LATENCY (all queries, warm state)")
    print("=" * 60)
    print(f"  P50:  {percentile(all_times, 50):.1f}ms")
    print(f"  P70:  {percentile(all_times, 70):.1f}ms")
    print(f"  P100: {percentile(all_times, 100):.1f}ms")
    print(f"  Mean: {statistics.mean(all_times):.1f}ms")

    print("\n" + "=" * 60)
    print("LATENCY BY TIER (which path was taken)")
    print("=" * 60)
    for tier in df["tier"].dropna().unique():
        times = df[df["tier"] == tier]["total_ms"].tolist()
        print(f"\n{tier} (n={len(times)}):")
        print(f"  P50:  {percentile(times, 50):.1f}ms")
        print(f"  P70:  {percentile(times, 70):.1f}ms")
        print(f"  P100: {percentile(times, 100):.1f}ms")

    print("\n" + "=" * 60)
    print("TIER DISTRIBUTION")
    print("=" * 60)
    print(df["tier"].value_counts(dropna=False))

    print(f"\nSaved: {OUTPUT_DIR / 'latency_final.csv'}")


if __name__ == "__main__":
    main()
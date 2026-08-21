from pathlib import Path
import logging
import pandas as pd
from rag.retriever import retrieve_batch

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_PATH = BASE_DIR / "data" / "sampled_passages_all.parquet"
OUTPUT_DIR = BASE_DIR / "eval" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = OUTPUT_DIR / "strategy_comparison_all.csv"
DETAIL_PATH = OUTPUT_DIR / "query_results_all.csv"

STRATEGIES = ["no_chunk_bge"]
TOP_K = 10
BATCH_SIZE = 64

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_test_queries(samples_per_lang: int = 150):
    df = pd.read_parquet(SAMPLE_PATH)
    df = df[df["query_id"].notna() & (df["is_selected"] == 1) & df["query_text"].notna()].copy()
    df = df.drop_duplicates(subset=["query_id", "language"])
    df["query_id"] = df["query_id"].astype(float)
    
    sampled_dfs = []
    for lang in df["language"].unique():
        lang_df = df[df["language"] == lang]
        n_sample = min(samples_per_lang, len(lang_df))
        sampled_dfs.append(lang_df.sample(n=n_sample, random_state=42))
    
    df = pd.concat(sampled_dfs, ignore_index=True)
    logger.info("Evaluation queries: %d (balanced %d/lang)", len(df), samples_per_lang)
    return df


def evaluate_strategy(df, strategy):
    logger.info("Starting strategy: %s", strategy)
    total = len(df)
    hits_at_1 = hits_at_5 = hits_at_10 = 0
    reciprocal_sum = 0.0
    detail_rows = []

    queries = df["query_text"].tolist()
    query_ids = df["query_id"].tolist()
    langs = df["language"].tolist()

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch_queries = queries[start:end]
        batch_ids = query_ids[start:end]
        batch_langs = langs[start:end]

        results_batch = retrieve_batch(batch_queries, strategy, k=TOP_K)

        for query_id, results, lang in zip(batch_ids, results_batch, batch_langs):
            first_relevant_rank = None
            for rank, result in enumerate(results, start=1):
                rqid = result["query_id"]
                if pd.isna(rqid):
                    continue
                if float(rqid) == float(query_id) and result["is_selected"] == 1:
                    first_relevant_rank = rank
                    break

            hit_1 = first_relevant_rank is not None and first_relevant_rank <= 1
            hit_5 = first_relevant_rank is not None and first_relevant_rank <= 5
            hit_10 = first_relevant_rank is not None and first_relevant_rank <= 10

            hits_at_1 += int(hit_1)
            hits_at_5 += int(hit_5)
            hits_at_10 += int(hit_10)
            reciprocal_sum += (1.0 / first_relevant_rank) if first_relevant_rank else 0.0

            detail_rows.append({
                "strategy": strategy, "language": lang, "query_id": query_id,
                "hit_at_1": int(hit_1), "hit_at_5": int(hit_5), "hit_at_10": int(hit_10),
                "first_relevant_rank": first_relevant_rank or 0,
            })

        logger.info("%s: %d/%d queries", strategy, end, total)

    return {
        "strategy": strategy, "queries": total,
        "recall_at_1": hits_at_1 / total, "recall_at_5": hits_at_5 / total,
        "recall_at_10": hits_at_10 / total, "mrr": reciprocal_sum / total,
    }, detail_rows


def main():
    test_df = load_test_queries()
    summaries, all_details = [], []

    for strategy in STRATEGIES:
        summary, details = evaluate_strategy(test_df, strategy)
        summaries.append(summary)
        all_details.extend(details)

    summary_df = pd.DataFrame(summaries).sort_values("recall_at_5", ascending=False)
    detail_df = pd.DataFrame(all_details)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    detail_df.to_csv(DETAIL_PATH, index=False)

    # Also break down recall by language, since that's the new dimension
    # we actually care about this time.
    detail_df["hit_5_bool"] = detail_df["hit_at_5"]
    by_lang = detail_df.groupby(["strategy", "language"])["hit_5_bool"].mean().reset_index()
    by_lang.columns = ["strategy", "language", "recall_at_5"]

    print("\n" + "=" * 75)
    print("OVERALL STRATEGY COMPARISON (3 languages combined)")
    print("=" * 75)
    display_df = summary_df.copy()
    for col in ["recall_at_1", "recall_at_5", "recall_at_10", "mrr"]:
        display_df[col] *= 100
    print(display_df.to_string(index=False, float_format=lambda x: f"{x:.2f}%"))

    print("\n" + "=" * 75)
    print("RECALL@5 BY LANGUAGE")
    print("=" * 75)
    by_lang["recall_at_5"] *= 100
    print(by_lang.to_string(index=False, float_format=lambda x: f"{x:.2f}%"))

    print(f"\nSaved: {SUMMARY_PATH}\nSaved: {DETAIL_PATH}")


if __name__ == "__main__":
    main()
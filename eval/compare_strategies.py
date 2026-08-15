from pathlib import Path
import logging

import pandas as pd

from rag.retriever import retrieve_batch, STRATEGIES


BASE_DIR = Path(__file__).resolve().parent.parent

SAMPLE_PATH = (
    BASE_DIR
    / "data"
    / "sampled_passages.parquet"
)

OUTPUT_DIR = BASE_DIR / "eval" / "results"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SUMMARY_PATH = OUTPUT_DIR / "strategy_comparison.csv"
DETAIL_PATH = OUTPUT_DIR / "query_results.csv"

TOP_K = 10
BATCH_SIZE = 64


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


def load_test_queries():

    df = pd.read_parquet(SAMPLE_PATH)

    required = {
        "query_id",
        "query_text",
        "is_selected",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    df = df[
        df["query_id"].notna()
        & (df["is_selected"] == 1)
        & df["query_text"].notna()
    ].copy()

    df = df.drop_duplicates(
        subset=["query_id"]
    )

    df["query_id"] = df["query_id"].astype(float)

    logger.info(
        "Evaluation queries: %d",
        len(df)
    )

    return df


def evaluate_strategy(df, strategy):

    logger.info(
        "Starting strategy: %s",
        strategy
    )

    total = len(df)

    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0

    reciprocal_sum = 0.0

    detail_rows = []

    queries = df["query_text"].tolist()
    query_ids = df["query_id"].tolist()

    for start in range(
        0,
        total,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            total
        )

        batch_queries = queries[start:end]
        batch_ids = query_ids[start:end]

        results_batch = retrieve_batch(
            batch_queries,
            strategy,
            k=TOP_K
        )

        for query_id, results in zip(
            batch_ids,
            results_batch
        ):

            first_relevant_rank = None

            for rank, result in enumerate(
                results,
                start=1
            ):

                result_query_id = result["query_id"]

                if pd.isna(result_query_id):
                    continue

                if (
                    float(result_query_id) == float(query_id)
                    and result["is_selected"] == 1
                ):
                    first_relevant_rank = rank
                    break

            hit_1 = (
                first_relevant_rank is not None
                and first_relevant_rank <= 1
            )

            hit_5 = (
                first_relevant_rank is not None
                and first_relevant_rank <= 5
            )

            hit_10 = (
                first_relevant_rank is not None
                and first_relevant_rank <= 10
            )

            if hit_1:
                hits_at_1 += 1

            if hit_5:
                hits_at_5 += 1

            if hit_10:
                hits_at_10 += 1

            reciprocal_rank = (
                1.0 / first_relevant_rank
                if first_relevant_rank
                else 0.0
            )

            reciprocal_sum += reciprocal_rank

            detail_rows.append({
                "strategy": strategy,
                "query_id": query_id,
                "hit_at_1": int(hit_1),
                "hit_at_5": int(hit_5),
                "hit_at_10": int(hit_10),
                "first_relevant_rank": (
                    first_relevant_rank
                    if first_relevant_rank
                    else 0
                ),
                "reciprocal_rank": reciprocal_rank,
            })

        processed = end

        logger.info(
            "%s: %d/%d queries",
            strategy,
            processed,
            total
        )

    summary = {
        "strategy": strategy,
        "queries": total,
        "recall_at_1": hits_at_1 / total,
        "recall_at_5": hits_at_5 / total,
        "recall_at_10": hits_at_10 / total,
        "mrr": reciprocal_sum / total,
    }

    return summary, detail_rows


def main():

    test_df = load_test_queries()

    summaries = []
    all_details = []

    for strategy in STRATEGIES:

        summary, details = evaluate_strategy(
            test_df,
            strategy
        )

        summaries.append(summary)
        all_details.extend(details)

    summary_df = pd.DataFrame(
        summaries
    )

    detail_df = pd.DataFrame(
        all_details
    )

    summary_df = summary_df.sort_values(
        "recall_at_5",
        ascending=False
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False
    )

    detail_df.to_csv(
        DETAIL_PATH,
        index=False
    )

    display_df = summary_df.copy()

    for column in [
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "mrr",
    ]:
        display_df[column] *= 100

    print("\n")
    print("=" * 75)
    print("RAG CHUNKING STRATEGY EVALUATION")
    print("=" * 75)

    print(
        display_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}%"
        )
    )

    print("\nFiles saved:")
    print(SUMMARY_PATH)
    print(DETAIL_PATH)


if __name__ == "__main__":
    main()
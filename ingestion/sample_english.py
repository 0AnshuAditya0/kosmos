import pandas as pd

INPUT_PATH = "data/flat_passages_english.parquet"
OUTPUT_PATH = "data/sampled_passages_english.parquet"

NEGATIVES_PER_QUERY = 2
TARGET_TOTAL = 20000


def sample_language(df_lang: pd.DataFrame) -> pd.DataFrame:
    positives = df_lang[df_lang["is_selected"] == 1]
    negatives = df_lang[df_lang["is_selected"] == 0]

    sampled_negatives = (
        negatives.groupby("query_id", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), NEGATIVES_PER_QUERY), random_state=42))
    )

    return pd.concat([positives, sampled_negatives], ignore_index=True)


if __name__ == "__main__":
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} total English rows")

    sampled = sample_language(df)
    print(f"After keeping all positives + capped negatives: {len(sampled)} rows")

    if len(sampled) > TARGET_TOTAL:
        frac = TARGET_TOTAL / len(sampled)
        sampled = sampled.sample(frac=frac, random_state=42)

    sampled = sampled.reset_index(drop=True)
    sampled.to_parquet(OUTPUT_PATH, index=False)

    print(f"\nFinal sampled size: {len(sampled)} rows")
    print(sampled["is_selected"].value_counts())
    print(f"\nSaved to {OUTPUT_PATH}")
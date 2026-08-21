import pandas as pd
from huggingface_hub import hf_hub_download


def load_and_flatten_english(filename: str) -> pd.DataFrame:
    path = hf_hub_download(
        repo_id="ai4bharat/MSMARCO-XI",
        repo_type="dataset",
        filename=filename,
    )
    df = pd.read_parquet(path)

    records = []
    for _, row in df.iterrows():
        passages = row["passages"]
        english = passages["English_passages"]
        selected = passages["is_selected"]

        for i in range(len(english)):
            records.append({
                "query_id": row["query_id"],
                "query_type": row["query_type"],
                "language": "eng",
                "query_text": row["Eng_Query"],
                "answer_text": row["Eng_Answer"],
                "passage_text": english[i],
                "is_selected": int(selected[i]),
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    print("Extracting English from Hindi-file source rows...")
    eng_from_hin = load_and_flatten_english("validation/hinval.parquet")
    print(f"  -> {len(eng_from_hin)} rows")

    print("Extracting English from Bengali-file source rows...")
    eng_from_ben = load_and_flatten_english("validation/benval.parquet")
    print(f"  -> {len(eng_from_ben)} rows")

    combined = pd.concat([eng_from_hin, eng_from_ben], ignore_index=True)
    combined = combined.drop_duplicates(subset=["query_id", "passage_text"])

    combined.to_parquet("data/flat_passages_english.parquet", index=False)
    print(f"\nSaved {len(combined)} unique English passage rows to data/flat_passages_english.parquet")
    print(combined.head())
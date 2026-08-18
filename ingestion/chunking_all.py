import re
import pandas as pd

INPUT_PATH = "data/sampled_passages_all.parquet"
OUTPUT_PATH = "data/chunks_all.parquet"

SENTENCE_SPLIT_RE = re.compile(r'(?<=[।.!?])\s+')


def no_chunk(text: str) -> list:
    return [text]


def sentence_aware_chunk(text: str) -> list:
    sentences = SENTENCE_SPLIT_RE.split(text.strip())
    return [s for s in sentences if s.strip()]


STRATEGIES = {
    "no_chunk": no_chunk,
    "sentence_aware": sentence_aware_chunk,
}


if __name__ == "__main__":
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} sampled passages (hin+ben+eng)")

    all_chunks = []
    chunk_id = 0

    for strategy_name, strategy_fn in STRATEGIES.items():
        print(f"Running strategy: {strategy_name}")
        for _, row in df.iterrows():
            pieces = strategy_fn(row["passage_text"])
            for piece in pieces:
                if not piece.strip():
                    continue
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "query_id": row["query_id"],
                    "language": row["language"],
                    "is_selected": row["is_selected"],
                    "strategy": strategy_name,
                    "chunk_text": piece,
                })
                chunk_id += 1

    result = pd.DataFrame(all_chunks)
    result.to_parquet(OUTPUT_PATH, index=False)

    print(f"\nTotal chunks produced: {len(result)}")
    print(result.groupby(["strategy", "language"]).size())
    print(f"\nSaved to {OUTPUT_PATH}")
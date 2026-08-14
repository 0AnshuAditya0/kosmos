import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INPUT_PATH = "data/chunks.parquet"
INDEX_DIR = "index"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

BATCH_SIZE = 256


def build_index_for_strategy(model, df_strategy: pd.DataFrame, strategy_name: str):
    print(f"\n--- {strategy_name}: {len(df_strategy)} chunks ---")

    texts = df_strategy["chunk_text"].tolist()

    print("Embedding...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss_path = f"{INDEX_DIR}/{strategy_name}.faiss"
    faiss.write_index(index, faiss_path)

    meta_path = f"{INDEX_DIR}/{strategy_name}_meta.parquet"
    df_strategy.reset_index(drop=True).to_parquet(meta_path, index=False)

    print(f"Saved index -> {faiss_path}")
    print(f"Saved metadata -> {meta_path}")


if __name__ == "__main__":
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} total chunks")

    for strategy_name in df["strategy"].unique():
        df_strategy = df[df["strategy"] == strategy_name]
        build_index_for_strategy(model, df_strategy, strategy_name)

    print("\nAll indexes built.")
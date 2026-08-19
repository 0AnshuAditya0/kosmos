import pandas as pd
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

INPUT_PATH = "data/chunks_all.parquet"
INDEX_DIR = "index"
MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 64


def main():
    print(f"CUDA available: {torch.cuda.is_available()}")

    print(f"Loading {MODEL_NAME} in fp16...")
    model = SentenceTransformer(MODEL_NAME)
    if torch.cuda.is_available():
        model = model.half()  

    df = pd.read_parquet(INPUT_PATH)
    df = df[df["strategy"] == "no_chunk"].reset_index(drop=True)
    print(f"Embedding {len(df)} no_chunk passages (hin+ben+eng)...")

    texts = df["chunk_text"].tolist()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype="float32")
    dim = embeddings.shape[1]
    print(f"Embedding dimension: {dim}")

    # Scalar quantization (int8) - ~4x smaller than flat float32, small
    # recall cost, well worth it given our memory constraints.
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings)
    index.add(embeddings)

    faiss.write_index(index, f"{INDEX_DIR}/no_chunk_bge.faiss")
    df.to_parquet(f"{INDEX_DIR}/no_chunk_bge_meta.parquet", index=False)

    print(f"\nSaved index -> {INDEX_DIR}/no_chunk_bge.faiss")
    print(f"Saved metadata -> {INDEX_DIR}/no_chunk_bge_meta.parquet")
    print(f"Index file size check: run 'dir index' to confirm actual size on disk")


if __name__ == "__main__":
    main()
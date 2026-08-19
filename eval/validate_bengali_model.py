import pandas as pd
import numpy as np

QUERY = "আপনি কি পাত্রে লাল পেঁয়াজ জন্মাতে পারেন?" 
TRUE_QUERY_ID = 78076.0

df = pd.read_parquet("data/sampled_passages_all.parquet")
ben_df = df[df["language"] == "ben"]

true_passage_row = ben_df[(ben_df["query_id"] == TRUE_QUERY_ID) & (ben_df["is_selected"] == 1)]
if true_passage_row.empty:
    raise SystemExit("Could not find the true positive passage for this query_id - check the ID.")
true_passage = true_passage_row.iloc[0]["passage_text"]

unrelated_samples = ben_df[
    (ben_df["query_id"] != TRUE_QUERY_ID) & (ben_df["is_selected"] == 0)
].sample(n=8, random_state=1)["passage_text"].tolist()

candidates = [true_passage] + unrelated_samples
labels = ["TRUE MATCH"] + ["unrelated"] * len(unrelated_samples)


def test_model(model_name, model_loader):
    print(f"\n{'=' * 70}\nModel: {model_name}\n{'=' * 70}")
    model = model_loader()
    all_texts = [QUERY] + candidates
    embeddings = model.encode(all_texts, normalize_embeddings=True)
    embeddings = np.asarray(embeddings)
    query_vec = embeddings[0]
    cand_vecs = embeddings[1:]
    scores = cand_vecs @ query_vec

    ranked = sorted(zip(scores, labels, candidates), key=lambda x: -x[0])
    for score, label, text in ranked:
        marker = " <-- TRUE MATCH" if label == "TRUE MATCH" else ""
        print(f"  {score:.4f}  {text[:50]}{marker}")

    true_score = scores[0]
    true_rank = [i for i, (s, l, t) in enumerate(ranked, 1) if l == "TRUE MATCH"][0]
    spread = scores.max() - scores.min()
    print(f"\n  True match rank: {true_rank}/{len(candidates)}")
    print(f"  Score spread (max-min): {spread:.4f}  (higher = more discriminative)")


def load_minilm():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def load_bge_m3():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("BAAI/bge-m3")


if __name__ == "__main__":
    test_model("Current: paraphrase-multilingual-MiniLM-L12-v2", load_minilm)
    test_model("Candidate: BGE-M3", load_bge_m3)
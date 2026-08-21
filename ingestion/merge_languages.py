
import pandas as pd

hin_ben = pd.read_parquet("data/sampled_passages.parquet")
eng = pd.read_parquet("data/sampled_passages_english.parquet")

combined = pd.concat([hin_ben, eng], ignore_index=True)
combined.to_parquet("data/sampled_passages_all.parquet", index=False)

print(f"Hindi+Bengali: {len(hin_ben)} rows")
print(f"English: {len(eng)} rows")
print(f"Combined total: {len(combined)} rows")
print(combined["language"].value_counts())
print("\nSaved to data/sampled_passages_all.parquet")
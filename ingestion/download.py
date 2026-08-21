from huggingface_hub import hf_hub_download

hin_path = hf_hub_download(
    repo_id="ai4bharat/MSMARCO-XI",
    repo_type="dataset",
    filename="validation/hinval.parquet"
)
print("Hindi file saved at:", hin_path)

ben_path = hf_hub_download(
    repo_id="ai4bharat/MSMARCO-XI",
    repo_type="dataset",
    filename="validation/benval.parquet"
)
print("Bengali file saved at:", ben_path)
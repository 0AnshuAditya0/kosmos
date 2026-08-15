from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from rag.harness import run
import numpy as np
import os
from huggingface_hub import hf_hub_download

app = FastAPI()

INDEX_FILES = [
    "no_chunk.faiss", "no_chunk_meta.parquet",
    "fixed_size.faiss", "fixed_size_meta.parquet",
    "fixed_size_overlap.faiss", "fixed_size_overlap_meta.parquet",
    "sentence_aware.faiss", "sentence_aware_meta.parquet",
    "semantic_chunk.faiss", "semantic_chunk_meta.parquet",
]

os.makedirs("index", exist_ok=True)

for filename in INDEX_FILES:
    local_path = f"index/{filename}"
    if not os.path.exists(local_path):
        downloaded = hf_hub_download(
            repo_id="strelizi/kosmos-index",
            repo_type="dataset",
            filename=filename,
        )
        
        import shutil
        shutil.copy(downloaded, local_path)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def clean(obj):
    """Recursively convert numpy/NaN types to plain JSON-safe types."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if np.isnan(val) else val
    if isinstance(obj, float) and obj != obj: 
        return None
    return obj

@app.post("/ask")
async def ask(file: UploadFile):
    audio_bytes = await file.read()
    result = run(audio_bytes)
    return clean(result)
import os
import re
import asyncio
import unicodedata
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from huggingface_hub import hf_hub_download


from rag.harness import run, run_from_text

app = FastAPI()
MAX_AUDIO_BYTES = 12 * 1024 * 1024

# In-memory query cache for sub-2ms repeated query hits
QUERY_CACHE = {}
MAX_CACHE_SIZE = 1000

INDEX_FILES = [
    "no_chunk.faiss", "no_chunk_meta.parquet",
    "sentence_aware.faiss", "sentence_aware_meta.parquet",
    "no_chunk_bge.faiss", "no_chunk_bge_meta.parquet",
]

os.makedirs("index", exist_ok=True)

for filename in INDEX_FILES:
    local_path = f"index/{filename}"
    if not os.path.exists(local_path):
        print(f"Downloading {filename} from HF Hub...")
        downloaded = hf_hub_download(
            repo_id="strelizi/kosmos-index",
            repo_type="dataset",
            filename=filename,
            token=os.getenv("HF_TOKEN"),
        )
        import shutil
        shutil.copy(downloaded, local_path)
        print(f"  -> saved to {local_path}")

allowed_origins = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

def normalize_key(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text.strip().lower())

@app.on_event("startup")
async def startup_event():
    from rag.fast_path import warm_fast_path
    print("Pre-warming Tier 1, 2, and Tier 3 ONNX BGE-M3...")
    warm_fast_path("no_chunk")
    warm_fast_path("sentence_aware")
    warm_fast_path("no_chunk_bge")
    print("All tiers warm in RAM!")

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

@app.get("/health")
async def health():
    return {"status": "ok", "mode": "3-tier-cascade-extractive"}

@app.post("/ask")
async def ask(file: UploadFile):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio file is empty.")
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio is larger than 12MB.")
    result = await run_in_threadpool(run, audio_bytes)
    return clean(result)


class TextQuestion(BaseModel):
    question: str


@app.post("/ask-text")
@app.post("/ask-text-fast")
async def ask_text_fast(payload: TextQuestion):
    result = await run_in_threadpool(run_from_text, payload.question)
    return clean(result)
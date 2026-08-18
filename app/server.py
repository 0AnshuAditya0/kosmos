from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rag.harness import run, run_from_text, run_extractive_from_text
import numpy as np
import os
from huggingface_hub import hf_hub_download, login


app = FastAPI()
MAX_AUDIO_BYTES = 12 * 1024 * 1024
FRONTEND_DIST = "frontend/dist"

INDEX_FILES = [
    "no_chunk.faiss", "no_chunk_meta.parquet",
    "sentence_aware.faiss", "sentence_aware_meta.parquet"
]

os.makedirs("index", exist_ok=True)

for filename in INDEX_FILES:
    local_path = f"index/{filename}"
    if not os.path.exists(local_path):
        try:
            downloaded = hf_hub_download(
                repo_id="strelizi/kosmos-index",
                repo_type="dataset",
                filename=filename,
                token=os.getenv("HF_TOKEN"),
            )
        except Exception as e:
            print(f"failed downloading {filename}: {e}")
            raise
        import shutil
        shutil.copy(downloaded, local_path)

allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def preload_model():
    from rag.fast_path import warm_fast_path
    for strategy in ["no_chunk", "sentence_aware"]:
        warm_fast_path(strategy)


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
    return {"status": "ok", "mode": "deterministic-extractive"}


@app.post("/ask")
async def ask(file: UploadFile):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio file is empty.")
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio is larger than 12 MB.")
    result = await run_in_threadpool(run, audio_bytes)
    return clean(result)


class TextQuestion(BaseModel):
    question: str


@app.post("/ask-text")
async def ask_text(payload: TextQuestion):
    result = await run_in_threadpool(run_from_text, payload.question)
    return clean(result)


@app.post("/ask-text-fast")
async def ask_text_fast(payload: TextQuestion):
    result = await run_in_threadpool(run_extractive_from_text, payload.question)
    return clean(result)


@app.get("/")
async def frontend():
    index_html = os.path.join(FRONTEND_DIST, "index.html")
    if os.path.exists(index_html):
        return FileResponse(index_html)
    return {"status": "ok", "docs": "/docs"}


if os.path.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")

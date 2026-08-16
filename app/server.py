from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag.harness import run, run_from_text, run_extractive_from_text
import numpy as np
import os
from huggingface_hub import hf_hub_download

app = FastAPI()

# Only download what the deployed app actually uses (BEST_STRATEGY = "no_chunk"
# in rag/harness.py). The other 4 strategies were only needed for the offline
# eval/compare_strategies.py comparison, not for serving live requests.
INDEX_FILES = [
    "no_chunk.faiss",
    "no_chunk_meta.parquet",
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


@app.on_event("startup")
async def preload_model():
    """
    Load the embedding model once when the server starts, instead of
    lazily on the first incoming request. Without this, the FIRST request
    after every server restart pays the full model-load cost (10-15s+),
    which is exactly the slow "cold start" we kept seeing in testing.
    Preloading here means every real user request hits a warm model.
    """
    from rag.retriever import get_model
    get_model()


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
    return {"status": "ok"}


@app.post("/ask")
async def ask(file: UploadFile):
    audio_bytes = await file.read()
    result = run(audio_bytes)
    return clean(result)


class TextQuestion(BaseModel):
    question: str


@app.post("/ask-text")
async def ask_text(payload: TextQuestion):
    """
    Text-input fallback path: skips STT entirely, runs the same
    retrieval -> guardrails -> generation flow used by /ask.
    """
    result = run_from_text(payload.question)
    return clean(result)


@app.post("/ask-text-fast")
async def ask_text_fast(payload: TextQuestion):
    """
    LLM-free path: hybrid (dense + BM25) retrieval + extractive sentence
    selection. No Groq call. Much faster and fully deterministic latency,
    at the cost of less fluent (verbatim, not composed) answers.
    """
    result = run_extractive_from_text(payload.question)
    return clean(result)
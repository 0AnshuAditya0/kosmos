import os
import shutil
import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

# --- 1. IDEMPOTENT INDEX DOWNLOAD ---
INDEX_FILES = [
    "no_chunk.faiss", "no_chunk_meta.parquet",
    "sentence_aware.faiss", "sentence_aware_meta.parquet",
    "no_chunk_bge.faiss", "no_chunk_bge_meta.parquet",
]

def ensure_indexes():
    os.makedirs("index", exist_ok=True)
    token = os.getenv("HF_TOKEN")
    for filename in INDEX_FILES:
        local_path = f"index/{filename}"
        if not os.path.exists(local_path):
            print(f"Downloading {filename} from HF Hub...")
            downloaded = hf_hub_download(
                repo_id="strelizi/kosmos-index",
                repo_type="dataset",
                filename=filename,
                token=token,
            )
            shutil.copy(downloaded, local_path)

ensure_indexes()

# --- 2. FASTAPI SETUP ---
fastapi_app = FastAPI()
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextQuestion(BaseModel):
    question: str

@fastapi_app.post("/ask-text-fast")
async def ask_text_fast(payload: TextQuestion):
    from rag.harness import run_from_text
    result = await run_in_threadpool(run_from_text, payload.question)
    return result

# --- 3. GRADIO UI ---
def answer_question(question: str):
    from rag.harness import run_from_text
    res = run_from_text(question)
    
    answer = res.get("answer", "No answer verified.")
    tier = res.get("tier", "N/A")
    timing = res.get("timing", {}).get("retrieval_ms", 0)
    
    status_str = f"**Tier Used:** {tier} | **Retrieval Latency:** {timing:.1f} ms"
    return answer, status_str

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚡ Tiered Cascade RAG")
    
    with gr.Row():
        with gr.Column():
            q_input = gr.Textbox(
                label="Ask a Question", 
                placeholder="who were the founders of the naacp ?",
                lines=2
            )
            submit_btn = gr.Button("Query Pipeline", variant="primary")
        
        with gr.Column():
            a_output = gr.Textbox(label="Extracted Answer", lines=4)
            meta_output = gr.Markdown("Ready...")

    submit_btn.click(
        fn=answer_question, 
        inputs=[q_input], 
        outputs=[a_output, meta_output]
    )

app = gr.mount_gradio_app(fastapi_app, demo, path="/")
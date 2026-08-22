import os
import shutil
import gradio as gr
import spaces
from huggingface_hub import hf_hub_download

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
            print(f"Downloading {filename}...")
            downloaded = hf_hub_download(
                repo_id="strelizi/kosmos-index",
                repo_type="dataset",
                filename=filename,
                token=token,
            )
            shutil.copy(downloaded, local_path)
    print("All indexes ready.")

ensure_indexes()


@spaces.GPU(duration=30)
def answer_question(question: str):
    from rag.harness import run_from_text
    res = run_from_text(question)
    answer = res.get("answer") or "No answer verified — the system declined rather than guess."
    tier = res.get("tier", "N/A")
    timing = res.get("timing", {}).get("retrieval_ms", 0)
    status_str = f"**Tier used:** {tier}  |  **Retrieval latency:** {timing:.1f} ms"
    return answer, status_str


with gr.Blocks(title="Kosmos - Tiered Cascade RAG") as demo:
    gr.Markdown("# Kosmos — Voice/Text RAG (Hindi, Bengali, English)")
    gr.Markdown("3-tier retrieval cascade: fast MiniLM tiers first, BGE-M3 escalation only when needed. No LLM generation - answers are verbatim retrieved evidence.")

    with gr.Row():
        with gr.Column():
            q_input = gr.Textbox(
                label="Ask a question (Hindi, Bengali, or English)",
                placeholder="कॉर्पोरेशन क्या है?",
                lines=2,
            )
            submit_btn = gr.Button("Ask", variant="primary")

        with gr.Column():
            a_output = gr.Textbox(label="Answer", lines=4)
            meta_output = gr.Markdown("Ready...")

    submit_btn.click(fn=answer_question, inputs=[q_input], outputs=[a_output, meta_output])

demo.launch()
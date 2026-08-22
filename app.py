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


def _format_result(res, transcribed_question=None, detected_language=None):
    answer = res.get("answer") or "No answer verified — the system declined rather than guess."
    tier = res.get("tier", "N/A")
    timing = res.get("timing", {})
    retrieval_ms = timing.get("retrieval_ms", 0) or 0
    stt_ms = timing.get("stt_ms", 0) or 0
    total_ms = stt_ms + retrieval_ms

    status_lines = []
    if transcribed_question:
        status_lines.append(f"**Transcribed question:** {transcribed_question}")
    if detected_language:
        status_lines.append(f"**Detected language:** {detected_language}")

    status_lines.append(f"**Tier used:** {tier}")
    status_lines.append(
        f"**STT:** {stt_ms:.1f} ms  +  **Retrieval:** {retrieval_ms:.1f} ms  "
        f"=  **Total:** {total_ms:.1f} ms"
    )

    return answer, "\n\n".join(status_lines)


@spaces.GPU(duration=30)
def answer_from_text(question: str):
    from rag.harness import run_from_text
    res = run_from_text(question)
    return _format_result(res)


@spaces.GPU(duration=30)
def answer_from_audio(audio_path: str):
    if audio_path is None:
        return "Please record or upload audio first.", "No audio received."

    from stt.sarvam_stt import transcribe
    from rag.harness import run_from_text

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    transcription = transcribe(audio_bytes)
    question = transcription["text"]
    language = transcription.get("language")

    res = run_from_text(question)
    return _format_result(res, transcribed_question=question, detected_language=language)


with gr.Blocks(title="Kosmos - Voice RAG") as demo:
    gr.Markdown("# Kosmos — Voice-Enabled RAG (Hindi, Bengali, English)")
    gr.Markdown(
        "Speak (or type) a question. Pipeline: Sarvam speech-to-text → "
        "3-tier retrieval cascade (fast MiniLM tiers first, BGE-M3 escalation "
        "only when needed) → verbatim grounded answer. No LLM generation."
    )

    with gr.Tab("Voice input"):
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Record or upload your question",
                )
                voice_btn = gr.Button("Ask (voice)", variant="primary")
            with gr.Column():
                voice_answer = gr.Textbox(label="Answer", lines=4)
                voice_meta = gr.Markdown("Ready...")

        voice_btn.click(fn=answer_from_audio, inputs=[audio_input], outputs=[voice_answer, voice_meta])

    with gr.Tab("Text input (fallback)"):
        with gr.Row():
            with gr.Column():
                q_input = gr.Textbox(
                    label="Ask a question (Hindi, Bengali, or English)",
                    placeholder="कॉर्पोरेशन क्या है?",
                    lines=2,
                )
                text_btn = gr.Button("Ask (text)", variant="secondary")
            with gr.Column():
                text_answer = gr.Textbox(label="Answer", lines=4)
                text_meta = gr.Markdown("Ready...")

        text_btn.click(fn=answer_from_text, inputs=[q_input], outputs=[text_answer, text_meta])

demo.launch()
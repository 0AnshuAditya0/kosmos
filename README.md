
# hhgoa2 — Voice-Enabled RAG (HH Goa 2026, Task 2)

Voice question → Sarvam STT → chunked+indexed retrieval (FAISS) over MSMARCO-XI
(Hindi + English subset) → grounded answer generation, wrapped in a harness
with guardrails.

## Status
Planning/skeleton stage. No components implemented yet.

## Team
- Ingestion + Retrieval Lead: TBD — sampling, chunking strategies, FAISS index
- STT + Harness Lead: TBD — Sarvam integration, orchestration layer
- Guardrails + Generation + Deploy Lead: TBD — generation, guardrails, HF Spaces, latency eval

## Pipeline shape
Voice input -> Speech-to-text (Sarvam) -> Chunking/Retrieval (FAISS) -> Answer generation (LLM)
All stages wrapped by rag/harness.py (retries, structured I/O, error handling).

## Folder map
- `data/`        streamed dataset subset cache (Hindi + English, never the full 55.6GB)
- `ingestion/`   sampling from HF (streaming) + multiple chunking strategies
- `index/`       embedding + FAISS index build scripts, persisted index files
- `stt/`         Sarvam speech-to-text wrapper
- `rag/`         retriever, generator, harness (orchestration)
- `guardrails/`  off-topic / unsafe-input / groundedness checks
- `eval/`        latency benchmarking (P50/P70/P100), retrieval quality checks
- `app/`         frontend + API entrypoint for the live deployed demo (HF Spaces)

## Decisions locked so far
- STT: Sarvam (Indic-first, matches dataset)
- Vector DB: FAISS, local/in-process (free, system-design-friendly, no hosted dependency)
- Languages: Hindi + English
- Dataset access: streaming only (`load_dataset(..., streaming=True)` + skip/take),
  never downloading the full 55.6GB
- Deployment target: Hugging Face Spaces (free, hosts data + app in one place)

## Open decisions
- Embedding model (must handle Hindi + English well)
- LLM provider for generation
- Exact subset size per language
- Chunking strategies to implement and compare
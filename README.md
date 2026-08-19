# Kosmos — verified multilingual voice retrieval

Kosmos answers English, Hindi, and Bengali questions from a local, memory-optimized FAISS index.
The live answer path is deterministic and **LLM-free**: it retrieves candidates using `BAAI/bge-m3`,
requires direct lexical evidence verification, and returns an exact source sentence.
When evidence is insufficient, it declines (`off_topic`) instead of hallucinating.

## Architecture

`voice audio -> Sarvam STT -> Compiled ONNX Embedding -> FAISS Vector Search -> Lexical Evidence Verification -> Source Sentence`

- **Text path**: Zero LLM generation latency, direct deterministic neural extraction.
- **Compiled Engine**: ONNX Runtime with operator fusion and intra-op thread tuning.
- **Query Cache**: In-memory LRU vector cache for instant repeated lookups (< 1ms).
- **Index**: Flat / Quantized FAISS index (`no_chunk.faiss`) covering all passages.

## Latency Benchmark Results (Strict Production Budget)

Measured across 50 test queries:

| Stage | Avg (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Embedding (ONNX)** | 24.41 ms | **22.39 ms** | 40.77 ms | 45.49 ms | PASS |
| **Vector Search (FAISS)** | 23.64 ms | **22.83 ms** | 27.37 ms | 30.68 ms | PASS |
| **Total Pipeline** | 48.06 ms | **45.62 ms** | **67.67 ms** | **71.45 ms** | **PASS (< 100ms)** |

## Run Locally

1. Install dependencies:
   ```powershell
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Configure environment:
   Copy `.env.example` to `.env` and configure `SARVAM_API_KEY` for voice STT.

3. Start backend API:
   ```powershell
   uvicorn app.server:app --reload --port 8000
   ```

4. Start frontend:
   ```powershell
   cd frontend
   npm install
   npm run dev
   ```

## Production Docker Deployment

```powershell
docker build -t kosmos .
docker run --rm -p 7860:7860 --env-file .env kosmos
```

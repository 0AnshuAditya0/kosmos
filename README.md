# Kosmos — verified voice retrieval

Kosmos answers Hindi and Bengali questions from a local, in-memory FAISS index.
The live answer path is deliberately **LLM-free**: it retrieves candidates once,
requires direct lexical evidence, and returns a verbatim sentence from a source.
When that evidence is missing, it declines instead of inventing an answer.

## Architecture

`voice audio -> Sarvam STT -> local embedding -> FAISS -> evidence check -> source sentence`

Text requests skip STT. There is no Groq call on either live route.

The index contains 40,000 passages (20,000 Hindi and 20,000 Bengali). The model
and the serving index load at startup; the first request should therefore not pay
that cold-start cost.

## Run locally

1. Create a virtual environment and install packages:

   ```powershell
   py -3.11 -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env`, then add a fresh Sarvam key only if you want
   to test voice input. Text testing does not need any API key.

3. Start the API (the initial startup loads the local embedding model and index):

   ```powershell
   uvicorn app.server:app --reload --port 8000
   ```

4. In a second terminal, start the UI:

   ```powershell
   cd frontend
   npm install
   npm run dev
   ```

   Open the Vite URL shown in that terminal. Its development proxy forwards
   API requests to `http://localhost:8000`.

## Test the API

Use the text path first; it is the latency path being optimized:

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/ask-text-fast `
  -ContentType 'application/json' `
  -Body '{"question":"टेस्ला कॉइल क्या है?"}'
```

An unsupported question should return `off_topic`, not an unrelated answer.
Try a known dataset-style question too:

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/ask-text-fast `
  -ContentType 'application/json' `
  -Body '{"question":"स्पाइरुलिना में पोटेशियम कितना होता है"}'
```

Check readiness with `http://localhost:8000/health` and interactive API docs at
`http://localhost:8000/docs`.

## Production Docker deployment

The Docker image builds the React frontend and serves it from FastAPI, so one
container exposes both UI and API on port 7860.

```powershell
docker build -t kosmos .
docker run --rm -p 7860:7860 --env-file .env kosmos
```

Open `http://localhost:7860`. Do not put `.env` in Git or in a public Space;
configure `SARVAM_API_KEY` as a deployment secret.

## Important latency note

The post-transcript text route avoids LLM/network generation and is the route to
benchmark for the retrieval target. Full voice-to-answer time still includes the
user's recording duration and Sarvam's external STT latency, so it cannot
honestly be reported as a sub-200 ms end-to-end interaction.

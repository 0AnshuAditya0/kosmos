import { useState, useRef } from "react";
import "./App.css";

const API_URL = "http://localhost:8000/ask";

const STATUS_LABELS = {
  success: "Answered",
  ungrounded: "Declined — insufficient grounding",
  off_topic: "Declined — off topic",
  unsafe_input: "Declined — unsafe input",
  error: "Error",
};

const STAGES = ["Voice Input", "Transcription", "Search", "Verified Answer"];

export default function App() {
  const [recording, setRecording] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (e) => chunksRef.current.push(e.data);
      recorder.onstop = handleStop;
      recorder.start();
      mediaRecorderRef.current = recorder;
      setRecording(true);
    } catch (err) {
      setResult({ status: "error", error: "Microphone access denied or unavailable." });
    }
  }

  function stopRecording() {
    mediaRecorderRef.current.stop();
    mediaRecorderRef.current.stream.getTracks().forEach((t) => t.stop());
    setRecording(false);
  }

  async function handleStop() {
    const blob = new Blob(chunksRef.current, { type: "audio/webm" });
    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("file", blob, "question.webm");

    try {
      const res = await fetch(API_URL, { method: "POST", body: formData });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();
      setResult(data);
      setHistory((h) => [data, ...h].slice(0, 10));
    } catch (err) {
      setResult({ status: "error", error: String(err.message || err) });
    } finally {
      setLoading(false);
    }
  }

  const currentStage = loading
    ? 1
    : result
      ? (result.status === "success" ? 4 : result.status === "error" ? 0 : 3)
      : 0;

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-dot" />
          Kosmos
        </div>
        <nav>
          <a className="active">Ask</a>
          <a>History</a>
          <a>Guardrails</a>
          <a>Settings</a>
        </nav>
      </aside>

      <main className="main">
        <div className="stage-track">
          {STAGES.map((s, i) => (
            <div key={s} className={`stage ${i < currentStage ? "done" : ""} ${i === currentStage ? "active" : ""}`}>
              <span className="stage-dot" />
              {s}
              {i < STAGES.length - 1 && <span className="stage-line" />}
            </div>
          ))}
        </div>

        <div className="mic-zone">
          <button
            className={`mic-btn ${recording ? "recording" : ""} ${loading ? "pulsing" : ""}`}
            onClick={recording ? stopRecording : startRecording}
            disabled={loading}
          >
            🎙
          </button>
          <p className="mic-hint">
            {recording ? "Listening — tap to stop" : loading ? "Thinking..." : "Tap to ask a question"}
          </p>
        </div>

        {result && result.status !== "error" && (
          <>
            <div className="transcript-block">
              <p className="q-text">{result.question}</p>
            </div>

            <div className="answer-card">
              <div className="answer-card-head">
                <span className="answer-label">Answer</span>
                <span className={`status-pill status-${result.status}`}>
                  {STATUS_LABELS[result.status] || result.status}
                </span>
              </div>

              <p className="answer-body">
                {result.answer || "No answer — the system declined rather than guess."}
              </p>

              {result.timing && (
                <div className="answer-footer">
                  <span className="latency-pill">Total {result.timing.total_ms}ms</span>
                  <span className="mini-timing">
                    STT {result.timing.stt_ms}ms · Retrieval {result.timing.retrieval_ms}ms · Gen {result.timing.generation_ms}ms
                  </span>
                </div>
              )}
            </div>
          </>
        )}

        {result && result.status === "error" && (
          <div className="answer-card error">
            <p>{result.error}</p>
          </div>
        )}

        {history.length > 1 && (
          <div className="history">
            <h4>Previous</h4>
            {history.slice(1).map((h, i) => (
              <div key={i} className="history-row">
                <span className={`mini-badge status-${h.status}`} />
                <span>{h.question}</span>
              </div>
            ))}
          </div>
        )}
      </main>

      <aside className="sources-panel">
        <h3>Grounded sources {result?.sources?.length ? `(${result.sources.length})` : ""}</h3>
        {!result?.sources?.length && <p className="empty-hint">Sources will appear here once you ask a question.</p>}
        {result?.sources?.map((s, i) => (
          <div key={i} className="source-card">
            <div className="source-top">
              <span>{i + 1}.</span>
              <span className="source-lang">{s.language}</span>
              {s.is_selected === 1 && <span className="gold-badge">ground truth</span>}
            </div>
            <p className="source-score">score {s.score?.toFixed(3)}</p>
            <p className="source-text">{s.chunk_text}</p>
          </div>
        ))}
      </aside>
    </div>
  );
}
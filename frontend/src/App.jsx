import { useState, useRef } from "react";
import "./App.css";

const API_URL = "http://localhost:8000/ask";

const STATUS_LABELS = {
  success: "Grounded answer",
  ungrounded: "Low confidence — insufficient grounding",
  off_topic: "No context found — declined",
  unsafe_input: "Unsafe query — declined",
  error: "Error / retry",
};

const SAMPLE_QUESTIONS = [
  { text: "What is a Tesla coil?", tag: "Physics" },
  { text: "Summarize the key ideas of retrieval-augmented generation.", tag: "AI systems" },
  { text: "टेस्ला कॉइल क्या है?", tag: "Hindi" },
  { text: "How can I find the key summaries of prominent books?", tag: "General" },
];

// Static examples for the "preview a response state" chips on the landing
// page - these are NOT live queries, just illustrate what each guardrail
// outcome looks like before the user has asked anything for real.
const DEMO_STATES = {
  success: {
    status: "success",
    question: "What is a Tesla coil?",
    answer: "Tesla coils are used to produce spectacular high-voltage long sparking displays. Voltage can exceed 1,000,000 volts and is discharged as an electrical arc.",
    timing: { stt_ms: 820, retrieval_ms: 105, generation_ms: 512, total_ms: 1437 },
    sources: [{ language: "hin", score: 0.81, is_selected: 1, chunk_text: "टेस्ला कॉइल का उपयोग शानदार उच्च वोल्टेज..." }],
  },
  ungrounded: {
    status: "ungrounded",
    question: "Yeah probably, I guess so",
    answer: null,
    timing: { stt_ms: 640, retrieval_ms: 98, generation_ms: 430, total_ms: 1168 },
    sources: [{ language: "hin", score: 0.41, is_selected: 0, chunk_text: "अप्रासंगिक पाठ खंड..." }],
  },
  off_topic: {
    status: "off_topic",
    question: "What's your favorite color?",
    answer: null,
    timing: { stt_ms: 700, retrieval_ms: 88, generation_ms: 0, total_ms: 788 },
    sources: [],
  },
  unsafe_input: {
    status: "unsafe_input",
    question: "[flagged input]",
    answer: null,
    timing: { stt_ms: 610, retrieval_ms: 0, generation_ms: 0, total_ms: 610 },
    sources: [],
  },
  error: {
    status: "error",
    error: "Upstream service timeout - retried twice, then failed.",
  },
};

const STAGES = ["Voice Input", "Transcription", "Search", "Verified Answer"];

export default function App() {
  const [recording, setRecording] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [textInput, setTextInput] = useState("");
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (e) => chunksRef.current.push(e.data);
      recorder.onstop = () => handleSubmitAudio();
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

  async function handleSubmitAudio() {
    const blob = new Blob(chunksRef.current, { type: "audio/webm" });
    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("file", blob, "question.webm");

    await submit(formData);
  }
  async function handleSubmitText(question) {
    if (!question || !question.trim()) return;
    setLoading(true);
    setResult(null);
    setTextInput("");

    try {
      const res = await fetch(API_URL.replace("/ask", "/ask-text"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
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

  async function submit(formData) {
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

  const showLanding = !result && !loading;

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-icon">👁</span>
          Kosmos
        </div>
        <nav className="top-nav">
          <span className="active">Voice RAG</span>
          <span>Sources</span>
          <span>Performance</span>
          <span>About</span>
        </nav>
        <div className="status-chip">
          <span className="status-dot" />
          System Ready
        </div>
      </header>

      <main className="main">
        {showLanding && (
          <div className="landing">
            <div className="landing-icon">👁</div>
            <h1>Ask anything</h1>
            <p className="landing-sub">
              Speak naturally. Kosmos retrieves grounded information and answers with evidence.
            </p>

            <div className="sample-grid">
              {SAMPLE_QUESTIONS.map((q, i) => (
                <button key={i} className="sample-card" onClick={() => handleSubmitText(q.text)}>
                  <span className="sample-text">{q.text}</span>
                  <span className="sample-tag">{q.tag}</span>
                </button>
              ))}
            </div>

            <div className="preview-row">
              <p className="preview-label">PREVIEW A RESPONSE STATE</p>
              <div className="preview-chips">
                <button className="preview-chip" onClick={() => setResult(DEMO_STATES.success)}>Grounded answer</button>
                <button className="preview-chip" onClick={() => setResult(DEMO_STATES.ungrounded)}>Low confidence</button>
                <button className="preview-chip" onClick={() => setResult(DEMO_STATES.off_topic)}>No context</button>
                <button className="preview-chip" onClick={() => setResult(DEMO_STATES.unsafe_input)}>Unsafe query</button>
                <button className="preview-chip" onClick={() => setResult(DEMO_STATES.error)}>Error / retry</button>
              </div>
            </div>
          </div>
        )}

        {!showLanding && (
          <>
            <div className="stage-track">
              {STAGES.map((s, i) => (
                <div key={s} className={`stage ${i < currentStage ? "done" : ""} ${i === currentStage ? "active" : ""}`}>
                  <span className="stage-dot" />
                  {s}
                  {i < STAGES.length - 1 && <span className="stage-line" />}
                </div>
              ))}
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
                        STT {result.timing.stt_ms ?? "-"}ms · Retrieval {result.timing.retrieval_ms ?? "-"}ms · Gen {result.timing.generation_ms ?? "-"}ms
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

            {result?.sources?.length > 0 && (
              <div className="sources-inline">
                <h3>Grounded sources ({result.sources.length})</h3>
                {result.sources.map((s, i) => (
                  <div key={i} className="source-card">
                    <div className="source-top">
                      <span>{i + 1}.</span>
                      <span className="source-lang">{s.language}</span>
                      {s.is_selected === 1 && <span className="gold-badge">ground truth</span>}
                      <span className="source-score">score {s.score?.toFixed(3)}</span>
                    </div>
                    <p className="source-text">{s.chunk_text}</p>
                  </div>
                ))}
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
          </>
        )}
      </main>

      <div className="input-bar">
        <button
          className={`mic-pill ${recording ? "recording" : ""}`}
          onClick={recording ? stopRecording : startRecording}
          disabled={loading}
        >
          🎙
        </button>
        <input
          type="text"
          placeholder="Ask a question..."
          value={textInput}
          onChange={(e) => setTextInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSubmitText(textInput)}
          disabled={loading || recording}
        />
        <button
          className="send-btn"
          onClick={() => handleSubmitText(textInput)}
          disabled={loading || recording || !textInput.trim()}
        >
          ↑
        </button>
      </div>
      <p className="input-hint">
        {recording ? "Listening — tap mic to stop" : loading ? "Processing..." : "Tap to speak · text input available as fallback"}
      </p>

      {showLanding && (
        <details className="tech-pipeline">
          <summary>
            Technical pipeline
            <span>⌄</span>
          </summary>
          <div className="tech-pipeline-body">
            Voice input → Sarvam speech-to-text → FAISS retrieval (no_chunk
            strategy, 40k indexed passages, Hindi + Bengali) → guardrail
            checks (unsafe input, off-topic, groundedness) → Groq-generated
            grounded answer. Full pipeline runs through a retry-enabled
            harness with per-stage latency tracking.
          </div>
        </details>
      )}
    </div>
  );
}
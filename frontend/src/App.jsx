import { useRef, useState } from "react";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "/ask";
const QUICK_CHECKS = [
  { question: "क्या आप पात्र में लाल प्याज उगा सकते हैं?", kind: "grounded", label: "Verified Hindi · Gardening" },
  { question: "कैटडॉग क्या है?", kind: "grounded", label: "Verified Hindi · Culture" },
  { question: "पुट्ट नौका क्या है", kind: "grounded", label: "Verified Hindi · Definition" },
  { question: "टेस्ला कॉइल क्या है?", kind: "decline", label: "Expected decline · absent from index" },
];

const NAV = ["Voice RAG", "Sources", "Performance", "About"];

function Status({ result }) {
  if (!result) return null;
  const declined = result.status !== "success";
  return <span className={`status ${declined ? "declined" : "grounded"}`}>
    {declined ? "No verified context — declined" : "Grounded answer"}
  </span>;
}

function SourceList({ sources }) {
  const [open, setOpen] = useState(false);
  if (!sources?.length) return null;
  return <section className="context">
    <button className="context-toggle" onClick={() => setOpen(!open)}>
      <span>▱ &nbsp; Retrieved context <b>{sources.length}</b></span><span>{open ? "⌃" : "⌄"}</span>
    </button>
    {open && <div className="source-list">
      {sources.map((source, index) => <article className="source" key={`${source.chunk_id}-${index}`}>
        <div><span className="source-number">{index + 1}</span><span>{source.language === "ben" ? "Bengali" : "Hindi"}</span><span className="evidence">Evidence {Math.round((source.lexical_score || 0) * 100)}%</span></div>
        <p>{source.chunk_text}</p>
      </article>)}
    </div>}
  </section>;
}

function VoicePage({ result, loading, submitText, recording, onMic, input, setInput }) {
  const isLanding = !result && !loading;
  return <main className="page voice-page">
    {isLanding ? <section className="hero">
      <div className="eye">◉</div>
      <p className="eyebrow">VOICE-ENABLED, SOURCE-FIRST RETRIEVAL</p>
      <h1>Ask with confidence.</h1>
      <p className="lede">Kosmos returns a source sentence when it finds direct evidence—and declines when it does not.</p>
      <div className="reviewer-box">
        <div><b>Reviewer quick checks</b><span>Real indexed questions plus a deliberate refusal test</span></div>
        <div className="check-grid">{QUICK_CHECKS.map((item) => <button key={item.question} className={`quick-check ${item.kind}`} onClick={() => submitText(item.question)}>
          <span className="check-state">{item.kind === "grounded" ? "✓ Expected answer" : "× Expected decline"}</span><strong>{item.question}</strong><small>{item.label}</small>
        </button>)}</div>
      </div>
    </section> : <section className="result-wrap">
      <div className="steps"><span className="done">● Voice input</span><i/><span className="done">● Transcription</span><i/><span className="done">● Search</span><i/><span className={result ? "active" : ""}>● Verified answer</span></div>
      {loading ? <div className="loading-card"><div className="pulse"/><b>Checking indexed evidence…</b><p>No LLM generation is running.</p></div> : <>
        <div className="query-meta"><span>{result.question}</span><small>{result.language || "Text"} · {result.timing?.stt_ms ? "Voice" : "Typed"}</small></div>
        <article className="answer-card"><header><span className="answer-label">ANSWER</span><Status result={result}/></header>
          <p className="answer">{result.answer || "No answer — the system declined rather than guess."}</p>
          <footer><b>Total {result.timing?.total_ms ?? "–"}ms</b><span>STT {result.timing?.stt_ms ?? 0}ms · Retrieval {result.timing?.retrieval_ms ?? "–"}ms · Generation 0ms</span><em>Verbatim source answer</em></footer>
        </article>
        <SourceList sources={result.sources}/>
      </>}
    </section>}
    <InputBar recording={recording} onMic={onMic} input={input} setInput={setInput} submitText={submitText} loading={loading}/>
  </main>;
}

function InputBar({ recording, onMic, input, setInput, submitText, loading }) {
  return <div className="input-area"><div className="input-bar"><button className={`mic ${recording ? "recording" : ""}`} onClick={onMic} disabled={loading}>🎙</button><input value={input} disabled={loading || recording} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && submitText(input)} placeholder="Ask a question…"/><button className="send" disabled={!input.trim() || loading || recording} onClick={() => submitText(input)}>↑</button></div><small>{recording ? "Listening — tap again to submit" : "Tap to speak · text input available as fallback"}</small></div>;
}

function InfoPage({ page }) {
  const content = {
    Sources: <><h1>Sources</h1><p className="lede left">Kosmos searches a local multilingual passage index. Answers are never composed beyond the retrieved source sentence.</p><div className="metrics"><Metric label="Indexed passages" value="40,000"/><Metric label="Languages" value="Hindi + Bengali"/><Metric label="Vector index" value="FAISS"/></div><div className="info-card"><b>Evidence rule</b><p>A passage must contain direct content-term overlap with the user’s question. Semantic similarity by itself is not enough to produce an answer.</p></div></>,
    Performance: <><span className="target">ϟ POST-TRANSCRIPT TARGET</span><h1>Performance</h1><p className="lede left">The serving route has no Groq call and reports stage timing on every response.</p><div className="metrics"><Metric label="Generation" value="0 ms"/><Metric label="Serving model" value="In memory"/><Metric label="Quality policy" value="Decline > guess"/></div><div className="info-card"><b>Honest measurement</b><p>The 200 ms objective applies after text is available. Voice-to-answer includes recording time and Sarvam network STT, which is displayed separately rather than hidden.</p></div></>,
    About: <><span className="target">HH GOA 2026</span><h1>A voice retrieval assistant that answers with evidence.</h1><p className="lede left">Kosmos turns Hindi and Bengali questions into traceable, source-first answers with a deterministic refusal path.</p><div className="features"><Metric label="Voice-first" value="Sarvam STT"/><Metric label="Multilingual" value="Hindi + Bengali"/><Metric label="Grounded" value="Source sentence"/><Metric label="Fast path" value="No LLM"/></div></>,
  };
  return <main className="page info-page">{content[page]}</main>;
}
function Metric({ label, value }) { return <div className="metric"><small>{label}</small><strong>{value}</strong></div>; }

export default function App() {
  const [page, setPage] = useState("Voice RAG"), [result, setResult] = useState(null), [loading, setLoading] = useState(false), [recording, setRecording] = useState(false), [input, setInput] = useState("");
  const recorder = useRef(null), chunks = useRef([]);
  async function submitText(question) { if (!question?.trim()) return; setLoading(true); setResult(null); setInput(""); try { const response = await fetch(`${API_URL}-text-fast`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question }) }); setResult(await response.json()); } catch (error) { setResult({ status: "error", question, error: String(error), timing: {} }); } finally { setLoading(false); } }
  async function submitAudio() { const data = new FormData(); data.append("file", new Blob(chunks.current, { type: "audio/webm" }), "question.webm"); setLoading(true); setResult(null); try { const response = await fetch(API_URL, { method: "POST", body: data }); setResult(await response.json()); } catch (error) { setResult({ status: "error", error: String(error), timing: {} }); } finally { setLoading(false); } }
  async function onMic() { if (recording) { recorder.current.stop(); recorder.current.stream.getTracks().forEach((track) => track.stop()); setRecording(false); return; } try { const stream = await navigator.mediaDevices.getUserMedia({ audio: true }); const current = new MediaRecorder(stream); chunks.current = []; current.ondataavailable = (event) => chunks.current.push(event.data); current.onstop = submitAudio; current.start(); recorder.current = current; setRecording(true); } catch { setResult({ status: "error", error: "Microphone access was denied.", timing: {} }); } }
  return <div className="shell"><header><button className="brand" onClick={() => setPage("Voice RAG")}>◉ <b>Kosmos</b></button><nav>{NAV.map((item) => <button className={page === item ? "selected" : ""} key={item} onClick={() => setPage(item)}>{item}</button>)}</nav><span className="ready">● System ready</span></header>{page === "Voice RAG" ? <VoicePage {...{result, loading, submitText, recording, onMic, input, setInput}}/> : <InfoPage page={page}/>}</div>;
}

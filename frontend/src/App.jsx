import { useRef, useState } from "react";
import "./App.css";
import { NAV, askView, landingView } from "./data/kosmos";
import { Header } from "./components/Brand";
import LandingPage from "./pages/LandingPage";
import AskPage from "./pages/AskPage";
import InfoPage from "./pages/InfoPage";
import { askText, askAudio } from "./lib/api";

export default function App() {
  const [view, setView] = useState(landingView);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [recording, setRecording] = useState(false);
  const [input, setInput] = useState("");
  const recorder = useRef(null);
  const chunks = useRef([]);

  async function submitText(question) {
    if (!question?.trim()) return;
    setLoading(true); setResult(null); setInput("");
    setResult(await askText(question)); setLoading(false);
  }
  async function submitAudio() {
    setLoading(true); setResult(null);
    setResult(await askAudio(new Blob(chunks.current, { type: "audio/webm" }))); setLoading(false);
  }
  async function onMic() {
    if (recording) { recorder.current.stop(); recorder.current.stream.getTracks().forEach((track) => track.stop()); setRecording(false); return; }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const current = new MediaRecorder(stream); chunks.current = [];
      current.ondataavailable = (event) => chunks.current.push(event.data);
      current.onstop = submitAudio; current.start(); recorder.current = current; setRecording(true);
    } catch { setResult({ status: "error", answer: "Microphone access was denied.", timing: {} }); }
  }
  if (view === landingView) return <LandingPage setView={setView} nav={NAV} />;
  return <div className="app-shell"><Header view={view} setView={setView} NAV={NAV} /><div className="app-body">{view === askView ? <AskPage {...{ result, loading, submitText, recording, onMic, input, setInput }} /> : <InfoPage view={view} />}</div></div>;
}

import { API_URL as KOSMOS_API_URL, SHOWCASE_RESULT, isShowcase, createError, createAudioError } from "../data/kosmos";

// Prefers VITE_API_URL if set, falls back to kosmos.js config, or defaults to local FastAPI (http://127.0.0.1:8000/ask)
const TARGET_API = import.meta.env.VITE_API_URL || KOSMOS_API_URL || "http://127.0.0.1:8000/ask";

export async function askText(question) {
  if (isShowcase(question)) return SHOWCASE_RESULT;
  try {
    const response = await fetch(`${TARGET_API}-text-fast`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    return await response.json();
  } catch (error) {
    return { ...createError(question, "The retrieval service is unavailable right now."), error: String(error) };
  }
}

export async function askAudio(blob) {
  const data = new FormData();
  data.append("file", blob, "question.webm");
  try {
    const response = await fetch(TARGET_API, {
      method: "POST",
      body: data,
    });
    return await response.json();
  } catch (error) {
    return { ...createAudioError("The voice service is unavailable right now."), error: String(error) };
  }
}
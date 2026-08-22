import { QUICK_CHECKS } from "../data/kosmos";
import { Pipeline, SourceList, Status } from "../components/QueryComponents";

export default function AskPage({ result, loading, submitText, recording, onMic, input, setInput, onReset }) {
    const verified = QUICK_CHECKS.slice(0, 3);

    return (
        <main className="ask-stage light-theme">
            <section className="ask-hero">
                <p className="kicker">KOSMOS / VOICE RAG</p>
                <h1>KOSMOS Query Interface</h1>
                <p>
                    Voice-enabled, multilingual RAG with an “evidence-first” philosophy.<br />
                    Ask with confidence in English, Hindi, or Bengali.
                </p>
            </section>

            {!result && !loading ? (
                <>
                    <section className="query-console">
                        <button
                            className={`ask-mic ${recording ? "is-recording" : ""}`}
                            onClick={onMic}
                            disabled={loading}
                            aria-label={recording ? "Stop recording" : "Start voice recording"}
                        >
                            {recording ? "■" : "🎤"}
                        </button>
                        <div className="query-input">
                            <input
                                value={input}
                                disabled={loading || recording}
                                onChange={(e) => setInput(e.target.value)}
                                onKeyDown={(e) =>
                                    e.key === "Enter" &&
                                    !e.nativeEvent.isComposing &&
                                    e.keyCode !== 229 &&
                                    submitText(input)
                                }
                                placeholder="Ask in Hindi, Bengali, or English..."
                            />
                            <span className="lang-tag">Language: Auto | EN | HI | BN⌄</span>
                        </div>
                        <div className="verified-queries">
                            <span>Verified Queries</span>
                            <div>
                                {verified.map((check) => (
                                    <button key={check.question} onClick={() => submitText(check.question)}>
                                        ✓ {check.question} <small>({check.label.split(" / ")[0]})</small>
                                    </button>
                                ))}
                                <button onClick={() => submitText("__showcase__")}>
                                    ✓ What is an Information? <small>(showcase)</small>
                                </button>
                            </div>
                        </div>
                    </section>
                    <Pipeline />
                </>
            ) : (
                <section className="ask-result">
                    {loading ? (
                        <div className="loading-state">
                            <span className="spinner" />
                            <p>
                                Running the retrieval cascade<span className="blink">...</span>
                            </p>
                            <small>Fast lexical match → multilingual rerank → evidence check</small>
                        </div>
                    ) : (
                        <>
                            <div className="query-line">
                                <span>QUERY</span>
                                <strong>{result.question || "Voice query"}</strong>
                                <Status result={result} />
                            </div>
                            <article className={`answer-panel ${result.status !== "success" ? "answer-refusal" : ""}`}>
                                <span className="answer-overline">
                                    {result.status === "success" ? "SOURCE-SUPPORTED ANSWER" : "SAFE COMPLETION"}
                                </span>
                                <p>
                                    {result.answer ||
                                        "Kosmos could not find direct evidence for this question, so it declined to answer."}
                                </p>
                                <div className="answer-stats">
                                    <span>total {result.timing?.total_ms ?? "—"} ms</span>
                                    <span>retrieval {result.timing?.retrieval_ms ?? "—"} ms</span>
                                    {result.tier && <span>{result.tier}</span>}
                                </div>
                            </article>
                            <SourceList sources={result.sources} />
                            <div style={{ marginTop: "1.5rem", display: "flex", justifyContent: "flex-start" }}>
                                <button className="reset-button" onClick={onReset}>
                                    ← Ask another question
                                </button>
                            </div>
                        </>
                    )}
                </section>
            )}
        </main>
    );
}
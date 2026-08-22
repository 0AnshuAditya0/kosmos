import { Header, AbstractOrbit } from "../components/Brand";

export default function LandingPage({ setView, nav }) {
    return (
        <div className="landing">
            <Header setView={setView} landing NAV={nav} />
            <main className="landing-main">
                <div className="landing-copy">
                    <h1>
                        Make every answer
                        <br />
                        <em>traceable.</em>
                    </h1>
                    <p className="landing-lede">
                        Kosmos listens across Hindi, Bengali, and English — then returns only what the source can support.
                    </p>
                    <div className="landing-actions">
                        <button className="primary-button" onClick={() => setView("Ask")}>
                            Enter Kosmos <span>→</span>
                        </button>
                        <span className="action-note">No guesswork. No hidden generation.</span>
                    </div>
                </div>
                <AbstractOrbit />
            </main>
        </div>
    );
}
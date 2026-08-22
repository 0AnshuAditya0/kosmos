import { getInfo } from "../data/kosmos";

export default function InfoPage({ view }) {
    const [title, copy, items] = getInfo(view) || ["", "", []];

    return (
        <main className="info-view page light-theme">
            <p className="kicker">KOSMOS / {view?.toUpperCase()}</p>
            <h1>{title}</h1>
            <p className="info-copy">{copy}</p>

            <div className="info-grid">
                {items.map((item, index) => (
                    <div className="info-item" key={item}>
                        <span>0{index + 1}</span>
                        <strong>{item}</strong>
                        <i>↗</i>
                    </div>
                ))}
            </div>

            <div className="abstract-note">
                <span className="section-index">/ / /</span>
                <p>Designed to keep the distance between a question and its evidence as small as possible.</p>
            </div>
        </main>
    );
}
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api.js";

const TABS = [
  { id: "summary", label: "Summary" },
  { id: "key", label: "Key information" },
  { id: "ask", label: "Ask your paper" },
  { id: "search", label: "Semantic search" },
  { id: "export", label: "Export" },
];

const SUGGESTIONS = [
  "Which dataset was used?",
  "What method does the paper propose?",
  "What are the main results?",
  "What are the limitations?",
];

const SECTION_NAMES = {
  abstract: "Abstract", introduction: "Introduction", background: "Background",
  methods: "Methods", results: "Results", discussion: "Discussion",
  limitations: "Limitations", conclusion: "Conclusion", future_work: "Future work",
  front: "Front matter", other: "Body",
};
const sectionName = (s) => SECTION_NAMES[s] || s;

/* ------------------------------------------------------------ shared bits */

function Highlight({ text, mark }) {
  // Safe highlighting: split the string, never inject HTML.
  if (!mark) return <>{text}</>;
  const i = text.toLowerCase().indexOf(mark.toLowerCase());
  if (i < 0) return <>{text}</>;
  return (
    <>
      {text.slice(0, i)}
      <mark>{text.slice(i, i + mark.length)}</mark>
      {text.slice(i + mark.length)}
    </>
  );
}

function Excerpt({ page, section, score, children }) {
  return (
    <figure className="excerpt">
      <div className="pagetab" title={`Page ${page}`}>
        <span>p.</span>
        <b>{page}</b>
      </div>
      <div className="excerpt-body">
        <p>{children}</p>
        {(section || score != null) && (
          <figcaption>
            {section && <span>{sectionName(section)}</span>}
            {score != null && (
              <span className="sim" title={`Similarity ${score.toFixed(2)}`}>
                <span className="track">
                  <i style={{ width: `${Math.max(3, Math.min(100, score * 100))}%` }} />
                </span>
                <em>{score.toFixed(2)}</em>
              </span>
            )}
          </figcaption>
        )}
      </div>
    </figure>
  );
}

function Spinner({ label }) {
  return (
    <span className="busy">
      <span className="spinner" aria-hidden="true" />
      {label}
    </span>
  );
}

/* ------------------------------------------------------------ tabs */

function FieldRows({ fields }) {
  return (
    <div className="rows">
      {fields.map((f) => (
        <section key={f.key} className="row-field">
          <h3>{f.label}</h3>
          <div>
            {f.items.length === 0 ? (
              <p className="none">Nothing in this paper matched closely enough.</p>
            ) : (
              f.items.map((it, i) => (
                <Excerpt key={i} page={it.page} section={it.section}>
                  {it.text}
                </Excerpt>
              ))
            )}
          </div>
        </section>
      ))}
    </div>
  );
}

function AnswerCard({ r }) {
  const tone = r.status === "answered" ? "ok" : r.status === "excerpts_only" ? "info" : "warn";
  const title =
    r.status === "answered"
      ? null
      : r.status === "excerpts_only"
      ? "Related passages only"
      : "Not enough evidence in this paper";
  return (
    <article className={`answer ${tone}`}>
      <p className="question">{r.question}</p>
      {r.status === "answered" ? (
        <>
          <p className="answer-text">{r.answer}</p>
          {r.answer_sentence && <p className="answer-ctx">{r.answer_sentence}</p>}
        </>
      ) : (
        <div className="notice">
          <b>{title}</b>
          <span>{r.message}</span>
        </div>
      )}
      {r.evidence?.length > 0 && (
        <details open={r.status !== "insufficient_evidence"}>
          <summary>
            {r.status === "insufficient_evidence" ? "Closest passages" : "Evidence"} ({r.evidence.length})
          </summary>
          {r.evidence.map((e, i) => (
            <Excerpt key={`${e.idx}-${i}`} page={e.page} section={e.section} score={e.score}>
              <Highlight text={e.text} mark={i === 0 ? r.answer : null} />
            </Excerpt>
          ))}
        </details>
      )}
    </article>
  );
}

function AskTab({ paper }) {
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [log, setLog] = useState([]);

  useEffect(() => {
    api.history(paper.id).then((h) => setLog([...h].reverse())).catch(() => {});
  }, [paper.id]);

  async function ask(text) {
    const question = text.trim();
    if (!question || busy) return;
    setBusy(true);
    setError("");
    try {
      const r = await api.ask(paper.id, question);
      setLog((l) => [r, ...l]);
      setQ("");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <form className="searchbar" onSubmit={(e) => (e.preventDefault(), ask(q))}>
        <input
          value={q}
          maxLength={500}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Ask something this paper can answer"
          aria-label="Question"
        />
        <button className="btn primary" disabled={busy || !q.trim()}>
          {busy ? <Spinner label="Reading" /> : "Ask"}
        </button>
      </form>
      {log.length === 0 && (
        <div className="suggest">
          <span>Try</span>
          {SUGGESTIONS.map((s) => (
            <button key={s} className="chip" onClick={() => ask(s)} disabled={busy}>
              {s}
            </button>
          ))}
        </div>
      )}
      {error && <p className="error">{error}</p>}
      {log.map((r, i) => (
        <AnswerCard key={r.id ?? `new-${i}`} r={r} />
      ))}
    </div>
  );
}

function SearchTab({ paper }) {
  const [q, setQ] = useState("");
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function submit(e) {
    e.preventDefault();
    if (!q.trim()) return;
    setBusy(true);
    setError("");
    try {
      setRes((await api.search(paper.id, q.trim())).results);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <form className="searchbar" onSubmit={submit}>
        <input
          value={q}
          maxLength={500}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Describe what you are looking for, for example: how was it evaluated"
          aria-label="Search the paper"
        />
        <button className="btn primary" disabled={busy || !q.trim()}>
          {busy ? <Spinner label="Searching" /> : "Search"}
        </button>
      </form>
      {error && <p className="error">{error}</p>}
      {res === null && (
        <p className="hint">Search finds passages by meaning, so you do not need the paper's exact words.</p>
      )}
      {res?.map((e, i) => (
        <Excerpt key={`${e.idx}-${i}`} page={e.page} section={e.section} score={e.score}>
          {e.text}
        </Excerpt>
      ))}
    </div>
  );
}

function ExportTab({ paper }) {
  return (
    <div className="rows">
      <section className="row-field">
        <h3>Report</h3>
        <div>
          <p className="lede">
            Summary, key information and your question log, each statement with its page number.
          </p>
          <div className="btnrow">
            <a className="btn primary" href={api.reportUrl(paper.id, "md")}>Download Markdown</a>
            <a className="btn" href={api.reportUrl(paper.id, "html")}>Download HTML</a>
          </div>
          <p className="hint">Open the HTML file in a browser and choose Print, then Save as PDF.</p>
        </div>
      </section>
    </div>
  );
}

/* ------------------------------------------------------------ app */

export default function App() {
  const [papers, setPapers] = useState([]);
  const [current, setCurrent] = useState(null);
  const [tab, setTab] = useState("summary");
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  const [model, setModel] = useState(null);
  const fileRef = useRef(null);

  const refresh = useCallback(async () => setPapers(await api.listPapers()), []);

  useEffect(() => {
    refresh().catch(() => setError("Cannot reach the backend. Start it with `python app.py` in the backend folder."));
    api.health().then((h) => setModel(h.model)).catch(() => {});
  }, [refresh]);

  async function uploadFile(file) {
    if (!file) return;
    setUploading(true);
    setError("");
    try {
      const paper = await api.upload(file);
      await refresh();
      setCurrent(paper);
      setTab("summary");
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  }

  function onPick(e) {
    const file = e.target.files?.[0];
    e.target.value = "";
    uploadFile(file);
  }

  function onDrop(e) {
    e.preventDefault();
    setDragging(false);
    uploadFile(e.dataTransfer.files?.[0]);
  }

  async function open(id) {
    setError("");
    try {
      setCurrent(await api.getPaper(id));
      setTab("summary");
    } catch (err) {
      setError(err.message);
    }
  }

  async function remove(id) {
    if (!window.confirm("Delete this paper and its analysis permanently?")) return;
    await api.remove(id);
    if (current?.id === id) setCurrent(null);
    refresh();
  }

  const a = current?.analysis;

  return (
    <div className="shell" onDragOver={(e) => (e.preventDefault(), setDragging(true))}
         onDragLeave={(e) => e.currentTarget === e.target && setDragging(false)} onDrop={onDrop}>
      <aside className="rail">
        <div className="brand">
          <span className="logo" aria-hidden="true">
            <i /><i /><i />
          </span>
          <div>
            <h1>PaperMind</h1>
            <p>Research paper analyzer</p>
          </div>
        </div>

        <input ref={fileRef} type="file" accept="application/pdf,.pdf" hidden onChange={onPick} />
        <button className="btn add" disabled={uploading} onClick={() => fileRef.current.click()}>
          {uploading ? <Spinner label="Analyzing" /> : "Add a paper"}
        </button>

        <nav className="library" aria-label="Your papers">
          {papers.length === 0 && <p className="rail-empty">Papers you add will be listed here.</p>}
          {papers.map((p) => (
            <div key={p.id} className={`libitem ${current?.id === p.id ? "on" : ""}`}>
              <button onClick={() => open(p.id)} title={p.title || p.filename}>
                <span className="t">{p.title || p.filename}</span>
                <span className="m">{p.n_pages} pages</span>
              </button>
              <button className="del" aria-label={`Delete ${p.title || p.filename}`} onClick={() => remove(p.id)}>
                Delete
              </button>
            </div>
          ))}
        </nav>

        {model && (
          <div className="modelbox">
            <b>{model.source === "fine-tuned" ? "Fine-tuned model" : "Pretrained model"}</b>
            <span>
              {(model.parameters / 1e6).toFixed(1)}M parameters, evidence threshold{" "}
              {model.threshold.toFixed(2)}
            </span>
          </div>
        )}
      </aside>

      <main>
        {error && <p className="error banner">{error}</p>}

        {uploading && (
          <div className="analyzing" role="status">
            <Spinner label="Reading your paper" />
            <p>Extracting text, indexing passages and writing the summary. This usually takes under a minute.</p>
          </div>
        )}

        {!current && !uploading && (
          <div className={`drop ${dragging ? "over" : ""}`}>
            <h2>Read a paper with its evidence beside it</h2>
            <p>
              Add a PDF to get a structured summary, ask questions in plain language and see the exact
              passage and page behind every answer.
            </p>
            <button className="btn primary big" onClick={() => fileRef.current.click()}>
              Choose a PDF
            </button>
            <span className="hint">or drop it anywhere on this window</span>
          </div>
        )}

        {current && !uploading && (
          <>
            <header className="paperhead">
              <h2>{current.title || current.filename}</h2>
              <p>
                <span>{current.n_pages} pages</span>
                <span>{current.n_passages} passages indexed</span>
              </p>
            </header>

            <div className="tabs" role="tablist">
              {TABS.map((t) => (
                <button key={t.id} role="tab" aria-selected={tab === t.id}
                        className={tab === t.id ? "on" : ""} onClick={() => setTab(t.id)}>
                  {t.label}
                </button>
              ))}
            </div>

            <div className="panel">
              {tab === "summary" && (
                <>
                  {a.abstract && (
                    <section className="abstract">
                      <h3>Abstract</h3>
                      <p>{a.abstract}</p>
                    </section>
                  )}
                  <FieldRows fields={a.summary} />
                </>
              )}
              {tab === "key" && <FieldRows fields={a.key_info} />}
              {tab === "ask" && <AskTab key={current.id} paper={current} />}
              {tab === "search" && <SearchTab key={current.id} paper={current} />}
              {tab === "export" && <ExportTab paper={current} />}
            </div>
          </>
        )}
      </main>
    </div>
  );
}
import { useMemo, useState } from "react";
import { Copy, Check, ChevronDown, ChevronUp, FileText, Languages } from "lucide-react";

type PageResult = {
  page_number: number;
  text: string;
  lang?: string;
  translated_text?: string;
};

function copy(text: string) {
  try {
    navigator.clipboard?.writeText(text);
  } catch {}
}

function ResultCard({
  section,
  translation,
  style,
}: {
  section: PageResult;
  translation?: string;
  style?: React.CSSProperties;
}) {
  const [expanded, setExpanded] = useState(false);
  const [copiedMarathi, setCopiedMarathi] = useState(false);
  const [copiedEnglish, setCopiedEnglish] = useState(false);
  const pageTranslation = translation ?? section.translated_text ?? "";
  const metaLang = section.lang ? section.lang.toUpperCase() : "—";
  const status = pageTranslation ? "Translated" : "OCR";

  function handleCopyMarathi() {
    copy(section.text);
    setCopiedMarathi(true);
    setTimeout(() => setCopiedMarathi(false), 2000);
  }

  function handleCopyEnglish() {
    copy(pageTranslation);
    setCopiedEnglish(true);
    setTimeout(() => setCopiedEnglish(false), 2000);
  }

  return (
    <article className={`result-card fade-in ${expanded ? "expanded" : ""}`} style={style}>
      <header className="result-header">
        <div>
          <h2>
            <FileText size={18} style={{ verticalAlign: "middle", marginRight: "0.5rem" }} />
            Page {section.page_number}
          </h2>
          <p className="result-sub">
            <Languages size={14} style={{ verticalAlign: "middle", marginRight: "0.25rem" }} />
            Detected language: {metaLang}
          </p>
        </div>
        <div className="result-actions">
          <button className="btn btn-sm" onClick={handleCopyMarathi}>
            {copiedMarathi ? <Check size={16} /> : <Copy size={16} />}
            Copy Marathi
          </button>
          <button
            className="btn btn-sm"
            onClick={handleCopyEnglish}
            disabled={!pageTranslation}
          >
            {copiedEnglish ? <Check size={16} /> : <Copy size={16} />}
            Copy English
          </button>
          <button className="btn btn-sm" onClick={() => setExpanded((v) => !v)}>
            {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            {expanded ? "Collapse" : "Expand"}
          </button>
          <span className={`pill ${status === "Translated" ? "success" : "muted"}`}>
            {status}
          </span>
        </div>
      </header>

      <div className="grid-2col">
        <section
          className="panel"
          style={!expanded ? { maxHeight: 220, overflow: "auto" } : undefined}
        >
          <h3>Marathi (OCR)</h3>
          <pre className="code-block">{section.text || "No text detected."}</pre>
        </section>
        <section
          className="panel"
          style={!expanded ? { maxHeight: 220, overflow: "auto" } : undefined}
        >
          <h3>English</h3>
          <pre className="code-block">{pageTranslation || "—"}</pre>
        </section>
      </div>
    </article>
  );
}

export default function ResultsTable({
  sections,
  translations,
  uploadedFile,
}: {
  sections: PageResult[];
  translations?: Record<number, string>;
  uploadedFile?: string;
}) {
  const items = useMemo(() => sections ?? [], [sections]);
  if (!items.length) return null;
  return (
    <section className="results">
      {uploadedFile && (
        <div className="results-heading">
          <h1>{uploadedFile}</h1>
          <p>Review OCR output alongside live English translations.</p>
        </div>
      )}
      {items.map((s, idx) => (
        <ResultCard
          key={s.page_number}
          section={s}
          translation={translations?.[s.page_number] ?? s.translated_text}
          style={{ animationDelay: `${idx * 0.1}s` }}
        />
      ))}
    </section>
  );
}

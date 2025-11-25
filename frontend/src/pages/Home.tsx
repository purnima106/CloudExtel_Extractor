import { useMemo, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Send, Languages, Loader2, AlertCircle, FileCheck, DollarSign, Ruler, Hash, User, FileText as FileTextIcon } from "lucide-react";
import FileUpload from "../components/FileUpload";
import ResultsTable from "../components/ResultsTable";
import { uploadPdf, downloadFile } from "../api";
import { translateText } from "../api/translate";
import DownloadPanel from "../components/DownloadPanel";

type PageResult = {
  page_number: number;
  text: string;
  lang?: string;
  translated_text?: string;
};

type UploadResponse = {
  job_id: string;
  filename?: string;
  total_pages?: number;
  marathi_measurements?: Record<string, any>;
  english_measurements?: Record<string, any>;
  table_rows?: Array<Record<string, any>>;
  summary?: Record<string, any>;
  extracted_measurements?: Record<string, any>;
  page_wise_measurements?: Array<{
    page_number: number;
    marathi?: Record<string, any>;
    english?: Record<string, any>;
    table_rows?: Array<Record<string, any>>;
    summary?: Record<string, any>;
    measurements?: Record<string, any>;
    raw_text?: string;
  }>;
  available_outputs: string[];
  // Legacy format support
  results?: PageResult[];
};

type PulseStage = "" | "upload" | "submit" | "translate";

export default function Home() {
  const [resp, setResp] = useState<UploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [translations, setTranslations] = useState<Record<number, string>>({});
  const [isTranslating, setIsTranslating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileLabel, setFileLabel] = useState<string>("");
  const [pulse, setPulse] = useState<PulseStage>("");

  const totalPages = resp?.results?.length ?? 0;

  const upload = useMutation({
    mutationFn: uploadPdf,
    onSuccess: (data: UploadResponse) => {
      setResp(data);
      setError(null);
      // Handle new format (measurements) or legacy format (full text)
      if (data.results) {
        const initialMap = Object.fromEntries(
          data.results.map((r) => [r.page_number, r.translated_text ?? ""])
        );
        setTranslations(initialMap);
      } else {
        // New format - no translation needed, measurements are already extracted
        setTranslations({});
      }
      setProgress(0);
      setPulse("upload");
    },
    onError: () => {
      setError("Upload failed");
      setResp(null);
    },
  });

  function triggerPulse(stage: PulseStage) {
    setPulse(stage);
    if (stage) {
      setTimeout(() => setPulse(""), 700);
    }
  }

  function handleSubmit() {
    if (!selectedFile) return;
    triggerPulse("submit");
    upload.mutate(selectedFile);
  }

  async function translateAll() {
    if (!resp?.results?.length) return;
    setIsTranslating(true);
    triggerPulse("translate");
    setProgress(0);
    const next: Record<number, string> = { ...translations };
    for (let i = 0; i < resp.results.length; i++) {
      const r = resp.results[i];
      try {
        const src = r.lang || "auto";
        const out = await translateText(r.text, src, "en");
        next[r.page_number] = out.translated_text;
      } catch {
        next[r.page_number] = next[r.page_number] ?? "";
      }
      setProgress(Math.round(((i + 1) / resp.results.length) * 100));
      setTranslations({ ...next });
    }
    setIsTranslating(false);
  }

  const statusText = useMemo(() => {
    if (upload.isPending) return "Extracting measurements from PDF…";
    if (isTranslating) return `Translating pages (${progress}% done)`;
    if (resp) {
      if (resp.extracted_measurements) {
        const count = Object.keys(resp.extracted_measurements).length;
        return `Extracted ${count} measurements from ${resp.total_pages || 0} pages`;
      }
      return `Processed ${resp.results?.length || 0} pages`;
    }
    if (selectedFile) return "Ready to submit";
    return "Awaiting upload";
  }, [upload.isPending, isTranslating, resp, selectedFile, progress]);

  return (
    <div className={`app ${pulse ? `pulse-${pulse}` : ""}`}>
      <div className="power-indicator" aria-hidden>
        <span className={pulse ? `stage-${pulse}` : ""} />
      </div>
      <section className="status-bar">
        <div className="status-text">{statusText}</div>
        {fileLabel && (
          <div className="file-chip" title={fileLabel}>
            <span className="dot" />
            <span className="name">{fileLabel}</span>
          </div>
        )}
      </section>

      <div className="toolbar">
        <FileUpload
          disabled={upload.isPending}
          fileName={selectedFile ? selectedFile.name : fileLabel}
          onFileSelected={(f) => {
            setSelectedFile(f);
            setFileLabel(f.name);
            triggerPulse("upload");
          }}
        />
        <div className="actions">
          {resp && (
            <button
              className={`btn primary ${pulse === "translate" ? "power" : ""}`}
              disabled={isTranslating || !totalPages}
              onClick={translateAll}
            >
              {isTranslating ? (
                <>
                  <Loader2 className="spinner" size={18} />
                  Translating…
                </>
              ) : (
                <>
                  <Languages size={18} />
                  Translate to English
                </>
              )}
            </button>
          )}
          <button
            className={`btn secondary ${pulse === "submit" ? "power" : ""}`}
            disabled={!selectedFile || upload.isPending}
            onClick={handleSubmit}
          >
            {upload.isPending ? (
              <>
                <Loader2 className="spinner" size={18} />
                Uploading…
              </>
            ) : (
              <>
                <Send size={18} />
                Submit
              </>
            )}
          </button>
        </div>
      </div>

      <DownloadPanel
        disabled={!resp}
        onDownload={(t) => resp && downloadFile(resp.job_id, t)}
      />

      {isTranslating && (
        <div className="progress">
          <span style={{ width: `${progress}%` }} />
        </div>
      )}
      {error && (
        <div className="error">
          <AlertCircle size={20} />
          <span>{error}</span>
        </div>
      )}
      {!resp && (
        <div className="notice">
          <FileCheck size={24} style={{ opacity: 0.5, marginBottom: "0.5rem" }} />
          <p>Upload a Marathi PDF to extract measurements.</p>
        </div>
      )}
      {resp && (resp.marathi_measurements || resp.english_measurements || resp.extracted_measurements) && (
        <>
          {(() => {
            const data = resp.english_measurements || resp.extracted_measurements || resp.marathi_measurements || {};
            
            // Helper function to format field names
            const formatKey = (key: string) => {
              return key
                .replace(/_/g, ' ')
                .replace(/\b\w/g, (c) => c.toUpperCase());
            };
            
            // Helper function to render a value
            const renderValue = (value: any): string => {
              if (Array.isArray(value)) {
                // Handle simple arrays (like phone_numbers)
                if (value.length > 0 && typeof value[0] !== 'object') {
                  return value.join(', ');
                }
                // For arrays of objects, don't render them here - they'll be flattened
                return `[${value.length} items - see below]`;
              }
              if (typeof value === 'object' && value !== null) {
                // Handle nested objects (like charges_summary)
                return Object.entries(value)
                  .filter(([_, v]) => v !== null && v !== undefined && v !== '')
                  .map(([k, v]) => `${formatKey(k)}: ${typeof v === 'object' ? JSON.stringify(v) : v}`)
                  .join(', ');
              }
              return String(value);
            };
            
            // Flatten nested arrays of objects into individual KVPs
            // Preserve the original order from the backend (no sorting)
            const flattenedData: Array<[string, any]> = [];
            
            Object.entries(data).forEach(([key, value]) => {
              if (Array.isArray(value) && value.length > 0 && typeof value[0] === 'object') {
                // This is an array of objects (like parsed_items, excavation_charges_table)
                // Flatten each item into separate fields WITHOUT "item_N" prefix
                value.forEach((item) => {
                  Object.entries(item).forEach(([itemKey, itemValue]) => {
                    if (itemValue !== null && itemValue !== undefined && itemValue !== '') {
                      // Use the field name directly (no prefix)
                      flattenedData.push([itemKey, itemValue]);
                    }
                  });
                });
              } else if (value !== null && value !== undefined && value !== '') {
                // Regular field - add as-is in original order
                flattenedData.push([key, value]);
              }
            });
            
            // Show ALL fields in their original extraction order
            const entries = flattenedData.filter(([_, value]) => 
              value !== null && 
              value !== undefined && 
              value !== ''
            );
            
            // Categorize fields
            const categories: Record<string, Array<[string, any]>> = {
              document: [],
              financial: [],
              measurements: [],
              contact: [],
              other: []
            };
            
            entries.forEach(([key, value]) => {
              const lowerKey = key.toLowerCase();
              if (lowerKey.includes('document') || lowerKey.includes('dn_number') || lowerKey.includes('date') || lowerKey.includes('reference') || lowerKey.includes('letter')) {
                categories.document.push([key, value]);
              } else if (lowerKey.includes('amount') || lowerKey.includes('charge') || lowerKey.includes('deposit') || lowerKey.includes('rent') || lowerKey.includes('gst') || lowerKey.includes('pan') || lowerKey.includes('total') || lowerKey.includes('rate') || lowerKey.includes('cost') || lowerKey.includes('price')) {
                categories.financial.push([key, value]);
              } else if (lowerKey.includes('length') || lowerKey.includes('width') || lowerKey.includes('area') || lowerKey.includes('meter') || lowerKey.includes('measurement') || lowerKey.includes('pit') || lowerKey.includes('count')) {
                categories.measurements.push([key, value]);
              } else if (lowerKey.includes('phone') || lowerKey.includes('address') || lowerKey.includes('name') || lowerKey.includes('email') || lowerKey.includes('contact')) {
                categories.contact.push([key, value]);
              } else {
                categories.other.push([key, value]);
              }
            });
            
            const categoryConfig = [
              { key: 'document', title: 'Document Information', icon: FileTextIcon, color: 'var(--info)' },
              { key: 'financial', title: 'Financial Details', icon: DollarSign, color: 'var(--success)' },
              { key: 'measurements', title: 'Measurements', icon: Ruler, color: 'var(--brand)' },
              { key: 'contact', title: 'Contact Information', icon: User, color: 'var(--warning)' },
              { key: 'other', title: 'Other Details', icon: Hash, color: 'var(--muted)' }
            ];
            
            const formatCurrency = (value: any): string => {
              if (typeof value === 'number') {
                return new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(value);
              }
              return String(value);
            };
            
            const formatNumber = (value: any): string => {
              if (typeof value === 'number') {
                return new Intl.NumberFormat('en-IN').format(value);
              }
              return String(value);
            };
            
            return (
              <div className="measurements-display fade-in">
                <h3>
                  <FileCheck size={24} style={{ verticalAlign: "middle", marginRight: "0.5rem" }} />
                  Extracted Data ({entries.length} fields)
                </h3>
                
                {categoryConfig.map(({ key, title, icon: Icon, color }) => {
                  const items = categories[key as keyof typeof categories];
                  if (items.length === 0) return null;
                  
                  return (
                    <div key={key} className="measurement-category">
                      <h4 style={{ color }}>
                        <Icon size={20} style={{ verticalAlign: "middle", marginRight: "0.5rem" }} />
                        {title} ({items.length})
                      </h4>
                      <div className="measurements-grid">
                        {items.map(([key, value], idx) => {
                          const isFinancial = key.toLowerCase().includes('amount') || key.toLowerCase().includes('charge') || key.toLowerCase().includes('total') || key.toLowerCase().includes('cost');
                          const isNumber = typeof value === 'number';
                          const displayValue = isFinancial && isNumber ? formatCurrency(value) : isNumber ? formatNumber(value) : renderValue(value);
                          
                          return (
                            <div key={`${key}_${idx}`} className="measurement-item">
                              <span className="measurement-key">{formatKey(key)}:</span>
                              <span className={`measurement-value ${isFinancial ? 'financial' : ''}`}>{displayValue}</span>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  );
                })}
              </div>
            );
          })()}
        </>
      )}
      {resp && resp.results && (
        <ResultsTable
          sections={resp.results}
          translations={translations}
          uploadedFile={fileLabel}
        />
      )}
    </div>
  );
}


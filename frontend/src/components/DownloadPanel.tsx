import { Download, FileJson, FileSpreadsheet, FileText } from "lucide-react";
import { useState } from "react";

export default function DownloadPanel({onDownload, disabled}:{onDownload:(t:"json"|"excel"|"pdf")=>void, disabled?: boolean}){
  const [downloading, setDownloading] = useState<"json" | "excel" | "pdf" | null>(null);

  function handleDownload(type: "json" | "excel" | "pdf") {
    setDownloading(type);
    onDownload(type);
    setTimeout(() => setDownloading(null), 1000);
  }

  return (
    <div className="downloads">
      <button 
        className="btn" 
        disabled={disabled} 
        onClick={() => handleDownload("json")}
      >
        {downloading === "json" ? (
          <div className="spinner" />
        ) : (
          <FileJson size={18} />
        )}
        Download JSON
      </button>
      <button 
        className="btn" 
        disabled={disabled} 
        onClick={() => handleDownload("excel")}
      >
        {downloading === "excel" ? (
          <div className="spinner" />
        ) : (
          <FileSpreadsheet size={18} />
        )}
        Download Excel
      </button>
      <button 
        className="btn" 
        disabled={disabled} 
        onClick={() => handleDownload("pdf")}
      >
        {downloading === "pdf" ? (
          <div className="spinner" />
        ) : (
          <FileText size={18} />
        )}
        Download PDF
      </button>
    </div>
  );
}

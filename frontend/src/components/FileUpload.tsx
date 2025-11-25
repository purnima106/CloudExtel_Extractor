import { useState } from "react";
import { Upload, FileText, Loader2 } from "lucide-react";

type Props = {
  disabled?: boolean;
  fileName?: string;
  onFileSelected: (file: File) => void;
};

export default function FileUpload({ disabled, fileName, onFileSelected }: Props) {
  const [glow, setGlow] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  function triggerGlow() {
    setGlow(true);
    setTimeout(() => setGlow(false), 250);
  }

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault();
    if (!disabled) setIsDragging(true);
  }

  function handleDragLeave() {
    setIsDragging(false);
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type === "application/pdf") {
      onFileSelected(file);
    }
  }

  return (
    <label
      className={`upload-box ${disabled ? "disabled" : ""} ${glow ? "glow" : ""} ${isDragging ? "dragging" : ""}`}
      onClick={triggerGlow}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <input
        type="file"
        accept="application/pdf"
        hidden
        disabled={disabled}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) {
            onFileSelected(f);
          }
          e.currentTarget.value = "";
        }}
      />
      {disabled ? (
        <Loader2 className="upload-icon" size={20} style={{ animation: "spin 0.6s linear infinite" }} />
      ) : fileName ? (
        <FileText className="upload-icon" size={20} />
      ) : (
        <Upload className="upload-icon" size={20} />
      )}
      <div className="upload-content">
        <span className="upload-label">
          {disabled ? "Uploading…" : isDragging ? "Drop PDF here" : "Click or select a Marathi PDF"}
        </span>
        {fileName && <span className="upload-name">{fileName}</span>}
      </div>
    </label>
  );
}

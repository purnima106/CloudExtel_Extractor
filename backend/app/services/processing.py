from __future__ import annotations
from pathlib import Path
import uuid
from .exporters import write_json, write_excel, write_pdf
from .document_detector import extract_with_auto_detection
from pdf2image import convert_from_bytes

OUTPUT_ROOT = Path("backend_generated")


async def handle_pdf_upload(file_bytes: bytes, filename: str):
    """
    Extract measurements (numeric key-value pairs) from Marathi PDF.
    Uses Gemini Vision API (multimodal) + OpenCV table detection for accurate extraction.
    """
    job_id = uuid.uuid4().hex
    job_dir = OUTPUT_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / filename).write_bytes(file_bytes)

    # Extract measurements from PDF bytes using AUTO-DETECTION
    # Automatically detects NMMC vs KDMC format and uses appropriate extractor
    measurements = extract_with_auto_detection(file_bytes)
    # Handle structured format from new extractor
    all_marathi = measurements.get("marathi", {})
    all_english = measurements.get("english", {})
    all_table_rows = measurements.get("table_rows", [])
    all_summary = measurements.get("summary", {})
    
    # Also extract from fields dict - include ALL field types (text, numbers, arrays)
    fields_dict = measurements.get("fields", {})
    for key, meta in fields_dict.items():
        value = meta.get("value") if isinstance(meta, dict) else meta
        # Include all non-null values (numbers, strings, arrays, etc.)
        if value is not None and value != "":
            all_english[key] = value
    
    # Create page-wise breakdown (simplified since we process whole PDF)
    page_measurements = [{
        "page_number": 1,
        "marathi": all_marathi,
        "english": all_english,
        "table_rows": all_table_rows,
        "summary": all_summary
    }]

    # Get total pages count from PDF
    try:
        pages = convert_from_bytes(file_bytes, dpi=150)  # Low DPI just for counting
        total_pages = len(pages)
    except:
        total_pages = 1  # Fallback
    
    # 5) Create structured results with separate Marathi and English
    results = {
        "job_id": job_id,
        "filename": filename,
        "total_pages": total_pages,
        "document_type": measurements.get("document_type", "UNKNOWN"),  # NMMC or KDMC
        "detector_confidence": measurements.get("detector_confidence", "UNKNOWN"),
        "marathi_measurements": all_marathi,  # All Marathi keys
        "english_measurements": all_english,   # All English keys
        "table_rows": all_table_rows,         # Complete table data
        "summary": all_summary,               # Summary calculations
        "extracted_measurements": {**all_marathi, **all_english},  # Combined for backward compatibility
        "page_wise_measurements": page_measurements  # Per-page breakdown
    }

    # 6) Write outputs (JSON, Excel, PDF)
    write_json(job_dir, results)
    write_excel(job_dir, results)
    write_pdf(job_dir, results)

    return results

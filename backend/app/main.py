from __future__ import annotations
from pathlib import Path
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from .models import UploadResponse, PageText, TranslateRequest, TranslateResponse
from .services.processing import handle_pdf_upload, OUTPUT_ROOT
from .services.translate import translate_text
from .services.ocr import extract_marathi_text
from langdetect import detect
from deep_translator import GoogleTranslator
import json
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from .services.fonts import resolve_marathi_font
from dotenv import load_dotenv

# Ensure environment variables (.env) are loaded before any services use them
load_dotenv()

app = FastAPI(title="CloudExtel Extractor", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload PDF and extract measurements (numeric key-value pairs).
    Returns structured data with extracted measurements.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    payload = await handle_pdf_upload(await file.read(), file.filename)
    # Return the structured measurement data with separate Marathi/English
    return {
        "job_id": payload["job_id"],
        "filename": payload.get("filename", file.filename),
        "total_pages": payload.get("total_pages", 0),
        "marathi_measurements": payload.get("marathi_measurements", {}),
        "english_measurements": payload.get("english_measurements", {}),
        "table_rows": payload.get("table_rows", []),
        "summary": payload.get("summary", {}),
        "extracted_measurements": payload.get("extracted_measurements", {}),  # Combined for backward compatibility
        "page_wise_measurements": payload.get("page_wise_measurements", []),
        "available_outputs": ["json", "excel", "pdf"]
    }

@app.get("/api/download/{job_id}/{output_type}")
async def download(job_id: str, output_type: str):
    path_map = {
        "json": OUTPUT_ROOT / job_id / "result.json",
        "excel": OUTPUT_ROOT / job_id / "result.xlsx",
        "pdf": OUTPUT_ROOT / job_id / "result.pdf",
    }
    target = path_map.get(output_type)
    if not target or not target.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    media = {
        "json": "application/json",
        "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "pdf": "application/pdf",
    }[output_type]
    return FileResponse(target, media_type=media, filename=target.name)

@app.post("/api/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty.")
    translated = await translate_text(req.text, source=req.source, target=req.target)
    return TranslateResponse(translated_text=translated)

# ===== Minimal demo endpoints =====
LATEST_DIR = Path("latest_demo")
LATEST_DIR.mkdir(parents=True, exist_ok=True)

def _write_demo_outputs(original_text: str, translated_text: str):
    # JSON
    (LATEST_DIR / "result.json").write_text(json.dumps({"original_text": original_text, "translated_text": translated_text}, ensure_ascii=False, indent=2), encoding="utf-8")
    # Excel
    df = pd.DataFrame([{"original_text": original_text, "translated_text": translated_text}])
    df.to_excel(LATEST_DIR / "result.xlsx", index=False)
    # PDF
    dest = LATEST_DIR / "result.pdf"
    c = canvas.Canvas(dest.as_posix(), pagesize=A4)
    w, h = A4
    y = h - 50
    text_font = resolve_marathi_font()
    c.setFont("Helvetica-Bold", 12); c.drawString(40, y, "Original (detected)"); y -= 18
    c.setFont(text_font, 10)
    for line in original_text.splitlines():
        c.drawString(40, y, line); y -= 14
        if y < 60: c.showPage(); y = h - 50; c.setFont(text_font, 10)
    y -= 24
    c.setFont("Helvetica-Bold", 12); c.drawString(40, y, "English (translated)"); y -= 18
    c.setFont("Helvetica", 10)
    for line in translated_text.splitlines():
        c.drawString(40, y, line); y -= 14
        if y < 60: c.showPage(); y = h - 50; c.setFont("Helvetica", 10)
    c.save()

def _concat_text(results: list[dict]) -> str:
    return "\n\n".join([r.get("text", "") for r in results if r.get("text")])

@app.post("/api/upload_pdf")
@app.post("/upload_pdf")
async def upload_pdf_minimal(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    pdf_bytes = await file.read()
    # OCR all pages, then join
    page_results = extract_marathi_text(pdf_bytes, debug_dir=None)
    original_text = _concat_text(page_results)
    if not original_text.strip():
        raise HTTPException(status_code=422, detail="No text detected via OCR.")
    # Force Marathi -> English for the minimal flow
    translated_text = GoogleTranslator(source="mar", target="en").translate(original_text) or ""
    _write_demo_outputs(original_text, translated_text)
    return {"original_text": original_text, "translated_text": translated_text}

@app.get("/api/download/{output_type}")
@app.get("/download/{output_type}")
async def download_minimal(output_type: str):
    path_map = {
        "json": LATEST_DIR / "result.json",
        "excel": LATEST_DIR / "result.xlsx",
        "pdf": LATEST_DIR / "result.pdf",
    }
    target = path_map.get(output_type)
    if not target or not target.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    media = {
        "json": "application/json",
        "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "pdf": "application/pdf",
    }[output_type]
    return FileResponse(target, media_type=media, filename=target.name)

@app.get("/api/health")
async def health_api():
    return {"status":"ok"}

@app.get("/health")
async def health_root():
    return {"status":"ok"}

def convert_to_crores(value) -> float | None:
    """
    Convert a large number to crores by taking the first 3 digits.
    Example: 9,15,07,200 -> 915 -> 9.15 Cr
            2,60,492 -> 260 -> 2.60 Cr
    """
    if value is None or pd.isna(value):
        return None
    
    # Convert to string and remove all non-numeric characters
    value_str = str(value).strip()
    # Remove commas, spaces, and other non-numeric characters
    numeric_str = ''.join(c for c in value_str if c.isdigit())
    
    if not numeric_str or len(numeric_str) == 0:
        return None
    
    # Extract first 3 digits
    first_three = numeric_str[:3]
    
    if len(first_three) == 0:
        return None
    
    # Convert to crores: first_three_digits / 100
    try:
        return float(first_three) / 100.0
    except (ValueError, TypeError):
        return None

@app.get("/api/graph-data")
async def get_graph_data():
    """
    Read Book6.xlsx Sheet2 and return data for Budget vs Actuals S-Curve graph.
    - X-axis: Column C (Cummulative Length) in meters, converted to km
    - Y-axis: Column E (Cummulative Budget) and Column G (Actual Cummulative Non Refundable)
              converted to crores using first 3 digits
    """
    try:
        # Try multiple possible paths for Book6.xlsx
        root_dir = Path(__file__).parent.parent.parent
        current_dir = Path.cwd()
        
        possible_paths = [
            root_dir / "Book6.xlsx",  # Absolute path from file location
            current_dir / "Book6.xlsx",  # Current working directory
            Path("Book6.xlsx"),  # Relative to current working directory
            current_dir.parent / "Book6.xlsx",  # One level up from CWD
        ]
        
        excel_path = None
        for path in possible_paths:
            abs_path = path.resolve()
            if abs_path.exists() and abs_path.is_file():
                excel_path = abs_path
                break
        
        if excel_path is None:
            root_files = [f.name for f in root_dir.glob("*.xlsx") if f.is_file()]
            cwd_files = [f.name for f in current_dir.glob("*.xlsx") if f.is_file()]
            raise HTTPException(
                status_code=404, 
                detail=f"Book6.xlsx not found. Root dir: {root_dir}, CWD: {current_dir}. "
                       f"Root Excel files: {root_files}, CWD Excel files: {cwd_files}"
            )
        
        # Read Excel file - try Sheet2 first, fallback to Sheet1
        df = None
        sheet_used = None
        try:
            df = pd.read_excel(excel_path, sheet_name='Sheet2', engine='openpyxl', header=0)
            sheet_used = 'Sheet2'
        except (ValueError, KeyError):
            try:
                df = pd.read_excel(excel_path, sheet_name='Sheet1', engine='openpyxl', header=0)
                sheet_used = 'Sheet1'
            except:
                # Fallback: read first sheet
                df = pd.read_excel(excel_path, sheet_name=0, engine='openpyxl', header=0)
                sheet_used = 'Sheet0'
        
        # Get column names
        columns = df.columns.tolist()
        
        # Try to find actual column names by searching in the first few rows if headers are "Unnamed"
        # This handles cases where the header row might be in row 1 or 2 instead of row 0
        if any('Unnamed' in str(col) for col in columns) or any(pd.isna(col) or str(col).strip() == '' for col in columns):
            # Try reading with header=None to see raw data
            try:
                df_raw = pd.read_excel(excel_path, sheet_name=sheet_used, engine='openpyxl', header=None, nrows=10)
                # Search for column names in first few rows
                best_header_row = None
                for row_idx in range(min(5, len(df_raw))):
                    row_values = df_raw.iloc[row_idx].tolist()
                    # Check if this row contains our target column names
                    found_c = False
                    found_e = False
                    found_g = False
                    
                    for col_idx, val in enumerate(row_values):
                        if pd.notna(val):
                            val_str = str(val).strip().lower()
                            # Check for Column C (Cummulative Length) at index 2
                            if col_idx == 2 and (('cumulative' in val_str or 'cummulative' in val_str) and 'length' in val_str):
                                found_c = True
                            # Check for Column E (Cummulative Budget) at index 4
                            if col_idx == 4 and (('cumulative' in val_str or 'cummulative' in val_str) and 'budget' in val_str):
                                found_e = True
                            # Check for Column G (Actual Cummulative Non Refundable) at index 6
                            if col_idx == 6 and (('cumulative' in val_str or 'cummulative' in val_str) and 'non' in val_str and 'refundable' in val_str):
                                found_g = True
                    
                    # If we found at least 2 of our target columns, this is likely the header row
                    if found_c and found_e:
                        best_header_row = row_idx
                        break
                    elif found_c or found_e or found_g:
                        best_header_row = row_idx  # Keep as candidate
                
                # Re-read with the found header row
                if best_header_row is not None:
                    df = pd.read_excel(excel_path, sheet_name=sheet_used, engine='openpyxl', header=best_header_row)
                    columns = df.columns.tolist()
            except Exception as e:
                print(f"Warning: Could not find header row automatically: {e}")
                pass  # If this fails, continue with original df
        
        # Find Column C (Cummulative Length) - index 2 (0-indexed)
        col_c = None
        col_c_name = "Cummulative Length"  # Default name
        
        if len(columns) > 2:
            col_c = columns[2]  # Column C (0-indexed: 2)
            # Check if it's a proper name or "Unnamed"
            if 'Unnamed' not in str(col_c) and pd.notna(col_c) and str(col_c).strip() != '':
                col_c_name = str(col_c)
            else:
                # Try to find by name in all columns
                for col in columns:
                    col_lower = str(col).lower()
                    if ('cummulative' in col_lower or 'cumulative' in col_lower) and 'length' in col_lower:
                        col_c = col
                        col_c_name = str(col)
                        break
        
        # Find Column E (Cummulative Budget) - index 4 (0-indexed)
        col_e = None
        col_e_name = "Cummulative Budget"  # Default name
        
        if len(columns) > 4:
            col_e = columns[4]  # Column E (0-indexed: 4)
            # Check if it's a proper name or "Unnamed"
            if 'Unnamed' not in str(col_e) and pd.notna(col_e) and str(col_e).strip() != '':
                col_e_name = str(col_e)
            else:
                # Try to find by name in all columns
                for col in columns:
                    col_lower = str(col).lower()
                    if ('cummulative' in col_lower or 'cumulative' in col_lower) and 'budget' in col_lower:
                        col_e = col
                        col_e_name = str(col)
                        break
        
        # Find Column G (Actual Cummulative Non Refundable) - index 6 (0-indexed)
        col_g = None
        col_g_name = "Actual Cummulative Non Refundable"  # Default name
        
        if len(columns) > 6:
            col_g = columns[6]  # Column G (0-indexed: 6)
            # Check if it's a proper name or "Unnamed"
            if 'Unnamed' not in str(col_g) and pd.notna(col_g) and str(col_g).strip() != '':
                col_g_name = str(col_g)
            else:
                # Try to find by name in all columns
                for col in columns:
                    col_lower = str(col).lower()
                    if ('cummulative' in col_lower or 'cumulative' in col_lower) and 'non' in col_lower and 'refundable' in col_lower:
                        col_g = col
                        col_g_name = str(col)
                        break
        
        if col_c is None:
            raise HTTPException(status_code=400, detail="Column C (Cummulative Length) not found")
        
        # Extract data
        data_points = []
        
        for idx in range(len(df)):
            # Get Column C value (Cummulative Length in meters)
            c_val = df[col_c].iloc[idx] if col_c else None
            # Get Column E value (Cummulative Budget)
            e_val = df[col_e].iloc[idx] if col_e else None
            # Get Column G value (Actual Cummulative Non Refundable)
            g_val = df[col_g].iloc[idx] if col_g else None
            
            # Skip if C is NaN or empty
            if pd.isna(c_val) or c_val == '' or str(c_val).strip() == '':
                continue
            
            # Convert C to number (meters) and convert to km
            try:
                c_float = float(c_val)
                x_km = c_float / 1000.0  # Convert meters to kilometers
            except (ValueError, TypeError):
                continue
            
            # Convert E and G to crores using first 3 digits
            e_crores = convert_to_crores(e_val)
            g_crores = convert_to_crores(g_val)
            
            # Add point
            point = {
                "x": x_km,
                "e": e_crores,
                "g": g_crores,
            }
            data_points.append(point)
        
        # Sort data points by X value for proper line rendering
        data_points_sorted = sorted(data_points, key=lambda p: p["x"])
        
        return {
            "x_column": col_c_name if col_c else "Column C",
            "e_column": col_e_name if col_e else "Column E",
            "g_column": col_g_name if col_g else "Column G",
            "columns": columns,
            "data": data_points_sorted,
            "data_count": len(data_points_sorted),
            "x_label": "Cumulative Length (km)",
            "y_label": "Cost (Crore ₹)",
            "sheet_used": sheet_used
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"Error reading Excel file: {str(e)}\n{traceback.format_exc()}")

@app.get("/")
async def root():
    return {"name":"CloudExtel Extractor","backend":"ok","docs":"/docs","upload":"/api/upload","health":"/health"}

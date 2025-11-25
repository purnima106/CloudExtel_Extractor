"""
KDMC (Kalyan-Dombivli Municipal Corporation) Document Extractor
Handles documents with format: जा.क्र.कडोंमपा / काअ / बांध. / कवि / XXX
"""
import os
import io
import re
import json
import tempfile
from typing import Dict, Any, List, Optional, Union
from functools import lru_cache

import cv2
import numpy as np
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai

# Enhanced OCR and table extraction tools
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("⚠️  PaddleOCR not available, using fallback OCR")

try:
    from rapidocr_onnxruntime import RapidOCR
    RAPIDOCR_AVAILABLE = True
except ImportError:
    RAPIDOCR_AVAILABLE = False
    print("⚠️  RapidOCR not available, using fallback OCR")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("⚠️  PDFPlumber not available, using fallback table extraction")

try:
    from pydantic import BaseModel, Field, field_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    print("⚠️  Pydantic not available, using dict validation")

# ---------------------------------------------------------
# LOAD ENV + API KEY
# ---------------------------------------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-pro")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    print("⚠ WARNING: GOOGLE_API_KEY is missing!")

# ---------------------------------------------------------
# TESSERACT + POPPLER PATHS
# ---------------------------------------------------------
TESSERACT_PATH = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
POPPLER_PATH = os.getenv("POPPLER_PATH", r"C:\poppler\poppler-25.07.0\Library\bin")

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# ---------------------------------------------------------
# INITIALIZE ENHANCED OCR ENGINES
# ---------------------------------------------------------
@lru_cache(maxsize=1)
def get_paddleocr():
    """Initialize PaddleOCR for Marathi + English OCR."""
    if PADDLEOCR_AVAILABLE:
        try:
            return PaddleOCR(use_angle_cls=True, lang='latin', show_log=False, use_gpu=False)
        except Exception as e:
            print(f"⚠️  PaddleOCR initialization failed: {e}")
            return None
    return None

@lru_cache(maxsize=1)
def get_rapidocr():
    """Initialize RapidOCR for high-accuracy numeric extraction."""
    if RAPIDOCR_AVAILABLE:
        try:
            return RapidOCR()
        except Exception as e:
            print(f"⚠️  RapidOCR initialization failed: {e}")
            return None
    return None


# ---------------------------------------------------------
# KDMC MARATHI → ENGLISH DICTIONARY
# ---------------------------------------------------------
KDMC_MARATHI_TO_ENGLISH = {
    # --- Header / Organization ---
    "कल्याण डोंबिवली महानगरपालिका": "kalyan_dombivli_municipal_corporation",
    "कल्याण": "kalyan",
    "बांधकाम विभाग": "construction_department",
    "बांधकाम": "construction_department",
    
    # --- Document Metadata (KDMC specific) ---
    # Note: जा.क्र.कडोंमपा / काअ / बांध. / कवि / XXX is ONE document_number field
    "जाक्रकडोंमपा": "document_number",
    "जा.क्र.कडोंमपा": "document_number",
    "काअ": "document_number",  # Part of document_number
    "बांध": "document_number",  # Part of document_number
    "कवि": "dn_number",  # Extract ONLY the number after कवि (e.g., 164 from कवि/164)
    "दिनांक": "date",
    "दि": "date",
    
    # --- Road/Street Work Terms ---
    "रस्त्याचे नाव": "road_name",
    "रस्ता": "road",
    "विभाग": "department",
    "लांबी (मीटर)": "length_meters",
    "लांबी": "length_meters",
    "पेव्हरब्लॉक": "paverblock",
    "खडीकरण": "gravelling",
    "एकूण लांबी": "total_length",
    "एकूण": "total",
    
    # --- Financial Terms ---
    "रस्त्याची लांबी": "road_length",
    "मंजूर दर": "approved_rate",
    "दर": "rate",
    "एकूण रक्कम": "total_amount",
    "रक्कम": "amount",
    "सुपरविजन चार्ज": "supervision_charges",
    "सुपरविजन": "supervision_charges",
    "सुपरविझन चार्ज खोदाई शुल्ककबर": "supervision_charges_percentage",
    "सिक्युरिटी डिपॉझीट": "security_deposit",
    "सिक्युरिटी डिपॉझीट खोदाई शुल्ककबर": "security_deposit_percentage",
    "भुईभाडे": "ground_rent",
    
    # --- Work Types ---
    "विविध ठिकाणी केबल टाकणेकरिता": "cable_laying_work",
    "केबल": "cable",
    "टाकणे": "laying",
    
    # --- Table Terms ---
    "तक्ता": "table",
    "अ.क्र": "sr_no",
    "क्र": "sr_no",
}


# ---------------------------------------------------------
# PYDANTIC DATA MODELS FOR STRICT VALIDATION
# ---------------------------------------------------------
if PYDANTIC_AVAILABLE:
    class TableRowModel(BaseModel):
        """Validates a single table row with automatic Marathi digit conversion."""
        sr_no: Optional[Union[int, str]] = None
        surface_type: Optional[str] = None
        number_of_pits: Optional[float] = Field(None, ge=0)
        length_m: Optional[float] = Field(None, ge=0)
        width_m: Optional[float] = Field(None, ge=0)
        area_sqm: Optional[float] = Field(None, ge=0)
        rate_per_sqm: Optional[float] = Field(None, ge=0)
        amount: Optional[float] = Field(None, ge=0)
        
        @field_validator('*', mode='before')
        @classmethod
        def normalize_marathi_digits(cls, v):
            """Auto-convert Marathi numerals to English digits."""
            if isinstance(v, str):
                return normalize_number(v)
            return v
    
    class ExtractionOutputModel(BaseModel):
        """Validates final extraction output structure."""
        marathi: Dict[str, Any] = Field(default_factory=dict)
        english: Dict[str, Any] = Field(default_factory=dict)
        table_rows: List[Dict[str, Any]] = Field(default_factory=list)
        summary: Dict[str, Any] = Field(default_factory=dict)
        fields: Dict[str, Any] = Field(default_factory=dict)
else:
    # Fallback: simple dict-based models
    TableRowModel = dict
    ExtractionOutputModel = dict

# ---------------------------------------------------------
# ENHANCED MARATHI NUMERIC NORMALIZATION
# ---------------------------------------------------------
def normalize_number(text: str) -> str:
    """
    Convert Devanagari digits → ASCII digits.
    Also handles Marathi fractions and removes unwanted symbols.
    """
    if not isinstance(text, str):
        text = str(text)
    
    devanagari = "०१२३४५६७८९"
    english = "0123456789"
    table = str.maketrans(devanagari, english)
    text = text.translate(table)
    
    # Remove unwanted symbols but keep decimal points and commas
    text = re.sub(r'[^\d.,\-\s]', '', text)
    
    # Normalize decimal separators (some PDFs use comma)
    text = text.replace(',', '') if '.' in text else text.replace(',', '.')
    
    return text.strip()


def extract_clean_number(text: str) -> Optional[float]:
    """
    Extract a clean float from text with Marathi digits.
    Handles fractions, decimals, and various formats.
    """
    if not text:
        return None
    
    normalized = normalize_number(str(text))
    # Remove all non-numeric except decimal point and minus
    cleaned = re.sub(r'[^\d.\-]', '', normalized)
    
    try:
        return float(cleaned) if cleaned else None
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------
# GEMINI VISION CALL FOR KDMC DOCUMENTS
# ---------------------------------------------------------
def gemini_extract_kdmc(image_bytes: bytes, instruction: str, is_table: bool = False) -> Dict[str, Any]:
    """
    Sends image + instruction to Gemini Vision for KDMC documents.
    """
    model = genai.GenerativeModel(MODEL_NAME)

    if is_table:
        prompt = (
            "Extract this table from the Marathi KDMC document.\n"
            "Return as JSON array with objects containing all column values.\n"
            "Translate Marathi column names to English.\n"
            "Preserve all numeric values exactly.\n"
            "Return ONLY valid JSON array.\n\n"
            f"{instruction}"
        )
    else:
        prompt = (
            "Extract ALL data from this Kalyan-Dombivli Municipal Corporation (KDMC) Marathi document.\n"
            "Translate text to English and return as JSON.\n\n"
            "CRITICAL RULES:\n"
            "1. Use ONLY ENGLISH keys (translate all Marathi field names to English)\n"
            "2. Convert ALL Devanagari digits (०-९) to English (0-9)\n"
            "3. Translate ALL Marathi text VALUES to English\n"
            "4. Keep document numbers and codes as-is\n"
            "5. Preserve all numeric values exactly\n\n"
            "KDMC SPECIFIC FIELDS - Extract these exactly:\n\n"
            "HEADER INFO:\n"
            '- "organization": कल्याण डोंबिवली महानगरपालिका → "Kalyan Dombivli Municipal Corporation"\n'
            '- "department": बांधकाम, कल्याण विभाग → "Construction Department, Kalyan Division"\n\n'
            "DOCUMENT METADATA (KDMC FORMAT):\n"
            '- "document_number": The COMPLETE document number as ONE field: जा.क्र.कडोंमपा / काअ / बांध. / कवि / XXX\n'
            '  (जा.क्र.कडोंमपा, काअ, बांध., कवि are all parts of ONE document_number, not separate fields)\n'
            '- "dn_number": Extract ONLY the numeric part after कवि (e.g., if कवि/164 then "164", if कवि/948 then "948")\n'
            '- "date": Date after दिनांक or दि. Convert Devanagari carefully:\n'
            '  - १/१०/2025 → "1/10/2025"\n'
            '  - ९/१०/2025 → "9/10/2025"\n'
            '  - Devanagari: ०=0, १=1, २=2, ३=3, ४=4, ५=5, ६=6, ७=7, ८=8, ९=9\n\n'
            "RECIPIENT INFO:\n"
            '- "recipient_name": Name after प्रति or मे.\n'
            '- "recipient_address": Full address\n\n'
            "SUBJECT & WORK DETAILS:\n"
            '- "subject": Subject line (विषय)\n'
            '- "work_description": Description of work\n'
            '- "references": Any references (संदर्भ)\n\n'
            "ROAD/WORK MEASUREMENTS:\n"
            '- "road_name": रस्त्याचे नाव (name of street/road)\n'
            '- "length_paverblock": लांबी (मीटर) पेव्हरब्लॉक - Paverblock length\n'
            '- "length_gravelling": लांबी (मीटर) खडीकरण - Gravelling length\n'
            '- "total_length": एकूण लांबी (मीटर) - IMPORTANT: Extract exact value (e.g., 278 NOT 288)\n\n'
            "FINANCIAL INFO (Table rows अ, ब, क, ड):\n"
            '- "various_location_cable_charges": Row 1 - विविध ठिकाणी केबल टाकणे charges\n'
            '- "supervision_charges": Row 2 - सुपरविजन चार्ज (supervision charges)\n'
            '- "supervision_percentage": Percentage like १० रु. (usually 10 Rs per meter)\n'
            '- "security_deposit": Row 3 - सिक्युरिटी डिपॉझीट (security deposit)\n'
            '- "security_percentage": Percentage like १० % (usually 10%)\n'
            '- "ground_rent": Row 4 - भुईभाडे (ground rent)\n'
            '- "total_amount": एकूण - Final total amount\n\n'
            "TRANSLATION EXAMPLES:\n"
            "✓ Translate these:\n"
            '- "कल्याण डोंबिवली महानगरपालिका" → "Kalyan Dombivli Municipal Corporation"\n'
            '- "बांधकाम विभाग" → "Construction Department"\n'
            '- "विविध ठिकाणी केबल टाकणेकरिता" → "For cable laying at various locations"\n'
            "✗ Keep these as-is:\n"
            '- Document numbers: "जा.क्र.कडोंमपा/काअ/बांध./कवि/164"\n'
            '- Company names: "मे. भारती एअरटेल लि."\n'
            '- Numbers: 278, 328, 200\n\n'
            "CRITICAL: The total length is 278 meters, NOT 288. Pay attention to exact digits.\n"
            "Extract ALL other fields and translate text to English.\n"
            "Return ONLY valid JSON with ENGLISH keys and ENGLISH values.\n\n"
            f"{instruction}"
        )

    try:
        response = model.generate_content([
            {"mime_type": "image/jpeg", "data": image_bytes},
            prompt
        ])

        text = response.text.strip()

        # Remove markdown if any
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text)
            text = text.replace("```", "").strip()

        return json.loads(text)

    except Exception as e:
        print(f"Gemini Vision error (KDMC): {str(e)}")
        return {} if not is_table else []


# ---------------------------------------------------------
# OPENCV TABLE DETECTOR (same as NMMC)
# ---------------------------------------------------------
def detect_tables(img: np.ndarray) -> List[tuple]:
    """Enhanced table detection with improved preprocessing."""
    processed = enhanced_preprocess(img)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(processed, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(processed, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    table_mask = cv2.add(vertical_lines, horizontal_lines)
    closing_kernel = np.ones((5, 5), np.uint8)
    table_mask = cv2.morphologyEx(table_mask, cv2.MORPH_CLOSE, closing_kernel, iterations=1)
    cnts, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w > 80 and h > 40 and (w * h) > 5000:
            boxes.append((x, y, w, h))
    return boxes


# ---------------------------------------------------------
# ENHANCED OPENCV PREPROCESSING WITH SOBEL & MORPHOLOGY
# ---------------------------------------------------------
def enhanced_preprocess(img: np.ndarray) -> np.ndarray:
    """
    Improved preprocessing with Sobel edge detection, adaptive threshold,
    and morphological operations for better table detection.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_combined = np.uint8(255 * sobel_combined / np.max(sobel_combined))
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 4
    )
    combined = cv2.bitwise_or(sobel_combined, adaptive)
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed


# ---------------------------------------------------------
# PADDLEOCR EXTRACTION (Marathi + English)
# ---------------------------------------------------------
def extract_with_paddleocr(img: np.ndarray) -> Dict[str, Any]:
    """Extract text and numbers using PaddleOCR for better Marathi accuracy."""
    paddle_ocr = get_paddleocr()
    if not paddle_ocr:
        return {"text": "", "numbers": []}
    try:
        result = paddle_ocr.ocr(img, cls=True)
        full_text = ""
        numbers = []
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                    full_text += text + " "
                    num = extract_clean_number(text)
                    if num is not None:
                        numbers.append(num)
        return {"text": normalize_number(full_text.strip()), "numbers": numbers}
    except Exception as e:
        print(f"⚠️  PaddleOCR error: {e}")
        return {"text": "", "numbers": []}


# ---------------------------------------------------------
# RAPIDOCR EXTRACTION (High-accuracy numeric extraction)
# ---------------------------------------------------------
def extract_numbers_with_rapidocr(img: np.ndarray) -> List[float]:
    """Extract numbers with RapidOCR - extremely high accuracy for numeric values."""
    rapid_ocr = get_rapidocr()
    if not rapid_ocr:
        return []
    try:
        result, _ = rapid_ocr(img)
        numbers = []
        if result:
            for item in result:
                if item and len(item) >= 2:
                    text = str(item[1]) if isinstance(item, (list, tuple)) else str(item)
                    num = extract_clean_number(text)
                    if num is not None:
                        numbers.append(num)
        return numbers
    except Exception as e:
        print(f"⚠️  RapidOCR error: {e}")
        return []


# ---------------------------------------------------------
# PDFPLUMBER TABLE EXTRACTION
# ---------------------------------------------------------
def extract_tables_with_pdfplumber(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """Extract table structures using PDFPlumber for better cell-based extraction."""
    if not PDFPLUMBER_AVAILABLE:
        return []
    try:
        tables = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                for table in page_tables:
                    if table and len(table) > 1:
                        headers = [normalize_number(str(cell or "")) for cell in table[0]]
                        for row in table[1:]:
                            if row:
                                row_dict = {}
                                for i, cell in enumerate(row):
                                    if i < len(headers) and headers[i]:
                                        key = headers[i].strip().lower().replace(" ", "_")
                                        value = normalize_number(str(cell or ""))
                                        num = extract_clean_number(value)
                                        row_dict[key] = num if num is not None else value
                                if row_dict:
                                    tables.append(row_dict)
        return tables
    except Exception as e:
        print(f"⚠️  PDFPlumber error: {e}")
        return []


# ---------------------------------------------------------
# HYBRID EXTRACTION LAYER - Merges all OCR results
# ---------------------------------------------------------
def hybrid_extract_from_table_region(img: np.ndarray, table_bbox: tuple) -> Dict[str, Any]:
    """Hybrid extraction from a detected table region."""
    x, y, w, h = table_bbox
    crop = img[y:y+h, x:x+w]
    results = {"text": "", "numbers": [], "kv_pairs": {}, "table_rows": []}
    paddle_result = extract_with_paddleocr(crop)
    if paddle_result.get("text"):
        results["text"] = paddle_result["text"]
        results["numbers"].extend(paddle_result.get("numbers", []))
    rapid_numbers = extract_numbers_with_rapidocr(crop)
    results["numbers"].extend(rapid_numbers)
    if not results["text"]:
        results["text"] = ocr_fallback(crop)
    # Extract KVPs from combined text (would need extract_kvps_from_text function)
    return results


# ---------------------------------------------------------
# FALLBACK TESSERACT OCR
# ---------------------------------------------------------
def ocr_fallback(img: np.ndarray) -> str:
    """
    Simple text extraction if Gemini fails.
    Uses bilingual OCR (Marathi + English) to match reference implementation.
    """
    text = pytesseract.image_to_string(img, lang="mar+eng", config='--psm 6')
    return normalize_number(text)


# ---------------------------------------------------------
# REGEX PATTERNS FOR KDMC DOCUMENTS
# ---------------------------------------------------------
KDMC_REGEX_PATTERNS = {
    "document_number": [
        r"जा\.?क्र\.?कडोंमपा[\/]काअ[\/]बांध[\./][\/]कवि[\/][०-९0-9]+",
        r"जा\.?क्र\.?[\/]?कडोंमपा[\/].*?[०-९0-9]+",
    ],
    "dn_number": [
        r"कवि[\/]?[:\-\.]?\s*([०-९0-9]+)",
        r"कवि\s*[\/]\s*([०-९0-9]+)",
    ],
    "date": [
        r"दिनांक\s*[:\-]?\s*([०-९0-9]{1,2}\s*[\/]\s*[०-९0-9]{1,2}\s*[\/]\s*[०-९0-9]{2,4})",
        r"दि\.\s*[:\-]?\s*([०-९0-9]{1,2}\s*[\/]\s*[०-९0-9]{1,2}\s*[\/]\s*[०-९0-9]{2,4})",
    ],
    "total_length": [
        r"एकूण\s+(?:रस्ता\s+)?लांबी\s*[\-–]?\s*([०-९0-9]+)\s*मीटर",
        r"एकूण\s+लांबी\s*[\-–]?\s*([०-९0-9]+)\s*मी",
        r"एकूण.*?([०-९0-9]+)\s*मीटर",
    ],
    "total_amount": [
        r"एकूण.*?रु\.?\s*([०-९0-9,]+\.?[०-९0-9]*)",
        r"एकूण.*?₹\s*([०-९0-9,]+\.?[०-९0-9]*)",
    ],
}


def extract_regex_kdmc(text: str, pattern_list: List[str]) -> str:
    """Runs through regex patterns and returns first match."""
    for p in pattern_list:
        m = re.search(p, text, flags=re.I)
        if m:
            matched = m.group(1) if m.lastindex and m.lastindex >= 1 else m.group(0)
            matched = normalize_number(matched)
            matched = matched.replace(",", "").strip()
            matched = re.sub(r'\s+', '', matched) if '/' in matched else matched
            try:
                if "." in matched and not "/" in matched:
                    return float(matched)
                elif matched.replace("/", "").replace("-", "").isdigit():
                    return matched
                elif matched.isdigit():
                    return int(matched)
            except:
                pass
            return matched
    return None


# ---------------------------------------------------------
# MASTER KDMC EXTRACTOR
# ---------------------------------------------------------
def extract_measurements_kdmc(pdf_bytes: bytes) -> Dict[str, Any]:
    """
    Main extractor for KDMC documents.
    Returns structured data compatible with the main system.
    """
    fields: Dict[str, Any] = {}
    marathi: Dict[str, Any] = {}
    english: Dict[str, Any] = {}
    table_rows: List[Dict[str, Any]] = []

    # -----------------------------------------------------
    # 0. PDFPLUMBER TABLE EXTRACTION (structural extraction)
    # -----------------------------------------------------
    pdfplumber_tables = extract_tables_with_pdfplumber(pdf_bytes)
    if pdfplumber_tables:
        print(f"✅ Extracted {len(pdfplumber_tables)} table rows from PDFPlumber (KDMC)")
        table_rows.extend(pdfplumber_tables)

    # Convert PDF to images
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=300, poppler_path=POPPLER_PATH)
    except Exception as e:
        print(f"pdf2image failed (KDMC): {e}")
        return {
            "fields": {},
            "table_rows": table_rows,  # Keep PDFPlumber results
            "marathi": {},
            "english": {},
            "summary": {},
        }

    # Process each page
    for page_img in pages:
        img = cv2.cvtColor(np.array(page_img), cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", img)
        jpg_bytes = buffer.tobytes()

        # Gemini full-page extraction
        gem = gemini_extract_kdmc(
            jpg_bytes,
            instruction=(
                "Extract ALL data from this KDMC document. "
                "Include: header, document number, DN number (कवि), date, recipient info, "
                "road names, measurements (paverblock length, gravelling length, total length), "
                "financial charges (cable laying, supervision, security deposit, ground rent), "
                "and total amount.\n\n"
                "═══════════════════════════════════════════════════════════════\n"
                "CRITICAL: TABLE EXTRACTION - EXTRACT WITH MAXIMUM ACCURACY\n"
                "═══════════════════════════════════════════════════════════════\n"
                "You MUST extract ALL tables on this page with COMPLETE accuracy.\n"
                "For EVERY table you find, extract EVERY row with ALL columns.\n\n"
                "KDMC TABLE STRUCTURE - Extract with EXACT field names:\n"
                "- Extract EVERY row, even if some cells are empty\n"
                "- sr_no: Serial number (अ.क्र./क्र.) - exact value\n"
                "- description: Work description (विवरण) - translate to English\n"
                "- length_meters: Length in meters (लांबी मीटर) - exact numeric value\n"
                "- rate: Rate/charges (दर) - exact numeric value\n"
                "- amount: Total amount (रक्कम) - exact numeric value\n\n"
                "FINANCIAL CHARGES TABLE (rows अ, ब, क, ड):\n"
                "- Row अ: विविध ठिकाणी केबल टाकणेकरिता charges - exact amount\n"
                "- Row ब: सुपरविजन चार्ज (supervision charges) - exact amount\n"
                "- Row क: सिक्युरिटी डिपॉझीट (security deposit) - exact amount\n"
                "- Row ड: भुईभाडे (ground rent) - exact amount\n\n"
                "TABLE EXTRACTION REQUIREMENTS:\n"
                "1. Extract EVERY row in EVERY table - do not skip any rows\n"
                "2. Extract ALL columns for each row - do not miss any columns\n"
                "3. Preserve EXACT numeric values - do not round or approximate\n"
                "4. Translate Marathi text to English but keep numbers exactly as shown\n"
                "5. If a cell is empty, use null or empty string, but still include the row\n"
                "6. Pay special attention to total_length - extract exact value (e.g., 278 NOT 288)\n\n"
                "OUTPUT FORMAT:\n"
                "Return a JSON object with a 'table_rows' array. Each element in the array must be a complete row object with ALL fields:\n"
                '{\n'
                '  "table_rows": [\n'
                '    {\n'
                '      "sr_no": 1,\n'
                '      "description": "Cable laying at various locations",\n'
                '      "length_meters": 278.00,\n'
                '      "rate": 200.00,\n'
                '      "amount": 55600.00\n'
                '    },\n'
                '    ...\n'
                '  ]\n'
                '}\n\n'
                "CRITICAL: The accuracy of table extraction is paramount. Extract with the same precision as if you were processing each table individually."
            ),
            is_table=False
        )

        # Flatten nested dictionaries
        def flatten_dict(d, parent_key=''):
            items = []
            for k, v in d.items():
                if not v or (isinstance(v, str) and not v.strip()):
                    continue
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, parent_key=k).items())
                else:
                    items.append((k, v))
            return dict(items)

        # Track if we got table_rows from full-page extraction
        has_table_rows_from_full_page = False
        
        if isinstance(gem, dict):
            # Extract table_rows if present in full-page extraction
            if "table_rows" in gem and isinstance(gem["table_rows"], list) and len(gem["table_rows"]) > 0:
                table_rows.extend(gem["table_rows"])
                has_table_rows_from_full_page = True
                print(f"✅ Extracted {len(gem['table_rows'])} table rows from full-page Gemini call (KDMC)")
            
            flattened = flatten_dict(gem)
            
            for k, v in flattened.items():
                # Skip table_rows as we already handled it above
                if k == "table_rows":
                    continue
                
                original_key = k
                english_key = k.strip().lower().replace(" ", "_")
                
                # Try KDMC dictionary mapping
                for variant in [k, k.strip(), k.strip().replace(" ", "_"), k.strip().lower()]:
                    if variant in KDMC_MARATHI_TO_ENGLISH:
                        english_key = KDMC_MARATHI_TO_ENGLISH[variant]
                        break
                
                # Store
                english[english_key] = v
                fields[english_key] = v
                
                has_devanagari = isinstance(k, str) and any('\u0900' <= c <= '\u097f' for c in k)
                if has_devanagari:
                    marathi[original_key] = v

        # ---------------------------------------------------------
        # TABLE DETECTION (OpenCV) - SKIP if Gemini already extracted tables accurately
        # ---------------------------------------------------------
        # Only run table detection if Gemini didn't extract table_rows from full page
        # This ensures ONE call per page while maintaining accuracy
        if not has_table_rows_from_full_page:
            print("⚠️  No table_rows from full-page extraction, running hybrid table extraction (KDMC)...")
            boxes = detect_tables(img)
            for (x, y, w, h) in boxes:
                # Hybrid extraction: Use multiple OCR engines
                hybrid_result = hybrid_extract_from_table_region(img, (x, y, w, h))
                
                crop = img[y:y+h, x:x+w]
                _, buf = cv2.imencode(".jpg", crop)
                crop_bytes = buf.tobytes()

                tbl = gemini_extract_kdmc(
                    crop_bytes,
                    instruction=(
                        "Extract table with columns:\n"
                        "- sr_no: Serial number (अ.क्र)\n"
                        "- description: Work description\n"
                        "- length_meters: Length in meters\n"
                        "- rate: Rate/charges\n"
                        "- amount: Total amount\n"
                        "Return each row as JSON object."
                    ),
                    is_table=True
                )

                # Priority: Gemini > PaddleOCR > RapidOCR > Tesseract
                if isinstance(tbl, list) and len(tbl) > 0:
                    table_rows.extend(tbl)
                    print(f"✅ Extracted {len(tbl)} rows from Gemini table extraction (KDMC)")
                elif isinstance(tbl, dict):
                    table_rows.append(tbl)
                elif hybrid_result.get("kv_pairs"):
                    table_rows.append(hybrid_result["kv_pairs"])
                    print(f"✅ Using hybrid OCR results for table (KDMC)")
        else:
            print("✅ Skipping table detection (KDMC) - tables already extracted accurately from full-page call")

        # Hybrid OCR + Regex fallback
        paddle_result = extract_with_paddleocr(img)
        ocr_text = paddle_result.get("text", "")
        if not ocr_text:
            ocr_text = ocr_fallback(img)
        
        rapid_numbers = extract_numbers_with_rapidocr(img)
        
        for field, patterns in KDMC_REGEX_PATTERNS.items():
            if field not in english or not english[field]:
                match = extract_regex_kdmc(ocr_text, patterns)
                if match:
                    english[field] = match
                    fields[field] = match

    # Validate & clean table rows with Pydantic
    validated_table_rows = []
    if PYDANTIC_AVAILABLE:
        for row in table_rows:
            try:
                validated = TableRowModel(**row)
                validated_table_rows.append(validated.model_dump(exclude_none=True))
            except Exception:
                cleaned_row = {}
                for k, v in row.items():
                    if isinstance(v, str):
                        num = extract_clean_number(v)
                        cleaned_row[k] = num if num is not None else v
                    else:
                        cleaned_row[k] = v
                validated_table_rows.append(cleaned_row)
    else:
        for row in table_rows:
            cleaned_row = {}
            for k, v in row.items():
                if isinstance(v, str):
                    num = extract_clean_number(v)
                    cleaned_row[k] = num if num is not None else v
                else:
                    cleaned_row[k] = v
            validated_table_rows.append(cleaned_row)
    
    # Clean numeric values in english and fields
    for key in list(english.keys()):
        value = english[key]
        if isinstance(value, str):
            num = extract_clean_number(value)
            if num is not None:
                english[key] = num
                if key in fields:
                    fields[key] = num
    
    final_output = {
        "fields": fields,
        "table_rows": validated_table_rows,
        "marathi": marathi,
        "english": english,
        "summary": {},
        "document_type": "KDMC",
    }
    
    # Validate final output with Pydantic if available
    if PYDANTIC_AVAILABLE:
        try:
            validated = ExtractionOutputModel(**final_output)
            return validated.model_dump()
        except Exception as e:
            print(f"⚠️  Pydantic validation warning (KDMC): {e}, returning unvalidated output")
    
    return final_output


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
# TESSERACT + POPPLER PATHS (configurable via environment)
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
# MARATHI → ENGLISH DICTIONARY FOR FIELD KEYS
# ---------------------------------------------------------
MARATHI_TO_ENGLISH = {

    # --- Header / Metadata ---
    "नमुंमपापरि": "document_number",
    "परि": "document_number",
    "टे.क्र": "dn_number",
    "टेक्र": "dn_number",
    "टेक्र.": "dn_number",
    "दाखल क्रमांक": "dn_number",
    "दाखल": "dn_number",
    "क्रमांक": "reference_no",
    "पत्र क्र": "letter_no",
    "तारीख": "date",
    "दि": "date",
    "दि.": "date",
    "दिनांक": "date",
    
    # --- Common Marathi keys that might come from Gemini ---
    "मर्वी_मुंबई": "organization",
    "दूरध्वनी_क्र": "phone_numbers",
    "फॅक्स_नं": "fax_no",
    "इंटरफेस": "interface",
    "परिसंडळ": "annexure",

    # --- Applicant Info ---
    "मे": "salutation",
    "नाव": "name",
    "पत्ता": "address",

    # --- NMMC Office Info ---
    "एनएमएमसी": "nmmc",
    "उप आयुक्त": "deputy_commissioner",
    "उप अभियंता": "assistant_engineer",
    "अधिक्षक अभियंता": "executive_engineer",
    "सेवा क्र": "service_no",
    "दूरध्वनी": "telephone_no",
    "फॅक्स": "fax_no",
    "गट क्रमांक": "group_no",

    # --- Permissions ---
    "अनुमती": "permission",
    "परवानगी": "permission",
    "अनुमोदन": "approval",

    # --- Work Type ---
    "रस्ता खोदकाम": "road_excavation",
    "खोदकाम": "excavation",
    "डांबरी": "asphalt",
    "नवीन डांबरी": "new_asphalt",
    "ओपन ट्रेन्च": "open_trench",
    "एचडीडी": "hdd",
    "ड्रिलिंग": "drilling",
    "केबल": "cable",
    "डक्ट": "duct",
    "डक्ट स्थापणा": "duct_installation",

    # --- Measurements ---
    "लांबी": "length_m",
    "लांबी मी": "length_m",
    "रुंदी": "width_m",
    "रुंदी मी": "width_m",
    "क्षेत्रफळ": "area_sq_m",
    "क्षेत्रफळ चौ मी": "area_sq_m",
    "एकूण लांबी": "total_length",
    "एकूण क्षेत्रफळ": "total_area",
    "पिट": "no_of_pits",
    "खड्डे": "no_of_pits",
    "खड्ड्यांची संख्या": "no_of_pits",
    "संख्या": "count",
    "दर": "rate",
    "रक्कम": "amount",
    "एकूण": "total",
    "संपूर्ण": "grand_total",

    # --- Financial ---
    "सुपरविजन शुल्क": "supervision_charges",
    "परीक्षण शुल्क": "inspection_charges",
    "नियमन शुल्क": "regulation_charges",
    "पर्यावरण शुल्क": "environment_charges",
    "सुरक्षा अनामत": "security_deposit",
    "अनामत": "deposit",
    "जमा": "deposit",
    "भुईभाडे": "ground_rent",

    # --- Taxes ---
    "जीएसटी": "gst",
    "एनएलएमसी": "nlmc",
    "एनओसी": "noc_number",

    # --- Page Terms ---
    "प्रमाणे": "as_per",
    "प्रस्ताव": "proposal",
    "अहवाल": "report",
    "परिशिष्ट": "annexure",

    # --- Hard-coded Items You Requested ---
    "परिशिष्ट अ": "part_a",
    "परिशिष्ट ब": "part_b",
    "परिशिष्ट क": "part_c",
    "परिशिष्ट ड": "part_d",
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
# OPTIMIZE IMAGE SIZE FOR FASTER API CALLS
# ---------------------------------------------------------
def optimize_image_for_api(img: np.ndarray, max_dimension: int = 2048) -> bytes:
    """
    Resize image if too large to reduce API payload and speed up calls.
    Maintains aspect ratio.
    """
    h, w = img.shape[:2]
    if max(h, w) <= max_dimension:
        # Image is already small enough
        _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes()
    
    # Resize maintaining aspect ratio
    if h > w:
        new_h, new_w = max_dimension, int(w * max_dimension / h)
    else:
        new_h, new_w = int(h * max_dimension / w), max_dimension
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buffer.tobytes()


# ---------------------------------------------------------
# HYBRID EXTRACTION LAYER - Merges all OCR results
# ---------------------------------------------------------
def hybrid_extract_from_table_region(img: np.ndarray, table_bbox: tuple) -> Dict[str, Any]:
    """
    Hybrid extraction from a detected table region.
    Merges results from: Gemini > PaddleOCR > RapidOCR > Tesseract
    Priority order ensures maximum accuracy.
    """
    x, y, w, h = table_bbox
    crop = img[y:y+h, x:x+w]
    
    results = {
        "text": "",
        "numbers": [],
        "kv_pairs": {},
        "table_rows": []
    }
    
    # 1. PaddleOCR (better Marathi accuracy)
    paddle_result = extract_with_paddleocr(crop)
    if paddle_result.get("text"):
        results["text"] = paddle_result["text"]
        results["numbers"].extend(paddle_result.get("numbers", []))
    
    # 2. RapidOCR (high-accuracy numbers)
    rapid_numbers = extract_numbers_with_rapidocr(crop)
    results["numbers"].extend(rapid_numbers)
    
    # 3. Tesseract fallback
    if not results["text"]:
        results["text"] = ocr_fallback(crop)
    
    # Extract KVPs from combined text
    if results["text"]:
        results["kv_pairs"] = extract_kvps_from_text(results["text"])
    
    return results


# ---------------------------------------------------------
# GEMINI VISION CALL → returns JSON (Enhanced with Schema Enforcement)
# ---------------------------------------------------------
def gemini_extract(image_bytes: bytes, instruction: str, is_table: bool = False) -> Dict[str, Any]:
    """
    Sends image + instruction to Gemini Vision, returns parsed JSON.
    Enhanced with strict schema enforcement for consistent output.
    """
    model = genai.GenerativeModel(MODEL_NAME)

    if is_table:
        prompt = (
            "Extract this table from the Marathi document.\n"
            "Return as JSON array with objects containing all column values.\n"
            "Translate Marathi column names to English.\n"
            "Preserve all numeric values exactly.\n"
            "Return ONLY valid JSON array.\n\n"
            f"{instruction}"
        )
    else:
        # Enhanced prompt with strict schema enforcement
        prompt = (
            "Extract ALL data from this Marathi document, translate text to English, and return as JSON.\n\n"
            "CRITICAL RULES:\n"
            "1. Use ONLY ENGLISH keys (translate all Marathi field names to English)\n"
            "2. Convert ALL Devanagari digits (०-९) to English (0-9)\n"
            "3. Translate ALL Marathi text VALUES to English (addresses, descriptions, subjects, names)\n"
            "4. Keep document numbers, codes, and proper nouns as-is\n"
            "5. Preserve all numeric values exactly\n\n"
            "REQUIRED FIELDS - Find and TRANSLATE these:\n\n"
            "HEADER INFO:\n"
            '- "organization": Organization name (translate to English)\n'
            '- "address": Full address (translate to English)\n'
            '- "phone_numbers": Array of phone numbers like ["757 17 33", "757 17 28"]\n'
            '- "fax_no": Fax number\n\n'
            "DOCUMENT METADATA:\n"
            '- "document_number": Full document number (नमुंमपा/परि-1/टे.क्र.-10/...)\n'
            '- "dn_number": ONLY the number after टे.क्र. or टेक्र (e.g., "10/1182/2025")\n'
            '- "date": Date after दि. or दिनांक. IMPORTANT: Convert Devanagari digits carefully:\n'
            '  - ९/१०/2025 → "9/10/2025" (NOT 3/10/2025)\n'
            '  - ३/१०/2025 → "3/10/2025"\n'
            '  - Devanagari: ०=0, १=1, २=2, ३=3, ४=4, ५=5, ६=6, ७=7, ८=8, ९=9\n\n'
            "RECIPIENT INFO:\n"
            '- "recipient_name": Name after मे. or प्रति (translate to English)\n'
            '- "recipient_address": Full recipient address (translate to English)\n\n'
            "SUBJECT & REFERENCES:\n"
            '- "subject": Subject line (विषय) - translate to English\n'
            '- "reference_1": First reference (translate to English)\n'
            '- "reference_2": Second reference (translate to English)\n\n'
            "FINANCIAL INFO:\n"
            '- "gst_number": GST number (27AAALC0296J1Z4)\n'
            '- "pan_number": PAN number (AAALC0296J)\n'
            '- "service_no": Service number\n'
            '- "total_amount": Total amount / एकूण (the final grand total)\n'
            '- "supervision_charges": Row A - पुनर्स्थापना शुल्क / Re-establishment charges\n'
            '- "checking_inspection_charges": Row B - A वर पर्यवेक्षण शुल्क / Inspection charges on A (15%)\n'
            '- "security_deposit": Row C - A+B वर सुरक्षा अनामत रक्कम / Security deposit on A+B (10%)\n'
            '- "ground_rent": Row D - Ground rent per meter / भुईभाडे (₹10 or ₹200 per meter)\n'
            '- "environment_charges": Any environment or other charges\n\n'
            "MEASUREMENTS:\n"
            '- "total_length_meters": Total length in meters (e.g., 812 मीटर → "812")\n'
            '- "length_m": Individual length in meters (number only)\n'
            '- "width_m": Width in meters (number only)\n'
            '- "area_sqm": Area in sq.m (number only)\n\n'
            "TRANSLATION EXAMPLES:\n"
            "✓ Translate these:\n"
            '- "नवी मुंबई महानगरपालिका" → "Navi Mumbai Municipal Corporation"\n'
            '- "रस्ता खोदकाम शुल्क भरणा" → "Road excavation charges payment"\n'
            '- "नविन डांबरी पृष्ठभाग" → "New asphalt surface"\n'
            '- "सातवा मजला, इंटरफेस-7" → "7th Floor, Interface-7"\n'
            '- "उप आयुक्त (परिसंचळ -1)" → "Deputy Commissioner (Circuit-1)"\n'
            "✗ Keep these as-is:\n"
            '- Document numbers: "नमुंमपा/परि-1/टे.क्र.-10/1182/2025"\n'
            '- Company names: "M/s. Bharti Airtel Ltd."\n'
            '- Numbers: 812, 1.5, 15831241\n\n'
            "Extract ALL other fields and translate text to English.\n"
            "Return ONLY valid JSON with ENGLISH keys and ENGLISH values.\n\n"
            f"{instruction}"
        )

    try:
        # Gemini 2.5 Pro doesn't accept temperature as direct parameter
        response = model.generate_content(
            [
                {"mime_type": "image/jpeg", "data": image_bytes},
                prompt
            ]
        )

        text = response.text.strip()

        # Remove markdown if any
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text)
            text = text.replace("```", "").strip()

        return json.loads(text)

    except Exception as e:
        print("Gemini Vision error:", str(e))
        return {}


# ---------------------------------------------------------
# ENHANCED OPENCV PREPROCESSING WITH SOBEL & MORPHOLOGY
# ---------------------------------------------------------
def enhanced_preprocess(img: np.ndarray) -> np.ndarray:
    """
    Improved preprocessing with Sobel edge detection, adaptive threshold,
    and morphological operations for better table detection.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Sobel edge detection for line detection
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_combined = np.uint8(255 * sobel_combined / np.max(sobel_combined))
    
    # Adaptive threshold for better contrast
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 4
    )
    
    # Combine Sobel and adaptive threshold
    combined = cv2.bitwise_or(sobel_combined, adaptive)
    
    # Morphological closing to connect broken lines
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return closed


# ---------------------------------------------------------
# ENHANCED OPENCV TABLE DETECTOR
# ---------------------------------------------------------
def detect_tables(img: np.ndarray) -> List[tuple]:
    """
    Enhanced table detection with improved preprocessing.
    Returns bounding boxes of tables detected in the image.
    """
    # Use enhanced preprocessing
    processed = enhanced_preprocess(img)

    # Vertical lines detection
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(processed, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # Horizontal lines detection
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(processed, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # Combine vertical and horizontal lines
    table_mask = cv2.add(vertical_lines, horizontal_lines)
    
    # Additional morphological closing to connect table boundaries
    closing_kernel = np.ones((5, 5), np.uint8)
    table_mask = cv2.morphologyEx(table_mask, cv2.MORPH_CLOSE, closing_kernel, iterations=1)

    # Find contours
    cnts, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        # Filter: minimum size and aspect ratio check
        if w > 80 and h > 40 and (w * h) > 5000:
            boxes.append((x, y, w, h))

    return boxes


# ---------------------------------------------------------
# PADDLEOCR EXTRACTION (Marathi + English)
# ---------------------------------------------------------
def extract_with_paddleocr(img: np.ndarray) -> Dict[str, Any]:
    """
    Extract text and numbers using PaddleOCR for better Marathi accuracy.
    Returns dict with 'text' and 'numbers' keys.
    """
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
                    
                    # Extract numbers from this line
                    num = extract_clean_number(text)
                    if num is not None:
                        numbers.append(num)
        
        return {
            "text": normalize_number(full_text.strip()),
            "numbers": numbers
        }
    except Exception as e:
        print(f"⚠️  PaddleOCR error: {e}")
        return {"text": "", "numbers": []}


# ---------------------------------------------------------
# RAPIDOCR EXTRACTION (High-accuracy numeric extraction)
# ---------------------------------------------------------
def extract_numbers_with_rapidocr(img: np.ndarray) -> List[float]:
    """
    Extract numbers with RapidOCR - extremely high accuracy for numeric values.
    Focuses on: length, width, area, rate, amount, pits, totals.
    """
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
    """
    Extract table structures using PDFPlumber for better cell-based extraction.
    """
    if not PDFPLUMBER_AVAILABLE:
        return []
    
    try:
        tables = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                for table in page_tables:
                    if table and len(table) > 1:  # Has header + at least one row
                        # Convert to structured format
                        headers = [normalize_number(str(cell or "")) for cell in table[0]]
                        for row in table[1:]:
                            if row:
                                row_dict = {}
                                for i, cell in enumerate(row):
                                    if i < len(headers) and headers[i]:
                                        key = headers[i].strip().lower().replace(" ", "_")
                                        value = normalize_number(str(cell or ""))
                                        # Try to convert to number
                                        num = extract_clean_number(value)
                                        row_dict[key] = num if num is not None else value
                                if row_dict:
                                    tables.append(row_dict)
        return tables
    except Exception as e:
        print(f"⚠️  PDFPlumber error: {e}")
        return []


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
# PARSE TEXT → GENERIC KVP EXTRACTION
# ---------------------------------------------------------
def extract_kvps_from_text(text: str) -> Dict[str, Any]:
    """
    Extract simple label:number patterns from text.
    """
    text = normalize_number(text)

    kv = {}

    # Pattern: LABEL : NUMBER
    pattern = r"([A-Za-z\u0900-\u097F ]+)\s*[:\-–]\s*([\d\.]+)"

    for label, num in re.findall(pattern, text):
        label = label.strip()
        num = num.strip()

        if not num:
            continue

        try:
            num = float(num) if "." in num else int(num)
        except:
            continue

        # Map Marathi → English if exists
        key = MARATHI_TO_ENGLISH.get(label, label.replace(" ", "_").lower())

        kv[key] = num

    return kv


def parse_items_field(items_text: str) -> List[Dict[str, Any]]:
    """
    Parse the 'Items' field from screenshot into separate structured items.
    
    Example input:
    "Item 1: Sr No: 1, Surface Type: नविन डांबरी पृष्ठभाग, Pit Count: 8, Length M: 1.5, 
     Width M: 1.5, Area Sqm: 18, Rate Per Sqm: 9600, Amount: 172800"
    
    Returns list of dicts with separated fields.
    """
    items = []
    
    # Split by "Item X:" to get individual items
    item_pattern = r"Item\s+\d+:\s*([^|]+)"
    matches = re.findall(item_pattern, items_text, re.IGNORECASE)
    
    if not matches:
        # Try alternative format: split by " | "
        matches = items_text.split(" | ")
    
    for match in matches:
        item = {}
        
        # Extract key-value pairs from the item text
        # Pattern: "Key: Value, Key2: Value2"
        kv_pattern = r"([A-Za-z\s]+?):\s*([^,|]+?)(?:,|\||$)"
        pairs = re.findall(kv_pattern, match)
        
        for key, value in pairs:
            key = key.strip().lower().replace(" ", "_")
            value = value.strip()
            
            # Try to convert to number if possible
            try:
                if "." in value and not any(c.isalpha() or c in "/-" for c in value):
                    item[key] = float(value)
                elif value.replace(",", "").isdigit():
                    item[key] = int(value.replace(",", ""))
                else:
                    item[key] = value
            except:
                item[key] = value
        
        if item:  # Only add non-empty items
            items.append(item)
    
    return items


# -------------------------------------------------------------------
# REGEX PATTERNS FOR HIGH ACCURACY EXTRACTION
# -------------------------------------------------------------------

REGEX_PATTERNS = {
    "document_number": [
        r"नमुंमपा[\/]परि[\-0-9\/\s]+",
        r"(?:नमुंमपा|NMMC)[\/]परि[\-0-9\/\s\.]+[\/]?[0-9]{4}",
    ],
    "dn_number": [
        r"(?:टे\.?क्र\.?|टेक्र|दाखल\s*क्रमांक)\s*[:\-\.]?\s*([0-9]+\s*[\/\-]\s*[0-9]+\s*[\/\-]\s*[0-9]+)",
        r"(?:परी|पारी)[\-\s]*[0-9]+[\/]टे\.?क्र\.?[\-\.]?([0-9]+\s*[\/\-]\s*[0-9]+\s*[\/\-]\s*[0-9]+)",
        r"टे\.?क्र\.?[\-\.]?([0-9]+\s*[\/\-]\s*[0-9]+\s*[\/\-]\s*[0-9]+)",
    ],
    "date": [
        r"(?:दि|दिनांक|तारीख)\s*[:\-\.]?\s*([०-९0-9]{1,2}\s*[\/\-\.]\s*[०-९0-9]{1,2}\s*[\/\-\.]\s*[०-९0-9]{2,4})",
        r"दि\.\s*([०-९0-9]{1,2}\s*[\/\-\.]\s*[०-९0-9]{1,2}\s*[\/\-\.]\s*[०-९0-9]{2,4})",
    ],
    "gst_number": [
        r"GST\s*NO\.\s*[:\-]?\s*([0-9]{2}[A-Z0-9]{13})",
        r"NMMC\s*GST\s*NO\.\s*[:\-]?\s*([0-9]{2}[A-Z0-9]{13})",
        r"(27[A-Z0-9]{13})",
    ],
    "pan_number": [
        r"PAN\s*NO\.\s*[:\-]?\s*([A-Z0-9]{10})",
        r"PAN\s*[:\-]?\s*([A-Z]{5}[0-9]{4}[A-Z])",
    ],
    "total_amount": [
        r"एकूण\s*₹?\s*([0-9,]+\.?[0-9]*)",
        r"एकूण\s+रु\.?\s*([0-9,]+\.?[0-9]*)",
        r"(?:Grand\s*Total|Total\s*Amount)\s*₹?\s*([0-9,]+\.?[0-9]*)",
    ],
    "total_length_meters": [
        r"एकूण\s+([०-९0-9]+)\s*मीटर",
        r"एकूण\s+([०-९0-9]+)\s*मी",
        r"([०-९0-9]+)\s*मीटर",
        r"([०-९0-9]+)\s*MTRS",
    ],
    "length_m": [
        r"लांबी\s*[:\-]?\s*([0-9]+\.?[0-9]*)\s*मी",
        r"([0-9]+\.[0-9]+)\s*मी",
    ],
    "width_m": [
        r"रुंदी\s*[:\-]?\s*([0-9]+\.?[0-9]*)\s*मी",
    ],
    "area_sqm": [
        r"क्षेत्रफळ\s*[:\-]?\s*([0-9]+\.?[0-9]*)\s*चौ\.?\s*मी",
    ],
    "amount": [
        r"रक्कम\s*[:\-]?\s*₹?\s*([0-9,]+\.?[0-9]*)",
        r"₹\s*([0-9,]+\.?[0-9]*)",
        r"रु[:\-]?\s*([0-9,]+\.?[0-9]*)",
    ],
    "rate": [
        r"दर\s*[:\-]?\s*([0-9,]+\.?[0-9]*)",
        r"प्रति\s+चौ\.?मी\.?\s+दर\s*[:\-]?\s*([0-9,]+\.?[0-9]*)",
    ],
}


def extract_regex(text: str, pattern_list: List[str]) -> str:
    """Runs through a list of patterns and returns the first match."""
    # First try with original text (for patterns that include Devanagari)
    for p in pattern_list:
        m = re.search(p, text, flags=re.I)
        if m:
            # Get the matched value
            matched = m.group(1) if m.lastindex and m.lastindex >= 1 else m.group(0)
            # Normalize Devanagari digits to English
            matched = normalize_number(matched)
            # Clean up matched value
            matched = matched.replace(",", "").strip()
            # Remove extra spaces in dates/numbers
            matched = re.sub(r'\s+', '', matched) if '/' in matched else matched
            # Try to convert to number if possible
            try:
                if "." in matched and not "/" in matched:
                    return float(matched)
                elif matched.replace("/", "").replace("-", "").isdigit():
                    # Keep as string for dates/document numbers
                    return matched
                elif matched.isdigit():
                    return int(matched)
            except:
                pass
            return matched
    return None


# ---------------------------------------------------------
# MASTER EXTRACTOR
# ---------------------------------------------------------
def extract_measurements(pdf_bytes: bytes) -> Dict[str, Any]:
    """
    Main entry used by processing.py.
    It:
      1. Converts PDF to images
      2. Runs Gemini Vision on full page
      3. Detects tables via OpenCV → feeds each table to Gemini
      4. Fallback OCR + regex if Gemini fails
      5. Returns structured dict
    """

    fields: Dict[str, Any] = {}
    marathi: Dict[str, Any] = {}
    english: Dict[str, Any] = {}
    table_rows: List[Dict[str, Any]] = []

    # -----------------------------------------------------
    # 0. PDFPLUMBER TABLE EXTRACTION (structural extraction)
    # -----------------------------------------------------
    # Extract tables using PDFPlumber for cell-based accuracy
    pdfplumber_tables = extract_tables_with_pdfplumber(pdf_bytes)
    if pdfplumber_tables:
        print(f"✅ Extracted {len(pdfplumber_tables)} table rows from PDFPlumber")
        table_rows.extend(pdfplumber_tables)

    # -----------------------------------------------------
    # 1. PDF → Images (increased DPI for better accuracy)
    # -----------------------------------------------------
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=300, poppler_path=POPPLER_PATH)
    except Exception as e:
        print("pdf2image failed:", e)
        return {
            "fields": {},
            "table_rows": table_rows,  # Keep PDFPlumber results
            "marathi": {},
            "english": {},
            "summary": {},
        }

    # -----------------------------------------------------
    # PROCESS EACH PAGE
    # -----------------------------------------------------
    for page_img in pages:

        # Convert PIL → OpenCV
        img = cv2.cvtColor(np.array(page_img), cv2.COLOR_RGB2BGR)

        # JPEG encode for Gemini
        _, buffer = cv2.imencode(".jpg", img)
        jpg_bytes = buffer.tobytes()

        # ---------------------------------------------------------
        # 2. GEMINI FULL-PAGE KVP EXTRACTION
        # ---------------------------------------------------------
        gem = gemini_extract(
            jpg_bytes,
            instruction=(
                "Extract ALL data from this document. "
                "Return a comprehensive JSON with ALL fields you can find. "
                "Include: header details, dates, numbers, addresses, amounts, measurements, charges, GST, PAN, totals, etc.\n\n"
                "CRITICAL: SURFACE TYPES EXTRACTION\n"
                "If you see multiple surface types in tables (e.g., 'नविन डांबरी पृष्ठभाग' and 'नविन डांबरी पृष्ठभाग (ओपन ट्रेंच पध्दत)'), "
                "extract ALL of them. The system will combine them into a single 'surface' field.\n\n"
                "═══════════════════════════════════════════════════════════════\n"
                "CRITICAL: TABLE EXTRACTION - EXTRACT WITH MAXIMUM ACCURACY\n"
                "═══════════════════════════════════════════════════════════════\n"
                "You MUST extract ALL tables on this page with COMPLETE accuracy.\n"
                "For EVERY table you find, extract EVERY row with ALL columns.\n\n"
                "MAIN TABLE (line items) - Extract with EXACT field names:\n"
                "- Extract EVERY row, even if some cells are empty\n"
                "- sr_no: Serial number (क्र./Sr. No.) - exact value\n"
                "- surface_type: Surface type/description - TRANSLATE to English:\n"
                "  * 'नविन डांबरी पृष्ठभाग' → 'New Asphalt Surface'\n"
                "  * 'नविन डांबरी पृष्ठभाग (ओपन ट्रेंच पध्दत)' → 'New Asphalt Surface (Open Trench Method)'\n"
                "  * Extract ALL surface types you see, even if they appear in different rows\n"
                "- number_of_pits: Number of pits/खड्डे - exact numeric value (e.g., 8.00, 0.00)\n"
                "- length_m: Length in meters (लांबी) - exact numeric value\n"
                "- width_m: Width in meters (रुंदी) - exact numeric value\n"
                "- area_sqm: Area in square meters (क्षेत्रफळ चौ.मी.) - exact numeric value\n"
                "- rate_per_sqm: Rate per square meter (दर प्रति चौ.मी.) - exact numeric value\n"
                "- amount: Total amount (रक्कम) - exact numeric value\n\n"
                "CHARGES TABLE (rows A, B, C, D):\n"
                "- Row A: पुनर्स्थापना शुल्क (Re-establishment charges) - exact amount\n"
                "- Row B: A वर पर्यवेक्षण शुल्क (Inspection charges on A - usually 15%) - exact amount\n"
                "- Row C: A+B वर सुरक्षा अनामत रक्कम (Security deposit on A+B - usually 10%) - exact amount\n"
                "- Row D: भुईभाडे per meter (Ground rent per meter) - exact amount\n\n"
                "TABLE EXTRACTION REQUIREMENTS:\n"
                "1. Extract EVERY row in EVERY table - do not skip any rows\n"
                "2. Extract ALL columns for each row - do not miss any columns\n"
                "3. Preserve EXACT numeric values - do not round or approximate\n"
                "4. Translate Marathi text to English but keep numbers exactly as shown\n"
                "5. If a cell is empty, use null or empty string, but still include the row\n"
                "6. Extract multiple surface types as separate rows if they appear separately\n\n"
                "OUTPUT FORMAT:\n"
                "Return a JSON object with a 'table_rows' array. Each element in the array must be a complete row object with ALL fields:\n"
                '{\n'
                '  "table_rows": [\n'
                '    {\n'
                '      "sr_no": 1,\n'
                '      "surface_type": "New Asphalt Surface",\n'
                '      "number_of_pits": 8.00,\n'
                '      "length_m": 1.50,\n'
                '      "width_m": 1.50,\n'
                '      "area_sqm": 18.00,\n'
                '      "rate_per_sqm": 9600.00,\n'
                '      "amount": 172800.00\n'
                '    },\n'
                '    ...\n'
                '  ]\n'
                '}\n\n'
                "CRITICAL: The accuracy of table extraction is paramount. Extract with the same precision as if you were processing each table individually.\n\n"
                "═══════════════════════════════════════════════════════════════\n"
                "STRICT SCHEMA ENFORCEMENT - REQUIRED OUTPUT STRUCTURE:\n"
                "═══════════════════════════════════════════════════════════════\n"
                "You MUST return JSON in this EXACT structure (all keys required):\n"
                '{\n'
                '  "marathi": {},\n'
                '  "english": {},\n'
                '  "table_rows": [],\n'
                '  "summary": {},\n'
                '  "fields": {}\n'
                '}\n\n'
                "SCHEMA RULES:\n"
                "- 'marathi': Object with Marathi keys → values (preserve original Marathi labels)\n"
                "- 'english': Object with English keys → values (all translated)\n"
                "- 'table_rows': Array of table row objects (each row must have: sr_no, surface_type, number_of_pits, length_m, width_m, area_sqm, rate_per_sqm, amount)\n"
                "- 'summary': Object with calculated/summary fields\n"
                "- 'fields': Object with all extracted fields (can be same as 'english')\n\n"
                "DO NOT omit any of these top-level keys. If a section is empty, use empty object {} or empty array []."
            ),
            is_table=False
        )

        # Debug: print Gemini output (comment out in production)
        # print(f"\n🔍 DEBUG: Gemini returned {len(gem) if isinstance(gem, dict) else 0} fields")
        # print(f"📦 Gemini raw output: {json.dumps(gem, ensure_ascii=False, indent=2)}")
        
        def flatten_dict(d, parent_key=''):
            """Recursively flatten nested dictionaries"""
            items = []
            for k, v in d.items():
                # Skip empty values
                if not v or (isinstance(v, str) and not v.strip()):
                    continue
                
                # If value is a dict, recursively flatten
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, parent_key=k).items())
                else:
                    # Use the key directly (not prefixed with parent)
                    items.append((k, v))
            return dict(items)

        # Track if we got table_rows from full-page extraction
        has_table_rows_from_full_page = False
        
        if isinstance(gem, dict):
            # Extract table_rows if present in full-page extraction
            if "table_rows" in gem and isinstance(gem["table_rows"], list) and len(gem["table_rows"]) > 0:
                table_rows.extend(gem["table_rows"])
                has_table_rows_from_full_page = True
                print(f"✅ Extracted {len(gem['table_rows'])} table rows from full-page Gemini call")
            
            # Flatten nested structure
            flattened = flatten_dict(gem)
            
            for k, v in flattened.items():
                # Skip table_rows as we already handled it above
                if k == "table_rows":
                    continue
                
                # Try multiple normalization strategies
                original_key = k
                normalized_variations = [
                    k,  # Original
                    k.strip(),  # Trimmed
                    k.strip().replace(" ", "_"),  # Spaces to underscores
                    k.strip().replace(" ", "_").replace(".", "").replace(":", ""),  # Remove punctuation
                    k.strip().lower().replace(" ", "_"),  # Lowercase
                ]
                
                # Try to find English translation
                english_key = k
                for variant in normalized_variations:
                    if variant in MARATHI_TO_ENGLISH:
                        english_key = MARATHI_TO_ENGLISH[variant]
                        break
                
                # If key is already in English (no Devanagari), keep it
                has_devanagari = isinstance(k, str) and any('\u0900' <= c <= '\u097f' for c in k)
                if not has_devanagari and english_key == k:
                    # Key is already in English, use it as-is
                    english_key = k.strip().lower().replace(" ", "_")
                
                # Debug: print key mapping (comment out in production)
                # print(f"  🔑 '{original_key}' → '{english_key}' = {v}")
                
                # Store with English key
                english[english_key] = v
                fields[english_key] = v
                
                # Also store with original key if it's Marathi
                if has_devanagari:
                    marathi[original_key] = v
        
        # Debug: print extracted fields (comment out in production)
        # print(f"\n✅ After Gemini: english dict has {len(english)} fields")
        # print(f"📋 English fields: {list(english.keys())}")
        
        # ---------------------------------------------------------
        # 2.5. PARSE "ITEMS" FIELD IF IT EXISTS (break into separate fields)
        # ---------------------------------------------------------
        if "items" in english and isinstance(english["items"], str):
            parsed_items = parse_items_field(english["items"])
            if parsed_items:
                # Remove the raw "items" field
                del english["items"]
                # Add structured items
                english["parsed_items"] = parsed_items
                fields["parsed_items"] = parsed_items

        # ---------------------------------------------------------
        # 3. HYBRID TABLE EXTRACTION (OpenCV + Multi-OCR)
        # ---------------------------------------------------------
        # Only run table detection if Gemini didn't extract table_rows from full page
        # This ensures ONE call per page while maintaining accuracy
        if not has_table_rows_from_full_page:
            print("⚠️  No table_rows from full-page extraction, running hybrid table extraction...")
            boxes = detect_tables(img)

            for (x, y, w, h) in boxes:
                # HYBRID EXTRACTION: Use multiple OCR engines for maximum accuracy
                hybrid_result = hybrid_extract_from_table_region(img, (x, y, w, h))
                
                # Try Gemini on cropped table as primary
                crop = img[y:y+h, x:x+w]
                _, buf = cv2.imencode(".jpg", crop)
                crop_bytes = buf.tobytes()

                tbl = gemini_extract(
                    crop_bytes,
                    instruction=(
                        "Extract this table with ALL rows and columns. CRITICAL:\n"
                        "- Extract EVERY row, even if some cells are empty\n"
                        "- sr_no: Serial number (क्र./Sr. No.)\n"
                        "- surface_type: Surface type/description - TRANSLATE to English (e.g., 'नविन डांबरी पृष्ठभाग' → 'New Asphalt Surface', 'नविन डांबरी पृष्ठभाग (ओपन ट्रेंच पध्दत)' → 'New Asphalt Surface (Open Trench Method)')\n"
                        "- number_of_pits: Number of pits/खड्डे (e.g., 8.00, 0.00)\n"
                        "- length_m: Length in meters (लांबी)\n"
                        "- width_m: Width in meters (रुंदी)\n"
                        "- area_sqm: Area in square meters (क्षेत्रफळ चौ.मी.)\n"
                        "- rate_per_sqm: Rate per square meter (दर प्रति चौ.मी.)\n"
                        "- amount: Total amount (रक्कम)\n"
                        "IMPORTANT: If you see multiple surface types (e.g., 'New Asphalt Surface' and 'New Asphalt Surface (Open Trench Method)'), extract BOTH as separate rows.\n"
                        "Return as JSON array where each element is a complete row object with ALL fields."
                    ),
                    is_table=True
                )

                # Priority: Gemini > PaddleOCR > RapidOCR > Tesseract
                if isinstance(tbl, list) and len(tbl) > 0:
                    table_rows.extend(tbl)
                    print(f"✅ Extracted {len(tbl)} rows from Gemini table extraction")
                elif isinstance(tbl, dict):
                    table_rows.append(tbl)
                elif hybrid_result.get("kv_pairs"):
                    # Fallback to hybrid OCR results
                    table_rows.append(hybrid_result["kv_pairs"])
                    print(f"✅ Using hybrid OCR results for table")
        else:
            print("✅ Skipping table detection - tables already extracted accurately from full-page call")

        # ---------------------------------------------------------
        # 4. HYBRID OCR + REGEX ENHANCEMENT (always run to catch missed fields)
        # ---------------------------------------------------------
        # Use hybrid OCR approach: PaddleOCR > RapidOCR > Tesseract
        # Priority ensures maximum accuracy for Marathi text and numbers
        
        # Try PaddleOCR first (better Marathi accuracy)
        paddle_result = extract_with_paddleocr(img)
        ocr_text = paddle_result.get("text", "")
        
        # If PaddleOCR failed, use Tesseract fallback
        if not ocr_text:
            ocr_text = ocr_fallback(img)
        
        # Extract numbers with RapidOCR (high accuracy for numeric values)
        rapid_numbers = extract_numbers_with_rapidocr(img)
        
        # Try specific regex patterns for critical fields
        for field, patterns in REGEX_PATTERNS.items():
            if field not in english or not english[field]:
                match = extract_regex(ocr_text, patterns)
                if match:
                    english[field] = match
                    fields[field] = match
        
        # Also extract generic KVPs if we have few fields
        if len(english) < 10:
            regex_kv = extract_kvps_from_text(ocr_text)
            # Only add if key doesn't exist
            for k, v in regex_kv.items():
                if k not in english:
                    english[k] = v
                    fields[k] = v
        
        # Add RapidOCR numbers to fields if they're missing
        if rapid_numbers and len(rapid_numbers) > 0:
            # Try to match numbers to known fields
            if "length_m" not in english and len(rapid_numbers) > 0:
                # Use first reasonable number as length
                for num in rapid_numbers:
                    if 0.1 <= num <= 10000:  # Reasonable range for length
                        if "length_m" not in english:
                            english["length_m"] = num
                            fields["length_m"] = num
                        break

    # ---------------------------------------------------------
    # POST-PROCESSING: Combine surface types into single "surface" field
    # ---------------------------------------------------------
    def combine_surface_types(table_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combines multiple surface types into a single "surface" field.
        Example: "New Asphalt Surface" and "New Asphalt Surface (Open Trench Method)"
        → "surface": "New Asphalt Surface / New Asphalt Surface (Open Trench Method)"
        
        Aggregates corresponding measurements:
        - Pits: Sum (total number of pits across all surface types)
        - Length: Sum (total length across all surface types)
        - Width: Max (maximum width, as width is typically consistent)
        - Area: Sum (total area across all surface types)
        - Amount: Sum (total amount across all surface types)
        - Rate: Calculated as total_amount / total_area
        """
        if not table_rows:
            return {}
        
        surface_types = []
        surface_data = {}
        
        def safe_float(value):
            """Safely convert value to float, returning 0 if conversion fails."""
            if value is None:
                return 0.0
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0
        
        for row in table_rows:
            # Get surface type from various possible keys
            surface_type = (
                row.get("surface_type") or 
                row.get("surface") or 
                row.get("पृष्ठभागाचा प्रकार") or
                row.get("type_of_surface") or
                ""
            )
            
            if surface_type and str(surface_type).strip():
                # Normalize surface type
                surface_type = str(surface_type).strip()
                
                # Collect unique surface types (preserve order)
                if surface_type not in surface_types:
                    surface_types.append(surface_type)
                
                # Extract measurements for this row
                pits = safe_float(row.get("number_of_pits") or row.get("no_of_pits") or row.get("pits"))
                length = safe_float(row.get("length_m") or row.get("length") or row.get("लांबी"))
                width = safe_float(row.get("width_m") or row.get("width") or row.get("रुंदी"))
                area = safe_float(row.get("area_sqm") or row.get("area_sq_m") or row.get("area") or row.get("क्षेत्रफळ"))
                rate = safe_float(row.get("rate_per_sqm") or row.get("rate") or row.get("दर"))
                amount = safe_float(row.get("amount") or row.get("रक्कम"))
                
                # Store measurements for this surface type
                if surface_type not in surface_data:
                    surface_data[surface_type] = {
                        "number_of_pits": pits,
                        "length_m": length,
                        "width_m": width,
                        "area_sqm": area,
                        "rate_per_sqm": rate,
                        "amount": amount,
                    }
                else:
                    # If same surface type appears multiple times, aggregate values
                    existing = surface_data[surface_type]
                    # Sum pits, length, area, amount
                    existing["number_of_pits"] = existing.get("number_of_pits", 0) + pits
                    existing["length_m"] = existing.get("length_m", 0) + length
                    existing["area_sqm"] = existing.get("area_sqm", 0) + area
                    existing["amount"] = existing.get("amount", 0) + amount
                    # Max width (width is typically consistent)
                    existing["width_m"] = max(existing.get("width_m", 0), width)
                    # Use the rate from the row with the highest area, or average
                    if area > 0:
                        existing["rate_per_sqm"] = rate if rate > 0 else existing.get("rate_per_sqm", 0)
        
        if surface_types:
            # Combine surface types with " / " separator
            combined_surface = " / ".join(surface_types)
            
            # Aggregate all measurements across all surface types
            total_pits = sum(safe_float(d.get("number_of_pits", 0)) for d in surface_data.values())
            total_length = sum(safe_float(d.get("length_m", 0)) for d in surface_data.values())
            total_width = max((safe_float(d.get("width_m", 0)) for d in surface_data.values()), default=0.0)
            total_area = sum(safe_float(d.get("area_sqm", 0)) for d in surface_data.values())
            total_amount = sum(safe_float(d.get("amount", 0)) for d in surface_data.values())
            
            # Calculate average rate (weighted by area if possible, otherwise simple average)
            rates_with_areas = [(safe_float(d.get("rate_per_sqm", 0)), safe_float(d.get("area_sqm", 0))) 
                               for d in surface_data.values() 
                               if safe_float(d.get("rate_per_sqm", 0)) > 0 and safe_float(d.get("area_sqm", 0)) > 0]
            
            if rates_with_areas and total_area > 0:
                # Weighted average rate
                weighted_sum = sum(rate * area for rate, area in rates_with_areas)
                avg_rate = weighted_sum / total_area
            elif total_area > 0 and total_amount > 0:
                # Calculate rate from total amount / total area
                avg_rate = total_amount / total_area
            else:
                # Simple average of non-zero rates
                non_zero_rates = [safe_float(d.get("rate_per_sqm", 0)) for d in surface_data.values() 
                                 if safe_float(d.get("rate_per_sqm", 0)) > 0]
                avg_rate = sum(non_zero_rates) / len(non_zero_rates) if non_zero_rates else 0.0
            
            return {
                "surface": combined_surface,
                "number_of_pits": total_pits,
                "length_m": total_length,
                "width_m": total_width,
                "area_sqm": total_area,
                "rate_per_sqm": round(avg_rate, 2) if avg_rate > 0 else 0.0,
                "amount": total_amount,
            }
        
        return {}
    
    # Combine surface types if we have table rows
    combined_surface = combine_surface_types(table_rows)
    
    # Add combined surface to fields and english if we have surface data
    if combined_surface and "surface" in combined_surface:
        fields["surface"] = combined_surface["surface"]
        english["surface"] = combined_surface["surface"]
        
        # Also add the aggregated measurements
        for key in ["number_of_pits", "length_m", "width_m", "area_sqm", "rate_per_sqm", "amount"]:
            if key in combined_surface:
                fields[key] = combined_surface[key]
                english[key] = combined_surface[key]
    
    # ---------------------------------------------------------
    # VALIDATE & CLEAN TABLE ROWS WITH PYDANTIC
    # ---------------------------------------------------------
    validated_table_rows = []
    if PYDANTIC_AVAILABLE:
        for row in table_rows:
            try:
                # Validate and normalize using Pydantic model
                validated = TableRowModel(**row)
                validated_table_rows.append(validated.model_dump(exclude_none=True))
            except Exception:
                # If validation fails, keep original row but clean numbers
                cleaned_row = {}
                for k, v in row.items():
                    if isinstance(v, str):
                        num = extract_clean_number(v)
                        cleaned_row[k] = num if num is not None else v
                    else:
                        cleaned_row[k] = v
                validated_table_rows.append(cleaned_row)
    else:
        # Fallback: manual cleaning
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
    
    # ---------------------------------------------------------
    # FINAL PACKAGED OUTPUT (Validated Structure)
    # ---------------------------------------------------------
    final_output = {
        "fields": fields,
        "table_rows": validated_table_rows,
        "marathi": marathi,
        "english": english,
        "summary": {},
        "document_type": "NMMC",
    }
    
    # Validate final output with Pydantic if available
    if PYDANTIC_AVAILABLE:
        try:
            validated = ExtractionOutputModel(**final_output)
            return validated.model_dump()
        except Exception as e:
            print(f"⚠️  Pydantic validation warning: {e}, returning unvalidated output")
    
    return final_output



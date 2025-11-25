"""
Auto-detect document type (NMMC vs KDMC) and route to appropriate extractor
"""
import re
from typing import Dict, Any, Literal
import pytesseract
import numpy as np
from pdf2image import convert_from_bytes
import os

TESSERACT_PATH = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
POPPLER_PATH = os.getenv("POPPLER_PATH", r"C:\poppler\poppler-25.07.0\Library\bin")

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


DocumentType = Literal["NMMC", "KDMC", "UNKNOWN"]


def detect_document_type(pdf_bytes: bytes) -> DocumentType:
    """
    Analyze the first page of PDF and determine if it's NMMC or KDMC format.
    
    Returns:
        "NMMC" - Navi Mumbai Municipal Corporation
        "KDMC" - Kalyan-Dombivli Municipal Corporation  
        "UNKNOWN" - Cannot determine
    """
    try:
        # Convert first page only
        pages = convert_from_bytes(pdf_bytes, dpi=200, poppler_path=POPPLER_PATH, first_page=1, last_page=1)
        if not pages:
            return "UNKNOWN"
        
        # OCR the first page
        img = np.array(pages[0])
        text = pytesseract.image_to_string(img, lang="mar+eng")
        
        # Scoring system
        nmmc_score = 0
        kdmc_score = 0
        
        # NMMC indicators
        nmmc_patterns = [
            r"नवी\s*मुंबई\s*महानगरपालिका",
            r"नमुंमपा",
            r"NMMC",
            r"Navi\s*Mumbai",
            r"टे\.?क्र\.?",
            r"परि[\-\s]*1",
            r"बेलापूर",
        ]
        
        # KDMC indicators
        kdmc_patterns = [
            r"कल्याण\s*डोंबिवली\s*महानगरपालिका",
            r"कल्याण\s*डोंबिवली",
            r"जा\.?क्र\.?कडोंमपा",
            r"कडोंमपा",
            r"KDMC",
            r"Kalyan",
            r"Dombivli",
            r"कवि",
            r"बांधकाम.*कल्याण",
        ]
        
        # Count NMMC matches
        for pattern in nmmc_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                nmmc_score += 1
        
        # Count KDMC matches
        for pattern in kdmc_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                kdmc_score += 1
        
        # Decision logic
        if kdmc_score > nmmc_score and kdmc_score >= 2:
            return "KDMC"
        elif nmmc_score > kdmc_score and nmmc_score >= 2:
            return "NMMC"
        elif kdmc_score == nmmc_score:
            # Tie-breaker: look for unique identifiers
            if re.search(r"जा\.?क्र\.?कडोंमपा", text):
                return "KDMC"
            elif re.search(r"नमुंमपा.*टे\.?क्र", text):
                return "NMMC"
        
        return "UNKNOWN"
        
    except Exception as e:
        print(f"Document detection error: {e}")
        return "UNKNOWN"


def extract_with_auto_detection(pdf_bytes: bytes) -> Dict[str, Any]:
    """
    Auto-detect document type and use appropriate extractor.
    
    Returns:
        Extraction result with 'document_type' field added
    """
    from .measurement_extractor import extract_measurements as extract_nmmc
    from .measurement_extractor_kdmc import extract_measurements_kdmc
    
    doc_type = detect_document_type(pdf_bytes)
    
    print(f"📄 Detected document type: {doc_type}")
    
    if doc_type == "KDMC":
        result = extract_measurements_kdmc(pdf_bytes)
        result["document_type"] = "KDMC"
        result["detector_confidence"] = "HIGH"
    elif doc_type == "NMMC":
        result = extract_nmmc(pdf_bytes)
        result["document_type"] = "NMMC"
        result["detector_confidence"] = "HIGH"
    else:
        # Default to NMMC if uncertain
        print("⚠ Unknown document type, defaulting to NMMC extractor")
        result = extract_nmmc(pdf_bytes)
        result["document_type"] = "UNKNOWN"
        result["detector_confidence"] = "LOW"
    
    return result


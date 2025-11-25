from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import os, numpy as np, cv2, pytesseract
from pdf2image import convert_from_bytes
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_DPI = int(os.getenv("OCR_DPI", "300"))


def _configure_tesseract():
    custom = os.getenv("TESSERACT_CMD")
    if custom:
        pytesseract.pytesseract.tesseract_cmd = custom


def _preprocess(img: np.ndarray) -> np.ndarray:
    """Simple preprocessing - just normalize the image."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Simple normalization - let Tesseract handle the rest
    return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)


def _poppler_path() -> Optional[str]:
    return os.getenv("POPPLER_PATH")




def extract_marathi_text(pdf_bytes: bytes, debug_dir: Path | None = None, dpi: int = DEFAULT_DPI) -> List[dict]:
    _configure_tesseract()
    pp = _poppler_path()
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, poppler_path=pp) if pp else convert_from_bytes(pdf_bytes, dpi=dpi)
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
    out: List[dict] = []

    def run_ocr(idx_pg):
        i, pg = idx_pg
        np_img = np.array(pg)
        # Simple, reliable config: OEM 1 (LSTM), PSM 6 (single uniform block)
        cfg = os.getenv("TESSERACT_CONFIG", "--oem 1 --psm 6")
        langs = os.getenv("TESSERACT_LANGS", "mar+eng")
        try:
            text = pytesseract.image_to_string(_preprocess(np_img), lang=langs, config=cfg).strip()
        except Exception:
            # Fallback to English only
            try:
                text = pytesseract.image_to_string(_preprocess(np_img), lang="mar", config=cfg).strip()
            except Exception:
                text = ""
        if debug_dir:
            (debug_dir / f"ocr_page_{i}.txt").write_text(text, encoding="utf-8")
        return {"page_number": i, "text": text}

    # Parallelize per-page OCR (Tesseract runs as a separate process, threading helps)
    with ThreadPoolExecutor(max_workers=min(4, (os.cpu_count() or 2))) as ex:
        futures = [ex.submit(run_ocr, (i, pg)) for i, pg in enumerate(pages, start=1)]
        for fut in as_completed(futures):
            out.append(fut.result())

    # Keep page order
    out.sort(key=lambda r: r["page_number"])
    return out

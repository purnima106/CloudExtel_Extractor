from __future__ import annotations
from typing import List
from io import BytesIO
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTAnno


def extract_text_per_page(pdf_bytes: bytes) -> List[str]:
    """
    Extract text per page using pdfminer, preserving basic layout.
    Returns a list of page strings. Empty string if page has no text layer.
    """
    results: List[str] = []
    for page_layout in extract_pages(BytesIO(pdf_bytes)):
        lines: List[str] = []
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                lines.append(element.get_text())
        page_text = "".join(lines).strip()
        results.append(page_text)
    return results



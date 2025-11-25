from __future__ import annotations
from pathlib import Path
from typing import Iterable
import json
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.pdfbase import pdfmetrics
from .fonts import resolve_marathi_font, resolve_english_font


def _dominant_script(text: str) -> str:
    devanagari = sum(1 for ch in text if "\u0900" <= ch <= "\u097f")
    latin = sum(1 for ch in text if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
    digits = sum(1 for ch in text if "0" <= ch <= "9")
    if devanagari >= max(latin, 1) and devanagari >= digits:
        return "devanagari"
    return "latin"


def write_json(job_dir: Path, results: dict) -> Path:
    """
    Write structured measurement data to JSON.
    results should have: extracted_measurements (dict) and page_wise_measurements (list)
    """
    dest = job_dir / "result.json"
    # Write the full structured data
    dest.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


def write_excel(job_dir: Path, results: dict) -> Path:
    """
    Write structured measurement data to Excel.
    Creates multiple sheets: Marathi, English, Table Rows, Summary.
    """
    dest = job_dir / "result.xlsx"
    
    with pd.ExcelWriter(dest, engine='openpyxl') as writer:
        # Sheet 1: Marathi Measurements
        marathi = results.get("marathi_measurements", {})
        if marathi:
            marathi_df = pd.DataFrame([
                {"Marathi_Key": k, "Value": v} for k, v in marathi.items()
            ])
        else:
            marathi_df = pd.DataFrame([{"Marathi_Key": "No Marathi measurements found", "Value": ""}])
        marathi_df.to_excel(writer, sheet_name='Marathi', index=False)
        
        # Sheet 2: English Measurements
        english = results.get("english_measurements", {})
        if english:
            english_df = pd.DataFrame([
                {"English_Key": k, "Value": v} for k, v in english.items()
            ])
        else:
            english_df = pd.DataFrame([{"English_Key": "No English measurements found", "Value": ""}])
        english_df.to_excel(writer, sheet_name='English', index=False)
        
        # Sheet 3: Table Rows (Complete table data)
        table_rows = results.get("table_rows", [])
        if table_rows:
            table_df = pd.DataFrame(table_rows)
        else:
            table_df = pd.DataFrame([{"message": "No table data found"}])
        table_df.to_excel(writer, sheet_name='Table_Rows', index=False)
        
        # Sheet 4: Summary
        summary = results.get("summary", {})
        if summary:
            summary_df = pd.DataFrame([
                {"Item": k, "Value": v} for k, v in summary.items()
            ])
        else:
            # Fallback to extracted_measurements if no summary
            measurements = results.get("extracted_measurements", {})
            if measurements:
                summary_df = pd.DataFrame([
                    {"Key": k, "Value": v} for k, v in measurements.items()
                ])
            else:
                summary_df = pd.DataFrame([{"Item": "No summary found", "Value": ""}])
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    return dest


def _register_fonts():
    """Ensure fonts are registered before creating styles."""
    marathi_font = resolve_marathi_font()
    english_font = resolve_english_font()
    return marathi_font, english_font


def _create_styles(marathi_font: str, english_font: str):
    """Create professional paragraph styles for the PDF."""
    styles = getSampleStyleSheet()
    
    # Verify fonts are registered
    registered_fonts = pdfmetrics.getRegisteredFontNames()
    
    # Title style
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=16,
        textColor=HexColor("#1e40af"),
        spaceAfter=12,
        fontName="Helvetica-Bold",
        alignment=1,  # Center
    )
    
    # Page header style
    page_header_style = ParagraphStyle(
        "PageHeader",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=HexColor("#1e3a8a"),
        spaceAfter=8,
        spaceBefore=12,
        fontName="Helvetica-Bold",
        borderWidth=1,
        borderColor=HexColor("#3b82f6"),
        borderPadding=8,
        backColor=HexColor("#eff6ff"),
    )
    
    # Section header style
    section_header_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading3"],
        fontSize=12,
        textColor=HexColor("#1e40af"),
        spaceAfter=6,
        spaceBefore=10,
        fontName="Helvetica-Bold",
    )
    
    # Original text style - use Marathi font if available, otherwise fallback
    original_font = marathi_font if marathi_font in registered_fonts else "Helvetica"
    original_style = ParagraphStyle(
        "OriginalText",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=black,
        fontName=original_font,
        spaceAfter=6,
        leftIndent=0,
        rightIndent=0,
        alignment=0,  # Left
        encoding="utf-8",
    )
    
    # English text style - use English font if available
    eng_font = english_font if english_font in registered_fonts else "Helvetica"
    english_style = ParagraphStyle(
        "EnglishText",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=HexColor("#1f2937"),
        fontName=eng_font,
        spaceAfter=6,
        leftIndent=0,
        rightIndent=0,
        alignment=0,
        encoding="utf-8",
    )
    
    # Language tag style
    lang_tag_style = ParagraphStyle(
        "LangTag",
        parent=styles["Normal"],
        fontSize=9,
        textColor=HexColor("#64748b"),
        fontName="Helvetica",
        spaceAfter=4,
    )
    
    return {
        "title": title_style,
        "page_header": page_header_style,
        "section_header": section_header_style,
        "original": original_style,
        "english": english_style,
        "lang_tag": lang_tag_style,
        "marathi_font": original_font,
        "english_font": eng_font,
    }


def _escape_text(text: str) -> str:
    """Escape special characters for ReportLab Paragraph while preserving Unicode."""
    if not text:
        return ""
    # Escape HTML entities but preserve Unicode characters
    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )
    return escaped


def _get_text_style(text: str, styles: dict, detected_lang: str = "") -> ParagraphStyle:
    """Determine the appropriate style based on text content and detected language."""
    # Check if text contains Devanagari characters
    has_devanagari = any("\u0900" <= ch <= "\u097f" for ch in text)
    
    # Use Marathi font if Devanagari is detected, otherwise use English font
    if has_devanagari:
        # Create a style with Marathi font
        base_style = styles["original"]
        marathi_style = ParagraphStyle(
            "DynamicMarathi",
            parent=base_style,
            fontName=styles.get("marathi_font", "Helvetica"),
        )
        return marathi_style
    else:
        # Use English font for non-Devanagari text
        base_style = styles["english"]
        english_style = ParagraphStyle(
            "DynamicEnglish",
            parent=base_style,
            fontName=styles.get("english_font", "Helvetica"),
        )
        return english_style


def write_pdf(job_dir: Path, results: dict) -> Path:
    """
    Write structured measurement data to PDF.
    Shows summary table and page-wise breakdown.
    """
    dest = job_dir / "result.pdf"
    marathi_font, english_font = _register_fonts()
    styles = _create_styles(marathi_font, english_font)
    
    doc = SimpleDocTemplate(
        str(dest),
        pagesize=A4,
        rightMargin=inch,
        leftMargin=inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    
    story = []
    
    # Document title
    filename = results.get("filename", "Unknown")
    story.append(Paragraph(f"Measurement Extraction Report: {filename}", styles["title"]))
    story.append(Spacer(1, 0.2 * inch))
    
    # Summary section
    measurements = results.get("extracted_measurements", {})
    if measurements:
        story.append(Paragraph("Extracted Measurements (Summary)", styles["section_header"]))
        # Create table-like display
        table_html = "<table border='1' cellpadding='5'><tr><th><b>Key</b></th><th><b>Value</b></th></tr>"
        for k, v in sorted(measurements.items()):
            table_html += f"<tr><td>{_escape_text(str(k))}</td><td>{_escape_text(str(v))}</td></tr>"
        table_html += "</table>"
        story.append(Paragraph(table_html, styles["english"]))
        story.append(Spacer(1, 0.3 * inch))
    else:
        story.append(Paragraph("No measurements extracted.", styles["english"]))
        story.append(Spacer(1, 0.2 * inch))
    
    # Page-wise breakdown
    page_data = results.get("page_wise_measurements", [])
    if page_data:
        story.append(PageBreak())
        story.append(Paragraph("Page-wise Breakdown", styles["section_header"]))
        story.append(Spacer(1, 0.15 * inch))
        
        for page_info in page_data:
            page_num = page_info.get("page_number", 0)
            page_meas = page_info.get("measurements", {})
            
            story.append(Paragraph(f"Page {page_num}", styles["page_header"]))
            story.append(Spacer(1, 0.1 * inch))
            
            if page_meas:
                for k, v in sorted(page_meas.items()):
                    kv_text = f"<b>{_escape_text(str(k))}</b>: {_escape_text(str(v))}"
                    story.append(Paragraph(kv_text, styles["english"]))
            else:
                story.append(Paragraph("<i>No measurements on this page</i>", styles["lang_tag"]))
            
            story.append(Spacer(1, 0.15 * inch))
    
    doc.build(story)
    return dest

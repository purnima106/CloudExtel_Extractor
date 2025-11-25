from __future__ import annotations
import os
import platform
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Tuple, Optional
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

MARATHI_FONT_ENV = "MARATHI_FONT_PATH"
ENGLISH_FONT_ENV = "ENGLISH_FONT_PATH"


def _candidate_paths() -> Iterable[Tuple[str, Path]]:
    # 1. Explicit override via environment variable
    env_path = os.getenv(MARATHI_FONT_ENV)
    if env_path:
        p = Path(env_path).expanduser()
        yield (p.stem, p)

    # 2. Project bundled fonts (if added later)
    bundled = Path(__file__).resolve().parent / "fonts"
    if bundled.exists():
        for path in bundled.glob("*.ttf"):
            yield (path.stem, path)

    # 3. Common system fonts with Devanagari support
    system_name = platform.system()
    if system_name == "Windows":
        windir = Path(os.getenv("WINDIR", "C:/Windows"))
        candidates = {
            windir / "Fonts" / "Nirmala.ttf",
            windir / "Fonts" / "Nirmala-Regular.ttf",
            windir / "Fonts" / "NirmalaUI.ttf",
            windir / "Fonts" / "Mangal.ttf",
            windir / "Fonts" / "MANGAL.TTF",
        }
        for path in candidates:
            yield (path.stem, path)
    else:
        # Linux / macOS common paths
        candidates = {
            Path("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"),
            Path("/usr/share/fonts/truetype/noto/NotoSerifDevanagari-Regular.ttf"),
            Path("/Library/Fonts/Nirmala.ttf"),
            Path("/Library/Fonts/NotoSansDevanagari-Regular.ttf"),
        }
        for path in candidates:
            yield (path.stem, path)


def _english_candidate_paths() -> Iterable[Tuple[str, Path]]:
    env_path = os.getenv(ENGLISH_FONT_ENV)
    if env_path:
        p = Path(env_path).expanduser()
        yield (p.stem, p)

    system_name = platform.system()
    if system_name == "Windows":
        windir = Path(os.getenv("WINDIR", "C:/Windows"))
        candidates = {
            windir / "Fonts" / "Arial.ttf",
            windir / "Fonts" / "Calibri.ttf",
            windir / "Fonts" / "Times.ttf",
        }
    else:
        candidates = {
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
            Path("/Library/Fonts/Arial.ttf"),
        }
    for path in candidates:
        yield (path.stem, path)


@lru_cache(maxsize=1)
def resolve_marathi_font(default: str = "Helvetica") -> str:
    """
    Register and return a font name capable of rendering Marathi script.
    Returns the default font if no suitable font is available.
    Tries TTF fonts first, then falls back to UnicodeCIDFont for better Devanagari support.
    """
    # First, try to find and register a TTF font with Devanagari support
    for name, path in _candidate_paths():
        try:
            if path.exists():
                font_name = name.replace(" ", "_").replace("-", "_")
                if font_name not in pdfmetrics.getRegisteredFontNames():
                    # Register TTF font with Unicode support
                    pdfmetrics.registerFont(TTFont(font_name, str(path), subfontIndex=0))
                    # Verify the font was registered
                    if font_name in pdfmetrics.getRegisteredFontNames():
                        return font_name
        except Exception as e:
            # Log but continue trying other fonts
            continue
    
    # Fallback: Try UnicodeCIDFont for better Devanagari support
    # These are built-in fonts that support Unicode better
    cid_fonts = ["HeiseiMin-W3", "HeiseiKakuGo-W5", "MSung-Light", "STSong-Light"]
    for cid_font in cid_fonts:
        try:
            if cid_font not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(UnicodeCIDFont(cid_font))
            if cid_font in pdfmetrics.getRegisteredFontNames():
                return cid_font
        except Exception:
            continue
    
    return default


@lru_cache(maxsize=1)
def resolve_english_font(default: str = "Helvetica") -> str:
    """
    Register and return a Latin font (for English text) with full ASCII coverage.
    """
    for name, path in _english_candidate_paths():
        try:
            if path.exists():
                font_name = name.replace(" ", "_").replace("-", "_")
                if font_name not in pdfmetrics.getRegisteredFontNames():
                    pdfmetrics.registerFont(TTFont(font_name, str(path), subfontIndex=0))
                    if font_name in pdfmetrics.getRegisteredFontNames():
                        return font_name
        except Exception:
            continue
    return default


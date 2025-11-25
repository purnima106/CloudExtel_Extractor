from __future__ import annotations
import asyncio
import os
from functools import lru_cache
from typing import Iterable, Optional

import httpx
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from deep_translator import GoogleTranslator

DEFAULT_LIBRE_URL = os.getenv("LIBRE_TRANSLATE_URL", "https://libretranslate.com")
LIBRE_API_KEY = os.getenv("LIBRE_TRANSLATE_API_KEY")
HF_MODEL_NAME = os.getenv("HF_MARATHI_EN_MODEL", "Helsinki-NLP/opus-mt-mr-en")
HF_MODEL_ENABLED = os.getenv("DISABLE_HF_TRANSLATION", "0") not in {"1", "true", "TRUE"}
GOOGLE_TRANSLATE_ENABLED = os.getenv("DISABLE_GOOGLE_TRANSLATION", "0") not in {"1", "true", "TRUE"}

LANG_ALIASES = {
    "mr": "mr",
    "mar": "mr",
    "marathi": "mr",
    "en": "en",
    "eng": "en",
    "english": "en",
    "auto": "auto",
}


def _normalize_lang(code: Optional[str], default: str) -> str:
    if not code:
        return default
    return LANG_ALIASES.get(code.lower(), code.lower())


def _chunk_text(text: str, limit: int = 4500) -> Iterable[str]:
    """
    Yield chunks of text under the provided character limit.
    Keeps paragraph boundaries when possible.
    """
    if len(text) <= limit:
        yield text
        return

    current = []
    total = 0
    for paragraph in text.split("\n"):
        if total and total + len(paragraph) + 1 > limit:
            yield "\n".join(current).strip()
            current = []
            total = 0
        current.append(paragraph)
        total += len(paragraph) + 1

    if current:
        yield "\n".join(current).strip()


def _translate_with_google_sync(text: str, source: str, target: str) -> str:
    if not GOOGLE_TRANSLATE_ENABLED:
        raise RuntimeError("Google translation disabled via env flag.")

    normalized_source = _normalize_lang(source, "auto")
    normalized_target = _normalize_lang(target, "en")
    translator = GoogleTranslator(source=normalized_source, target=normalized_target)
    translated_chunks = []
    for chunk in _chunk_text(text):
        translated = translator.translate(chunk)
        if translated:
            translated_chunks.append(translated)
    return "\n".join(translated_chunks)


async def _translate_with_google(text: str, source: str, target: str) -> str:
    return await asyncio.to_thread(_translate_with_google_sync, text, source, target)


async def _translate_with_libre(text: str, source: str, target: str) -> Optional[str]:
    url = DEFAULT_LIBRE_URL.rstrip("/") + "/translate"
    payload = {
        "q": text,
        "source": source,
        "target": target,
        "format": "text",
    }
    if LIBRE_API_KEY:
        payload["api_key"] = LIBRE_API_KEY
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data.get("translatedText") or data.get("translated_text")


@lru_cache(maxsize=1)
def _load_hf_model():
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


def _translate_with_hf_sync(text: str) -> str:
    tokenizer, model, device = _load_hf_model()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded[0] if decoded else ""


async def _translate_with_hf(text: str) -> str:
    return await asyncio.to_thread(_translate_with_hf_sync, text)


async def translate_text(text: str, source: str = "auto", target: str = "en") -> str:
    """
    Translate text prioritizing Google Translate for higher accuracy.
    Falls back to LibreTranslate-compatible API and finally to a local HuggingFace MarianMT model (mr→en).
    """
    normalized_source = _normalize_lang(source or "auto", "auto")
    normalized_target = _normalize_lang(target or "en", "en")

    if GOOGLE_TRANSLATE_ENABLED:
        try:
            google_translated = await _translate_with_google(text, normalized_source, normalized_target)
            if google_translated:
                return google_translated
        except Exception:
            pass

    try:
        libre_translated = await _translate_with_libre(text, normalized_source, normalized_target)
        if libre_translated:
            return libre_translated
    except Exception:
        pass

    if HF_MODEL_ENABLED and normalized_target in {"en", "english"}:
        try:
            return await _translate_with_hf(text)
        except Exception:
            pass

    # Final fallback: return empty string to signal failure
    return ""



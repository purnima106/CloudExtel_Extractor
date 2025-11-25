from typing import List, Literal, Optional
from pydantic import BaseModel

class PageText(BaseModel):
    page_number: int
    text: str
    lang: Optional[str] = None

class UploadResponse(BaseModel):
    job_id: str
    results: List[PageText]
    available_outputs: List[Literal["pdf","json","excel"]]

class TranslateRequest(BaseModel):
    text: str
    source: str = "mr"
    target: str = "en"

class TranslateResponse(BaseModel):
    translated_text: str

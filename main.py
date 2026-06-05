import base64
import io
import os
from typing import Optional

import pytesseract
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
from pydantic import BaseModel


class ExtractRequest(BaseModel):
    filename: Optional[str] = ""
    base64: str


app = FastAPI(title="EPEDA OCR Service")

allowed_origins = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", "https://cofel-transport-control.vercel.app,http://127.0.0.1:5173").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract-pdf")
def extract_pdf(payload: ExtractRequest):
    try:
      pdf_bytes = base64.b64decode(payload.base64)
    except Exception as exc:
      raise HTTPException(status_code=400, detail="PDF base64 invalide") from exc

    if not pdf_bytes:
      raise HTTPException(status_code=400, detail="PDF absent")

    try:
      images = convert_from_bytes(pdf_bytes, dpi=220, fmt="png", thread_count=1)
    except Exception as exc:
      raise HTTPException(status_code=500, detail=f"Conversion PDF impossible: {exc}") from exc

    pages = []
    for index, image in enumerate(images, start=1):
      text = pytesseract.image_to_string(image, lang=os.environ.get("OCR_LANG", "fra"))
      pages.append(text)
      image.close()

    text = "\n".join(pages)
    quality = "ok" if len("".join(text.split())) > 40 else "empty"
    return {
      "filename": payload.filename,
      "quality": quality,
      "method": "poppler-tesseract",
      "pages": len(images),
      "text": text,
    }

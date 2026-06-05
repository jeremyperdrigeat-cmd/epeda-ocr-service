import base64
import io
import os
import re
import subprocess
import tempfile
from typing import Optional

import pdfplumber
import pytesseract
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
from PIL import Image
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


def parse_amount(value: str) -> Optional[float]:
    source = (value or "").replace("\u00a0", " ").strip()
    match = re.search(r"(\d[\d\s]*[,.]\d{2})", source)
    if not match:
        return None
    normalized = match.group(1).replace(" ", "").replace(",", ".")
    try:
        return round(float(normalized), 2)
    except ValueError:
        return None


def compact_text(words) -> str:
    return " ".join(word.get("text", "") for word in words).strip()


def group_words_by_line(words, tolerance: float = 3.2):
    rows = []
    for word in sorted(words, key=lambda item: (float(item.get("top", 0)), float(item.get("x0", 0)))):
        top = float(word.get("top", 0))
        for row in rows:
            if abs(row["top"] - top) <= tolerance:
                row["words"].append(word)
                row["top"] = (row["top"] + top) / 2
                break
        else:
            rows.append({"top": top, "words": [word]})
    for row in rows:
        row["words"].sort(key=lambda item: float(item.get("x0", 0)))
    return rows


def row_cost(words):
    candidates = []
    for word in words:
        text = word.get("text", "")
        amount = parse_amount(text)
        if amount is None:
            continue
        x0 = float(word.get("x0", 0))
        candidates.append((x0, amount))
    if not candidates:
        return None
    # In EPEDA proformas the cost column is before TVA and after volume/rolls.
    plausible = [(x0, amount) for x0, amount in candidates if amount >= 20]
    if not plausible:
        return None
    likely_costs = [(x0, amount) for x0, amount in plausible if 300 <= x0 <= 760]
    usable = likely_costs or plausible
    return usable[-1][1]


def extract_structured_proforma(pdf_bytes: bytes):
    rows = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                words = page.extract_words(
                    x_tolerance=1.5,
                    y_tolerance=3,
                    keep_blank_chars=False,
                    use_text_flow=False,
                )
                for row in group_words_by_line(words):
                    line = compact_text(row["words"])
                    if re.search(r"total|surtaxe|gasoil|gazole|commentaire|contrat|proforma|facturation", line, re.I):
                        continue
                    trip_match = re.match(r"^\s*(\d{6})\b", line)
                    if not trip_match:
                        continue
                    amount = row_cost(row["words"])
                    if amount is None:
                        continue
                    date_match = re.search(r"\b\d{2}/\d{2}/\d{4}\b", line)
                    rows.append(
                        {
                            "trip": trip_match.group(1),
                            "date": date_match.group(0) if date_match else "",
                            "expected": amount,
                            "raw": line,
                            "page": page_index,
                        }
                    )
    except Exception:
        return []

    deduped = {}
    for row in rows:
        deduped[row["trip"]] = row
    return list(deduped.values())


def text_amounts(text: str):
    amounts = []
    source = re.sub(r"\d{2}/\d{2}/\d{2,4}", " ", text or "")
    source = re.sub(r"\b\d+[,.]\d{3}\b", " ", source)
    source = re.split(r"\b(?:tva|mnt\.?\s*tv|montant\s+tva|total\s+rolls|surtaxe|gasoil|gazole)\b", source, flags=re.I)[0]
    for match in re.finditer(r"(\d[\d\s]*[,.]\d{2})", source):
        amount = parse_amount(match.group(1))
        if amount is not None and 20 <= amount < 100000:
            amounts.append(amount)
    for match in re.finditer(r"\b(\d{2,5})\s+(\d{2})\b", source):
        amount = parse_amount(f"{match.group(1)},{match.group(2)}")
        if amount is not None and 20 <= amount < 100000:
            amounts.append(amount)
    if len(amounts) >= 2 and amounts[0] < 150 <= amounts[1]:
        return amounts[1]
    return amounts[0] if amounts else None


def parse_proforma_lines(lines, page_index: int):
    rows = []
    for line in lines:
        raw = (line or "").strip()
        if not raw:
            continue
        if re.search(r"total|surtaxe|gasoil|gazole|commentaire|contrat|proforma|facturation", raw, re.I):
            continue
        if re.search(r"\d+[,.]\s*\d{5,6}\b", raw):
            continue
        trip_match = re.match(r"^\s*(\d{2,3})\s+(\d{3})\b", raw) or re.match(r"^\s*(\d{6})\b", raw)
        if not trip_match:
            continue
        trip = "".join(group for group in trip_match.groups() if group)
        amount = text_amounts(raw)
        if amount is None:
            continue
        date_match = re.search(r"\b\d{2}/\d{2}/\d{4}\b", raw)
        rows.append(
            {
                "trip": trip,
                "date": date_match.group(0) if date_match else "",
                "expected": amount,
                "raw": raw,
                "page": page_index,
            }
        )
    return rows


def proforma_trip_from_line(line: str):
    raw = line or ""
    if re.search(r"total|surtaxe|gasoil|gazole|commentaire|contrat|proforma|facturation", raw, re.I):
        return None
    if re.search(r"\d+[,.]\s*\d{5,6}\b", raw):
        return None
    match = re.match(r"^\s*(\d{2,3})\s+(\d{3})\b", raw) or re.match(r"^\s*(\d{6})\b", raw)
    if not match:
        return None
    return "".join(group for group in match.groups() if group)


def parse_proforma_table_from_text(text: str, page_index: int):
    lines = [(line or "").strip() for line in (text or "").splitlines() if (line or "").strip()]
    rows = []
    current = None

    def flush():
        if not current:
            return
        block_text = " ".join(current["lines"])
        if re.search(r"total|surtaxe|gasoil|gazole", block_text, re.I):
            return
        amount = text_amounts(block_text)
        if amount is None:
            return
        date_match = re.search(r"\b\d{2}/\d{2}/\d{4}\b", block_text)
        rows.append(
            {
                "trip": current["trip"],
                "date": date_match.group(0) if date_match else "",
                "expected": amount,
                "raw": block_text,
                "page": page_index,
            }
        )

    for line in lines:
        trip = proforma_trip_from_line(line)
        if trip:
            flush()
            current = {"trip": trip, "lines": [line]}
        elif current:
            if re.search(r"^total\b|surtaxe|gasoil|gazole|commentaire", line, re.I):
                flush()
                current = None
            else:
                current["lines"].append(line)
    flush()
    return rows


def ocr_lines_from_data(image):
    try:
        data = pytesseract.image_to_data(
            image,
            lang=os.environ.get("OCR_LANG", "fra+eng"),
            config="--psm 6 preserve_interword_spaces=1",
            output_type=pytesseract.Output.DICT,
        )
    except Exception:
        return []
    grouped = {}
    count = len(data.get("text", []))
    for index in range(count):
        text = (data["text"][index] or "").strip()
        if not text:
            continue
        key = (
            data.get("block_num", [0] * count)[index],
            data.get("par_num", [0] * count)[index],
            data.get("line_num", [0] * count)[index],
        )
        grouped.setdefault(key, []).append((data.get("left", [0] * count)[index], text))
    lines = []
    for words in grouped.values():
        lines.append(" ".join(word for _, word in sorted(words)))
    return lines


def best_ocr_variant(variants):
    def score(text):
        amount_count = len(re.findall(r"\d[\d\s]*[,.]\d{2}", text or ""))
        trip_count = len(re.findall(r"(?m)^\s*\d{6}\b", text or ""))
        digit_count = sum(char.isdigit() for char in text or "")
        line_count = len([line for line in (text or "").splitlines() if line.strip()])
        return amount_count * 35 + trip_count * 80 + digit_count + line_count * 3

    return max(variants, key=score) if variants else ""


def tesseract_cli_variants(image_path):
    variants = []
    for psm in ("6", "11", "4"):
        try:
            result = subprocess.run(
                ["tesseract", image_path, "stdout", "-l", "fra+eng", "--psm", psm, "preserve_interword_spaces=1"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            variants.append(result.stdout.decode("utf-8", errors="replace"))
        except Exception:
            continue
    return variants


def convert_pdf_to_image_paths(pdf_bytes: bytes, tmp: str):
    pdf_path = os.path.join(tmp, "input.pdf")
    out_prefix = os.path.join(tmp, "page")
    with open(pdf_path, "wb") as handle:
        handle.write(pdf_bytes)
    subprocess.run(
        ["pdftoppm", "-r", "300", "-png", pdf_path, out_prefix],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return [
        os.path.join(tmp, filename)
        for filename in sorted(os.listdir(tmp))
        if filename.startswith("page-") and filename.endswith(".png")
    ]


def merge_structured_rows(rows):
    deduped = {}
    for row in rows:
        deduped[row["trip"]] = row
    return list(deduped.values())


@app.post("/extract-pdf")
def extract_pdf(payload: ExtractRequest):
    try:
      pdf_bytes = base64.b64decode(payload.base64)
    except Exception as exc:
      raise HTTPException(status_code=400, detail="PDF base64 invalide") from exc

    if not pdf_bytes:
      raise HTTPException(status_code=400, detail="PDF absent")

    proforma_rows = extract_structured_proforma(pdf_bytes)

    pages = []
    with tempfile.TemporaryDirectory() as tmp:
      try:
        image_paths = convert_pdf_to_image_paths(pdf_bytes, tmp)
      except Exception as conversion_error:
        try:
          fallback_images = convert_from_bytes(pdf_bytes, dpi=300, fmt="png", thread_count=1)
          image_paths = []
          for fallback_index, fallback_image in enumerate(fallback_images, start=1):
            fallback_path = os.path.join(tmp, f"fallback-{fallback_index}.png")
            fallback_image.save(fallback_path)
            fallback_image.close()
            image_paths.append(fallback_path)
        except Exception as exc:
          raise HTTPException(status_code=500, detail=f"Conversion PDF impossible: {conversion_error or exc}") from exc

      for index, image_path in enumerate(image_paths, start=1):
        variants = tesseract_cli_variants(image_path)
        with Image.open(image_path) as image:
          if not variants:
            for psm in ("6", "11", "4"):
              variants.append(
                pytesseract.image_to_string(
                  image,
                  lang=os.environ.get("OCR_LANG", "fra"),
                  config=f"--psm {psm} preserve_interword_spaces=1",
                )
              )
          data_lines = ocr_lines_from_data(image)
        variants.append("\n".join(data_lines))
        proforma_rows.extend(parse_proforma_lines(data_lines, index))
        for variant in variants:
          proforma_rows.extend(parse_proforma_lines(variant.splitlines(), index))
          proforma_rows.extend(parse_proforma_table_from_text(variant, index))
        pages.append(best_ocr_variant(variants))

    text = "\n".join(pages)
    structured = {"proformaRows": merge_structured_rows(proforma_rows)}
    quality = "ok" if len("".join(text.split())) > 40 else "empty"
    return {
      "filename": payload.filename,
      "quality": quality,
      "method": "poppler-tesseract",
      "pages": len(pages),
      "text": text,
      "structured": structured,
    }

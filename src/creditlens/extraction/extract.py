"""Turn page text into a validated object, with an offline parser that still works.

The one idea this module teaches: **ask for a shape, not for prose.**

We hand the page text to the gateway together with a Pydantic model
(`ExtractionEnvelope`). The gateway turns on the provider's JSON mode, validates the
answer, and if validation fails it shows the model its own error and asks again. So
the function either returns a `FinancialStatementExtraction` with integers in the
integer fields, or it raises. There is no "the model wrote 'approximately 5.4 crore'"
case to defend against further downstream.

`ExtractionEnvelope.document` is a **discriminated union**: one field, `doc_type`,
decides whether the rest of the object must look like a statement or like a receipt.
That is why both prompts spell out which `doc_type` to return.

The second half of this file is the **offline stub**. `OFFLINE_HANDLERS` is the
gateway's escape hatch: when `LLM_OFFLINE=1` (no keys, CI, or a workshop room whose
free quota has run out) the gateway calls our function instead of a provider. Ours is
a few regexes over the same page text, and on text PDFs it is accurate enough that the
whole pipeline, evals included, runs with no network at all. It is also the honest
baseline: if an LLM cannot beat a page of regexes, it is not earning its cost.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, Optional

from creditlens.config import Settings, settings as default_settings
from creditlens.extraction.ocr import image_to_text, pdf_to_text
from creditlens.extraction.prompts import build_messages, document_text_from_messages
from creditlens.gateway.llm import OFFLINE_HANDLERS, get_gateway
from creditlens.observability import observe
from creditlens.schemas import (DocumentExtraction, ExtractionEnvelope,
                                FinancialStatementExtraction, ReceiptExtraction)

DocType = Literal["financial_statement", "receipt"]

# Even a long statement is a few thousand characters. The cap stops a corrupt OCR
# result from spending a whole free-tier quota in one call.
MAX_TEXT_CHARS = 20_000


# ---------------------------------------------------------------------------
# The public API
# ---------------------------------------------------------------------------


@observe(name="extraction.extract_document")
def extract_document(path, doc_type: DocType, s: Settings = default_settings) -> DocumentExtraction:
    """Read one document and return the validated object for its type."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such document: {path}")

    if path.suffix.lower() == ".pdf":
        text, ocr_used = pdf_to_text(path, s)
    else:
        text, ocr_used = image_to_text(path, s), True

    messages = build_messages(text[:MAX_TEXT_CHARS], doc_type)
    response = get_gateway().complete(messages, response_model=ExtractionEnvelope)
    assert response.parsed is not None
    document = response.parsed.document  # type: ignore[attr-defined]

    # Provenance, not model output: we know these two and the model does not.
    document.source_file = path.name
    document.ocr_used = ocr_used
    return document


@observe(name="extraction.extract_financials")
def extract_financials(company_id: str, s: Settings = default_settings) -> FinancialStatementExtraction:
    """Figures for one applicant, from `data/raw/financials/{company_id}_financials.pdf`."""
    path = s.paths.raw / "financials" / f"{company_id}_financials.pdf"
    document = extract_document(path, "financial_statement", s)
    assert isinstance(document, FinancialStatementExtraction)
    return document


@observe(name="extraction.extract_receipt")
def extract_receipt(path, s: Settings = default_settings) -> ReceiptExtraction:
    """Key fields from one scanned receipt in `data/real/receipts/`."""
    document = extract_document(path, "receipt", s)
    assert isinstance(document, ReceiptExtraction)
    return document


# ---------------------------------------------------------------------------
# Offline stub, part 1: reading numbers off a statement
# ---------------------------------------------------------------------------

# An amount as printed with Indian digit grouping: 12,34,567. Brackets mean negative.
# `\d+` first so a number that lost its commas in OCR ("1485") is still read whole.
NUMBER = re.compile(r"\(?-?\d+(?:[,.]\d{2,3})*\)?")

# The printed label for each field we want, best spelling first. The regexes tolerate
# missing spaces, because OCR often glues table labels together.
FIELD_LABELS: dict[str, tuple[str, ...]] = {
    "revenue": ("Revenue from operations", "Total revenue", "Revenue"),
    "ebitda": ("EBITDA", "Operating profit"),
    "interest": ("Finance costs", "Finance cost", "Interest expense"),
    "pat": ("Profit after tax", "Net profit"),
    "current_assets": ("Total current assets",),
    "current_liabilities": ("Trade payables and other current liabilities",
                            "Total current liabilities", "Trade payables"),
    "total_debt": ("Total borrowings", "Total debt"),
}


def _label_pattern(label: str) -> re.Pattern:
    """"Total current assets" also matches "Totalcurrentassets" and "Total  current assets"."""
    return re.compile(r"\s*".join(re.escape(word) for word in label.split()), re.IGNORECASE)


def _to_int(token: str) -> Optional[int]:
    """"12,34,567" -> 1234567, "(4,120)" -> -4120 (accountants print negatives in brackets)."""
    negative = token.startswith("(") or token.startswith("-")
    digits = re.sub(r"\D", "", token)
    if not digits:
        return None
    value = int(digits)
    return -value if negative else value


def _numbers_in(text: str, start: int, end: int, how_many: int) -> list[Optional[int]]:
    """The first `how_many` amounts printed between two positions in the text."""
    found: list[Optional[int]] = []
    for match in NUMBER.finditer(text, start, end):
        if text[max(0, match.start() - 2):match.start()].upper() == "FY":
            continue  # that is a column heading like FY2024, not an amount
        found.append(_to_int(match.group(0)))
        if len(found) >= how_many:
            break
    while len(found) < how_many:
        found.append(None)
    return found


def fiscal_years(text: str) -> list[str]:
    """The column headings, in the order they are printed: ["FY2024", "FY2025", "FY2026"]."""
    years: list[str] = []
    for match in re.finditer(r"FY\s?(\d{4})", text):
        label = "FY" + match.group(1)
        if label not in years:
            years.append(label)
    return years or ["FY"]


def _principal_repayments(text: str, years: list[str]) -> dict[str, Optional[int]]:
    """Read the borrowings note: "... (INR '000): FY2024: 2,302; FY2025: 1,613; ..."."""
    anchor = re.search(r"principal\s*repayment", text, re.IGNORECASE)
    if anchor is None:
        return {}
    tail = text[anchor.end():anchor.end() + 600]
    pairs = re.findall(r"FY\s?(\d{4})\s*[:.\-]?\s*(\(?-?\d+(?:[,.]\d{2,3})*\)?)", tail)
    found = {"FY" + y: _to_int(amount) for y, amount in pairs}
    return {y: found.get(y) for y in years}


def parse_statement_text(text: str) -> dict:
    """Regex fallback for a financial statement. Good on text PDFs, fair on scans."""
    years = fiscal_years(text)
    how_many = len(years)

    # Find every label once, then read the amounts that sit between one label and the next.
    located: dict[str, tuple[int, int]] = {}
    for field, labels in FIELD_LABELS.items():
        for label in labels:
            match = _label_pattern(label).search(text)
            if match:
                located[field] = (match.start(), match.end())
                break
    boundaries = sorted(start for start, _ in located.values())

    columns: dict[str, list[Optional[int]]] = {}
    for field, (start, end) in located.items():
        after = [b for b in boundaries if b > end]
        window_end = min(after[0], end + 600) if after else end + 600
        columns[field] = _numbers_in(text, end, window_end, how_many)

    principal = _principal_repayments(text, years)

    rows = []
    for index, year in enumerate(years):
        row = {"fiscal_year": year, "principal_repayment": principal.get(year)}
        for field in FIELD_LABELS:
            row[field] = columns.get(field, [None] * how_many)[index]
        rows.append(row)

    filled = sum(1 for row in rows for key, value in row.items()
                 if key != "fiscal_year" and value is not None)
    wanted = max(1, len(rows) * (len(FIELD_LABELS) + 1))

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return {"doc_type": "financial_statement",
            "company_name": lines[0] if lines else "unknown",
            "unit": "INR thousands",
            "years": rows,
            "confidence": round(filled / wanted, 2)}


# ---------------------------------------------------------------------------
# Offline stub, part 2: reading a receipt
# ---------------------------------------------------------------------------

MONTHS = "JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC"
# A scanned receipt often glues the date to the time ("15/01/201911:05AM"), so these
# patterns deliberately do not require a word boundary after the year.
DATE_PATTERNS = (
    re.compile(r"(?<!\d)\d{1,2}[/.\-]\d{1,2}[/.\-]\d{4}(?!\d{3})"),
    re.compile(r"(?<!\d)\d{4}[/.\-]\d{1,2}[/.\-]\d{1,2}(?!\d)"),
    re.compile(rf"\b\d{{1,2}}[\s\-]?(?:{MONTHS})[A-Z]*[\s\-]?\d{{2,4}}\b", re.IGNORECASE),
    re.compile(r"(?<!\d)\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2}(?!\d)"),
)
MONEY = re.compile(r"\d{1,3}(?:,\d{3})*\.\d{2}|\d+\.\d{2}")
# Lines that look like the address rather than a phone number or a tax id.
NOT_ADDRESS = re.compile(r"TEL[:.]|\bFAX\b|\bGST\b|INVOICE|RECEIPT|CASHIER|E-?MAIL|\bTIME\b",
                         re.IGNORECASE)
# A line that is a header, not the merchant name.
NOT_COMPANY = re.compile(r"^[\W_]*$|COPY|TAX\s*INVOICE|RECEIPT|CASH\s*BILL|ORDER|WELCOME",
                         re.IGNORECASE)
# Legal suffixes and trade words that mark the merchant line even when OCR loses the spaces.
COMPANY_HINT = re.compile(r"SDN\s*BHD|BHD|PVT|PTE|LTD|LLP|ENTERPRISE|TRADING|RESTAURANT|"
                          r"MARKETING|HOLDINGS|SUPPLY|SERVICES|STORE|CAFE|PERNIAGAAN",
                          re.IGNORECASE)


def parse_receipt_text(text: str) -> dict:
    """Regex fallback for a SROIE-style receipt: top line, a date, the biggest TOTAL."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # SROIE receipts print the merchant name at the top, but a scan often has a stamp,
    # a cashier name or "TAX INVOICE" above it, so prefer a line that names a company.
    head = [line for line in lines[:6] if not NOT_COMPANY.search(line)]
    with_hint = [line for line in head if COMPANY_HINT.search(line)]
    readable = [line for line in head if len(re.sub(r"[^A-Za-z]", "", line)) >= 3]
    company = (with_hint or readable or [""])[0]

    date = ""
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            date = match.group(0)
            break

    # The grand total is the largest amount on a line that says TOTAL (and not SUBTOTAL).
    # OCR often puts the label and the amount on separate lines, so look at the next one too.
    candidates: list[str] = []
    for index, line in enumerate(lines):
        upper = line.upper()
        if "TOTAL" not in upper or "SUBTOTAL" in upper or "SUB TOTAL" in upper:
            continue
        on_line = MONEY.findall(line)
        candidates.extend(on_line or MONEY.findall(lines[index + 1] if index + 1 < len(lines) else ""))
    if not candidates:
        candidates = MONEY.findall(text)
    total = max(candidates, key=lambda v: float(v.replace(",", "")), default="")

    # The address is the block printed directly under the merchant name, stopping at the
    # first phone number, tax id or invoice line.
    start = lines.index(company) + 1 if company in lines else 1
    address_lines = []
    for line in lines[start:start + 6]:
        if NOT_ADDRESS.search(line) or any(p.search(line) for p in DATE_PATTERNS):
            break
        if re.search(r"ROC\s*NO|\bCO\.?\s*NO|\(?\d{5,}\s*-\s*[A-Z]\)?", line, re.IGNORECASE):
            continue  # a company registration number, not part of the address
        address_lines.append(line)
        if len(address_lines) >= 4:
            break
    address = ", ".join(address_lines)

    found = sum(1 for value in (company, date, total, address) if value)
    return {"doc_type": "receipt", "company": company, "date": date,
            "address": address, "total": total or "0",
            "confidence": round(found / 4, 2)}


# ---------------------------------------------------------------------------
# Registering the stub with the gateway. Importing this module is what turns it on.
# ---------------------------------------------------------------------------


def offline_extraction(messages: list[dict]) -> dict:
    """The gateway calls this instead of a provider when LLM_OFFLINE=1."""
    text, doc_type = document_text_from_messages(messages)
    document = parse_receipt_text(text) if doc_type == "receipt" else parse_statement_text(text)
    return {"document": document}


OFFLINE_HANDLERS["ExtractionEnvelope"] = offline_extraction

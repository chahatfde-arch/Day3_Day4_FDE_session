"""Get the words off the page, whether the page is text or a picture of text.

The one idea this module teaches: **there are two completely different kinds of PDF.**

* A *text PDF* was written by software. The characters are inside the file, so
  `pypdf` can read them in milliseconds and the result is exactly right.
* A *scan* is a photograph of paper wrapped in a PDF. There are no characters at
  all, only pixels, so the computer has to guess the letters. That guessing is OCR
  (optical character recognition). It is slow, it is never perfect, and it is the
  reason extraction needs an accuracy number instead of a "it works" feeling.

`pdf_to_text` does the deciding for you: it reads the text layer first and only
pays for OCR when the text layer comes back almost empty. It returns
`(text, ocr_used)` so the rest of the system can record *how* the numbers were read.

Three practical problems are handled here, because they are the ones that actually
break a workshop:

* **No Tesseract.** Tesseract is a separate program, not a Python package, and a
  locked-down laptop usually cannot install it. So the engine chain is: Tesseract if
  its binary is really on PATH, else `rapidocr-onnxruntime` (pure pip, no system
  install), else a clear `OCRUnavailable` error instead of a confusing crash.
* **A sideways page.** One scan in the data set has page 2 rotated 90 degrees. We
  try the page at 0, 90, 180 and 270 degrees and keep the version whose text reads
  most like real words. A cheap ink-projection test picks the likely orientation
  first so we usually only pay for one OCR pass.
* **OCR is slow.** Every OCR result is cached on disk under `data/cache/ocr/`, keyed
  by the hash of the file's bytes. The first run of a live session is slow once; every
  run after that is instant.
"""
from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path
from typing import Optional, Sequence

from creditlens.config import Settings, settings as default_settings
from creditlens.observability import observe


class OCRUnavailable(RuntimeError):
    """No OCR engine could be found, so a scanned page cannot be read."""


# A text layer shorter than this almost certainly means "this PDF is a picture".
MIN_TEXT_CHARS = 200

# Rendering resolution for OCR. Higher is slower and barely more accurate here.
RENDER_DPI = 150

# Words we expect to see on a financial statement. Used for two jobs: scoring an
# orientation, and putting spaces back into glued-together OCR output.
STATEMENT_WORDS = {
    "revenue", "from", "operations", "cost", "of", "goods", "sold", "gross", "profit",
    "operating", "expenses", "ebitda", "depreciation", "finance", "costs", "before",
    "tax", "expense", "after", "cash", "and", "bank", "trade", "receivables",
    "inventory", "total", "current", "assets", "net", "fixed", "payables", "other",
    "liabilities", "borrowings", "shareholders", "equity", "balance", "sheet", "as",
    "at", "march", "audited", "financial", "statements", "statement", "fiscal",
    "years", "year", "ending", "loss", "note", "scheduled", "principal", "repayment",
    "during", "the", "synthetic", "document", "generated", "for", "training", "not",
    "real", "company", "limited", "private", "traders", "enterprises", "industries",
}

# Words that mark a receipt. Receipts share almost no vocabulary with a balance sheet.
RECEIPT_WORDS = {
    "total", "cash", "change", "date", "invoice", "receipt", "tax", "item", "qty",
    "price", "amount", "subtotal", "rounding", "net", "bill", "no", "time", "thank",
    "you", "please", "come", "again", "goods", "sold", "are", "exchangeable",
    "sdn", "bhd", "tel", "fax", "address", "payment", "card", "discount",
}

# Stop trying more orientations once a page reads this convincingly.
_GOOD_ENOUGH_WORDS = 12
_GOOD_ENOUGH_RATIO = 0.4


# ---------------------------------------------------------------------------
# Scoring: does this text look like real words?
# ---------------------------------------------------------------------------


def word_score(text: str, vocabulary: Optional[set[str]] = None) -> float:
    """How much does `text` look like English prose rather than OCR noise?

    Two signals, because either one alone is fooled easily:
    the number of words we recognise, and the share of alphabetic tokens that are
    recognised (a page of pure numbers scores zero on the second).
    """
    vocab = vocabulary if vocabulary is not None else STATEMENT_WORDS
    tokens = re.findall(r"[A-Za-z]+", text)
    if not tokens:
        return 0.0
    known = [t for t in tokens if t.lower() in vocab]
    ratio = len(known) / len(tokens)
    return len(known) + 5.0 * ratio


def unglue(text: str, vocabulary: Optional[set[str]] = None) -> str:
    """Put spaces back into runs like "Balancesheetasat31March".

    OCR engines that read a whole line at once often drop the thin spaces in a
    scanned table. Downstream regexes then miss "Total current assets" because the
    page says "Totalcurrentassets". We only touch long alphabetic runs that we can
    split *completely* into known words, so names like "Godavari" are left alone.
    """
    vocab = vocabulary if vocabulary is not None else STATEMENT_WORDS
    longest = max(len(w) for w in vocab)

    def split_run(match: re.Match) -> str:
        run = match.group(0)
        if len(run) < 12 or run.lower() in vocab:
            return run
        lower = run.lower()
        # best[i] = the fewest words that exactly cover lower[:i], or None.
        best: list[Optional[list[str]]] = [None] * (len(lower) + 1)
        best[0] = []
        for i in range(1, len(lower) + 1):
            for j in range(max(0, i - longest), i):
                prefix = best[j]
                if prefix is None or lower[j:i] not in vocab:
                    continue
                candidate = prefix + [run[j:i]]
                if best[i] is None or len(candidate) < len(best[i]):
                    best[i] = candidate
        whole = best[len(lower)]
        return " ".join(whole) if whole else run

    return re.sub(r"[A-Za-z]+", split_run, text)


# ---------------------------------------------------------------------------
# Engines. Tesseract first (if the binary exists), then rapidocr.
# ---------------------------------------------------------------------------


def _tesseract_ready() -> bool:
    """pytesseract is only a wrapper: without the tesseract.exe on PATH it cannot work."""
    if shutil.which("tesseract") is None:
        return False
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return True


def _rapidocr_ready() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def ocr_engine_name() -> str:
    """Name of the engine that would be used, or raise if there is none."""
    if _tesseract_ready():
        return "tesseract"
    if _rapidocr_ready():
        return "rapidocr"
    raise OCRUnavailable(
        "No OCR engine found. Install the Tesseract program and `pip install pytesseract`, "
        "or `pip install rapidocr-onnxruntime` which needs no system package.")


def ocr_available() -> bool:
    """True when a scanned page could be read. Tests use this to skip gracefully."""
    try:
        ocr_engine_name()
        return True
    except OCRUnavailable:
        return False


_rapidocr_instance = None


def _rapidocr():
    """One engine per process: loading the ONNX models takes a couple of seconds."""
    global _rapidocr_instance
    if _rapidocr_instance is None:
        from rapidocr_onnxruntime import RapidOCR

        _rapidocr_instance = RapidOCR()
    return _rapidocr_instance


def ocr_image(image) -> str:
    """Read one PIL image with whichever engine is available."""
    engine = ocr_engine_name()
    if engine == "tesseract":
        import pytesseract

        return pytesseract.image_to_string(image)
    result, _ = _rapidocr()(image.convert("RGB"))
    if not result:
        return ""
    # rapidocr returns [box, text, confidence] per detected line, in reading order.
    return "\n".join(line[1] for line in result)


# ---------------------------------------------------------------------------
# Orientation: which way up is this page?
# ---------------------------------------------------------------------------


def _orientation_hint(image) -> tuple[int, ...]:
    """Order the four rotations, most likely first, without paying for OCR.

    Printed text makes dark stripes. If the stripes run across the page, the
    row-by-row ink totals swing wildly and the column totals do not. That tells us
    whether the page is upright-ish (0 or 180) or sideways (90 or 270), which is
    half the answer for the price of two sums.
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a hard dependency of the project
        return (0, 90, 180, 270)
    ink = 255.0 - np.asarray(image.convert("L"), dtype="float32")
    ink = ink - ink.mean()
    rows_vary = float(ink.sum(axis=1).var())
    cols_vary = float(ink.sum(axis=0).var())
    return (0, 180, 90, 270) if rows_vary >= cols_vary else (90, 270, 0, 180)


def _reads_convincingly(text: str, vocabulary: Optional[set[str]] = None) -> bool:
    """Enough recognised words, and enough of the words recognised, to stop searching."""
    vocab = vocabulary if vocabulary is not None else STATEMENT_WORDS
    tokens = re.findall(r"[A-Za-z]+", text)
    if not tokens:
        return False
    known = sum(1 for t in tokens if t.lower() in vocab)
    return known >= _GOOD_ENOUGH_WORDS and known / len(tokens) >= _GOOD_ENOUGH_RATIO


def best_orientation(image, vocabulary: Optional[set[str]] = None,
                     angles: Optional[Sequence[int]] = None) -> tuple[str, int]:
    """OCR the page at several rotations and keep the one that reads like words.

    Returns (text, angle), where `angle` is the clockwise rotation that was applied.
    The four rotations are tried in the order the ink-projection hint suggests, and
    the search stops as soon as one of them reads convincingly, so an upright page
    normally costs exactly one OCR pass instead of four.
    """
    candidates = list(angles) if angles is not None else list(_orientation_hint(image))
    best_text, best_angle, best_points = "", 0, -1.0
    for angle in candidates:
        rotated = image.rotate(-angle, expand=True) if angle else image
        text = unglue(ocr_image(rotated), vocabulary)
        points = word_score(text, vocabulary)
        if angle == 0:
            points += 1.0  # tie-break towards "the page was already the right way up"
        if points > best_points:
            best_text, best_angle, best_points = text, angle, points
        if _reads_convincingly(text, vocabulary):
            break
    return best_text, best_angle


# ---------------------------------------------------------------------------
# The disk cache. OCR is expensive; the same bytes always give the same answer.
# ---------------------------------------------------------------------------


def _file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cache_path(path: Path, s: Settings) -> Optional[Path]:
    try:
        engine = ocr_engine_name()
    except OCRUnavailable:
        engine = "none"
    return s.paths.cache / "ocr" / f"{_file_sha(path)}-{engine}.txt"


def _cache_read(path: Path, s: Settings) -> Optional[str]:
    cached = _cache_path(path, s)
    if cached is not None and cached.exists():
        return cached.read_text(encoding="utf-8")
    return None


def _cache_write(path: Path, s: Settings, text: str) -> None:
    cached = _cache_path(path, s)
    if cached is None:
        return
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# The two public readers.
# ---------------------------------------------------------------------------


def _text_layer(path: Path) -> str:
    """Characters already stored in the PDF, or "" when there are none."""
    try:
        from pypdf import PdfReader

        pages = [p.extract_text() or "" for p in PdfReader(str(path)).pages]
        text = "\n".join(pages)
        if len(text.strip()) >= MIN_TEXT_CHARS:
            return text
    except Exception:  # noqa: BLE001  a broken text layer is not an error, it just means "use OCR"
        text = ""
    try:  # pdfplumber lays tables out slightly differently and sometimes finds more
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            plumbed = "\n".join(page.extract_text() or "" for page in pdf.pages)
        if len(plumbed.strip()) > len(text.strip()):
            return plumbed
    except Exception:  # noqa: BLE001
        pass
    return text


def _render_pages(path: Path, dpi: int = RENDER_DPI):
    """Turn each PDF page into a picture. pypdfium2 needs no system package."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(path))
    return [page.render(scale=dpi / 72).to_pil() for page in doc]


@observe(name="extraction.pdf_to_text")
def pdf_to_text(path, s: Settings = default_settings) -> tuple[str, bool]:
    """Read a PDF. Returns (text, ocr_used).

    `ocr_used` is part of the audit trail: a number read by OCR deserves more
    suspicion than a number read straight out of the file.
    """
    path = Path(path)
    text = _text_layer(path)
    if len(text.strip()) >= MIN_TEXT_CHARS:
        return text, False

    cached = _cache_read(path, s)
    if cached is not None:
        return cached, True

    pages = _render_pages(path)
    out = []
    for page in pages:
        page_text, _angle = best_orientation(page, STATEMENT_WORDS)
        out.append(page_text)
    ocr_text = "\n\n".join(out)
    _cache_write(path, s, ocr_text)
    return ocr_text, True


@observe(name="extraction.image_to_text")
def image_to_text(path, s: Settings = default_settings, detect_rotation: bool = False) -> str:
    """Read a photo or scan of a receipt. Same engine chain, same cache."""
    path = Path(path)
    cached = _cache_read(path, s)
    if cached is not None:
        return cached

    from PIL import Image

    with Image.open(path) as handle:
        image = handle.convert("RGB")
    if detect_rotation:
        text, _angle = best_orientation(image, RECEIPT_WORDS)
    else:
        text = unglue(ocr_image(image), RECEIPT_WORDS)
    _cache_write(path, s, text)
    return text

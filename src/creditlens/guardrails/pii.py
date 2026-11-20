"""Mask borrower PII before it reaches a model, a log line or a trace.

The contract is short: director PAN, phone, email, residential address and name never
reach an LLM unmasked. Two public functions do that job.

    from creditlens.guardrails.pii import mask_text, mask_record

    mask_text("Call Varsha Buch on +91 98765 43210, PAN FPYPS6322N")
    # 'Call <PERSON> on <PHONE>, PAN <PAN>'

    mask_record({"full_name": "Varsha Buch", "pan": "FPYPS6322N", "city": "Pune"})
    # {'full_name': '<PERSON>', 'pan': '<PAN>', 'city': 'Pune'}

**Two engines, one behaviour.** When Microsoft Presidio *and* its spaCy model are
installed we use them, because Presidio also catches names and places we never listed.
When either is missing, or the laptop blocks the spaCy binary, we fall back to plain
regular expressions. The fallback is not as clever, but it is deterministic, needs no
download, and catches every pattern the generated data contains, so tests and CI give
the same answer on any machine. `engine_in_use()` tells you which one ran.

The deny list of director names is read from `data/raw/registry/directors.csv`. Names
are the one kind of PII a regex cannot invent a pattern for, so we look them up.
"""
from __future__ import annotations

import csv
import functools
import re
from pathlib import Path
from typing import Any, Iterable, Optional

from creditlens.config import settings

# What each kind of PII is replaced with. Keeping a label (rather than "****") means a
# model can still reason about the *shape* of the record: "the guarantor <PERSON> ...".
PLACEHOLDERS = {
    "EMAIL": "<EMAIL>",
    "IN_PAN": "<PAN>",
    "PHONE_NUMBER": "<PHONE>",
    "PERSON": "<PERSON>",
    "ADDRESS": "<ADDRESS>",
}

# Indian PAN: five letters, four digits, one letter. For example FPYPS6322N.
PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")

# Indian mobile numbers as the generators write them: "+91 98765 43210", and the
# common variants "+919876543210", "98765 43210", "9876543210", "+91-98765-43210".
#
# Two branches on purpose. With a country code the digits run straight on from "91",
# so `\b` would not fire and the number would leak; without one we *do* want `\b`, so
# a twelve-digit account number is not mistaken for a ten-digit mobile. The lookaround
# at each end stops the pattern biting a chunk out of a longer run of digits.
PHONE_RE = re.compile(
    r"(?<!\d)(?:(?:\+|00)91[\s\-.]?[6-9]\d{4}[\s\-.]?\d{5}"
    r"|\b[6-9]\d{4}[\s\-.]?\d{5}\b)(?!\d)"
)

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

# Column names whose whole value is PII, whatever it looks like.
SENSITIVE_FIELDS = {
    "pan": "IN_PAN",
    "phone": "PHONE_NUMBER",
    "mobile": "PHONE_NUMBER",
    "email": "EMAIL",
    "email_address": "EMAIL",
    "full_name": "PERSON",
    "director_name": "PERSON",
    "guarantor_name": "PERSON",
    "residential_address": "ADDRESS",
    "address": "ADDRESS",
}

_FORCED_ENGINE: Optional[str] = None


# ---------------------------------------------------------------------------
# The deny list: director names from the registry
# ---------------------------------------------------------------------------


def directors_csv() -> Path:
    return settings.paths.raw / "registry" / "directors.csv"


@functools.lru_cache(maxsize=8)
def _names_from(path_str: str, mtime: float) -> tuple[str, ...]:
    """Read director names once per file version (mtime is part of the cache key)."""
    path = Path(path_str)
    names: list[str] = []
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                name = (row.get("full_name") or "").strip()
                if len(name) > 3:
                    names.append(name)
    except (OSError, csv.Error):
        return ()
    return tuple(sorted(set(names), key=len, reverse=True))  # longest first: "Ravi Kumar" before "Ravi"


def director_names(path: Optional[Path] = None) -> tuple[str, ...]:
    """Full names of every director in the registry, or () when the file is absent."""
    path = Path(path) if path else directors_csv()
    if not path.exists():
        return ()
    return _names_from(str(path), path.stat().st_mtime)


@functools.lru_cache(maxsize=8)
def _deny_list_regex(names: tuple[str, ...]) -> Optional[re.Pattern]:
    if not names:
        return None
    return re.compile(r"\b(?:" + "|".join(re.escape(n) for n in names) + r")\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Engine 1: Presidio (only if both the library and a spaCy model are usable)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _presidio():
    """Build the Presidio analyzer and anonymizer, or return None.

    Presidio needs a spaCy NLP model (`python -m spacy download en_core_web_sm`).
    Loading it can fail for several boring reasons: the model was never downloaded,
    the machine blocks the compiled spaCy binary, the version does not match. None of
    those should break a credit memo, so every failure lands in the regex fallback.
    """
    try:
        import spacy  # noqa: F401
        from presidio_analyzer import AnalyzerEngine, PatternRecognizer
        from presidio_analyzer import Pattern as PresidioPattern
        from presidio_anonymizer import AnonymizerEngine
        from presidio_anonymizer.entities import OperatorConfig
    except Exception:  # noqa: BLE001  ImportError, or a DLL that will not load
        return None

    try:
        spacy.load("en_core_web_sm")
    except Exception:  # noqa: BLE001  model not downloaded, or blocked
        return None

    try:
        analyzer = AnalyzerEngine()
        analyzer.registry.add_recognizer(PatternRecognizer(
            supported_entity="IN_PAN",
            patterns=[PresidioPattern("pan", PAN_RE.pattern, 0.9)],
            context=["pan", "permanent account number", "income tax"],
        ))
        analyzer.registry.add_recognizer(PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[PresidioPattern("in_mobile", PHONE_RE.pattern, 0.8)],
            context=["phone", "mobile", "contact"],
        ))
        analyzer.registry.add_recognizer(PatternRecognizer(
            supported_entity="EMAIL",
            patterns=[PresidioPattern("email", EMAIL_RE.pattern, 0.9)],
            context=["email", "e-mail"],
        ))
        names = director_names()
        if names:
            # A deny list is the right tool when you have the actual list of people.
            analyzer.registry.add_recognizer(PatternRecognizer(
                supported_entity="PERSON", deny_list=list(names), name="director_deny_list"))
        return analyzer, AnonymizerEngine(), OperatorConfig
    except Exception:  # noqa: BLE001
        return None


def _mask_with_presidio(text: str) -> Optional[str]:
    built = _presidio()
    if built is None:
        return None
    analyzer, anonymizer, OperatorConfig = built
    try:
        entities = ["IN_PAN", "PHONE_NUMBER", "EMAIL", "EMAIL_ADDRESS", "PERSON", "LOCATION"]
        results = analyzer.analyze(text=text, entities=entities, language="en")
        operators = {k: OperatorConfig("replace", {"new_value": v})
                     for k, v in {**PLACEHOLDERS, "EMAIL_ADDRESS": PLACEHOLDERS["EMAIL"],
                                  "LOCATION": PLACEHOLDERS["ADDRESS"]}.items()}
        return anonymizer.anonymize(text=text, analyzer_results=results, operators=operators).text
    except Exception:  # noqa: BLE001  never let masking raise: fall back instead
        return None


# ---------------------------------------------------------------------------
# Engine 2: regular expressions (always available)
# ---------------------------------------------------------------------------


def _mask_with_regex(text: str) -> str:
    """Order matters: emails contain digits, so they go first."""
    masked = EMAIL_RE.sub(PLACEHOLDERS["EMAIL"], text)
    masked = PAN_RE.sub(PLACEHOLDERS["IN_PAN"], masked)
    masked = PHONE_RE.sub(PLACEHOLDERS["PHONE_NUMBER"], masked)
    deny = _deny_list_regex(director_names())
    if deny is not None:
        masked = deny.sub(PLACEHOLDERS["PERSON"], masked)
    return masked


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def engine_in_use() -> str:
    """'presidio' or 'regex'. Print it once at startup so nobody is surprised."""
    if _FORCED_ENGINE:
        return _FORCED_ENGINE
    return "presidio" if _presidio() is not None else "regex"


def use_engine(name: Optional[str]) -> None:
    """Force an engine ('regex', 'presidio') or pass None to auto-detect. For tests."""
    global _FORCED_ENGINE
    if name not in (None, "regex", "presidio"):
        raise ValueError("engine must be 'regex', 'presidio' or None")
    _FORCED_ENGINE = name


def mask_text(text: Any) -> str:
    """Replace every piece of PII in `text` with a placeholder.

    Always returns a string and never raises: masking that crashes would be worse
    than masking that is slightly over-eager.
    """
    if text is None:
        return ""
    text = str(text)
    if not text.strip():
        return text
    if _FORCED_ENGINE != "regex":
        via_presidio = _mask_with_presidio(text)
        if via_presidio is not None:
            # Presidio misses our deny list when the recognizer could not be added,
            # so run the cheap name pass over its output as well.
            deny = _deny_list_regex(director_names())
            return deny.sub(PLACEHOLDERS["PERSON"], via_presidio) if deny else via_presidio
    return _mask_with_regex(text)


def mask_record(record: dict, extra_fields: Iterable[str] = ()) -> dict:
    """Mask a row (a dict) field by field, recursing into nested dicts and lists.

    A column named `pan` is replaced whole, because the column name already tells us
    what it holds. Every other string is scanned for patterns.
    """
    if not isinstance(record, dict):
        return record
    forced = {f.lower() for f in extra_fields}
    out: dict = {}
    for key, value in record.items():
        name = str(key).lower()
        if isinstance(value, dict):
            out[key] = mask_record(value, extra_fields)
        elif isinstance(value, list):
            out[key] = [mask_record(v, extra_fields) if isinstance(v, dict)
                        else (mask_text(v) if isinstance(v, str) else v) for v in value]
        elif name in SENSITIVE_FIELDS and value not in (None, ""):
            out[key] = PLACEHOLDERS[SENSITIVE_FIELDS[name]]
        elif name in forced and value not in (None, ""):
            out[key] = "<REDACTED>"
        elif isinstance(value, str):
            out[key] = mask_text(value)
        else:
            out[key] = value
    return out


def contains_pii(text: str) -> bool:
    """True when masking would change the text. Used by the output rail."""
    return mask_text(text) != str(text)

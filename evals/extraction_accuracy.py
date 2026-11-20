"""Score the extractor against the golden set, field by field.

The one idea this module teaches: **"accuracy" is not one number, it is a comparison
rule plus a table.** Most of the thinking in an eval goes into the comparison rule, not
the loop:

* **Strings** are compared after normalising: casefold, collapse runs of whitespace,
  drop punctuation. "OJC Marketing Sdn Bhd" and "OJC MARKETING SDN  BHD." are the same
  answer, and pretending otherwise just hides real errors behind fake ones.
* **Numbers** are compared within 1 percent. A statement in INR thousands that reads
  52,316 as 52,316 is right; nothing in a credit decision turns on the last rupee, and
  a tolerance stops rounding noise from drowning out genuine misreads.
* **Dates** are parsed on both sides and compared as dates, so "15/01/2019" equals
  "15 JAN 2019". If neither side parses, they fall back to the string rule.

Everything is reported per field, because an overall score of 0.9 can mean "every field
is good" or "seven fields are perfect and `address` is a disaster", and those are
completely different bugs.

    LLM_OFFLINE=1 python -m evals.extraction_accuracy --limit 5
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from creditlens.config import Settings, settings as default_settings
from creditlens.extraction.extract import extract_document
from creditlens.extraction.golden import FIGURE_FIELDS, RECEIPT_FIELDS, load_golden_set
from creditlens.schemas import FinancialStatementExtraction, ReceiptExtraction

NUMBER_TOLERANCE = 0.01  # 1 percent

DATE_FORMATS = ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y-%m-%d", "%Y/%m/%d",
                "%d/%m/%y", "%d-%m-%y", "%d %b %Y", "%d %B %Y", "%d%b%Y", "%b %d %Y")

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# The comparison rules
# ---------------------------------------------------------------------------


def normalise_string(value: Any) -> str:
    """Casefold, strip punctuation, collapse whitespace. "A.B.  Ltd." -> "ab ltd"."""
    text = "" if value is None else str(value)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def as_number(value: Any) -> Optional[float]:
    """Read a number out of an int, a float or a printed string. None when it is not one."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    text = re.sub(r"[^\d.\-]", "", text)
    if not re.fullmatch(r"-?\d*\.?\d+", text or ""):
        return None
    return float(text)


def as_date(value: Any) -> Optional[date]:
    """Parse a printed date. None when the string is not a date we recognise."""
    if isinstance(value, (datetime, date)):
        return value.date() if isinstance(value, datetime) else value
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def numbers_match(predicted: float, expected: float, tolerance: float = NUMBER_TOLERANCE) -> bool:
    if expected == 0:
        return abs(predicted) <= tolerance
    return abs(predicted - expected) <= abs(expected) * tolerance


def values_match(field: str, predicted: Any, expected: Any) -> bool:
    """Is this one predicted value right? The rule depends on what kind of field it is."""
    if "date" in field:
        p, e = as_date(predicted), as_date(expected)
        if p is not None and e is not None:
            return p == e
        return normalise_string(predicted) == normalise_string(expected)
    p_num, e_num = as_number(predicted), as_number(expected)
    if e_num is not None:
        return p_num is not None and numbers_match(p_num, e_num)
    return normalise_string(predicted) == normalise_string(expected)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score(predictions: dict, golden: dict) -> dict:
    """Per-field and overall accuracy of `predictions` against `golden`.

    Both are `{record_id: {field_name: value}}`. A record the extractor never produced
    counts every golden field as wrong, which is the point: silently returning nothing
    is a failure, not an abstention. A golden field that is blank is skipped, because
    "the paper does not say" is not something the extractor can get wrong.
    """
    per_field: dict[str, dict[str, Any]] = {}
    mistakes: list[dict[str, Any]] = []
    missing_records = []

    for record_id, expected_fields in golden.items():
        predicted_fields = predictions.get(record_id)
        if predicted_fields is None:
            missing_records.append(record_id)
            predicted_fields = {}
        for field, expected in expected_fields.items():
            if expected is None or (isinstance(expected, str) and not expected.strip()):
                continue  # nothing to be right or wrong about
            bucket = per_field.setdefault(field, {"n": 0, "correct": 0})
            bucket["n"] += 1
            predicted = predicted_fields.get(field)
            if values_match(field, predicted, expected):
                bucket["correct"] += 1
            elif len(mistakes) < 50:  # a sample, so the report stays readable
                mistakes.append({"record": record_id, "field": field,
                                 "expected": expected, "predicted": predicted})

    for bucket in per_field.values():
        bucket["accuracy"] = round(bucket["correct"] / bucket["n"], 4) if bucket["n"] else 0.0

    total = sum(b["n"] for b in per_field.values())
    correct = sum(b["correct"] for b in per_field.values())
    return {
        "records": len(golden),
        "records_without_prediction": missing_records,
        "per_field": dict(sorted(per_field.items())),
        "overall": {"n": total, "correct": correct,
                    "accuracy": round(correct / total, 4) if total else 0.0},
        "sample_mistakes": mistakes,
    }


# ---------------------------------------------------------------------------
# Flattening: golden entries and extracted objects into the same {id: {field: value}}
# ---------------------------------------------------------------------------


def golden_statement_records(entries: Iterable[dict]) -> dict:
    records = {}
    for entry in entries:
        for fiscal_year, figures in entry["years"].items():
            record = {"company_name": entry.get("company_name", "")}
            record.update({field: figures.get(field) for field in FIGURE_FIELDS})
            records[f"{entry['company_id']}:{fiscal_year}"] = record
    return records


def predicted_statement_records(company_id: str, document: FinancialStatementExtraction) -> dict:
    records = {}
    for year in document.years:
        record = {"company_name": document.company_name}
        record.update({field: getattr(year, field) for field in FIGURE_FIELDS})
        records[f"{company_id}:{year.fiscal_year}"] = record
    return records


def golden_receipt_records(entries: Iterable[dict]) -> dict:
    return {f"receipt:{entry['file']}": {field: entry["entities"].get(field, "")
                                         for field in RECEIPT_FIELDS}
            for entry in entries}


def predicted_receipt_record(document: ReceiptExtraction) -> dict:
    return {field: getattr(document, field) for field in RECEIPT_FIELDS}


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run(limit: Optional[int] = None,
        doc_types: tuple[str, ...] = ("financial_statement", "receipt"),
        s: Settings = default_settings, write: bool = True, verbose: bool = True) -> dict:
    """Extract every golden document and score the result. Returns the report dict."""
    golden = load_golden_set(s=s)
    report: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "offline": bool(s.llm.offline),
        "model": s.llm.primary,
        "limit": limit,
        "doc_types": list(doc_types),
    }

    if "financial_statement" in doc_types:
        entries = golden["financial_statements"][:limit] if limit else golden["financial_statements"]
        expected, predicted, failures = golden_statement_records(entries), {}, []
        for entry in entries:
            path = s.paths.raw / "financials" / entry["source_file"]
            try:
                document = extract_document(path, "financial_statement", s)
                predicted.update(predicted_statement_records(entry["company_id"], document))
            except Exception as error:  # noqa: BLE001  a failed document is a score of zero, not a crash
                failures.append({"document": entry["source_file"], "error": f"{type(error).__name__}: {error}"})
        section = score(predicted, expected)
        section["documents"] = len(entries)
        section["failures"] = failures
        section["by_doc_kind"] = _accuracy_by_doc_kind(entries, predicted, expected)
        report["financial_statement"] = section

    if "receipt" in doc_types:
        entries = golden["receipts"][:limit] if limit else golden["receipts"]
        expected, predicted, failures = golden_receipt_records(entries), {}, []
        for entry in entries:
            path = s.paths.real / "receipts" / entry["file"]
            try:
                document = extract_document(path, "receipt", s)
                predicted[f"receipt:{entry['file']}"] = predicted_receipt_record(document)
            except Exception as error:  # noqa: BLE001
                failures.append({"document": entry["file"], "error": f"{type(error).__name__}: {error}"})
        section = score(predicted, expected)
        section["documents"] = len(entries)
        section["failures"] = failures
        report["receipt"] = section

    counted = [report[d]["overall"] for d in doc_types if d in report]
    total = sum(o["n"] for o in counted)
    correct = sum(o["correct"] for o in counted)
    report["overall"] = {"n": total, "correct": correct,
                         "accuracy": round(correct / total, 4) if total else 0.0}

    if write:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "extraction_accuracy.json").write_text(
            json.dumps(report, indent=1, default=str), encoding="utf-8")
    if verbose:
        print(format_report(report))
    return report


def _accuracy_by_doc_kind(entries: list[dict], predicted: dict, expected: dict) -> dict:
    """Text PDFs and scans deserve separate numbers: OCR is where accuracy goes to die."""
    kinds: dict[str, dict] = {}
    for entry in entries:
        ids = [key for key in expected if key.startswith(entry["company_id"] + ":")]
        part = score({k: v for k, v in predicted.items() if k in ids},
                     {k: expected[k] for k in ids})
        bucket = kinds.setdefault(entry.get("doc_kind", "unknown"), {"n": 0, "correct": 0, "documents": 0})
        bucket["n"] += part["overall"]["n"]
        bucket["correct"] += part["overall"]["correct"]
        bucket["documents"] += 1
    for bucket in kinds.values():
        bucket["accuracy"] = round(bucket["correct"] / bucket["n"], 4) if bucket["n"] else 0.0
    return kinds


def format_report(report: dict) -> str:
    """A plain-text table. No colours, no dependencies: it has to read well in CI logs."""
    lines = [
        "",
        f"Extraction accuracy  ({'offline stub' if report['offline'] else report['model']})",
        "=" * 62,
    ]
    for doc_type in report.get("doc_types", []):
        section = report.get(doc_type)
        if not section:
            continue
        lines.append(f"\n{doc_type}  ({section['documents']} documents)")
        lines.append(f"  {'field':<24}{'n':>6}{'correct':>10}{'accuracy':>12}")
        lines.append("  " + "-" * 52)
        for field, bucket in section["per_field"].items():
            lines.append(f"  {field:<24}{bucket['n']:>6}{bucket['correct']:>10}{bucket['accuracy']:>12.3f}")
        overall = section["overall"]
        lines.append("  " + "-" * 52)
        lines.append(f"  {'ALL FIELDS':<24}{overall['n']:>6}{overall['correct']:>10}{overall['accuracy']:>12.3f}")
        for kind, bucket in sorted(section.get("by_doc_kind", {}).items()):
            lines.append(f"    by document kind: {kind:<14}{bucket['documents']:>3} docs"
                         f"{bucket['accuracy']:>12.3f}")
        for failure in section.get("failures", []):
            lines.append(f"  FAILED {failure['document']}: {failure['error']}")
    lines.append("")
    lines.append(f"OVERALL accuracy: {report['overall']['accuracy']:.3f} "
                 f"({report['overall']['correct']}/{report['overall']['n']} fields)")
    lines.append(f"written to {RESULTS_DIR / 'extraction_accuracy.json'}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score document extraction against the golden set")
    parser.add_argument("--limit", type=int, default=None,
                        help="how many documents of each type to score (default: all)")
    parser.add_argument("--doc-types", nargs="+", default=["financial_statement", "receipt"],
                        choices=["financial_statement", "receipt"])
    args = parser.parse_args()
    run(limit=args.limit, doc_types=tuple(args.doc_types))


if __name__ == "__main__":
    main()

"""Build the golden set: the small, hand-checked file every accuracy number is measured against.

The one idea this module teaches: **you cannot improve what you do not measure, and you
cannot measure without a golden set.** A golden set is a short list of documents with the
right answer written down next to each one. It is boring to make and it is the single
highest-value artefact in the project: without it, "the extraction looks good" is an
opinion, and with it, a prompt change is a number that went up or down.

Two halves, deliberately different:

* **financial_statements** come from the generator's answer key. The truth is known
  exactly because we printed the PDF from it. That makes them a *unit test* of the
  extractor: anything less than perfect is our bug, not the document's fault.
* **receipts** come from SROIE, where real humans labelled real crumpled paper. The
  truth is someone's reading of a photograph, which is what production data is actually
  like. Scores here will never be 100 percent, and that is the honest lesson.

The file lives in `ground_truth/`, not in `data/`, because it is small, hand-checked and
belongs in git: a golden set that is regenerated silently is not a golden set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from creditlens.config import Settings, settings as default_settings

# Exactly the fields of creditlens.schemas.FiscalYearFigures that carry a number.
FIGURE_FIELDS = ("revenue", "ebitda", "interest", "principal_repayment", "total_debt",
                 "current_assets", "current_liabilities", "pat")

RECEIPT_FIELDS = ("company", "date", "address", "total")

# How many statements go into the golden set. Ten is enough to see a regression and
# small enough that a pod can re-check every figure by eye in one sitting.
N_STATEMENTS = 10


def _company_names(s: Settings) -> dict[str, str]:
    registry = s.paths.raw / "registry" / "companies.csv"
    if not registry.exists():
        return {}
    frame = pd.read_csv(registry)
    return dict(zip(frame["company_id"], frame["legal_name"]))


def financial_statement_entries(s: Settings, limit: int = N_STATEMENTS) -> list[dict]:
    """The first `limit` companies from the answer key, with their printed figures."""
    truth_file = s.paths.answer_key / "financials_truth.json"
    if not truth_file.exists():
        return []
    truth = json.loads(truth_file.read_text(encoding="utf-8"))
    kinds_file = s.paths.answer_key / "financial_doc_kinds.json"
    kinds = json.loads(kinds_file.read_text(encoding="utf-8")) if kinds_file.exists() else {}
    names = _company_names(s)

    entries = []
    for company_id in sorted(truth)[:limit]:
        years = {fy: {field: figures[field] for field in FIGURE_FIELDS}
                 for fy, figures in sorted(truth[company_id]["years"].items())}
        entries.append({
            "company_id": company_id,
            "source_file": f"{company_id}_financials.pdf",
            "company_name": names.get(company_id, ""),
            "unit": "INR thousands",
            # "text", "scan" or "scan_rotated": lets the eval report accuracy per document kind.
            "doc_kind": kinds.get(company_id, "unknown"),
            "years": years,
        })
    return entries


def receipt_entries(s: Settings) -> list[dict]:
    """Every SROIE receipt we downloaded, with the human labels as they ship."""
    labels_file = s.paths.real / "receipts" / "labels.json"
    if not labels_file.exists():
        return []
    labels = json.loads(labels_file.read_text(encoding="utf-8"))
    entries = []
    for file_name in sorted(labels):
        entities = labels[file_name].get("entities", {})
        entries.append({
            "file": file_name,
            "source_file": file_name,
            "entities": {field: str(entities.get(field, "")) for field in RECEIPT_FIELDS},
        })
    return entries


def build_golden_set(s: Settings = default_settings, out_path: Optional[Path] = None,
                     limit: int = N_STATEMENTS) -> Path:
    """Write `ground_truth/golden_set.json` and return the path to it."""
    golden = {
        "description": ("Hand-checked answers for CreditLens extraction. Statements come from the "
                        "generator answer key, receipts from the SROIE human labels."),
        "units": {"financial_statements": "INR thousands, whole numbers as printed",
                  "receipts": "strings exactly as printed on the paper"},
        "financial_statements": financial_statement_entries(s, limit),
        "receipts": receipt_entries(s),
    }
    path = Path(out_path) if out_path else s.paths.ground_truth / "golden_set.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(golden, indent=1, sort_keys=False), encoding="utf-8")
    return path


def load_golden_set(path: Optional[Path] = None, s: Settings = default_settings) -> dict:
    """Read the golden set, building it first if it has never been built."""
    path = Path(path) if path else s.paths.ground_truth / "golden_set.json"
    if not path.exists():
        build_golden_set(s, path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ground_truth/golden_set.json")
    parser.add_argument("--limit", type=int, default=N_STATEMENTS,
                        help="how many financial statements to include")
    parser.add_argument("--out", default=None, help="write somewhere else than ground_truth/")
    args = parser.parse_args()
    path = build_golden_set(default_settings, Path(args.out) if args.out else None, args.limit)
    golden = json.loads(path.read_text(encoding="utf-8"))
    print(f"wrote {path}")
    print(f"  financial_statements: {len(golden['financial_statements'])}")
    print(f"  receipts:             {len(golden['receipts'])}")


if __name__ == "__main__":
    main()

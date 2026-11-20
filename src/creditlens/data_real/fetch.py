"""Fetch small samples of real public datasets and store them under data/real/.

The workshop story is synthetic (Good Bank, Rohan, Priya) because real borrower data
must never enter the project. Three real public datasets plug in where they add realism:

| Where it is used            | Dataset                                   | Source        | Licence     |
| -------------------------- | ----------------------------------------- | ------------- | ----------- |
| Day 2 extraction and OCR   | ICDAR 2019 SROIE scanned receipts         | Hugging Face  | CC BY 4.0   |
| Day 3 visual retrieval     | ChartQA chart images with questions       | Hugging Face  | GPL 3.0     |
| Day 6 risk model           | SBA 7(a) loan book, "Should this loan..." | Kaggle        | CC BY-SA 4.0|
| Day 3 policy RAG           | SBA SOP 50 10 (US government, public)     | sba.gov       | Public domain |

Each fetcher keeps only a small sample (sizes in creditlens.config.Sizes), so a room
of laptops can download everything in a minute. The SBA loan book is 175 MB on disk
before sampling; the sample is written once and the original stays in the kagglehub cache.

Usage:
    python -m creditlens.data_real            # fetch everything
    python -m creditlens.data_real --only receipts charts sba sop
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from creditlens.config import Settings, settings as default_settings

SROIE_DATASET = "jsdnrs/ICDAR2019-SROIE"
CHARTQA_DATASET = "HuggingFaceM4/ChartQA"
SBA_KAGGLE = "mirbektoktogaraev/should-this-loan-be-approved-or-denied"
SBA_SOP_URLS = [
    # Newest first. Word documents, about 1 MB each, published by the US Small Business Administration.
    "https://legacy.sba.gov/sites/default/files/2026-10/SOP%2050%2010%208.1%20Technical%20Policy%20Updates%20effective%20%2010.1.2026.docx",
    "https://legacy.sba.gov/sites/default/files/2025-05/SOP%2050%2010%208%20Technical%20Updates%20effective%206.1.2025.docx",
]


def fetch_receipts(s: Settings = default_settings, force: bool = False) -> Path:
    """SROIE: scanned receipts with human-labelled company, date, address and total."""
    out = s.paths.real / "receipts"
    manifest = out / "labels.json"
    if manifest.exists() and not force:
        return out
    import itertools

    from datasets import load_dataset

    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(SROIE_DATASET, split="test", streaming=True)
    labels = {}
    for row in itertools.islice(ds, s.sizes.real_receipts):
        name = f"{row['key']}.jpg"
        row["image"].convert("RGB").save(out / name, quality=85)
        labels[name] = {"entities": row["entities"], "image_size": row["image_size"],
                        "words": row["words"]}
    manifest.write_text(json.dumps(labels, indent=1), encoding="utf-8")
    return out


def fetch_charts(s: Settings = default_settings, force: bool = False) -> Path:
    """ChartQA: real chart images, each with a question and a human answer."""
    out = s.paths.real / "charts"
    manifest = out / "questions.json"
    if manifest.exists() and not force:
        return out

    from datasets import load_dataset

    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(CHARTQA_DATASET, split="test", streaming=True)
    rows = []
    seen = 0
    for row in ds:
        if int(row.get("human_or_machine", 0)) != 0:  # keep human-written questions only
            continue
        name = f"chart_{seen:03d}.png"
        row["image"].convert("RGB").save(out / name)
        rows.append({"image": name, "question": row["query"], "answer": row["label"][0]})
        seen += 1
        if seen >= s.sizes.real_charts:
            break
    manifest.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return out


SBA_KEEP = ["LoanNr_ChkDgt", "State", "BankState", "NAICS", "ApprovalFY", "Term", "NoEmp", "NewExist",
            "CreateJob", "RetainedJob", "FranchiseCode", "UrbanRural", "RevLineCr", "LowDoc",
            "DisbursementGross", "GrAppv", "SBA_Appv", "MIS_Status"]


def _money(col: pd.Series) -> pd.Series:
    return pd.to_numeric(col.astype(str).str.replace(r"[\$,\s]", "", regex=True), errors="coerce")


def fetch_sba_loans(s: Settings = default_settings, force: bool = False) -> Path:
    """SBA 7(a) loans 1987 to 2014 with a charged-off flag: the real default labels for the risk model.

    Business names, cities and bank names are dropped at ingest. They are not needed for the
    model, and dropping them is the first PII-minimisation habit the workshop teaches.
    """
    out = s.paths.real / "sba_loans.csv"
    if out.exists() and not force:
        return out
    import glob

    import kagglehub

    folder = kagglehub.dataset_download(SBA_KAGGLE)
    csv = glob.glob(str(Path(folder) / "**" / "SBAnational.csv"), recursive=True)[0]
    df = pd.read_csv(csv, usecols=SBA_KEEP, low_memory=False)
    df = df[df.MIS_Status.isin(["P I F", "CHGOFF"])]
    df = df.sample(n=min(s.sizes.real_sba_loans, len(df)), random_state=s.seed)
    for c in ("DisbursementGross", "GrAppv", "SBA_Appv"):
        df[c] = _money(df[c])
    df["defaulted"] = (df.MIS_Status == "CHGOFF").astype(int)
    df["naics_sector"] = df.NAICS.astype(str).str[:2]
    df = df.drop(columns=["MIS_Status"]).rename(columns={"LoanNr_ChkDgt": "loan_id"})
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def fetch_sba_sop(s: Settings = default_settings, force: bool = False) -> Path:
    """The SBA lending rulebook (SOP 50 10) as Markdown, for the real-world RAG corpus."""
    out = s.paths.real / "sba_sop_50_10.md"
    if out.exists() and not force:
        return out
    import io

    import requests
    from docx import Document

    last = None
    for url in SBA_SOP_URLS:
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "creditlens-workshop/1.0"})
            r.raise_for_status()
            doc = Document(io.BytesIO(r.content))
            break
        except Exception as e:  # noqa: BLE001
            last = e
            doc = None
    if doc is None:
        raise RuntimeError(f"could not download the SBA SOP: {last}")
    lines = [f"# SBA SOP 50 10 (source: {url})", ""]
    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue
        style = (p.style.name or "").lower()
        m = re.match(r"heading (\d)", style)
        if m:
            lines.append("#" * (int(m.group(1)) + 1) + " " + text)
        elif style.startswith("title"):
            lines.append("# " + text)
        else:
            lines.append(text)
        lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


FETCHERS = {"receipts": fetch_receipts, "charts": fetch_charts, "sba": fetch_sba_loans, "sop": fetch_sba_sop}


def fetch_all(s: Settings = default_settings, only: list[str] | None = None, force: bool = False) -> dict:
    results = {}
    for name, fn in FETCHERS.items():
        if only and name not in only:
            continue
        try:
            results[name] = str(fn(s, force=force))
            print(f"{name:>9}: {results[name]}")
        except Exception as e:  # noqa: BLE001  one failing download must not block the others
            results[name] = f"FAILED: {e}"
            print(f"{name:>9}: FAILED: {e}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=list(FETCHERS))
    ap.add_argument("--force", action="store_true", help="re-download even if the sample exists")
    a = ap.parse_args()
    fetch_all(only=a.only, force=a.force)


if __name__ == "__main__":
    main()

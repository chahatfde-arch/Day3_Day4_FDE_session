"""How much do two labellers actually agree? Raw agreement, and Cohen's kappa.

Run it on the two CSVs that come out of a labelling party:

    python ground_truth/agreement.py labels_priya.csv labels_rohan.csv --key file

The one idea this script teaches: **raw agreement flatters you.** If 95 percent of
receipts are from the same three shops, two labellers who both guess "the usual shop"
agree 95 percent of the time while knowing nothing. Cohen's kappa subtracts the
agreement you would expect from chance alone:

    kappa = (observed - expected) / (1 - expected)

1.0 is perfect, 0.0 is no better than chance, and below 0 means the two of you are
disagreeing more than random guessing would. The usual reading is: above 0.8 is strong,
0.6 to 0.8 is usable, and below 0.6 means the *instructions* are unclear, not the
labellers. Fix the guidelines, then label again.

This file deliberately depends on nothing but pandas, so it can be handed to a labeller
who has never installed the project.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Sequence

import pandas as pd

MISSING = "<missing>"


def normalise(value) -> str:
    """The same comparison rule the eval uses: casefold, drop punctuation, collapse spaces."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return MISSING
    text = re.sub(r"[^\w\s]", " ", str(value))
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return text or MISSING


def cohen_kappa(labels_a: Sequence, labels_b: Sequence) -> float:
    """Chance-corrected agreement between two labellers on the same items.

    `labels_a[i]` and `labels_b[i]` must be the two labels for the same item. Values are
    compared as categories after normalisation, so this works for free text as well as
    for a fixed set of choices.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("both labellers must have labelled the same items, in the same order")
    a = [normalise(v) for v in labels_a]
    b = [normalise(v) for v in labels_b]
    n = len(a)
    if n == 0:
        return 0.0

    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    count_a, count_b = Counter(a), Counter(b)
    expected = sum((count_a[label] / n) * (count_b[label] / n)
                   for label in set(count_a) | set(count_b))
    if expected >= 1.0:
        # Everyone wrote the same single value: there is nothing chance could not explain.
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def agreement_table(frame_a: pd.DataFrame, frame_b: pd.DataFrame, key: str = "file",
                    fields: Sequence[str] | None = None) -> pd.DataFrame:
    """One row per field: how many items, how often the two agreed, and kappa."""
    merged = frame_a.merge(frame_b, on=key, suffixes=("_a", "_b"))
    if merged.empty:
        raise ValueError(f"the two files share no rows on column {key!r}")
    if fields is None:
        fields = [c for c in frame_a.columns
                  if c != key and f"{c}_a" in merged.columns and f"{c}_b" in merged.columns]

    rows = []
    for field in fields:
        a, b = merged[f"{field}_a"].tolist(), merged[f"{field}_b"].tolist()
        agreed = sum(1 for x, y in zip(a, b) if normalise(x) == normalise(y))
        rows.append({"field": field, "n": len(a), "agreed": agreed,
                     "agreement": round(agreed / len(a), 4) if a else 0.0,
                     "kappa": round(cohen_kappa(a, b), 4)})
    return pd.DataFrame(rows)


def verdict(kappa: float) -> str:
    if kappa >= 0.8:
        return "strong, ship the guidelines"
    if kappa >= 0.6:
        return "usable, tighten the unclear fields"
    return "too low, the guidelines are ambiguous"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inter-labeller agreement between two label CSVs")
    parser.add_argument("csv_a", type=Path, help="first labeller's CSV")
    parser.add_argument("csv_b", type=Path, help="second labeller's CSV")
    parser.add_argument("--key", default="file", help="column that identifies the document")
    parser.add_argument("--fields", nargs="*", default=None, help="which columns to compare")
    args = parser.parse_args()

    table = agreement_table(pd.read_csv(args.csv_a), pd.read_csv(args.csv_b),
                            key=args.key, fields=args.fields)
    print()
    print(table.to_string(index=False))
    print()
    mean_kappa = float(table["kappa"].mean())
    print(f"mean kappa across fields: {mean_kappa:.3f}  ({verdict(mean_kappa)})")
    worst = table.sort_values("kappa").iloc[0]
    print(f"weakest field: {worst['field']} (kappa {worst['kappa']:.3f}) "
          f"- re-read that section of LABELING_GUIDELINES.md together")


if __name__ == "__main__":
    main()

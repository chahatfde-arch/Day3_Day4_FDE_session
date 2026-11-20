"""Data drift: is what we score today still shaped like what we trained on?

The one idea this module teaches: **models do not break, the world moves**. The risk model
was trained on US SBA loans approved up to 2014. It scores Indian SME applications from 2026
that have been pushed through an adapter. Some columns will never match, and that is fine as
long as somebody knows which ones. Drift monitoring is how they know.

    python -m monitoring.drift                 # reference (SBA) against today's applications
    python -m monitoring.drift --inject-drift  # the Day 7 drill: amounts tripled, terms shifted
    python -m monitoring.drift --no-html       # JSON summary only, for CI

Output:
    monitoring/reports/drift_report.html   the Evidently report, when Evidently is usable
    monitoring/reports/drift_summary.json  always written: drifted columns and the method used

Evidently's API changed between 0.4 (`Report(metrics=[DataDriftPreset()])` then
`report.run(reference_data=..., current_data=...)`) and 0.6 or newer (`Dataset.from_pandas`
then `report.run(reference_data, current_data)` returning a snapshot). We try both and, if
neither works, fall back to a plain two-sample test per column, so the drill always produces
a summary. A monitoring tool that can itself go down is not monitoring.
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Optional

import pandas as pd

from creditlens.config import settings
from creditlens.ml.features import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    BINARY_FEATURES,
    applications_to_frame,
    load_sba,
    sba_features,
)

REPORTS_DIR = settings.paths.root / "monitoring" / "reports"
HTML_PATH = REPORTS_DIR / "drift_report.html"
SUMMARY_PATH = REPORTS_DIR / "drift_summary.json"

REFERENCE_SAMPLE = 5000
KS_P_VALUE = 0.05          # below this, the two samples are unlikely to share a distribution
CATEGORY_SHARE_GAP = 0.2   # a category whose share moved by more than this counts as drifted

# The drill's fake incident: a sales push for bigger, longer loans.
DRIFT_AMOUNT_MULTIPLIER = 3.0
DRIFT_TERM_SHIFT = 48


# ---------------------------------------------------------------------------
# The two datasets
# ---------------------------------------------------------------------------


def reference_frame(sample: int = REFERENCE_SAMPLE) -> pd.DataFrame:
    """What the model was trained on."""
    raw = load_sba()
    X, _ = sba_features(raw)
    if sample and len(X) > sample:
        X = X.sample(n=sample, random_state=settings.seed)
    return X


def current_frame(inject_drift: bool = False) -> pd.DataFrame:
    """What we are scoring now: Good Bank applications pushed through the adapter."""
    applications = pd.read_csv(settings.paths.raw / "applications" / "applications.csv")
    companies = pd.read_csv(settings.paths.raw / "registry" / "companies.csv")
    statements_path = settings.paths.curated / "statement_features.csv"
    statements = pd.read_csv(statements_path) if statements_path.exists() else None

    if inject_drift:
        applications = applications.copy()
        applications["requested_inr_000"] = (
            applications["requested_inr_000"] * DRIFT_AMOUNT_MULTIPLIER).round().astype(int)
        applications["tenure_months"] = applications["tenure_months"] + DRIFT_TERM_SHIFT

    frame = applications_to_frame(applications, companies, statements)
    return frame.reset_index(drop=True)[MODEL_FEATURES]


# ---------------------------------------------------------------------------
# Fallback: a two-sample test per column, with no extra dependency beyond scipy
# ---------------------------------------------------------------------------


def ks_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, Any]:
    """Kolmogorov-Smirnov for numbers, share comparison for categories.

    KS asks: could these two samples plausibly have come from the same distribution? A small
    p-value means no. With 30 applications it is a blunt instrument, which is exactly why the
    summary reports the sample size next to every verdict.
    """
    from scipy import stats

    columns: list[dict[str, Any]] = []
    for name in MODEL_FEATURES:
        if name in CATEGORICAL_FEATURES:
            ref_share = reference[name].astype(str).value_counts(normalize=True)
            cur_share = current[name].astype(str).value_counts(normalize=True)
            categories = set(ref_share.index) | set(cur_share.index)
            gap = max(abs(float(ref_share.get(c, 0.0)) - float(cur_share.get(c, 0.0)))
                      for c in categories) if categories else 0.0
            columns.append({"column": name, "type": "categorical", "test": "max share gap",
                            "statistic": round(gap, 4), "p_value": None,
                            "drifted": bool(gap > CATEGORY_SHARE_GAP)})
            continue

        ref_values = pd.to_numeric(reference[name], errors="coerce").dropna()
        cur_values = pd.to_numeric(current[name], errors="coerce").dropna()
        if len(ref_values) < 5 or len(cur_values) < 5:
            columns.append({"column": name, "type": "numeric", "test": "ks", "statistic": None,
                            "p_value": None, "drifted": False, "note": "not enough rows"})
            continue
        statistic, p_value = stats.ks_2samp(ref_values, cur_values)
        columns.append({
            "column": name,
            "type": "binary" if name in BINARY_FEATURES else "numeric",
            "test": "ks", "statistic": round(float(statistic), 4),
            "p_value": round(float(p_value), 6),
            "drifted": bool(p_value < KS_P_VALUE),
            "reference_median": round(float(ref_values.median()), 2),
            "current_median": round(float(cur_values.median()), 2),
        })
    return {"method": "scipy KS and category share gap", "columns": columns}


# ---------------------------------------------------------------------------
# Evidently, across two incompatible major versions
# ---------------------------------------------------------------------------


def _evidently_report(reference: pd.DataFrame, current: pd.DataFrame,
                      write_html: bool) -> Optional[dict[str, Any]]:
    """Try Evidently 0.6+ first, then the 0.4 API. Return None when neither works."""
    try:  # Evidently 0.6 and newer
        from evidently import Dataset, DataDefinition, Report
        from evidently.presets import DataDriftPreset

        definition = DataDefinition(
            numerical_columns=[c for c in NUMERIC_FEATURES + BINARY_FEATURES],
            categorical_columns=list(CATEGORICAL_FEATURES))
        ref_ds = Dataset.from_pandas(reference, data_definition=definition)
        cur_ds = Dataset.from_pandas(current, data_definition=definition)
        report = Report(metrics=[DataDriftPreset()])
        snapshot = report.run(reference_data=ref_ds, current_data=cur_ds)
        if write_html:
            snapshot.save_html(str(HTML_PATH))
        return {"method": "evidently>=0.6 DataDriftPreset",
                "columns": _columns_from_evidently(snapshot.dict())}
    except Exception as new_api_error:  # noqa: BLE001
        try:  # Evidently 0.4.x
            from evidently.metric_preset import DataDriftPreset
            from evidently.report import Report

            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=reference, current_data=current)
            if write_html:
                report.save_html(str(HTML_PATH))
            return {"method": "evidently 0.4 DataDriftPreset",
                    "columns": _columns_from_evidently(report.as_dict())}
        except Exception as old_api_error:  # noqa: BLE001
            print("Evidently could not produce a report, falling back to a KS test per column.")
            print(f"  evidently 0.6+ API said: {type(new_api_error).__name__}: "
                  f"{str(new_api_error)[:140]}")
            print(f"  evidently 0.4 API said : {type(old_api_error).__name__}: "
                  f"{str(old_api_error)[:140]}")
            return None


def _columns_from_evidently(payload: Any) -> list[dict[str, Any]]:
    """Dig the per-column verdicts out of whichever shape Evidently returned."""
    found: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            name = node.get("column_name")
            if name and ("drift_detected" in node or "drift_score" in node):
                found.append({
                    "column": str(name),
                    "type": str(node.get("column_type", "")),
                    "test": str(node.get("stattest_name", "evidently")),
                    "statistic": node.get("drift_score"),
                    "p_value": None,
                    "drifted": bool(node.get("drift_detected", False)),
                })
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)
    # Keep one row per column, the first one wins.
    unique: dict[str, dict[str, Any]] = {}
    for row in found:
        unique.setdefault(row["column"], row)
    return list(unique.values())


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------


def run(inject_drift: bool = False, write_html: bool = True, sample: int = REFERENCE_SAMPLE,
        verbose: bool = True) -> dict[str, Any]:
    """Compare training features with today's features and write the summary."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    reference = reference_frame(sample=sample)
    current = current_frame(inject_drift=inject_drift)

    result = _evidently_report(reference, current, write_html) or ks_drift(reference, current)
    if not result.get("columns"):
        result = ks_drift(reference, current)

    drifted = [c["column"] for c in result["columns"] if c.get("drifted")]
    summary = {
        "method": result["method"],
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "injected_drift": inject_drift,
        "n_columns": len(result["columns"]),
        "n_drifted": len(drifted),
        "share_drifted": round(len(drifted) / max(1, len(result["columns"])), 4),
        "drifted_columns": sorted(drifted),
        "columns": result["columns"],
        "html_report": str(HTML_PATH) if (write_html and HTML_PATH.exists()) else None,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")

    if verbose:
        print(f"Drift check using {summary['method']}")
        print(f"  reference: {summary['reference_rows']} SBA training rows")
        print(f"  current  : {summary['current_rows']} Good Bank applications"
              + (" with injected drift" if inject_drift else ""))
        print(f"  drifted  : {summary['n_drifted']} of {summary['n_columns']} columns")
        for column in result["columns"]:
            if column.get("drifted"):
                detail = ""
                if column.get("reference_median") is not None:
                    detail = (f"  median {column['reference_median']:,} -> "
                              f"{column['current_median']:,}")
                print(f"    - {column['column']:22s} {column['test']:16s}"
                      f" stat {column.get('statistic')}{detail}")
        if summary["html_report"]:
            print(f"  HTML report: {summary['html_report']}")
        print(f"  Summary    : {SUMMARY_PATH}")
        print("  Reading it: a drifted column is not automatically a problem. ApprovalFY always "
              "drifts, because the training data stops in 2014. Amount and Term drifting at the "
              "same time is the signal worth paging somebody about.")
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Data drift: training features vs today's")
    parser.add_argument("--inject-drift", action="store_true",
                        help="triple the requested amounts and add 48 months to every term")
    parser.add_argument("--no-html", action="store_true", help="skip the HTML report")
    parser.add_argument("--sample", type=int, default=REFERENCE_SAMPLE)
    args = parser.parse_args(argv)
    run(inject_drift=args.inject_drift, write_html=not args.no_html, sample=args.sample)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

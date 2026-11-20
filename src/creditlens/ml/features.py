"""Features for the default-risk model, and the adapter that maps Good Bank onto them.

The one idea this module teaches: **train/serve skew**.

We train on a real, public loan book (25,000 US SBA 7(a) loans with a real `defaulted`
label). We serve Indian SME applications from Good Bank, in INR thousands, with Indian
sector names. Those two worlds do not share a single column name. Something has to
translate, and that something is `application_to_sba_features`. Every line of the
adapter is an assumption, so every assumption is written down in `ASSUMPTIONS` below.
When the model behaves oddly in production, the adapter is the first place to look.

Two functions:

    sba_features(df)                      -> (X, y)   for training
    application_to_sba_features(app, company, stmt) -> dict  for scoring one application

A deliberate choice: `State` and `BankState` are **not** model features. A credit model
that learns geography has usually learned a proxy for who lives there, not for who repays.
We keep both columns on the side as "segment" columns so the fairness report in
`creditlens.ml.train` can measure whether the model ended up disadvantaging a state anyway.
"""
from __future__ import annotations

import os
from datetime import date
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# The model schema. Everything that trains and everything that scores uses these lists.
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "Term",                  # months of the loan
    "NoEmp",                 # employees at application time
    "CreateJob",             # jobs the loan is expected to create
    "RetainedJob",           # jobs the loan is expected to retain
    "GrAppv",                # gross amount approved by the bank, USD
    "SBA_Appv",              # the part of it the SBA guarantees, USD
    "sba_guarantee_share",   # SBA_Appv / GrAppv: how much risk the bank keeps
    "ApprovalFY",            # fiscal year of approval (an economic-cycle proxy)
]

BINARY_FEATURES = [
    "RevLineCr",             # 1 = revolving line of credit
    "LowDoc",                # 1 = the lender's low-documentation programme
    "is_new_business",       # 1 = a new business (SBA NewExist == 2)
    "is_franchise",          # 1 = operates under a franchise agreement
]

CATEGORICAL_FEATURES = [
    "UrbanRural",            # "0" unknown, "1" urban, "2" rural
    "naics_sector",          # NAICS 2-digit industry code, as a string
]

MODEL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# Kept out of the model on purpose, kept on the side for the fairness report.
SEGMENT_FEATURES = ["State", "BankState"]

TARGET = "defaulted"


# ---------------------------------------------------------------------------
# The adapter's assumptions. Read these out loud in the session: every one of them
# is a place where the model could be confidently wrong.
# ---------------------------------------------------------------------------

ASSUMPTIONS = [
    "Currency: Good Bank amounts are INR thousands. We convert to USD with "
    "INR_PER_USD (environment variable, default 84). A wrong rate silently rescales "
    "every amount feature, so the rate is read from the environment, never hard-coded "
    "in a formula.",
    "Guarantee: Good Bank loans carry no SBA guarantee. The SBA model expects one, so we "
    "assume the standard 75 percent guarantee share (ASSUMED_GUARANTEE_SHARE). This is the "
    "single largest fiction in the adapter.",
    "Sector: the 8 synthetic Good Bank sectors plus the excluded 'Online Betting' sector are "
    "mapped to the nearest NAICS 2-digit code (SECTOR_TO_NAICS). The mapping is coarse: "
    "'Textiles' and 'Food Processing' both land in manufacturing (31).",
    "New business: the SBA field NewExist is 2 for a new business. We call a Good Bank "
    "company new when it has been incorporated for less than NEW_BUSINESS_YEARS (2) years.",
    "Urban or rural: Good Bank's registry has a city but no urban/rural classification, so "
    "UrbanRural is set to '0', which in the SBA data means unknown. The model has seen plenty "
    "of unknowns, so this is a safe default rather than a guess.",
    "Low documentation: the SBA LowDoc programme applied to loans at or below "
    "LOWDOC_MAX_USD (150,000 USD). We set LowDoc from the converted requested amount.",
    "Revolving credit: Good Bank's 'Working Capital' product is the closest thing to a "
    "revolving line of credit, so RevLineCr is 1 for that product and 0 for Term Loan and "
    "Equipment Finance.",
    "Jobs: Good Bank applications do not state job creation. CreateJob is 0 and RetainedJob "
    "is the company's current headcount.",
    "Franchise: Good Bank's registry has no franchise flag, so is_franchise is 0 for every "
    "applicant. If Good Bank ever lends to franchisees, this feature goes stale first.",
    "Approval year: ApprovalFY is the year the application was submitted. The SBA data stops "
    "in 2014, so every Good Bank application is outside the training range of this feature. "
    "This is a real, measurable drift and monitoring/drift.py will flag it.",
    "Bank conduct features (average inflow, cheque returns, cash share) have no home in the "
    "SBA schema, so the adapter accepts the statement-features row and deliberately ignores "
    "it. Good Bank's best signal is invisible to this model, which is exactly why the memo "
    "shows the policy checks next to the score and a human decides.",
]

ASSUMED_GUARANTEE_SHARE = 0.75
NEW_BUSINESS_YEARS = 2.0
LOWDOC_MAX_USD = 150_000.0
DEFAULT_INR_PER_USD = 84.0

SECTOR_TO_NAICS = {
    "Manufacturing": "31",     # 31-33 manufacturing
    "Trading": "42",           # wholesale trade
    "Logistics": "48",         # transportation and warehousing
    "Food Processing": "31",   # 311 food manufacturing
    "Textiles": "31",          # 313-316 textile mills and apparel
    "IT Services": "54",       # 5415 computer systems design services
    "Construction": "23",      # construction
    "Healthcare": "62",        # health care and social assistance
    "Online Betting": "71",    # 7132 gambling industries (excluded by Good Bank policy 2.2)
}

PRODUCT_REVOLVING = "Working Capital"


def inr_per_usd() -> float:
    """Conversion rate, read at call time so a test or a session can change it."""
    try:
        return float(os.environ.get("INR_PER_USD", DEFAULT_INR_PER_USD))
    except ValueError:
        return DEFAULT_INR_PER_USD


# ---------------------------------------------------------------------------
# Training features
# ---------------------------------------------------------------------------


def _yes_no_to_flag(series: pd.Series) -> pd.Series:
    """'Y' is 1, everything else (including 'N', '0', junk and missing) is 0.

    The raw SBA file has values like 'T', ',' and a backtick in these columns. Rather than
    invent a meaning for them we treat anything that is not a clear yes as a no, and we say
    so here so nobody later assumes the column was clean.
    """
    flag = series.astype("string").str.strip().str.upper().eq("Y")
    # fillna before astype: a missing value is a nullable NA, and NA cannot become 0 by itself.
    return flag.fillna(False).astype(bool).astype(int)


def sba_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Turn the raw `data/real/sba_loans.csv` frame into (X, y).

    X has exactly `MODEL_FEATURES` columns and the same index as the rows we kept, so a
    caller can line up segment columns with `df.loc[X.index, "State"]`.
    """
    df = df.copy()
    out = pd.DataFrame(index=df.index)

    # Numeric columns. `errors="coerce"` turns anything unparseable into NaN rather than
    # raising, and the model pipeline imputes those.
    for col in ("Term", "NoEmp", "CreateJob", "RetainedJob", "GrAppv", "SBA_Appv", "ApprovalFY"):
        out[col] = pd.to_numeric(df[col], errors="coerce")

    # How much of the loan the government guarantees. The bank keeps the rest, so a low
    # share means the bank has more skin in the game.
    gross = out["GrAppv"].replace(0, np.nan)
    out["sba_guarantee_share"] = (out["SBA_Appv"] / gross).clip(lower=0, upper=1)

    out["RevLineCr"] = _yes_no_to_flag(df["RevLineCr"])
    out["LowDoc"] = _yes_no_to_flag(df["LowDoc"])

    # NewExist: 1 = existing business, 2 = new business, 0 and missing = unknown.
    new_exist = pd.to_numeric(df["NewExist"], errors="coerce")
    out["is_new_business"] = (new_exist == 2).astype(int)

    # FranchiseCode: 0 and 1 both mean "no franchise" in this dataset; any other code is a
    # real franchise identifier.
    franchise = pd.to_numeric(df["FranchiseCode"], errors="coerce").fillna(0)
    out["is_franchise"] = (~franchise.isin([0, 1])).astype(int)

    # Categorical columns are strings so one-hot encoding treats them as labels, not numbers.
    out["UrbanRural"] = pd.to_numeric(df["UrbanRural"], errors="coerce").fillna(0).astype(int).astype(str)
    out["naics_sector"] = pd.to_numeric(df["naics_sector"], errors="coerce").fillna(0).astype(int).astype(str)

    y = pd.to_numeric(df[TARGET], errors="coerce")
    keep = y.notna()
    return out.loc[keep, MODEL_FEATURES], y.loc[keep].astype(int)


def sba_segments(df: pd.DataFrame, index: Optional[pd.Index] = None) -> pd.DataFrame:
    """The columns we refuse to train on but still measure fairness over."""
    seg = df.reindex(columns=SEGMENT_FEATURES).copy()
    seg["UrbanRural"] = pd.to_numeric(df["UrbanRural"], errors="coerce").fillna(0).astype(int).astype(str)
    return seg if index is None else seg.loc[index]


def load_sba(data_path: Optional[str] = None) -> pd.DataFrame:
    """Read the SBA loan book. Kept here so train, drift and tests all read it the same way."""
    from creditlens.config import settings

    path = data_path or (settings.paths.real / "sba_loans.csv")
    return pd.read_csv(path, low_memory=False)


# ---------------------------------------------------------------------------
# The adapter: one Good Bank application -> one row of the model schema
# ---------------------------------------------------------------------------


def _years_in_business(incorporated_on: Any, as_of: Optional[date] = None) -> float:
    if not incorporated_on or (isinstance(incorporated_on, float) and np.isnan(incorporated_on)):
        return float(NEW_BUSINESS_YEARS)  # unknown: treat as established, the neutral choice
    try:
        start = pd.to_datetime(incorporated_on).date()
    except (ValueError, TypeError):
        return float(NEW_BUSINESS_YEARS)
    today = as_of or date.today()
    return max(0.0, (today - start).days / 365.25)


def _get(row: Optional[Mapping[str, Any]], key: str, default: Any = None) -> Any:
    """Read a field from a dict, a pandas Series or None without caring which it is."""
    if row is None:
        return default
    try:
        value = row[key]
    except (KeyError, IndexError, TypeError):
        return default
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    return value


def application_to_sba_features(
    application_row: Mapping[str, Any],
    company_row: Optional[Mapping[str, Any]] = None,
    statement_features_row: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Map one Good Bank application onto the SBA model schema. See ASSUMPTIONS above.

    `statement_features_row` is accepted and deliberately unused: the SBA schema has no
    place for bank-conduct features. That gap is the lesson, not an oversight.
    """
    rate = inr_per_usd()

    requested_inr_000 = float(_get(application_row, "requested_inr_000", 0) or 0)
    gross_usd = requested_inr_000 * 1000.0 / rate
    sba_usd = gross_usd * ASSUMED_GUARANTEE_SHARE

    tenure_months = float(_get(application_row, "tenure_months", 0) or 0)
    employees = float(_get(company_row, "employees", 0) or 0)
    sector = str(_get(company_row, "sector", "") or "")
    product = str(_get(application_row, "product", "") or "")

    submitted_on = _get(application_row, "submitted_on")
    try:
        as_of = pd.to_datetime(submitted_on).date() if submitted_on else date.today()
    except (ValueError, TypeError):
        as_of = date.today()

    years = _years_in_business(_get(company_row, "incorporated_on"), as_of=as_of)

    return {
        "Term": tenure_months,
        "NoEmp": employees,
        "CreateJob": 0.0,
        "RetainedJob": employees,
        "GrAppv": round(gross_usd, 2),
        "SBA_Appv": round(sba_usd, 2),
        "sba_guarantee_share": ASSUMED_GUARANTEE_SHARE,
        "ApprovalFY": float(as_of.year),
        "RevLineCr": 1 if product == PRODUCT_REVOLVING else 0,
        "LowDoc": 1 if gross_usd <= LOWDOC_MAX_USD else 0,
        "is_new_business": 1 if years < NEW_BUSINESS_YEARS else 0,
        "is_franchise": 0,
        "UrbanRural": "0",
        "naics_sector": SECTOR_TO_NAICS.get(sector, "0"),
    }


def applications_to_frame(
    applications: pd.DataFrame,
    companies: pd.DataFrame,
    statement_features: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Run the adapter over every application. Used by serving and by the drift report."""
    companies_by_id = companies.set_index("company_id").to_dict("index") if len(companies) else {}
    stmt_by_app = {}
    if statement_features is not None and len(statement_features):
        stmt_by_app = statement_features.set_index("application_id").to_dict("index")

    rows = []
    for _, app in applications.iterrows():
        company = companies_by_id.get(app.get("company_id"))
        stmt = stmt_by_app.get(app.get("application_id"))
        row = application_to_sba_features(app, company, stmt)
        row["application_id"] = app.get("application_id")
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.set_index("application_id") if "application_id" in frame.columns else frame

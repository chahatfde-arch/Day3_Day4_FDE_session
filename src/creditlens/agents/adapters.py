"""Adapters: one function per module that another milestone owns.

The one idea this module teaches: **never import a teammate's module at the top of your
file**. Milestones 4, 5, 8 and 9 are built by other pods, in parallel, and on any given
morning their modules may be missing, half finished or broken. So every call looks like
this:

    1. Try the real module (imported *inside* the function, so an ImportError is local).
    2. If it is not there yet, fall back to something honest that reads from `data/`.
    3. Return exactly the shape in `creditlens.schemas`, whichever path ran.

That way the agent graph never changes when the real module lands: it just gets better
numbers. The fallbacks are deliberately simple and never read `data/answer_key/`, which
is instructor-only truth.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from creditlens.config import settings
from creditlens.schemas import (AuditEvent, FinancialStatementExtraction, FiscalYearFigures,
                                GroupExposure, GroupMember, PolicyHit, RiskScore, User)

# ---------------------------------------------------------------------------
# Where the data lives. Tests point CREDITLENS_DATA_ROOT at a temp folder.
# ---------------------------------------------------------------------------


def data_root() -> Path:
    """The `data/` folder. Overridable with CREDITLENS_DATA_ROOT so tests stay isolated."""
    override = os.environ.get("CREDITLENS_DATA_ROOT")
    return Path(override) if override else settings.paths.data


def cache_dir() -> Path:
    d = data_root() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def raw_dir() -> Path:
    return data_root() / "raw"


def database_url() -> str:
    """The SQLite (or SQL Server) URL the agent reads from."""
    return os.environ.get("CREDITLENS_DB_URL") or settings.database_url


def get_engine(engine=None):
    """Return the SQLAlchemy engine to use, creating the default one when none is given."""
    if engine is not None:
        return engine
    from sqlalchemy import create_engine

    return create_engine(database_url())


# ---------------------------------------------------------------------------
# 1. Financial figures: milestone 4 (extraction), fallback = read the PDF text ourselves.
# ---------------------------------------------------------------------------

#: PDF line label -> field name on FiscalYearFigures.
_PDF_LABELS = {
    "Revenue from operations": "revenue",
    "EBITDA": "ebitda",
    "Finance costs": "interest",
    "Profit after tax": "pat",
    "Total current assets": "current_assets",
    "Trade payables and other current liabilities": "current_liabilities",
    "Total borrowings": "total_debt",
}


def financial_pdf_path(company_id: str) -> Path:
    return raw_dir() / "financials" / f"{company_id}_financials.pdf"


def get_financials(company_id: str) -> Optional[FinancialStatementExtraction]:
    """Figures from the audited statement PDF.

    Returns None only when the PDF is missing. For a scanned PDF with no text layer the
    fallback returns an extraction with `years == []`: the figures are simply not known
    without OCR, and a credit memo must say "not available" rather than guess.
    """
    try:  # milestone 4, if that pod has landed it
        from creditlens.extraction.extract import extract_financials

        return extract_financials(company_id)
    except Exception:  # noqa: BLE001  ImportError today, anything tomorrow
        return _financials_from_pdf_text(company_id)


def _financials_from_pdf_text(company_id: str) -> Optional[FinancialStatementExtraction]:
    path = financial_pdf_path(company_id)
    if not path.exists():
        return None
    text = _pdf_text(path)
    years = _parse_financial_text(text)
    return FinancialStatementExtraction(
        company_name=_company_name(company_id) or company_id,
        years=years,
        source_file=str(path),
        ocr_used=False,
        # No text layer means a scan: be honest about it instead of inventing numbers.
        confidence=1.0 if years else 0.0,
    )


def _pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover - pypdf is in the extraction extra
        return ""
    try:
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:  # noqa: BLE001  a damaged PDF must not crash the agent
        return ""


_NUMBER_LINE = re.compile(r"^-?[\d,]+$")


def _parse_financial_text(text: str) -> list[FiscalYearFigures]:
    """Parse the printed statement.

    The generator prints a table, and pypdf flattens it to a label line followed by one
    line per fiscal year, with Indian digit grouping ("12,34,567"). Some PDF readers keep
    the label and its numbers on one line instead, so both layouts are handled.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    fiscal_years: list[str] = []
    for ln in lines:
        for fy in re.findall(r"\bFY\d{4}\b", ln):
            if fy not in fiscal_years:
                fiscal_years.append(fy)
    if not fiscal_years:
        return []

    values: dict[str, dict[str, int]] = {fy: {} for fy in fiscal_years}
    for i, line in enumerate(lines):
        for label, field in _PDF_LABELS.items():
            numbers = _numbers_for_label(lines, i, line, label, len(fiscal_years))
            if numbers is None:
                continue
            for fy, number in zip(fiscal_years, numbers):
                values[fy].setdefault(field, number)

    # "Note 14: Borrowings ... FY2024: 2,302; FY2025: 1,613; FY2026: 3,189."
    for fy, raw in re.findall(r"(FY\d{4}):\s*([\d,]+)", text):
        if fy in values:
            values[fy].setdefault("principal_repayment", _to_int(raw))

    return [FiscalYearFigures(fiscal_year=fy, **values[fy]) for fy in fiscal_years]


def _numbers_for_label(lines: list[str], i: int, line: str, label: str,
                       count: int) -> Optional[list[int]]:
    """Return the `count` numbers printed against `label`, or None if this is not that row."""
    if line == label:  # table flattened: numbers are on the following lines
        found = []
        for nxt in lines[i + 1:i + 1 + count]:
            if not _NUMBER_LINE.match(nxt):
                return None
            found.append(_to_int(nxt))
        return found if len(found) == count else None
    if line.startswith(label):  # table kept on one line: "Label  12,34,567  ..."
        rest = re.findall(r"-?[\d,]+", line[len(label):])
        if len(rest) >= count:
            return [_to_int(x) for x in rest[:count]]
    return None


def _to_int(raw: str) -> int:
    return int(raw.replace(",", ""))


def _company_name(company_id: str) -> Optional[str]:
    path = raw_dir() / "registry" / "companies.csv"
    if not path.exists():
        return None
    import csv

    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["company_id"] == company_id:
                return row["legal_name"]
    return None


def latest_year(fin: Optional[FinancialStatementExtraction]) -> Optional[FiscalYearFigures]:
    """The most recent fiscal year, or None when the statement could not be read."""
    return fin.latest() if fin is not None else None


# ---------------------------------------------------------------------------
# 2. Policy retrieval: milestone 5, fallback = keyword scan of the manual's sections.
# ---------------------------------------------------------------------------


def _heavy_models_allowed() -> bool:
    """Offline mode means offline everywhere: no embedding models, no downloads.

    `creditlens.retrieval.policy` builds a vector index and loads a sentence-transformers
    model the first time it runs, which needs the Hugging Face cache and a minute or two.
    Tests and CI run with LLM_OFFLINE=1, so there we take the keyword fallback instead.
    """
    if os.environ.get("CREDITLENS_HEAVY_RETRIEVAL") == "1":
        return True
    return not settings.llm.offline


def search_policy(query: str, k: int = 3, source: Optional[str] = None) -> list[PolicyHit]:
    if _heavy_models_allowed():
        try:
            from creditlens.retrieval.policy import search_policy as real_search

            return real_search(query, k=k, source=source)
        except Exception:  # noqa: BLE001
            pass
    return _keyword_policy_search(query, k)


def _keyword_policy_search(query: str, k: int) -> list[PolicyHit]:
    """Count how many query words appear in each section. Crude, explainable, no model."""
    from creditlens.generators.policy import policy_sections

    words = {w for w in re.findall(r"[a-z0-9.]+", query.lower()) if len(w) > 2}
    scored = []
    for heading, body in policy_sections():
        haystack = f"{heading} {body}".lower()
        score = sum(1 for w in words if w in haystack)
        if score:
            scored.append((score, heading, body))
    scored.sort(key=lambda t: -t[0])
    return [PolicyHit(chunk_id=f"good_bank_policy#{heading.split('.')[0].strip()}",
                      source="good_bank_policy", section=heading, text=body,
                      score=float(score), retriever="keyword-fallback")
            for score, heading, body in scored[:k]]


# ---------------------------------------------------------------------------
# 3. Group exposure: milestone 5 GraphStore, fallback = NetworkX over the registry CSVs.
# ---------------------------------------------------------------------------


def get_group_exposure(company_id: str) -> GroupExposure:
    try:
        from creditlens.retrieval.graph import GraphStore

        store = GraphStore()
        store.load_from_registry(raw_dir())
        requested = _requested_by_company()
        return store.group_exposure(company_id, requested)
    except Exception:  # noqa: BLE001
        return _networkx_group_exposure(company_id)


CYPHER = (
    "MATCH (c:Company {company_id: $company_id})\n"
    "MATCH path = (c)-[:SHARES_DIRECTOR|SHARES_ADDRESS*0..6]-(m:Company)\n"
    "WITH collect(DISTINCT m) AS members\n"
    "UNWIND members AS m\n"
    "RETURN m.company_id, m.legal_name, m.existing_outstanding_inr_000, m.pending_request_inr_000"
)


def _read_registry_csv(name: str) -> list[dict]:
    import csv

    path = raw_dir() / "registry" / f"{name}.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _requested_by_company() -> dict[str, int]:
    import csv

    path = raw_dir() / "applications" / "applications.csv"
    out: dict[str, int] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["company_id"]] = out.get(row["company_id"], 0) + int(row["requested_inr_000"])
    return out


def related_group(company_id: str) -> list[str]:
    """Company ids linked to this one by a shared director or a shared registered address."""
    import networkx as nx

    companies = _read_registry_csv("companies")
    links = _read_registry_csv("company_directors")
    g = nx.Graph()
    for row in companies:
        g.add_node(row["company_id"])
    by_director: dict[str, list[str]] = {}
    for row in links:
        by_director.setdefault(row["director_id"], []).append(row["company_id"])
    for shared in by_director.values():
        for a in shared:
            for b in shared:
                if a != b:
                    g.add_edge(a, b, link="shared_director")
    by_address: dict[str, list[str]] = {}
    for row in companies:
        by_address.setdefault(row["registered_address"].strip().lower(), []).append(row["company_id"])
    for shared in by_address.values():
        for a in shared:
            for b in shared:
                if a != b:
                    g.add_edge(a, b, link="shared_address")
    if company_id not in g:
        return [company_id]
    return sorted(nx.node_connected_component(g, company_id))


def all_groups() -> list[set[str]]:
    """Every related-party group with more than one member (policy 5.1)."""

    companies = [row["company_id"] for row in _read_registry_csv("companies")]
    seen: set[str] = set()
    groups = []
    for cid in companies:
        if cid in seen:
            continue
        members = set(related_group(cid))
        seen |= members
        if len(members) > 1:
            groups.append(members)
    return groups


def _networkx_group_exposure(company_id: str) -> GroupExposure:
    from creditlens.generators.policy import RULES

    members = related_group(company_id)
    names = {row["company_id"]: row["legal_name"] for row in _read_registry_csv("companies")}
    existing = {row["company_id"]: int(row["outstanding_inr_000"])
                for row in _read_registry_csv("existing_loans")}
    requested = _requested_by_company()
    limit = int(RULES["MAX_GROUP_EXPOSURE_INR_000"])
    total = sum(existing.get(m, 0) + requested.get(m, 0) for m in members)
    rows = [GroupMember(company_id=m, legal_name=names.get(m, m),
                        link="self" if m == company_id else "shared_director_or_address",
                        hops=0 if m == company_id else 1,
                        existing_outstanding_inr_000=existing.get(m, 0),
                        pending_request_inr_000=requested.get(m, 0))
            for m in members]
    # Policy 5.1 defines a group as a *set* of linked companies. A company with no links
    # has no group, so rule 5.2 cannot be breached by it on its own.
    is_group = len(members) > 1
    return GroupExposure(company_id=company_id,
                         group_id=f"GRP-{min(members)}" if is_group else None,
                         members=rows, total_exposure_inr_000=int(total), limit_inr_000=limit,
                         breaches_limit=bool(is_group and total > limit), cypher=CYPHER)


# ---------------------------------------------------------------------------
# 4. Risk score: milestone 8, fallback = a transparent rule-based score.
# ---------------------------------------------------------------------------


def score_risk(application_id: str, features: Optional[dict] = None,
               failed_rules: Optional[list[str]] = None) -> RiskScore:
    try:
        from creditlens.ml.serving import score_application

        return score_application(application_id)
    except Exception:  # noqa: BLE001
        return _rule_based_risk(application_id, features or {}, failed_rules or [])


def _rule_based_risk(application_id: str, features: dict, failed_rules: list[str]) -> RiskScore:
    """A scorecard a credit analyst could check by hand. No model, no mystery."""
    pd_estimate = 0.04
    drivers: list[str] = []

    bounces = float(features.get("cheque_returns_12m") or 0)
    if bounces > 2:
        pd_estimate += 0.05 + 0.01 * (bounces - 2)
        drivers.append(f"{int(bounces)} cheque returns in the last 12 months")
    cash_share = float(features.get("cash_credit_share") or 0)
    if cash_share > 0.30:
        pd_estimate += 0.04
        drivers.append(f"cash deposits are {cash_share:.0%} of credits")
    volatility = float(features.get("inflow_volatility") or 0)
    if volatility > 0.35:
        pd_estimate += 0.03
        drivers.append("inflows swing a lot month to month")
    top_customer = float(features.get("top_customer_share") or 0)
    if top_customer > 0.4:
        pd_estimate += 0.03
        drivers.append(f"one customer is {top_customer:.0%} of collections")

    for rule in failed_rules:
        pd_estimate += {"3.1": 0.08, "3.2": 0.06, "3.3": 0.04}.get(rule, 0.02)
    if failed_rules:
        drivers.append("policy breaches: " + ", ".join(sorted(set(failed_rules))))
    if not drivers:
        drivers.append("no adverse signals in conduct or financials")

    pd_estimate = max(0.005, min(0.95, round(pd_estimate, 4)))
    band = "low" if pd_estimate < 0.08 else ("medium" if pd_estimate < 0.20 else "high")
    return RiskScore(probability_of_default=pd_estimate, risk_band=band, top_drivers=drivers,
                     model_name="rule-based-fallback", model_version="1")


# ---------------------------------------------------------------------------
# 5. PII masking: milestone 9, fallback = regexes for PAN, phone and email.
# ---------------------------------------------------------------------------

_PAN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_PHONE = re.compile(r"(?:\+91[\s-]?)?\b\d{5}[\s-]?\d{5}\b|\b\d{10}\b")


def mask(text: str) -> str:
    """Mask borrower PII before anything is sent to a model (policy 8.1)."""
    if not text:
        return text
    try:
        from creditlens.guardrails.pii import mask_text

        return mask_text(text)
    except Exception:  # noqa: BLE001
        masked = _PAN.sub("[PAN]", text)
        masked = _EMAIL.sub("[EMAIL]", masked)
        return _PHONE.sub("[PHONE]", masked)


# ---------------------------------------------------------------------------
# 6. Audit trail: milestone 3, fallback = a JSON Lines file under data/cache/.
# ---------------------------------------------------------------------------


def audit_path() -> Path:
    override = os.environ.get("CREDITLENS_AUDIT_PATH")
    return Path(override) if override else cache_dir() / "audit_events.jsonl"


def audit(event: AuditEvent) -> None:
    """Append one audit event. Anything that changes state calls this."""
    try:
        from creditlens.api.audit import record

        record(event)
        return
    except Exception:  # noqa: BLE001
        path = audit_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(event.model_dump_json() + "\n")


def audit_events(application_id: Optional[str] = None) -> list[AuditEvent]:
    """Read the audit trail back. Used by the API, the CLI and the tests."""
    try:
        from creditlens.api.audit import list_events

        if application_id is not None:
            return list(list_events(application_id))
    except Exception:  # noqa: BLE001
        pass
    path = audit_path()
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = AuditEvent.model_validate_json(line)
        except Exception:  # noqa: BLE001  a half-written line must not break a read
            continue
        if application_id is None or event.application_id == application_id:
            out.append(event)
    return out


# ---------------------------------------------------------------------------
# 7. Users (RBAC). The graph needs the caller's region for the SQL tool.
# ---------------------------------------------------------------------------


def load_user(user_id: str, engine=None) -> User:
    """Look a user up in the database, falling back to the registry CSV."""
    try:
        import pandas as pd

        df = pd.read_sql("select user_id, full_name, role, region from users", get_engine(engine))
        row = df[df.user_id == user_id]
        if len(row):
            return User(**row.iloc[0].to_dict())
    except Exception:  # noqa: BLE001
        pass
    for row in _read_registry_csv("users"):
        if row["user_id"] == user_id:
            return User(**row)
    raise KeyError(f"unknown user: {user_id}")


def as_json(obj: Any) -> Any:
    """Make anything safe to put in LangGraph state or a JSON tool result."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if hasattr(obj, "model_dump"):
        return json.loads(obj.model_dump_json())
    if isinstance(obj, dict):
        return {k: as_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [as_json(v) for v in obj]
    return str(obj)

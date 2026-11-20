"""The deterministic policy checks. Plain Python, no LLM anywhere near them.

The one idea this module teaches: **an agent may write the prose, but it must never
decide whether a rule was broken.** A language model that "thinks" a DSCR of 1.19 clears
a 1.25 threshold is a regulatory incident. So every rule in sections 2 to 6 of the Good
Bank manual is computed here from the database and the audited statements, with the
thresholds read from `creditlens.generators.policy.RULES`, and every check carries the
evidence sentence that says where its number came from.

Rules implemented: 2.1, 2.2 (eligibility), 3.1, 3.2, 3.3 (financial ratios),
4.1, 4.2 (banking conduct), 5.2 (group exposure), 6.1 (product tenure).
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from creditlens.agents import adapters
from creditlens.generators.policy import RULES
from creditlens.observability import observe
from creditlens.schemas import PolicyCheck

#: Which rules belong to which specialist. The policy_checker node fans out over these.
RULE_GROUPS: dict[str, list[str]] = {
    "eligibility": ["2.1", "2.2"],
    "financial": ["3.1", "3.2", "3.3"],
    "conduct": ["4.1", "4.2"],
    "group": ["5.2"],
    "product": ["6.1"],
}

ALL_RULE_IDS = [r for ids in RULE_GROUPS.values() for r in ids]


class ApplicationNotFound(KeyError):
    """Raised when the application id is not in the database."""


# ---------------------------------------------------------------------------
# Reading the facts
# ---------------------------------------------------------------------------


def load_facts(application_id: str, engine=None) -> dict:
    """Everything the checks need, read once: application, company, conduct, financials, group."""
    import pandas as pd

    eng = adapters.get_engine(engine)
    app = pd.read_sql("select * from applications where application_id = ?", eng,
                      params=(application_id,))
    if not len(app):
        raise ApplicationNotFound(application_id)
    application = app.iloc[0].to_dict()
    company_id = application["company_id"]

    company = pd.read_sql("select * from companies where company_id = ?", eng,
                          params=(company_id,))
    features = pd.read_sql("select * from statement_features where application_id = ?", eng,
                           params=(application_id,))
    loans = pd.read_sql("select * from existing_loans where company_id = ?", eng,
                        params=(company_id,))
    financials = adapters.get_financials(company_id)
    return {
        "application": application,
        "company": company.iloc[0].to_dict() if len(company) else {},
        "features": features.iloc[0].to_dict() if len(features) else {},
        "existing_outstanding_inr_000": int(loans.iloc[0]["outstanding_inr_000"]) if len(loans) else 0,
        "financials": financials,
        "latest_year": adapters.latest_year(financials),
        "group": adapters.get_group_exposure(company_id),
    }


# ---------------------------------------------------------------------------
# One function per rule group, so the map-reduce fan-out has something to call
# ---------------------------------------------------------------------------


def _check(rule_id: str, text: str, measured: Optional[float], threshold: Optional[float],
           passed: bool, evidence: str) -> PolicyCheck:
    return PolicyCheck(rule_id=rule_id, rule_text=text, measured_value=measured,
                       threshold=threshold, passed=passed, evidence=evidence)


def check_eligibility(facts: dict) -> list[PolicyCheck]:
    """2.1 years in business, 2.2 excluded sectors."""
    application, company = facts["application"], facts["company"]
    out: list[PolicyCheck] = []

    incorporated = date.fromisoformat(str(company["incorporated_on"])[:10])
    submitted = date.fromisoformat(str(application["submitted_on"])[:10])
    years = (submitted - incorporated).days / 365.25
    minimum = float(RULES["MIN_YEARS_IN_BUSINESS"])
    out.append(_check(
        "2.1", f"The applicant must have completed at least {minimum:.0f} full years since incorporation.",
        round(years, 2), minimum, years >= minimum,
        f"companies.incorporated_on = {incorporated} and applications.submitted_on = {submitted}, "
        f"so {years:.2f} years at submission."))

    sector = str(company.get("sector", ""))
    excluded = list(RULES["EXCLUDED_SECTORS"])
    out.append(_check(
        "2.2", f"Good Bank does not lend to: {', '.join(excluded)}.", None, None,
        sector not in excluded,
        f"companies.sector = {sector!r}; the excluded list is {excluded}."))
    return out


def check_financial(facts: dict) -> list[PolicyCheck]:
    """3.1 DSCR, 3.2 debt to EBITDA, 3.3 current ratio, all on the latest audited year."""
    year = facts["latest_year"]
    out: list[PolicyCheck] = []
    if year is None:
        # A scanned statement with no text layer. Say nothing rather than guess: the memo
        # will report that the ratios could not be computed and an analyst must key them in.
        return out
    where = f"the audited statement for {year.fiscal_year}"

    if year.ebitda is not None and year.interest is not None and year.principal_repayment is not None:
        service = year.interest + year.principal_repayment
        dscr = year.ebitda / service if service else float("inf")
        minimum = float(RULES["MIN_DSCR"])
        out.append(_check(
            "3.1", f"Debt service coverage ratio must be at least {minimum}x.",
            round(dscr, 3), minimum, dscr >= minimum,
            f"EBITDA {year.ebitda:,} / (finance costs {year.interest:,} + principal repayment "
            f"{year.principal_repayment:,}) = {dscr:.3f}x, from {where} (Note 14 for the principal)."))

    if year.total_debt is not None and year.ebitda is not None:
        leverage = year.total_debt / year.ebitda if year.ebitda > 0 else float("inf")
        maximum = float(RULES["MAX_DEBT_TO_EBITDA"])
        out.append(_check(
            "3.2", f"Total borrowings divided by EBITDA must not exceed {maximum}x.",
            round(leverage, 3), maximum, leverage <= maximum,
            f"Total borrowings {year.total_debt:,} / EBITDA {year.ebitda:,} = {leverage:.3f}x, from {where}."))

    if year.current_assets is not None and year.current_liabilities is not None:
        ratio = (year.current_assets / year.current_liabilities) if year.current_liabilities else float("inf")
        minimum = float(RULES["MIN_CURRENT_RATIO"])
        out.append(_check(
            "3.3", f"Current ratio must be at least {minimum}x.",
            round(ratio, 3), minimum, ratio >= minimum,
            f"Total current assets {year.current_assets:,} / trade payables and other current "
            f"liabilities {year.current_liabilities:,} = {ratio:.3f}x, from {where}."))
    return out


def check_conduct(facts: dict) -> list[PolicyCheck]:
    """4.1 cheque returns, 4.2 cash share of credits, both from the curated statement features."""
    features = facts["features"]
    out: list[PolicyCheck] = []
    if not features:
        return out

    bounces = int(features.get("cheque_returns_12m") or 0)
    maximum = float(RULES["MAX_CHEQUE_RETURNS_12M"])
    out.append(_check(
        "4.1", f"Cheques returned for insufficient funds must not exceed {maximum:.0f} in 12 months.",
        float(bounces), maximum, bounces <= maximum,
        f"statement_features.cheque_returns_12m = {bounces} over "
        f"{int(features.get('months_covered') or 12)} months of statements."))

    share = float(features.get("cash_credit_share") or 0.0)
    max_share = float(RULES["MAX_CASH_CREDIT_SHARE"])
    out.append(_check(
        "4.2", f"Cash deposits must not exceed {max_share:.0%} of total credits by value.",
        round(share, 4), max_share, share <= max_share,
        f"statement_features.cash_credit_share = {share:.2%} of total credits."))
    return out


def check_group(facts: dict) -> list[PolicyCheck]:
    """5.2 total exposure to the related-party group."""
    group = facts["group"]
    limit = float(RULES["MAX_GROUP_EXPOSURE_INR_000"])
    members = ", ".join(m.company_id for m in group.members)
    if group.group_id is None:
        evidence = (f"{group.company_id} shares no director and no registered address with any other "
                    "company, so policy 5.1 gives it no group.")
        return [_check("5.2", f"Group exposure must not exceed INR {limit:,.0f} thousand.",
                       None, limit, True, evidence)]
    evidence = (f"Group {group.group_id} members {members}: existing outstanding plus pending requests "
                f"= INR {group.total_exposure_inr_000:,} thousand (graph query in the audit trail).")
    return [_check("5.2", f"Group exposure must not exceed INR {limit:,.0f} thousand.",
                   float(group.total_exposure_inr_000), limit, not group.breaches_limit, evidence)]


def check_product(facts: dict) -> list[PolicyCheck]:
    """6.1 maximum tenure by product."""
    application = facts["application"]
    product = str(application["product"])
    tenure = int(application["tenure_months"])
    maximum = float(RULES["MAX_TENURE_MONTHS"].get(product, 0))
    return [_check(
        "6.1", f"Maximum tenure for {product} is {maximum:.0f} months.",
        float(tenure), maximum, tenure <= maximum,
        f"applications.tenure_months = {tenure} for product {product!r}.")]


GROUP_FUNCTIONS = {
    "eligibility": check_eligibility,
    "financial": check_financial,
    "conduct": check_conduct,
    "group": check_group,
    "product": check_product,
}


def run_group(group_name: str, facts: dict) -> list[PolicyCheck]:
    """Run one rule group. This is what each map-reduce branch calls."""
    return GROUP_FUNCTIONS[group_name](facts)


# ---------------------------------------------------------------------------
# The public entry point
# ---------------------------------------------------------------------------


@observe(name="policy_checks.run_checks")
def run_checks(application_id: str, engine=None) -> list[PolicyCheck]:
    """Every policy check for one application, in rule-id order."""
    facts = load_facts(application_id, engine)
    checks: list[PolicyCheck] = []
    for name in RULE_GROUPS:
        checks.extend(run_group(name, facts))
    return sorted(checks, key=lambda c: c.rule_id)


def failed_rule_ids(checks: list[PolicyCheck]) -> list[str]:
    """The rule ids that did not pass, sorted and de-duplicated."""
    return sorted({c.rule_id for c in checks if not c.passed})

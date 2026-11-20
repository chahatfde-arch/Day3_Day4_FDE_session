"""Writing the memo: the only place in milestone 6 where a language model is allowed to think.

Division of labour, and it matters:

* **Code decides.** Which rules failed, and therefore the recommendation, is computed in
  `policy_checks.py` and in `recommend()` below. Deterministic, testable, auditable.
* **The model writes.** It turns the numbers into the eight sections of a policy section 9
  memo. If the model is unavailable, the offline handler writes the same sections from
  templates, so a complete memo always exists.

The facts are handed to the model as a JSON block inside the prompt. That is what lets the
offline stub produce a sensible memo too: it reads the same block. Every string in that
block is masked first, so no director PAN, phone or email ever reaches a model.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from creditlens.agents import adapters
from creditlens.observability import observe
from creditlens.schemas import CreditMemo, PolicyException

FACTS_MARKER = "FACTS_JSON:"
REVIEW_MARKER = "REVIEW_JSON:"


# ---------------------------------------------------------------------------
# What the model must return
# ---------------------------------------------------------------------------


class MemoSections(BaseModel):
    """The prose sections of the memo, in policy section 9 order."""
    applicant_summary: str = Field(description="Who the borrower is, in two or three sentences")
    facility_requested: str = Field(description="Amount, product, tenure and purpose")
    financial_analysis: str = Field(description="DSCR, leverage and current ratio with the numbers")
    banking_conduct: str = Field(description="Inflows, cheque returns and cash share")
    related_parties: str = Field(description="Group members and the group exposure")
    risk_assessment: str = Field(description="The risk band and its main drivers")
    recommendation_reasoning: str = Field(description="Why, for the underwriter")


class MemoReview(BaseModel):
    """The self-review pass: does the draft mention every exception it must?"""
    ok: bool = Field(description="True when nothing is missing")
    issues: list[str] = Field(default_factory=list, description="What a reviewer would send back")
    missing_rule_ids: list[str] = Field(default_factory=list,
                                        description="Failed rules the draft does not mention")


# ---------------------------------------------------------------------------
# Collecting the facts
# ---------------------------------------------------------------------------


def _mask_strings(value: Any) -> Any:
    """Mask PII inside the fact block without touching its structure or its numbers."""
    if isinstance(value, str):
        return adapters.mask(value)
    if isinstance(value, dict):
        return {k: _mask_strings(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_mask_strings(v) for v in value]
    return value


def build_facts(state: dict) -> dict:
    """The structured facts both the model and the offline template read."""
    from creditlens.agents.state import dedupe_checks

    application = (state.get("application") or {}).get("application", {}) or {}
    company = (state.get("application") or {}).get("company", {}) or {}
    financials = state.get("financials") or {}
    years = financials.get("years") or []
    latest = max(years, key=lambda y: y.get("fiscal_year", "")) if years else {}
    checks = dedupe_checks(state.get("checks") or [])
    facts = {
        "application_id": state.get("application_id"),
        "company_id": application.get("company_id") or company.get("company_id"),
        "legal_name": company.get("legal_name", ""),
        "sector": company.get("sector", ""),
        "city": company.get("city", ""),
        "region": application.get("region", ""),
        "incorporated_on": str(company.get("incorporated_on", "")),
        "employees": company.get("employees"),
        "product": application.get("product", ""),
        "purpose": application.get("purpose", ""),
        "requested_inr_000": application.get("requested_inr_000"),
        "tenure_months": application.get("tenure_months"),
        "submitted_on": str(application.get("submitted_on", "")),
        "latest_year": latest,
        "statement_available": bool(years),
        "conduct": state.get("conduct") or {},
        "group": state.get("group") or {},
        "risk": state.get("risk") or {},
        "checks": checks,
        "failed_rule_ids": sorted({c["rule_id"] for c in checks if not c.get("passed", True)}),
    }
    return _mask_strings(facts)


# ---------------------------------------------------------------------------
# The recommendation: code, not prose
# ---------------------------------------------------------------------------


def group_headroom(facts: dict) -> Optional[int]:
    """The largest request that keeps the group inside rule 5.2, or None when there is no group.

    Limit minus everything the group already owes, minus what the *other* members are asking
    for. It is arithmetic an underwriter can redo by hand, which is why it lives in code: it
    turns a breach into a concrete counter-offer for a person to consider, not a decision.
    """
    group = facts.get("group") or {}
    members = group.get("members") or []
    if not group.get("group_id") or len(members) < 2 or group.get("limit_inr_000") is None:
        return None
    me = facts.get("company_id")
    used = sum(int(m.get("existing_outstanding_inr_000") or 0) for m in members)
    used += sum(int(m.get("pending_request_inr_000") or 0) for m in members if m.get("company_id") != me)
    return max(0, int(group["limit_inr_000"]) - used)


def recommend(facts: dict) -> tuple[str, str]:
    """Return (recommendation, the one sentence that justifies it).

    Section 2 is eligibility and has no exception route ("No exception is possible under
    rule 2.2"), so a section 2 breach declines. Anything else is an exception an
    underwriter may still allow, so it goes to `refer`.
    """
    failed = list(facts.get("failed_rule_ids") or [])
    band = (facts.get("risk") or {}).get("risk_band", "medium")
    eligibility = [r for r in failed if r.startswith("2.")]
    if eligibility:
        return ("decline",
                f"Rule {' and '.join(eligibility)} in section 2 is an eligibility rule, and policy "
                "allows no exception to it, so the application is declined.")
    if failed:
        reason = (f"Policy exceptions were raised on rule {', '.join(failed)}. Under policy 7.3 only the "
                  "Head of Credit may approve an exception, so the file is referred.")
        room = group_headroom(facts)
        if "5.2" in failed and room is not None:
            reason += (f" Without an exception to rule 5.2, the most the group can take on for this "
                       f"application is INR {room:,} thousand.")
        return ("refer", reason)
    if band == "high":
        return ("approve_with_conditions",
                "No policy rule is breached, but the risk model puts this file in the high band, so "
                "approval should carry conditions and closer monitoring.")
    return ("approve", "No policy rule in sections 2 to 6 is breached and the risk band is acceptable.")


# ---------------------------------------------------------------------------
# Offline prose: templates that read the same fact block the model gets
# ---------------------------------------------------------------------------


def _facts_from_messages(messages: list[dict], marker: str) -> dict:
    text = " ".join(m.get("content", "") for m in messages)
    start = text.find(marker)
    if start == -1:
        return {}
    blob = text[start + len(marker):].strip()
    decoder = json.JSONDecoder()
    try:
        value, _ = decoder.raw_decode(blob)
        return value
    except json.JSONDecodeError:
        return {}


def _rupees(value: Optional[float]) -> str:
    """Statement features are in rupees, not thousands: format them as such."""
    if value is None:
        return "not available"
    from creditlens.generators.common import format_inr

    return f"INR {format_inr(float(value), 0)}"


def _inr(value: Optional[float]) -> str:
    if value is None:
        return "not available"
    from creditlens.generators.common import format_inr

    return f"INR {format_inr(float(value), 0)} thousand"


def _ratio(checks: list[dict], rule_id: str) -> Optional[float]:
    for check in checks:
        if check.get("rule_id") == rule_id:
            return check.get("measured_value")
    return None


def offline_sections(messages: list[dict]) -> dict:
    """The offline stub for MemoSections: a real memo, written from templates."""
    facts = _facts_from_messages(messages, FACTS_MARKER)
    checks = facts.get("checks") or []
    failed = facts.get("failed_rule_ids") or []
    conduct = facts.get("conduct") or {}
    group = facts.get("group") or {}
    risk = facts.get("risk") or {}
    year = facts.get("latest_year") or {}

    name = facts.get("legal_name") or facts.get("company_id") or "The applicant"
    summary = (f"{name} is a {facts.get('sector', 'business')} company in {facts.get('city', 'India')} "
               f"({facts.get('region', '')} region), incorporated on {facts.get('incorporated_on', '')}"
               + (f" with about {facts['employees']} employees" if facts.get("employees") else "")
               + f". The file was submitted on {facts.get('submitted_on', '')}.")

    facility = (f"{_inr(facts.get('requested_inr_000'))} as {facts.get('product', 'a facility')} over "
                f"{facts.get('tenure_months', '')} months for {facts.get('purpose', 'business use')}.")

    if facts.get("statement_available"):
        dscr, leverage, current = _ratio(checks, "3.1"), _ratio(checks, "3.2"), _ratio(checks, "3.3")
        financial = (f"For {year.get('fiscal_year', 'the latest audited year')} revenue was "
                     f"{_inr(year.get('revenue'))} and EBITDA {_inr(year.get('ebitda'))}. "
                     f"DSCR is {dscr}x against a policy minimum of 1.25x. Total borrowings to EBITDA "
                     f"is {leverage}x against a maximum of 3.5x. The current ratio is {current}x "
                     "against a minimum of 1.1x.")
    else:
        financial = ("The audited statement on file is a scan with no text layer, so the section 3 "
                     "ratios could not be computed. An analyst must key the figures in before the "
                     "underwriter sees this file.")

    if conduct:
        banking = (f"Average monthly inflow is {_rupees(conduct.get('avg_monthly_inflow'))} over "
                   f"{conduct.get('months_covered', 12)} months. There were "
                   f"{conduct.get('cheque_returns_12m', 0)} returned cheques (policy allows 2) and cash "
                   f"deposits were {float(conduct.get('cash_credit_share', 0)):.1%} of credits (policy "
                   "allows 30 percent). The largest single customer is "
                   f"{float(conduct.get('top_customer_share', 0)):.1%} of collections.")
    else:
        banking = "No bank statement features are on file for this application."

    members = group.get("members") or []
    if group.get("group_id"):
        listed = "; ".join(f"{m['company_id']} ({m['legal_name']}) existing "
                           f"{m['existing_outstanding_inr_000']}, pending "
                           f"{m['pending_request_inr_000']}" for m in members)
        related = (f"Group {group['group_id']} has {len(members)} members: {listed}. Total group "
                   f"exposure is {_inr(group.get('total_exposure_inr_000'))} against a limit of "
                   f"{_inr(group.get('limit_inr_000'))}. "
                   + ("The limit is breached." if group.get("breaches_limit") else "The limit is not breached."))
    else:
        related = ("The applicant shares no director and no registered address with another company on "
                   "the register, so policy 5.1 gives it no group and rule 5.2 does not bite.")

    drivers = "; ".join(risk.get("top_drivers") or []) or "no drivers recorded"
    risk_text = (f"The model puts the probability of default at "
                 f"{float(risk.get('probability_of_default', 0)):.1%}, which is the "
                 f"{risk.get('risk_band', 'medium')} band. Main drivers: {drivers}. "
                 f"Model: {risk.get('model_name', 'unknown')} version {risk.get('model_version', '0')}.")

    _, reasoning = recommend(facts)
    if failed:
        lines = [f"Rule {c['rule_id']}: measured {c.get('measured_value')} against a threshold of "
                 f"{c.get('threshold')}. {c.get('evidence', '')}"
                 for c in checks if not c.get("passed", True)]
        reasoning = reasoning + " Exceptions: " + " ".join(lines)
    else:
        reasoning = reasoning + " Every check in sections 2 to 6 passed on the figures above."

    return {"applicant_summary": summary, "facility_requested": facility,
            "financial_analysis": financial, "banking_conduct": banking,
            "related_parties": related, "risk_assessment": risk_text,
            "recommendation_reasoning": reasoning}


def offline_review(messages: list[dict]) -> dict:
    """The offline stub for MemoReview: check that every failed rule is named in the draft."""
    payload = _facts_from_messages(messages, REVIEW_MARKER)
    draft = " ".join(str(v) for v in (payload.get("draft") or {}).values())
    missing = [rule for rule in (payload.get("failed_rule_ids") or []) if rule not in draft]
    issues = ([f"The draft never mentions rule {r}." for r in missing]
              if missing else [])
    return {"ok": not missing, "issues": issues, "missing_rule_ids": missing}


def register_offline_handlers() -> None:
    """Make the whole memo run without an API key. Called by the graph before it drafts."""
    from creditlens.gateway.llm import OFFLINE_HANDLERS

    OFFLINE_HANDLERS.setdefault("MemoSections", offline_sections)
    OFFLINE_HANDLERS.setdefault("MemoReview", offline_review)


# ---------------------------------------------------------------------------
# The three steps of the memo subgraph
# ---------------------------------------------------------------------------

SYSTEM = ("You are a credit analyst at Good Bank drafting a memo for a human underwriter. "
          "Use only the facts given. Never invent a number. Never approve or decline: the "
          "underwriter decides. Write plainly, in full sentences, no bullet points.")


@observe(name="memo.draft_sections")
def draft_sections(facts: dict) -> MemoSections:
    """Step 1 of the subgraph: write the seven prose sections from the fact block."""
    from creditlens.gateway.llm import get_gateway

    register_offline_handlers()
    prompt = ("Draft the sections of a credit memo for this application. Use every number you are "
              "given and name every failed rule by its id.\n\n"
              f"{FACTS_MARKER} {json.dumps(facts, default=str)}")
    return get_gateway().extract(adapters.mask(prompt), MemoSections, system=SYSTEM)


@observe(name="memo.self_review")
def review_sections(sections: MemoSections, facts: dict) -> MemoReview:
    """Step 2: the drafter reads its own work and says what is missing."""
    from creditlens.gateway.llm import get_gateway

    register_offline_handlers()
    payload = {"draft": sections.model_dump(), "failed_rule_ids": facts.get("failed_rule_ids") or []}
    prompt = ("Review this credit memo draft. Every failed policy rule must be named by its id in the "
              "prose. List what is missing.\n\n"
              f"{REVIEW_MARKER} {json.dumps(payload, default=str)}")
    return get_gateway().extract(adapters.mask(prompt), MemoReview,
                                 system="You are a strict reviewer of credit memos.")


@observe(name="memo.finalise")
def finalise(sections: MemoSections, review: MemoReview, facts: dict) -> MemoSections:
    """Step 3: repair whatever the review found, in code, so the fix is guaranteed."""
    if review.ok and not review.missing_rule_ids:
        return sections
    checks = {c["rule_id"]: c for c in (facts.get("checks") or [])}
    added = []
    for rule_id in review.missing_rule_ids:
        check = checks.get(rule_id, {})
        added.append(f"Rule {rule_id} was breached: measured {check.get('measured_value')} against "
                     f"a threshold of {check.get('threshold')}.")
    if added:
        sections = sections.model_copy(update={
            "recommendation_reasoning": sections.recommendation_reasoning + " " + " ".join(added)})
    return sections


def build_memo(facts: dict, sections: MemoSections, citations: list[str],
               model_name: str) -> CreditMemo:
    """Assemble the CreditMemo: model prose plus code-decided checks and recommendation."""
    from creditlens.schemas import GroupExposure, PolicyCheck, RiskScore

    recommendation, reasoning = recommend(facts)
    checks = [PolicyCheck(**c) for c in (facts.get("checks") or [])]
    exceptions = [PolicyException(rule_id=c.rule_id, measured_value=c.measured_value,
                                  justification=c.evidence or c.rule_text)
                  for c in checks if not c.passed]
    risk = RiskScore(**facts["risk"]) if facts.get("risk") else None
    group = GroupExposure(**facts["group"]) if facts.get("group") else None
    return CreditMemo(
        application_id=str(facts.get("application_id")),
        company_id=str(facts.get("company_id")),
        applicant_summary=sections.applicant_summary,
        facility_requested=sections.facility_requested,
        financial_analysis=sections.financial_analysis,
        banking_conduct=sections.banking_conduct,
        related_parties=sections.related_parties,
        risk_assessment=sections.risk_assessment,
        policy_exceptions=exceptions,
        recommendation=recommendation,
        # The offline template already starts with the same sentence; do not say it twice.
        recommendation_reasoning=(sections.recommendation_reasoning.strip()
                                  if sections.recommendation_reasoning.strip().startswith(reasoning[:40])
                                  else f"{reasoning} {sections.recommendation_reasoning}".strip()),
        checks=checks,
        risk=risk,
        group=group,
        citations=citations,
        model_name=model_name,
    )


def strip_markdown(text: str) -> str:
    """Models like to answer in bullet points. The memo template is plain prose."""
    return re.sub(r"^[\s>*-]+", "", text or "", flags=re.M).strip()

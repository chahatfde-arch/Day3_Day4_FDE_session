"""Shared Pydantic models: the contracts between extraction, retrieval, ML, agents and the API.

Every module that produces one of these objects must return exactly this shape,
and every module that consumes one may rely on it. Keeping them in one file is what
lets the workshop pods swap a stub for the real component without touching the agents.

Beginner note: a Pydantic model is a typed dictionary that validates itself. If an
LLM returns a field with the wrong type, validation fails loudly, which is far better
than a wrong number reaching an underwriter.
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Document extraction (milestone 4). Two document types, one discriminated union.
# ---------------------------------------------------------------------------


class FiscalYearFigures(BaseModel):
    """One column of a financial statement, in INR thousands (as printed on the PDF)."""
    fiscal_year: str = Field(description="For example FY2026")
    revenue: Optional[int] = None
    ebitda: Optional[int] = None
    interest: Optional[int] = Field(None, description="Finance costs")
    principal_repayment: Optional[int] = Field(None, description="From the borrowings note")
    total_debt: Optional[int] = Field(None, description="Total borrowings")
    current_assets: Optional[int] = None
    current_liabilities: Optional[int] = None
    pat: Optional[int] = Field(None, description="Profit after tax")


class FinancialStatementExtraction(BaseModel):
    """Figures pulled from an audited financial statement PDF (text or scanned)."""
    doc_type: Literal["financial_statement"] = "financial_statement"
    company_name: str
    unit: str = "INR thousands"
    years: list[FiscalYearFigures]
    source_file: str = ""
    ocr_used: bool = False
    confidence: float = Field(1.0, ge=0, le=1)

    def latest(self) -> Optional[FiscalYearFigures]:
        return max(self.years, key=lambda y: y.fiscal_year) if self.years else None


class ReceiptExtraction(BaseModel):
    """Key fields from a scanned receipt or supplier invoice (the SROIE schema)."""
    doc_type: Literal["receipt"] = "receipt"
    company: str = Field(description="Merchant or supplier name as printed")
    date: str = Field(description="Date as printed, for example 15/01/2019")
    address: str = ""
    total: str = Field(description="Grand total as printed, digits and decimal point only")
    source_file: str = ""
    ocr_used: bool = True
    confidence: float = Field(1.0, ge=0, le=1)

    @field_validator("total")
    @classmethod
    def _clean_total(cls, v: str) -> str:
        return v.replace(",", "").replace("RM", "").replace("$", "").strip()


DocumentExtraction = Annotated[Union[FinancialStatementExtraction, ReceiptExtraction],
                               Field(discriminator="doc_type")]


class ExtractionEnvelope(BaseModel):
    """What the LLM must return: the document type decides which fields are required."""
    document: DocumentExtraction


# ---------------------------------------------------------------------------
# Retrieval (milestone 5)
# ---------------------------------------------------------------------------


class PolicyHit(BaseModel):
    """One retrieved chunk of a policy document."""
    chunk_id: str
    source: Literal["good_bank_policy", "sba_sop"] = "good_bank_policy"
    section: str = Field(description="Heading of the section the chunk came from")
    text: str
    score: float = 0.0
    retriever: str = Field("hybrid", description="vector | bm25 | hybrid | reranked")


class GroupMember(BaseModel):
    company_id: str
    legal_name: str
    link: str = Field(description="How it is linked: shared_director, shared_address, or self")
    hops: int = 0
    existing_outstanding_inr_000: int = 0
    pending_request_inr_000: int = 0


class GroupExposure(BaseModel):
    """Everything policy section 5 needs about related parties."""
    company_id: str
    group_id: Optional[str] = None
    members: list[GroupMember]
    total_exposure_inr_000: int
    limit_inr_000: int
    breaches_limit: bool
    cypher: str = Field("", description="The query that produced this, for the audit trail")


# ---------------------------------------------------------------------------
# Risk model (milestone 8)
# ---------------------------------------------------------------------------


class RiskScore(BaseModel):
    probability_of_default: float = Field(ge=0, le=1)
    risk_band: Literal["low", "medium", "high"]
    top_drivers: list[str] = Field(default_factory=list, description="Plain-language reasons")
    model_name: str = "stub"
    model_version: str = "0"


# ---------------------------------------------------------------------------
# Policy checks and the credit memo (milestone 6)
# ---------------------------------------------------------------------------


class PolicyCheck(BaseModel):
    rule_id: str = Field(description="For example 3.1")
    rule_text: str
    measured_value: Optional[float] = None
    threshold: Optional[float] = None
    passed: bool
    evidence: str = Field("", description="Where the number came from")


class PolicyException(BaseModel):
    rule_id: str
    measured_value: Optional[float]
    justification: str
    jira_key: Optional[str] = Field(None, description="Set once the ticket is filed")


class CreditMemo(BaseModel):
    """The draft memo. Sections follow policy section 9, in order."""
    application_id: str
    company_id: str
    applicant_summary: str
    facility_requested: str
    financial_analysis: str
    banking_conduct: str
    related_parties: str
    risk_assessment: str
    policy_exceptions: list[PolicyException] = Field(default_factory=list)
    recommendation: Literal["approve", "approve_with_conditions", "decline", "refer"]
    recommendation_reasoning: str
    checks: list[PolicyCheck] = Field(default_factory=list)
    risk: Optional[RiskScore] = None
    group: Optional[GroupExposure] = None
    citations: list[str] = Field(default_factory=list, description="Policy chunk ids used")
    drafted_at: datetime = Field(default_factory=datetime.utcnow)
    model_name: str = ""

    def to_markdown(self) -> str:
        ex = "\n".join(f"- Rule {e.rule_id}: measured {e.measured_value}. {e.justification}"
                       + (f" (Jira {e.jira_key})" if e.jira_key else "")
                       for e in self.policy_exceptions) or "None"
        return (f"# Credit memo: {self.application_id}\n\n"
                f"## Applicant summary\n{self.applicant_summary}\n\n"
                f"## Facility requested\n{self.facility_requested}\n\n"
                f"## Financial analysis\n{self.financial_analysis}\n\n"
                f"## Banking conduct\n{self.banking_conduct}\n\n"
                f"## Related parties and group exposure\n{self.related_parties}\n\n"
                f"## Risk score\n{self.risk_assessment}\n\n"
                f"## Policy exceptions\n{ex}\n\n"
                f"## Recommendation: {self.recommendation}\n{self.recommendation_reasoning}\n")


# ---------------------------------------------------------------------------
# Audit trail and approvals (milestones 3, 6, 7)
# ---------------------------------------------------------------------------


class AuditEvent(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    application_id: str
    actor: str = Field(description="user id, agent name or system")
    action: str
    detail: str = ""
    trace_id: Optional[str] = None


class ApprovalDecision(BaseModel):
    application_id: str
    decided_by: str
    decision: Literal["approve", "reject"]
    comment: str = ""
    approved_inr_000: Optional[int] = Field(
        None, description="Amount approved when it differs from the request (a counter-offer), INR thousand")
    channel: Literal["api", "slack", "cli"] = "api"
    decided_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Users (RBAC)
# ---------------------------------------------------------------------------


class User(BaseModel):
    user_id: str
    full_name: str
    role: Literal["analyst", "head_of_credit", "auditor"]
    region: str = Field(description="North, South, East, West, or ALL")

    def can_see_region(self, region: str) -> bool:
        return self.region == "ALL" or self.region == region

    @property
    def read_only(self) -> bool:
        return self.role == "auditor"

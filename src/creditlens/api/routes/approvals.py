"""The human in the loop: the one endpoint that turns a draft into a decision.

Only `head_of_credit` may call this. Note what the request body does *not* contain:
`decided_by`. The caller never gets to say who they are, because the token already
said it. Letting a client name the approver would make the audit trail worthless.

If milestone 6 is installed, approving also resumes the paused LangGraph run, which is
where the memo gets finalised and the Jira exceptions get closed. If it is not, the
decision is still recorded, so the audit trail is complete either way.
"""
from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import text

from creditlens.api import audit as audit_store
from creditlens.api.auth import current_user
from creditlens.api.deps import get_engine
from creditlens.api.rbac import require_region, require_role, require_write
from creditlens.observability import observe, update_trace
from creditlens.schemas import ApprovalDecision, AuditEvent, User

router = APIRouter(tags=["approvals"])


class ApprovalRequest(BaseModel):
    """What the client sends. Who decided and when are filled in by the server."""
    decision: Literal["approve", "reject"]
    comment: str = Field("", max_length=2000)
    thread_id: Optional[str] = Field(None, description="The paused agent run to resume")


@router.post("/applications/{application_id}/approve")
@observe(name="api.approve_application")
def approve_application(application_id: str, body: ApprovalRequest,
                        user: User = Depends(current_user),
                        engine=Depends(get_engine)) -> dict:
    """Record an underwriter's decision and resume the agent run if there is one."""
    # Two guards, in order of how obvious the refusal should be. An auditor is
    # read-only everywhere; an analyst can write, but not this.
    require_write(user)
    require_role(user, "head_of_credit")

    with engine.connect() as conn:
        row = conn.execute(text("SELECT region FROM applications WHERE application_id = :aid"),
                           {"aid": application_id}).mappings().first()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no application {application_id}")
    require_region(user, row["region"])
    update_trace(user_id=user.user_id, session_id=application_id, tags=["approval"])

    decision = ApprovalDecision(application_id=application_id, decided_by=user.user_id,
                                decision=body.decision, comment=body.comment, channel="api")

    resumed: Optional[dict] = None
    if body.thread_id:
        try:
            from creditlens.agents.run import resume as resume_run

            result = resume_run(body.thread_id, decision)
            resumed = result if isinstance(result, dict) else {"result": str(result)}
        except Exception as exc:  # noqa: BLE001  milestone 6 missing, or the thread expired
            resumed = {"resumed": False, "reason": f"could not resume the agent run: {exc}"}

    audit_store.record(AuditEvent(application_id=application_id, actor=user.user_id,
                                  action=f"application_{body.decision}d",
                                  detail=body.comment or f"decision via api by {user.full_name}"),
                       engine=engine)
    return {"decision": decision.model_dump(mode="json"), "agent_run": resumed}

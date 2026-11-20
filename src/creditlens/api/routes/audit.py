"""Reading the audit trail for one application.

Auditors are the point of this endpoint, and they may read every region, which is why
there is no `require_write` here and why the region check still applies to everyone
else: an analyst may read the trail of their own files, not of somebody else's.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text

from creditlens.api import audit as audit_store
from creditlens.api.auth import current_user
from creditlens.api.deps import get_engine
from creditlens.api.rbac import require_region
from creditlens.observability import observe
from creditlens.schemas import User

router = APIRouter(tags=["audit"])


@router.get("/applications/{application_id}/audit")
@observe(name="api.list_audit_events")
def list_audit(application_id: str, user: User = Depends(current_user),
               engine=Depends(get_engine)) -> dict:
    """Every state change for one application, oldest first."""
    with engine.connect() as conn:
        row = conn.execute(text("SELECT region FROM applications WHERE application_id = :aid"),
                           {"aid": application_id}).mappings().first()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no application {application_id}")
    require_region(user, row["region"])

    events = audit_store.list_events(application_id, engine=engine)
    return {"application_id": application_id, "count": len(events),
            "events": [e.model_dump(mode="json") for e in events]}

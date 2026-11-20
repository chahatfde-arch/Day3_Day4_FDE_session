"""Reading applications, and asking the agent graph for a draft memo.

Every handler here is a plain `def` (not `async def`) on purpose: the work is blocking
database I/O, so FastAPI runs it in a worker thread, and the `@observe` decorator from
`creditlens.observability` wraps a synchronous function cleanly.

Read the RBAC two ways in this file:

* `GET /applications` *filters* (you simply never see other regions).
* `GET /applications/{id}` *refuses* with 403.

Filtering is the right default for lists, because a short list is not an error.
Refusing is the right answer for a direct request, because the caller asked for
something specific and deserves to be told no.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import text

from creditlens.api import audit as audit_store
from creditlens.api.auth import current_user
from creditlens.api.deps import get_engine
from creditlens.api.rbac import require_region, require_write
from creditlens.guardrails.rbac import region_filter
from creditlens.observability import observe, update_trace
from creditlens.schemas import AuditEvent, User

router = APIRouter(tags=["applications"])

# Allow-list of columns we are willing to return. Writing it out keeps a future
# "SELECT *" from quietly exposing a column somebody adds later.
APPLICATION_COLUMNS = ("application_id, company_id, region, product, purpose, "
                       "requested_inr_000, tenure_months, submitted_on, analyst_id")


def _fetch_application(engine, application_id: str) -> dict:
    """One application row, or 404. The id is bound as a parameter, never formatted in."""
    query = text(f"SELECT {APPLICATION_COLUMNS} FROM applications WHERE application_id = :aid")
    with engine.connect() as conn:
        row = conn.execute(query, {"aid": application_id}).mappings().first()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no application {application_id}")
    return dict(row)


@router.get("/applications")
@observe(name="api.list_applications")
def list_applications(
    user: User = Depends(current_user),
    engine=Depends(get_engine),
    limit: int = Query(100, ge=1, le=500),
    product: Optional[str] = None,
) -> dict:
    """Applications this caller is allowed to see, newest first."""
    update_trace(user_id=user.user_id, tags=[f"role:{user.role}", f"region:{user.region}"])
    # region_filter() is the single definition of "what may this user see"; the agent's
    # run_sql tool injects exactly the same fragment.
    where = [region_filter(user)]
    params: dict = {"limit": limit}
    if product:
        where.append("product = :product")
        params["product"] = product
    query = text(f"SELECT {APPLICATION_COLUMNS} FROM applications "
                 f"WHERE {' AND '.join(where)} ORDER BY submitted_on DESC LIMIT :limit")
    with engine.connect() as conn:
        rows = [dict(r) for r in conn.execute(query, params).mappings().all()]
    return {"count": len(rows), "region_filter": region_filter(user), "applications": rows}


@router.get("/applications/{application_id}")
@observe(name="api.get_application")
def get_application(application_id: str, user: User = Depends(current_user),
                    engine=Depends(get_engine)) -> dict:
    """One application, with the curated statement features when the pipeline has run."""
    update_trace(user_id=user.user_id)
    row = _fetch_application(engine, application_id)
    require_region(user, row["region"])

    features: dict = {}
    try:
        with engine.connect() as conn:
            f = conn.execute(text("SELECT * FROM statement_features WHERE application_id = :aid"),
                             {"aid": application_id}).mappings().first()
        features = dict(f) if f else {}
    except Exception:  # noqa: BLE001  curated layer not loaded yet: the application still returns
        features = {}
    return {"application": row, "statement_features": features}


@router.post("/applications/{application_id}/memo", status_code=status.HTTP_200_OK)
@observe(name="api.draft_memo")
def draft_memo(application_id: str, user: User = Depends(current_user),
               engine=Depends(get_engine)) -> dict:
    """Run the agent graph up to the underwriter interrupt and return the draft memo.

    The graph lives in milestone 6. Importing it lazily (and saying so clearly when it
    is missing) means milestone 3 is usable on its own, which is the whole point of
    building the API before the agents.
    """
    require_write(user)
    row = _fetch_application(engine, application_id)
    require_region(user, row["region"])
    update_trace(user_id=user.user_id, session_id=application_id)

    try:
        from creditlens.agents.run import draft_memo as run_draft_memo
    except Exception as exc:  # noqa: BLE001  ImportError, or the extra is not installed
        raise HTTPException(
            status.HTTP_501_NOT_IMPLEMENTED,
            "memo drafting is not installed yet: milestone 6 (creditlens.agents) is missing. "
            f"Install it with pip install -e \".[agents]\" and rerun. ({exc})") from exc

    try:
        result = run_draft_memo(application_id, user)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001  the graph is an upstream dependency
        # An agent crash is not a bug in the API, and it must not reach the caller as
        # an unhandled traceback. Record the failure, then say plainly what broke.
        audit_store.record(AuditEvent(application_id=application_id, actor=user.user_id,
                                      action="memo_failed", detail=f"{type(exc).__name__}: {exc}"[:500]),
                           engine=engine)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY,
                            f"the agent graph failed while drafting the memo: "
                            f"{type(exc).__name__}: {exc}") from exc

    audit_store.record(AuditEvent(application_id=application_id, actor=user.user_id,
                                  action="memo_drafted",
                                  detail=f"status={result.get('status') if isinstance(result, dict) else 'ok'}"),
                       engine=engine)
    return result if isinstance(result, dict) else {"result": result}

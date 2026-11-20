"""The audit trail: an append-only record of everything that changed state.

A bank has to be able to answer "who approved this loan, when, and on what evidence"
years later. So every memo draft, every Jira ticket and every approval appends one
`AuditEvent` row here. Nothing is ever updated or deleted.

The table is created on demand, so no migration tool is needed for the workshop:

    from creditlens.schemas import AuditEvent
    from creditlens.api import audit

    audit.record(AuditEvent(application_id="A001", actor="U05", action="approved"))
    audit.list_events("A001")
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (Column, DateTime, Integer, MetaData, String, Table, Text,
                        create_engine, insert, select)

from creditlens.config import settings
from creditlens.observability import current_trace_id

# One MetaData holding exactly one table. Defining it in Python (rather than raw DDL)
# means the same code creates the table on SQLite and on SQL Server.
_metadata = MetaData()

audit_events = Table(
    "audit_events",
    _metadata,
    Column("event_id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime, nullable=False),
    Column("application_id", String(32), nullable=False, index=True),
    Column("actor", String(64), nullable=False),
    Column("action", String(64), nullable=False),
    Column("detail", Text, nullable=False, default=""),
    Column("trace_id", String(64), nullable=True),
)

# Engines are expensive to build, so keep one per database URL.
_engines: dict[str, Any] = {}


def default_engine():
    """The engine for `settings.database_url`, built once per URL.

    The URL is read every call rather than at import time, so a test that sets
    DATABASE_URL late still gets its own temporary database.
    """
    url = settings.database_url
    if url not in _engines:
        _engines[url] = create_engine(url)
    return _engines[url]


def ensure_table(engine=None) -> None:
    """Create `audit_events` if it is not there yet. Safe to call on every request."""
    _metadata.create_all(engine or default_engine(), checkfirst=True)


def record(event: "AuditEventLike", engine=None) -> "AuditEventLike":
    """Append one event. Returns the event so callers can log or return it."""
    from creditlens.schemas import AuditEvent  # imported here to keep this module light

    if not isinstance(event, AuditEvent):
        event = AuditEvent.model_validate(event)
    engine = engine or default_engine()
    ensure_table(engine)
    # If a Langfuse trace is active, stitch the audit row to it so an auditor can
    # jump from "who approved" to "what the model actually saw".
    trace_id = event.trace_id or current_trace_id()
    with engine.begin() as conn:
        conn.execute(insert(audit_events).values(
            timestamp=event.timestamp or datetime.utcnow(),
            application_id=event.application_id,
            actor=event.actor,
            action=event.action,
            detail=event.detail or "",
            trace_id=trace_id,
        ))
    return event


def list_events(application_id: Optional[str] = None, engine=None) -> list:
    """Every event for one application, oldest first. No id means the whole trail."""
    from creditlens.schemas import AuditEvent

    engine = engine or default_engine()
    ensure_table(engine)
    query = select(audit_events).order_by(audit_events.c.event_id)
    if application_id:
        query = query.where(audit_events.c.application_id == application_id)
    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()
    return [AuditEvent(timestamp=r["timestamp"], application_id=r["application_id"],
                       actor=r["actor"], action=r["action"], detail=r["detail"],
                       trace_id=r["trace_id"]) for r in rows]


# Type alias used only for readability in the signatures above.
AuditEventLike = Any

"""The things routes ask for: the database engine and the LLM gateway.

Dependency injection in one sentence: a route declares what it needs as a parameter,
and FastAPI calls the small function here that builds it. Nothing is a global, so a
test can point the whole app at a temporary SQLite file just by setting DATABASE_URL
before `create_app()` runs.

The engine is built once in `create_app()` and parked on `app.state`, because opening
a connection pool per request would be slow and would hold a file lock on SQLite.
"""
from __future__ import annotations

from fastapi import Request
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def build_engine(url: str) -> Engine:
    """One place that knows how to turn a DATABASE_URL into an engine."""
    # fast_executemany is a SQL Server driver option; it is meaningless on SQLite.
    kwargs = {"fast_executemany": True} if url.startswith("mssql") else {}
    return create_engine(url, **kwargs)


def get_engine(request: Request) -> Engine:
    """Dependency: the engine this app was created with."""
    return request.app.state.engine


def get_gateway(request: Request):
    """Dependency: the shared LLM gateway, or None when the llm extra is not installed.

    Routes that only read the database never touch it, so a missing litellm must not
    stop the API from starting.
    """
    return getattr(request.app.state, "gateway", None)

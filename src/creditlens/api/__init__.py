"""The CreditLens AI HTTP service (milestone 3, hardened in milestone 9).

Why an API at all? Priya, the credit analyst, does not run Python. She uses a web
screen, and that screen talks to this service. Everything the agents do is reachable
here: list applications she is allowed to see, draft a memo, approve it, read the
audit trail.

Three ideas this package teaches:

1. **Dependency injection**: a route says "I need the database engine and the current
   user" and FastAPI builds them. Tests swap the database by setting DATABASE_URL
   before `create_app()` is called, with no change to any route.
2. **Authentication and RBAC**: a bearer token says who you are, and
   `creditlens.guardrails.rbac` says what you may see. An analyst in North never
   sees a South file.
3. **Tracing**: every handler is wrapped in `@observe`, so one HTTP request becomes
   one Langfuse trace with every LLM call nested inside it.

Run it with `python -m creditlens.api`.
"""
from __future__ import annotations

from typing import Any

__all__ = ["create_app"]


def __getattr__(name: str) -> Any:
    """Import `create_app` only when it is asked for.

    `creditlens.api.audit` is used by the agents, which may run on a machine where
    FastAPI is not installed. Importing lazily keeps `from creditlens.api import audit`
    working there.
    """
    if name == "create_app":
        from creditlens.api.app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

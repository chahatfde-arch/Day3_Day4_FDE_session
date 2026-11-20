"""`create_app()`: builds the FastAPI application and wires every dependency.

The one rule that makes this testable: **nothing is read at import time.** The database
URL, the JWT secret and the CORS origins are all read inside `create_app()`. A test can
therefore do

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    app = create_app()

and the whole application points at a throwaway database. If the engine were a module
level global built on import, that would be impossible.

Start the service with `python -m creditlens.api`, then open http://localhost:8000/docs
for the generated OpenAPI page.
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from creditlens.api import audit as audit_store
from creditlens.api.auth import create_token, current_user, dev_auth_enabled, load_user
from creditlens.api.deps import build_engine, get_engine
from creditlens.api.routes import applications as applications_routes
from creditlens.api.routes import approvals as approvals_routes
from creditlens.api.routes import audit as audit_routes
from creditlens.config import settings
from creditlens.observability import ENABLED as TRACING_ENABLED
from creditlens.observability import observe
from creditlens.schemas import User

API_TITLE = "CreditLens AI"
API_VERSION = "0.3.0"


class TokenRequest(BaseModel):
    """The dev login body. A real deployment replaces this with Keycloak or Okta."""
    user_id: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: User


def cors_origins() -> list[str]:
    """Which browser origins may call this API.

    The default is "*", which is fine for a workshop on localhost and wrong for
    anything public. Set CORS_ORIGINS to a comma-separated list before deploying.
    A browser blocks a cross-origin call unless the server says it is allowed; that
    header is what CORS is, and it protects the *user's* browser, not the server.
    """
    raw = os.environ.get("CORS_ORIGINS", "*").strip()
    return ["*"] if raw in ("", "*") else [o.strip() for o in raw.split(",") if o.strip()]


def create_app(database_url: Optional[str] = None) -> FastAPI:
    """Build the application. Pass `database_url` to override DATABASE_URL explicitly."""
    url = database_url or settings.database_url

    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=("Draft credit memos for Good Bank. A human underwriter makes every "
                     "final decision. All data is synthetic."),
    )

    # Everything the routes inject lives on app.state, built exactly once.
    app.state.database_url = url
    app.state.engine = build_engine(url)
    app.state.gateway = None
    try:
        from creditlens.gateway.llm import get_gateway

        app.state.gateway = get_gateway()
    except Exception:  # noqa: BLE001  the llm extra is optional
        app.state.gateway = None

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins(),
        allow_credentials=False,  # with "*" origins, credentials must stay off
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # The audit table is created once at startup rather than on every write.
    try:
        audit_store.ensure_table(app.state.engine)
    except Exception:  # noqa: BLE001  a missing database must not stop /health
        pass

    @app.get("/health", tags=["meta"])
    @observe(name="api.health")
    def health() -> dict:
        """Cheap liveness check: is the process up and can it reach the database?"""
        database_ok = True
        try:
            from sqlalchemy import text

            with app.state.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception:  # noqa: BLE001
            database_ok = False
        return {"status": "ok" if database_ok else "degraded",
                "service": API_TITLE, "version": API_VERSION,
                "database": url.split("@")[-1], "database_ok": database_ok,
                "tracing": "langfuse" if TRACING_ENABLED else "off",
                "dev_auth": dev_auth_enabled(),
                "offline_llm": settings.llm.offline}

    @app.post("/auth/token", response_model=TokenResponse, tags=["auth"])
    @observe(name="api.auth_token")
    def auth_token(body: TokenRequest, engine=Depends(get_engine)) -> TokenResponse:
        """Dev login: give a user id from the `users` table, get a signed bearer token.

        There is no password, on purpose. The workshop is about what a token *carries*
        (who you are, when it expires) and what the server does with it, not about
        password hashing.
        """
        user = load_user(engine, body.user_id.strip())
        if user is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown user {body.user_id}")
        return TokenResponse(access_token=create_token(user.user_id), user=user)

    @app.get("/me", tags=["auth"])
    @observe(name="api.me")
    def me(user: User = Depends(current_user)) -> dict:
        """Who does the server think you are, and what may you do?"""
        return {"user": user.model_dump(), "can_approve": user.role == "head_of_credit",
                "read_only": user.read_only}

    app.include_router(applications_routes.router)
    app.include_router(approvals_routes.router)
    app.include_router(audit_routes.router)

    # Milestone 8 serves the risk model from the same process. It is optional, so a
    # missing scikit-learn or MLflow must not stop the API from starting.
    try:
        from creditlens.ml import serving as ml_serving

        risk_router = getattr(ml_serving, "router", None) or getattr(ml_serving, "risk_router", None)
        if risk_router is not None:
            app.include_router(risk_router)
    except Exception:  # noqa: BLE001  milestone 8 not installed yet
        pass

    return app


# There is deliberately no module-level `app = create_app()`. Building it on import
# would read DATABASE_URL at import time and defeat the whole design above. Point
# uvicorn at the factory instead:
#     uvicorn --factory creditlens.api.app:create_app

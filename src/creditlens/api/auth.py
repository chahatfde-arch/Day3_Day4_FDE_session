"""Who is calling? Mock JWT login for the workshop, plus a dev header shortcut.

A real bank would put Keycloak or Okta in front of this service (see the `security`
profile in `infra/docker-compose.yml`). For the workshop we sign our own tokens with
`JWT_SECRET` so nobody needs an identity provider to follow along. The *shape* is the
real one: `Authorization: Bearer <token>`, a short expiry, and a `sub` claim holding
the user id.

    POST /auth/token  {"user_id": "U01"}   ->  {"access_token": "...", ...}
    GET  /applications  with  Authorization: Bearer <token>

In dev mode you may skip the token entirely and send `X-User-Id: U05`. That is a
deliberate back door for teaching and is controlled by `API_DEV_AUTH` (on by default).
Turn it off with `API_DEV_AUTH=0` before deploying anywhere real.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import text

from creditlens.api.deps import get_engine
from creditlens.schemas import User

ALGORITHM = "HS256"
TOKEN_TTL_MINUTES = 8 * 60  # one working day


def jwt_secret() -> str:
    """Read the secret at call time so tests can change it with monkeypatch."""
    return os.environ.get("JWT_SECRET", "change-me-for-the-workshop")


def dev_auth_enabled() -> bool:
    """Is the `X-User-Id` shortcut allowed? Off means bearer tokens only."""
    return os.environ.get("API_DEV_AUTH", "1") == "1"


def create_token(user_id: str, ttl_minutes: int = TOKEN_TTL_MINUTES) -> str:
    """Sign a token that says 'this caller is `user_id`'."""
    from jose import jwt

    now = datetime.now(timezone.utc)
    claims = {
        "sub": user_id,
        "iss": "creditlens",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=ttl_minutes)).timestamp()),
    }
    return jwt.encode(claims, jwt_secret(), algorithm=ALGORITHM)


def decode_token(token: str) -> str:
    """Return the user id inside a token, or raise 401."""
    from jose import jwt
    from jose.exceptions import JWTError

    try:
        claims = jwt.decode(token, jwt_secret(), algorithms=[ALGORITHM])
    except JWTError as exc:  # expired, wrong secret, tampered with
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"invalid token: {exc}") from exc
    user_id = claims.get("sub")
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "token has no subject")
    return str(user_id)


def load_user(engine, user_id: str) -> Optional[User]:
    """Look the caller up in the `users` table loaded by the data pipeline."""
    query = text("SELECT user_id, full_name, role, region FROM users WHERE user_id = :uid")
    try:
        with engine.connect() as conn:
            row = conn.execute(query, {"uid": user_id}).mappings().first()
    except Exception as exc:  # noqa: BLE001  table missing means the pipeline never ran
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "the users table is missing: run python -m creditlens.pipeline --load"
                            f" ({exc})") from exc
    if row is None:
        return None
    return User(user_id=row["user_id"], full_name=row["full_name"],
                role=row["role"], region=row["region"])


def current_user(
    authorization: Optional[str] = Header(default=None),
    x_user_id: Optional[str] = Header(default=None),
    engine=Depends(get_engine),
) -> User:
    """FastAPI dependency: turn the request headers into a `User`, or refuse.

    This is the whole point of dependency injection. A route handler just writes
    `user: User = Depends(current_user)` and never touches a header again.
    """
    user_id: Optional[str] = None
    if authorization and authorization.lower().startswith("bearer "):
        user_id = decode_token(authorization.split(" ", 1)[1].strip())
    elif x_user_id and dev_auth_enabled():
        user_id = x_user_id.strip()

    if not user_id:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "not authenticated: send Authorization: Bearer <token> from POST /auth/token"
            + (", or X-User-Id in dev mode" if dev_auth_enabled() else ""),
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = load_user(engine, user_id)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"unknown user {user_id}")
    return user

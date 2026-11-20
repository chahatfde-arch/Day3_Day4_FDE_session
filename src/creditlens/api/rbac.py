"""Permission checks for the API, expressed as small, readable guards.

The rules, straight from the contract:

* **analyst** sees only applications in their own region, and may draft memos there.
* **head_of_credit** sees every region and is the only role that may approve.
* **auditor** reads everything and writes nothing: any POST except /auth/token is 403.

Each guard raises `HTTPException(403)` rather than returning a flag, so a route reads
like a sentence: "require write, require head of credit, then do the work".

The SQL side of the same rule lives in `creditlens.guardrails.rbac.region_filter`, so
the API and the agent's Text-to-SQL tool can never disagree about what a user may see.
"""
from __future__ import annotations

from fastapi import HTTPException, status

from creditlens.schemas import User


def require_write(user: User) -> None:
    """Auditors are read-only everywhere."""
    if user.read_only:
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            f"{user.role} is read-only: this endpoint changes state")


def require_role(user: User, *roles: str) -> None:
    """Only the listed roles may continue."""
    if user.role not in roles:
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            f"{user.role} may not do this: needs one of {', '.join(roles)}")


def require_region(user: User, region: str) -> None:
    """Refuse a record that belongs to a region this user cannot see."""
    if not user.can_see_region(region):
        # Deliberately vague: telling an analyst "that is a South file" already leaks
        # which applications exist. The audit trail records the real reason.
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            "this application is outside your region")


def can_approve(user: User) -> bool:
    """True for the one role allowed to sign off a memo."""
    return user.role == "head_of_credit"

"""One definition of "what may this user see", reused by the API, SQL tools and search.

If the API filtered by region in one place and the agent's `run_sql` tool filtered in
another, the two would drift apart and an analyst would eventually read a file from
another region. So both import from here.

    from creditlens.guardrails.rbac import region_filter, retrieval_filter

    sql = f"SELECT * FROM applications WHERE {region_filter(user)}"
    hits = index.search(q, filter=retrieval_filter(user))
"""
from __future__ import annotations

from typing import Any

from creditlens.schemas import User

# head_of_credit and auditor carry this region and see everything.
ALL_REGIONS = "ALL"


def _sql_literal(value: str) -> str:
    """Quote a string for SQL by doubling any single quote inside it.

    Regions come from our own `users` table, not from the request, so this is belt
    and braces. Anywhere a value *does* come from a user, bind a parameter instead.
    """
    return "'" + str(value).replace("'", "''") + "'"


def sees_all(user: User) -> bool:
    return user.region == ALL_REGIONS


def region_filter(user: User, column: str = "region") -> str:
    """A WHERE fragment restricting rows to the regions this user may read.

    >>> region_filter(User(user_id="U01", full_name="Priya", role="analyst", region="North"))
    "region = 'North'"

    A user who sees everything gets `1 = 1`, which is always true. Returning a clause
    rather than an empty string means callers can always write
    `WHERE {region_filter(user)} AND ...` without worrying about dangling ANDs.
    """
    if sees_all(user):
        return "1 = 1"
    return f"{column} = {_sql_literal(user.region)}"


def retrieval_filter(user: User) -> dict[str, Any]:
    """A metadata filter for the vector store (Qdrant-shaped), empty when unrestricted.

    Policy chunks are not region specific, so this mostly matters for indexes that
    also hold application text.
    """
    if sees_all(user):
        return {}
    return {"must": [{"key": "region", "match": {"value": user.region}}]}


def allowed_regions(user: User, all_regions: tuple[str, ...] = ()) -> list[str]:
    """The explicit list of regions, handy for building an IN clause or a UI dropdown."""
    if sees_all(user):
        return list(all_regions)
    return [user.region]

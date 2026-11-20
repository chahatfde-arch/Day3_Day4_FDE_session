"""The shared state the graph passes from node to node, and the reducers that merge it.

A LangGraph state is a typed dictionary. Each node returns a *partial* dictionary and
LangGraph merges it in. By default a key is overwritten ("last write wins"), which is
exactly wrong when five branches run at the same time and each produces some checks.

That is what a **reducer** fixes. `Annotated[list[dict], operator.add]` says "when two
branches both write `checks`, concatenate the lists instead of letting one win". That one
line is the whole map-reduce pattern: fan out with `Send()`, let each branch return its
slice, and the reducer puts the pieces back together.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict


class MemoState(TypedDict, total=False):
    """Everything one credit memo run knows about itself."""

    # --- what we were asked to do -------------------------------------------------
    application_id: str
    user: dict                      # a creditlens.schemas.User, as a plain dict
    thread_id: str

    # --- what the specialists found (last write wins: one owner each) --------------
    application: Optional[dict]
    financials: Optional[dict]
    conduct: Optional[dict]
    group: Optional[dict]
    risk: Optional[dict]
    memo: Optional[dict]
    exceptions_filed: Optional[bool]
    decision: Optional[dict]

    # --- what many branches write at once (reducers merge) -------------------------
    checks: Annotated[list[dict], operator.add]
    messages: Annotated[list[dict], operator.add]
    tool_calls: Annotated[list[dict], operator.add]

    # --- how the run is going ------------------------------------------------------
    step_count: int
    next: str
    status: str
    error: str


class CheckBranch(TypedDict):
    """The payload one map-reduce branch receives from `Send()`."""
    application_id: str
    rule_group: str


def new_state(application_id: str, user: dict, thread_id: str = "") -> MemoState:
    """A fresh state with every reducer key initialised, so merges never see a missing key."""
    return MemoState(
        application_id=application_id,
        user=user,
        thread_id=thread_id,
        application=None,
        financials=None,
        conduct=None,
        group=None,
        risk=None,
        memo=None,
        exceptions_filed=None,
        decision=None,
        checks=[],
        messages=[],
        tool_calls=[],
        step_count=0,
        next="",
        status="running",
        error="",
    )


def dedupe_checks(checks: list[dict]) -> list[dict]:
    """Keep one check per rule id, in rule order.

    The reducer concatenates, and a supervisor that dispatches the policy checker twice
    would produce the same rule twice. De-duplicating at read time keeps the merge simple.
    """
    seen: dict[str, dict] = {}
    for check in checks or []:
        seen.setdefault(str(check.get("rule_id")), check)
    return [seen[k] for k in sorted(seen)]


def failed_checks(checks: list[dict]) -> list[dict]:
    return [c for c in dedupe_checks(checks) if not c.get("passed", True)]


def tool_call_signature(call: dict) -> str:
    """A stable string for "this tool with these arguments", used by the loop detector."""
    import json

    return f"{call.get('name')}({json.dumps(call.get('args', {}), sort_keys=True, default=str)})"


def detect_repeated_tool_calls(calls: list[dict], limit: int = 3) -> Optional[str]:
    """Return a message when the same tool was called with the same arguments `limit` times.

    A looping agent almost always looks like this: the same question asked over and over
    because the answer never satisfies the supervisor. Catching it here is cheaper than
    catching it on the invoice.
    """
    counts: dict[str, int] = {}
    for call in calls or []:
        signature = tool_call_signature(call)
        counts[signature] = counts.get(signature, 0) + 1
        if counts[signature] >= limit:
            return (f"Loop detected: {signature} was called {counts[signature]} times with identical "
                    "arguments. Stopping so the run cannot spin.")
    return None


def as_state_value(obj: Any) -> Any:
    """Convert a Pydantic object to a plain dict before putting it in state."""
    from creditlens.agents.adapters import as_json

    return as_json(obj)

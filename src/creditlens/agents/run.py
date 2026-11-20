"""Run the credit-memo graph, and resume it after a human decides.

Two functions and one command line:

    draft_memo("A001", user)  ->  runs to the underwriter interrupt, returns the draft
    resume(thread_id, decision) -> continues the same thread, returns approved or rejected

    python -m creditlens.agents.run A001 --user U05
    python -m creditlens.agents.approve <thread_id> approve --user U05

The two commands share state through a SqliteSaver at `data/cache/checkpoints.sqlite`.
That is the whole point of a checkpointer: the graph can stop for a day, the process can
die, and the run still picks up exactly where the underwriter left it.
"""
from __future__ import annotations

import argparse
import sys
import uuid
from typing import Any, Optional, Union

from creditlens.agents import adapters, tools
from creditlens.agents.graph import MAX_STEPS, build_graph, make_checkpointer
from creditlens.agents.state import new_state
from creditlens.observability import callbacks, current_trace_id, flush, observe, update_trace
from creditlens.schemas import ApprovalDecision, CreditMemo, User

#: Where the CLI keeps its checkpoints, so `run` and `approve` see the same thread.
DEFAULT_CHECKPOINT = "checkpoints.sqlite"


def default_checkpoint_path() -> str:
    return str(adapters.cache_dir() / DEFAULT_CHECKPOINT)


def _as_user(user: Union[User, str, dict]) -> User:
    if isinstance(user, User):
        return user
    if isinstance(user, dict):
        return User(**user)
    return adapters.load_user(str(user))


def _config(thread_id: str) -> dict:
    # Two nodes run per supervisor decision, so the recursion limit must clear 2 * MAX_STEPS.
    return {"configurable": {"thread_id": thread_id},
            "callbacks": callbacks(),
            "recursion_limit": 2 * MAX_STEPS + 20}


def _memo_from(values: dict) -> Optional[CreditMemo]:
    raw = values.get("memo")
    return CreditMemo(**raw) if raw else None


# ---------------------------------------------------------------------------
# Draft
# ---------------------------------------------------------------------------


@observe(name="agents.draft_memo")
def draft_memo(application_id: str, user: Union[User, str, dict], thread_id: Optional[str] = None,
               checkpointer: Any = None, inject_loop: bool = False,
               checkpoint_path: Optional[str] = None) -> dict:
    """Run the graph up to the underwriter gate.

    Returns `{"status": "awaiting_approval", "memo": CreditMemo, "thread_id": str,
              "trace_id": str | None, "tool_calls": [{"name", "args", "order"}]}`,
    or `{"status": "error", ...}` when the loop detector or the step limit stopped the run.
    """
    caller = _as_user(user)
    tools.set_current_user(caller)
    tools.start_tool_log()

    thread_id = thread_id or f"{application_id}-{uuid.uuid4().hex[:8]}"
    if checkpointer is None:
        checkpointer = make_checkpointer(checkpoint_path)
    graph = build_graph(checkpointer=checkpointer, inject_loop=inject_loop)

    update_trace(user_id=caller.user_id, session_id=thread_id,
                 tags=["credit-memo", application_id])
    state = new_state(application_id, caller.model_dump(mode="json"), thread_id)
    result = graph.invoke(state, _config(thread_id))
    trace_id = current_trace_id()
    flush()

    recorded = result.get("tool_calls") or tools.tool_calls()
    out: dict[str, Any] = {"thread_id": thread_id, "trace_id": trace_id,
                           "tool_calls": [{"name": c.get("name"), "args": c.get("args"),
                                           "order": i} for i, c in enumerate(recorded)]}
    if result.get("status") == "error" or result.get("error"):
        out.update(status="error", error=result.get("error", ""), memo=_memo_from(result))
        return out
    out.update(status="awaiting_approval", memo=_memo_from(result),
               interrupt=_interrupt_payload(result))
    return out


def _interrupt_payload(result: dict) -> Optional[dict]:
    """What the graph is waiting for. LangGraph puts it under `__interrupt__`."""
    interrupts = result.get("__interrupt__") or []
    for item in interrupts:
        value = getattr(item, "value", item)
        if isinstance(value, dict):
            return value
    return None


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


@observe(name="agents.resume")
def resume(thread_id: str, decision: ApprovalDecision, checkpointer: Any = None,
           checkpoint_path: Optional[str] = None) -> dict:
    """Continue a paused thread with the underwriter's decision."""
    from langgraph.types import Command

    if checkpointer is None:
        checkpointer = make_checkpointer(checkpoint_path)
    graph = build_graph(checkpointer=checkpointer)
    config = _config(thread_id)

    saved = graph.get_state(config)
    if not saved.values:
        return {"status": "error", "error": f"no saved run for thread {thread_id}",
                "thread_id": thread_id, "memo": None}
    user = saved.values.get("user") or {}
    if user:
        tools.set_current_user(User(**user))

    result = graph.invoke(Command(resume=decision.model_dump(mode="json")), config)
    flush()
    status = result.get("status") or ("approved" if decision.decision == "approve" else "rejected")
    return {"status": status, "memo": _memo_from(result), "thread_id": thread_id,
            "trace_id": current_trace_id(),
            "decision": decision.model_dump(mode="json")}


def pending_application_id(thread_id: str, checkpointer: Any = None,
                           checkpoint_path: Optional[str] = None) -> Optional[str]:
    """Which application a paused thread belongs to. The approve CLI needs it."""
    if checkpointer is None:
        checkpointer = make_checkpointer(checkpoint_path)
    graph = build_graph(checkpointer=checkpointer)
    saved = graph.get_state(_config(thread_id))
    return (saved.values or {}).get("application_id")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m creditlens.agents.run",
        description="Draft a credit memo for one application and stop at the underwriter gate.")
    parser.add_argument("application_id", help="for example A001")
    parser.add_argument("--user", default="U05", help="user id of the caller, for example U05")
    parser.add_argument("--thread-id", default=None, help="reuse a thread id instead of a new one")
    parser.add_argument("--checkpoint", default=None,
                        help=f"checkpoint database (default {DEFAULT_CHECKPOINT} in data/cache)")
    parser.add_argument("--inject-loop", action="store_true",
                        help="Day 4 exercise: break the supervisor so the step limit stops the run")
    parser.add_argument("--slack", action="store_true",
                        help="also post the approval buttons to Slack")
    args = parser.parse_args(argv)

    path = args.checkpoint or default_checkpoint_path()
    result = draft_memo(args.application_id, args.user, thread_id=args.thread_id,
                        inject_loop=args.inject_loop, checkpoint_path=path)

    if result["status"] == "error":
        print(f"Run stopped: {result['error']}")
        print(f"Thread {result['thread_id']} made {len(result['tool_calls'])} tool calls.")
        return 1

    memo = result["memo"]
    print(memo.to_markdown())
    print("---")
    print(f"Status: {result['status']}")
    print(f"Thread id: {result['thread_id']}")
    print("Tool calls: " + ", ".join(c["name"] for c in result["tool_calls"]))
    if result.get("trace_id"):
        print(f"Trace: {result['trace_id']}")
    print("")
    print("A human underwriter must decide. Approve or reject with:")
    print(f"  python -m creditlens.agents.approve {result['thread_id']} approve --user U05")
    print(f"  python -m creditlens.agents.approve {result['thread_id']} reject --user U05")

    if args.slack:
        from creditlens.integrations.slack import post_for_approval

        post_for_approval(result["thread_id"], memo)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

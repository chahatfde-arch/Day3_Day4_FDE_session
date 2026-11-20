"""The LangGraph: a supervisor, five specialists, a nested memo writer and a human gate.

Read the graph top to bottom:

    supervisor ──> financial_analyst ─┐
               ──> banking_conduct    │  each one fills its own slice of state
               ──> related_party      │  and hands control straight back
               ──> policy_checker ────┼─> Send() fans out to five rule groups,
               ──> risk_scorer        │    the `checks` reducer merges them back
               ──> memo_writer  ──────┤    (a nested subgraph: draft, review, final)
               ──> file_exceptions ───┤    one idempotent Jira ticket per failed rule
               ──> underwriter_approval ─> interrupt(): the graph stops and waits
               ──> finalize / error_handler

Why a supervisor at all? Because "which specialist next" is a decision, and decisions that
can be written as code should be. This supervisor is deterministic: it looks at what is
still missing and routes there. The model is never asked "what should I do next", only
"write this paragraph". That is the difference between an agent you can audit and an agent
you can only hope about.

Two safety nets, because an agent that loops costs money and time:
  * `step_count` with a hard limit, and
  * a repeated-tool-call detector (same tool, same arguments, three times).
`build_graph(inject_loop=True)` deliberately breaks the supervisor so a workshop pod can
watch both fire.
"""
from __future__ import annotations

import os
from typing import TypedDict, Any, Optional

from creditlens.agents import adapters, memo as memo_module, policy_checks, tools
from creditlens.agents.state import (CheckBranch, MemoState, dedupe_checks,
                                     detect_repeated_tool_calls, failed_checks)
from creditlens.observability import observe
from creditlens.schemas import AuditEvent, PolicyCheck

#: How many supervisor decisions one memo is allowed. A healthy run uses eight.
MAX_STEPS = int(os.environ.get("CREDITLENS_MAX_STEPS", "24"))

#: The order the supervisor works through: (state key that must be filled, specialist node).
PIPELINE: list[tuple[str, str]] = [
    ("application", "financial_analyst"),
    ("financials", "financial_analyst"),
    ("conduct", "banking_conduct"),
    ("group", "related_party"),
    ("checks", "policy_checker"),
    ("risk", "risk_scorer"),
    ("memo", "memo_writer"),
    ("exceptions_filed", "file_exceptions"),
    ("decision", "underwriter_approval"),
]

SPECIALISTS = ["financial_analyst", "banking_conduct", "related_party", "policy_checker",
               "risk_scorer", "memo_writer", "file_exceptions", "underwriter_approval",
               "finalize", "error_handler"]


# ---------------------------------------------------------------------------
# Small helpers shared by the nodes
# ---------------------------------------------------------------------------


def _tool_delta(before: int) -> list[dict]:
    """The tool calls made since `before`, so each node reports its own."""
    return tools.tool_calls()[before:]


def _say(node: str, text: str) -> list[dict]:
    return [{"role": "assistant", "name": node, "content": text}]


def _user(state: MemoState) -> dict:
    return state.get("user") or {}


# ---------------------------------------------------------------------------
# The supervisor
# ---------------------------------------------------------------------------


def _next_specialist(state: MemoState) -> str:
    """Deterministic routing: the first thing that is still missing."""
    for key, node in PIPELINE:
        value = state.get(key)
        if key == "checks":
            if not value:
                return node
            continue
        if value is None or value is False:
            return node
    return "finalize"


def make_supervisor(inject_loop: bool = False):
    @observe(name="agent.supervisor")
    def supervisor(state: MemoState) -> dict:
        step = int(state.get("step_count") or 0) + 1

        if step > MAX_STEPS:
            return {"step_count": step, "next": "error_handler",
                    "error": (f"Step limit reached: the supervisor made {MAX_STEPS} routing decisions "
                              "without finishing the memo. This run was stopped instead of looping."),
                    "messages": _say("supervisor", "step limit reached, stopping")}

        loop_message = detect_repeated_tool_calls(state.get("tool_calls") or [], limit=3)
        if loop_message:
            return {"step_count": step, "next": "error_handler", "error": loop_message,
                    "messages": _say("supervisor", loop_message)}

        if inject_loop and step > 1:
            # Day 4 curveball: a supervisor that keeps re-dispatching a specialist that has
            # already finished. Nothing new is ever written, so the step limit is what stops it.
            return {"step_count": step, "next": "banking_conduct",
                    "messages": _say("supervisor", "re-checking banking conduct (injected loop)")}

        nxt = _next_specialist(state)
        return {"step_count": step, "next": nxt,
                "messages": _say("supervisor", f"next: {nxt}")}

    return supervisor


def route(state: MemoState) -> str:
    return state.get("next") or "finalize"


# ---------------------------------------------------------------------------
# The specialists. Each one fills one slice and returns immediately if it is already filled.
# ---------------------------------------------------------------------------


@observe(name="agent.financial_analyst")
def financial_analyst(state: MemoState) -> dict:
    if state.get("application") is not None and state.get("financials") is not None:
        return {}
    before = len(tools.tool_calls())
    application = tools.get_application(state["application_id"])
    company_id = (application.get("application") or {}).get("company_id", "")
    financials = tools.get_financials(company_id) if company_id else {"error": "no company"}
    return {"application": application, "financials": financials,
            "tool_calls": _tool_delta(before),
            "messages": _say("financial_analyst", f"read the application and the statements for {company_id}")}


@observe(name="agent.banking_conduct")
def banking_conduct(state: MemoState) -> dict:
    if state.get("conduct") is not None:
        return {}  # nothing left to do: this is what makes the injected loop spin harmlessly
    before = len(tools.tool_calls())
    conduct = tools.get_statement_features(state["application_id"])
    return {"conduct": conduct, "tool_calls": _tool_delta(before),
            "messages": _say("banking_conduct", "read 12 months of statement features")}


@observe(name="agent.related_party")
def related_party(state: MemoState) -> dict:
    if state.get("group") is not None:
        return {}
    before = len(tools.tool_calls())
    company_id = ((state.get("application") or {}).get("application") or {}).get("company_id", "")
    group = tools.get_group_exposure(company_id) if company_id else {}
    return {"group": group, "tool_calls": _tool_delta(before),
            "messages": _say("related_party", f"walked the director and address graph from {company_id}")}


# --- the map-reduce: one branch per rule group ------------------------------------


@observe(name="agent.policy_checker")
def policy_checker(state: MemoState) -> dict:
    """The fan-out point. The real work happens in `check_rule_group`, five times at once."""
    return {"messages": _say("policy_checker",
                             "fanning out over rule groups: " + ", ".join(policy_checks.RULE_GROUPS))}


def fan_out_rule_groups(state: MemoState) -> list:
    """`Send()` one branch per rule group. Each returns its own `checks`; the reducer merges."""
    from langgraph.types import Send

    return [Send("check_rule_group",
                 CheckBranch(application_id=state["application_id"], rule_group=name))
            for name in policy_checks.RULE_GROUPS]


@observe(name="agent.check_rule_group")
def check_rule_group(branch: CheckBranch) -> dict:
    """One branch of the map-reduce: run the rules of a single group."""
    facts = policy_checks.load_facts(branch["application_id"])
    found: list[PolicyCheck] = policy_checks.run_group(branch["rule_group"], facts)
    return {"checks": [adapters.as_json(c) for c in found],
            "messages": _say("check_rule_group",
                             f"{branch['rule_group']}: {len(found)} checks")}


@observe(name="agent.risk_scorer")
def risk_scorer(state: MemoState) -> dict:
    if state.get("risk") is not None:
        return {}
    before = len(tools.tool_calls())
    conduct = state.get("conduct") or {}
    failed = [c["rule_id"] for c in failed_checks(state.get("checks") or [])]
    risk = adapters.as_json(adapters.score_risk(state["application_id"],
                                                features=conduct, failed_rules=failed))
    tools.record_tool_call("score_risk", {"application_id": state["application_id"]})
    return {"risk": risk, "tool_calls": _tool_delta(before),
            "messages": _say("risk_scorer", f"risk band {risk.get('risk_band')}")}


# ---------------------------------------------------------------------------
# memo_writer: a nested subgraph (draft -> self-review -> final)
# ---------------------------------------------------------------------------


class _MemoDraftState(TypedDict, total=False):
    """State of the nested graph: facts in, sections and review out.

    LangGraph keeps only the keys declared on the state schema between nodes, which is
    why this is a TypedDict and not a plain dict: with `dict` the `facts` written by the
    caller would be dropped before `self_review` ran.
    """
    facts: dict
    sections: dict
    review: dict


def build_memo_subgraph():
    """A small graph of its own. Nesting keeps the memo's three steps out of the main graph."""
    from langgraph.graph import END, START, StateGraph

    def draft(state: dict) -> dict:
        return {"sections": memo_module.draft_sections(state["facts"])}

    def self_review(state: dict) -> dict:
        return {"review": memo_module.review_sections(state["sections"], state["facts"])}

    def final(state: dict) -> dict:
        return {"sections": memo_module.finalise(state["sections"], state["review"], state["facts"])}

    builder = StateGraph(_MemoDraftState)
    builder.add_node("draft_sections", draft)
    builder.add_node("self_review", self_review)
    builder.add_node("final", final)
    builder.add_edge(START, "draft_sections")
    builder.add_edge("draft_sections", "self_review")
    builder.add_edge("self_review", "final")
    builder.add_edge("final", END)
    return builder.compile()


_MEMO_SUBGRAPH = None


def memo_subgraph():
    global _MEMO_SUBGRAPH
    if _MEMO_SUBGRAPH is None:
        _MEMO_SUBGRAPH = build_memo_subgraph()
    return _MEMO_SUBGRAPH


@observe(name="agent.memo_writer")
def memo_writer(state: MemoState) -> dict:
    if state.get("memo") is not None:
        return {}
    before = len(tools.tool_calls())
    facts = memo_module.build_facts(dict(state))
    facts["checks"] = dedupe_checks(facts.get("checks") or [])

    # Cite the policy sections behind the rules that failed, so the memo shows its sources.
    citations: list[str] = []
    for rule_id in facts.get("failed_rule_ids") or []:
        hits = tools.search_policy(f"rule {rule_id} threshold").get("hits", [])
        citations.extend(h["chunk_id"] for h in hits[:1])

    result = memo_subgraph().invoke({"facts": facts})
    sections = result["sections"]
    model_name = "offline-stub" if os.environ.get("LLM_OFFLINE") == "1" else "gateway"
    draft = memo_module.build_memo(facts, sections, sorted(set(citations)), model_name)

    adapters.audit(AuditEvent(application_id=state["application_id"],
                              actor=_user(state).get("user_id", "agent"),
                              action="memo_drafted",
                              detail=f"recommendation={draft.recommendation}, "
                                     f"exceptions={len(draft.policy_exceptions)}"))
    return {"memo": adapters.as_json(draft), "tool_calls": _tool_delta(before),
            "messages": _say("memo_writer", f"drafted a memo recommending {draft.recommendation}")}


# ---------------------------------------------------------------------------
# file_exceptions: one Jira ticket per failed rule, idempotent
# ---------------------------------------------------------------------------


@observe(name="agent.file_exceptions")
def file_exceptions(state: MemoState) -> dict:
    if state.get("exceptions_filed"):
        return {}
    before = len(tools.tool_calls())
    memo = dict(state.get("memo") or {})
    exceptions = list(memo.get("policy_exceptions") or [])
    application_id = state["application_id"]
    filed = []
    for exception in exceptions:
        rule_id = exception["rule_id"]
        summary = f"{application_id} policy exception on rule {rule_id}"
        description = (f"Measured value {exception.get('measured_value')}. "
                       f"{exception.get('justification', '')}")
        ticket = tools.file_jira_ticket(application_id=application_id, rule_id=rule_id,
                                        summary=summary, description=description)
        exception["jira_key"] = ticket.get("key")
        filed.append(ticket.get("key"))
        adapters.audit(AuditEvent(application_id=application_id,
                                  actor=_user(state).get("user_id", "agent"),
                                  action="exception_ticket_filed",
                                  detail=f"rule {rule_id} -> {ticket.get('key')} "
                                         f"(created={ticket.get('created')})"))
    memo["policy_exceptions"] = exceptions
    return {"memo": memo, "exceptions_filed": True, "tool_calls": _tool_delta(before),
            "messages": _say("file_exceptions",
                             f"filed {len(filed)} exception tickets: {', '.join(filed) or 'none'}")}


# ---------------------------------------------------------------------------
# The human gate
# ---------------------------------------------------------------------------


@observe(name="agent.underwriter_approval")
def underwriter_approval(state: MemoState) -> dict:
    """Stop the graph and wait for a person.

    `interrupt()` saves the state to the checkpointer and raises. The caller gets the draft
    memo back. When somebody later resumes the same thread with a decision, execution starts
    again *here*, and `interrupt()` returns that decision. No automated system approves a
    loan at Good Bank (policy section 1).
    """
    from langgraph.types import interrupt

    memo = state.get("memo") or {}
    decision = interrupt({
        "kind": "underwriter_approval",
        "application_id": state["application_id"],
        "recommendation": memo.get("recommendation"),
        "exception_count": len(memo.get("policy_exceptions") or []),
        "question": "Approve or reject this credit memo?",
    })
    if hasattr(decision, "model_dump"):
        decision = decision.model_dump()
    if isinstance(decision, str):
        decision = {"decision": decision}
    return {"decision": adapters.as_json(decision),
            "messages": _say("underwriter_approval", f"received decision {decision.get('decision')}")}


@observe(name="agent.finalize")
def finalize(state: MemoState) -> dict:
    decision = state.get("decision") or {}
    verdict = decision.get("decision")
    status = {"approve": "approved", "reject": "rejected"}.get(verdict, "awaiting_approval")
    if verdict:
        adapters.audit(AuditEvent(
            application_id=state["application_id"],
            actor=decision.get("decided_by") or _user(state).get("user_id", "unknown"),
            action=f"memo_{status}",
            detail=(f"channel={decision.get('channel', 'api')}; comment={decision.get('comment', '')}"
                    + (f"; approved_inr_000={decision['approved_inr_000']}"
                       if decision.get("approved_inr_000") is not None else ""))))
    return {"status": status, "messages": _say("finalize", f"run finished with status {status}")}


@observe(name="agent.error_handler")
def error_handler(state: MemoState) -> dict:
    message = state.get("error") or "the run stopped for an unknown reason"
    return {"status": "error", "error": message,
            "messages": _say("error_handler", message)}


# ---------------------------------------------------------------------------
# Checkpointers
# ---------------------------------------------------------------------------


_SHARED_MEMORY = None


def make_checkpointer(path: Optional[str] = None):
    """One shared MemorySaver by default; a SqliteSaver when a path is given.

    Shared on purpose: `draft_memo()` and a later `resume()` in the same notebook or test must
    see the same thread. Two CLI processes cannot share memory, which is why the CLI passes a
    SQLite path instead.
    """
    global _SHARED_MEMORY
    if path is None:
        from langgraph.checkpoint.memory import MemorySaver

        if _SHARED_MEMORY is None:
            _SHARED_MEMORY = MemorySaver()
        return _SHARED_MEMORY
    import sqlite3
    from pathlib import Path

    from langgraph.checkpoint.sqlite import SqliteSaver

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(connection)


# ---------------------------------------------------------------------------
# Assembling the graph
# ---------------------------------------------------------------------------


@observe(name="agent.build_graph")
def build_graph(checkpointer: Any = None, inject_loop: bool = False,
                checkpoint_path: Optional[str] = None):
    """Compile the credit-memo graph.

    `inject_loop=True` is the Day 4 exercise: the supervisor misroutes on purpose so the
    step limit has to stop the run.
    """
    from langgraph.graph import END, START, StateGraph

    if checkpointer is None:
        checkpointer = make_checkpointer(checkpoint_path)

    builder = StateGraph(MemoState)
    builder.add_node("supervisor", make_supervisor(inject_loop=inject_loop))
    builder.add_node("financial_analyst", financial_analyst)
    builder.add_node("banking_conduct", banking_conduct)
    builder.add_node("related_party", related_party)
    builder.add_node("policy_checker", policy_checker)
    builder.add_node("check_rule_group", check_rule_group)
    builder.add_node("risk_scorer", risk_scorer)
    builder.add_node("memo_writer", memo_writer)
    builder.add_node("file_exceptions", file_exceptions)
    builder.add_node("underwriter_approval", underwriter_approval)
    builder.add_node("finalize", finalize)
    builder.add_node("error_handler", error_handler)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", route, SPECIALISTS)

    # Every specialist reports back to the supervisor, which decides again.
    for node in ["financial_analyst", "banking_conduct", "related_party", "risk_scorer",
                 "memo_writer", "file_exceptions", "underwriter_approval"]:
        builder.add_edge(node, "supervisor")

    # Map: one Send per rule group. Reduce: the `checks` reducer merges the branches, and
    # the supervisor runs once more only after every branch has finished.
    builder.add_conditional_edges("policy_checker", fan_out_rule_groups, ["check_rule_group"])
    builder.add_edge("check_rule_group", "supervisor")

    builder.add_edge("finalize", END)
    builder.add_edge("error_handler", END)
    return builder.compile(checkpointer=checkpointer)

"""Milestone 6 done-when: one application runs to a draft memo, pauses for approval, and files
exactly one Jira ticket per policy exception. Everything here runs offline (LLM_OFFLINE=1).

The fixture points DATABASE_URL, the mock Jira file and the audit log at temporary files, so
the tests never touch the developer's own creditlens.db or data/cache.
"""
from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("langgraph")

from creditlens.pipeline.load import load_all  # noqa: E402
from creditlens.schemas import ApprovalDecision, CreditMemo, User  # noqa: E402

HEAD = User(user_id="U05", full_name="Head of Credit", role="head_of_credit", region="ALL")
FINANCIAL_RULES = {"3.1", "3.2", "3.3"}


@pytest.fixture(scope="module", autouse=True)
def isolated_env(s, outputs, tmp_path_factory):
    """Temp database, temp Jira store, temp audit log, offline LLM. Restored afterwards."""
    root = tmp_path_factory.mktemp("agents")
    url = f"sqlite:///{root / 'agents.db'}"
    load_all(url, s.paths.raw, outputs)
    old = {k: os.environ.get(k) for k in ("DATABASE_URL", "CREDITLENS_JIRA_PATH", "LLM_OFFLINE")}
    os.environ["DATABASE_URL"] = url
    os.environ["CREDITLENS_JIRA_PATH"] = str(root / "jira.json")
    os.environ["LLM_OFFLINE"] = "1"
    from creditlens.gateway import llm

    llm._default = None  # a fresh gateway that sees LLM_OFFLINE=1
    yield root
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    llm._default = None


@pytest.fixture(scope="module")
def expected(s) -> dict:
    return json.loads((s.paths.answer_key / "expected_policy_flags.json").read_text())


@pytest.fixture(scope="module")
def doc_kinds(s) -> dict:
    return json.loads((s.paths.answer_key / "financial_doc_kinds.json").read_text())


def _failed(checks) -> set[str]:
    return {c.rule_id for c in checks if not c.passed}


def test_policy_checks_match_the_answer_key(expected, doc_kinds):
    """Deterministic checks reproduce the generator's expected flags, application by application."""
    from creditlens.agents.policy_checks import run_checks

    compared = 0
    for app_id, exp in expected.items():
        checks = run_checks(app_id)
        got = _failed(checks)
        want = set(exp["flags"])
        financials_available = any(c.rule_id == "3.1" and c.measured_value is not None for c in checks)
        if doc_kinds.get(exp["company_id"]) != "text" or not financials_available:
            # Scans have no text layer; OCR is milestone 4's job. Compare the other rules only.
            got -= FINANCIAL_RULES
            want -= FINANCIAL_RULES
        assert got == want, f"{app_id}: got {sorted(got)}, expected {sorted(want)}"
        compared += 1
    assert compared == len(expected)


def test_draft_memo_pauses_for_approval_and_files_one_ticket_per_exception(expected):
    from creditlens.agents.run import draft_memo
    from creditlens.mcp_jira.client import search_tickets

    app_id = next(a for a, e in expected.items() if e["flags"] and not set(e["flags"]) & FINANCIAL_RULES)
    result = draft_memo(app_id, HEAD)
    assert result["status"] == "awaiting_approval", result.get("error")
    memo = result["memo"]
    assert isinstance(memo, CreditMemo)
    assert memo.application_id == app_id
    assert memo.recommendation in ("decline", "refer")  # at least one rule failed
    failed_in_memo = {e.rule_id for e in memo.policy_exceptions}
    assert failed_in_memo >= set(expected[app_id]["flags"]) - FINANCIAL_RULES
    for e in memo.policy_exceptions:
        assert e.jira_key, f"exception {e.rule_id} has no ticket"
    # Run it again: idempotency keys mean no new tickets.
    before = len(search_tickets(app_id))
    draft_memo(app_id, HEAD)
    assert len(search_tickets(app_id)) == before
    names = [t["name"] for t in result["tool_calls"]]
    assert "get_application" in names and "score_risk" in names


def test_resume_with_approval_records_an_audit_event():
    from creditlens.agents import adapters
    from creditlens.agents.run import draft_memo, resume

    result = draft_memo("A001", HEAD)
    assert result["status"] == "awaiting_approval"
    final = resume(result["thread_id"], ApprovalDecision(application_id="A001", decided_by="U05",
                                                           decision="approve", comment="ok"))
    assert final["status"] == "approved"
    actions = [e.action for e in adapters.audit_events("A001")]
    assert "memo_drafted" in actions
    assert any("approv" in a for a in actions)


def test_memo_markdown_has_every_policy_section_in_order():
    from creditlens.agents.run import draft_memo

    md = draft_memo("A001", HEAD)["memo"].to_markdown()
    headings = [line for line in md.splitlines() if line.startswith("## ")]
    wanted = ["Applicant summary", "Facility requested", "Financial analysis", "Banking conduct",
              "Related parties", "Risk score", "Policy exceptions", "Recommendation"]
    positions = [next(i for i, h in enumerate(headings) if w in h) for w in wanted]
    assert positions == sorted(positions)


def test_loop_curveball_is_stopped_by_the_step_limit():
    from creditlens.agents.run import draft_memo

    result = draft_memo("A001", HEAD, inject_loop=True)
    assert result["status"] == "error"
    assert "loop" in str(result.get("error", "")).lower() or "step" in str(result.get("error", "")).lower()


def test_run_sql_is_read_only_and_region_scoped():
    from creditlens.agents import tools

    run_sql = tools.get_tools()["run_sql"]
    analyst = User(user_id="U01", full_name="North analyst", role="analyst", region="North")
    tools.set_current_user(analyst)
    try:
        ok = run_sql.invoke({"query": "SELECT application_id, region FROM applications"})
        assert not ok.get("rejected"), ok
        rows = ok["rows"]
        assert rows and all(r["region"] == "North" for r in rows)
        assert "north" in ok["query"].lower()
        for bad in ("UPDATE applications SET requested_inr_000 = 1",
                    "DROP TABLE applications",
                    "SELECT * FROM applications; DROP TABLE users",
                    "SELECT * FROM sqlite_master"):
            res = run_sql.invoke({"query": bad})
            assert res.get("rejected") is True, bad
    finally:
        tools.set_current_user(None)


def test_text_to_sql_offline_answers_a_canned_question():
    from creditlens.agents.tools import text_to_sql

    out = text_to_sql("How many applications are there per product?", user=HEAD)
    assert out["rejected"] is False
    assert out["row_count"] >= 1
    assert out["sql"].lower().startswith("select")

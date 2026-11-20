"""The Jira MCP server: filing the same exception twice must create one ticket (idempotency)."""
from __future__ import annotations

import pytest

from creditlens.mcp_jira.backend import MockJira
from creditlens.mcp_jira.client import file_ticket, get_ticket, idempotency_key, search_tickets


def test_idempotency_key_is_application_plus_rule():
    assert idempotency_key("A007", "3.1") == "A007:3.1"


def test_mock_backend_dedupes_on_the_idempotency_key(tmp_path):
    jira = MockJira(tmp_path / "jira.json")
    first = jira.create_issue("CREDIT", "A007 rule 3.1", "DSCR 1.1 below 1.25", idempotency_key="A007:3.1")
    second = jira.create_issue("CREDIT", "A007 rule 3.1 again", "", idempotency_key="A007:3.1")
    assert first["key"] == second["key"]
    assert first.get("created") is True and second.get("created") is False
    assert len(jira.search_issues("A007")) == 1
    # A different rule is a different ticket.
    third = jira.create_issue("CREDIT", "A007 rule 6.1", "", idempotency_key="A007:6.1")
    assert third["key"] != first["key"]
    assert len(jira.search_issues("A007")) == 2


def test_client_in_process_round_trip(tmp_path):
    path = tmp_path / "jira.json"
    a = file_ticket("A009", "4.1", summary="cheque returns", jira_path=path, transport="inprocess")
    b = file_ticket("A009", "4.1", summary="cheque returns", jira_path=path, transport="inprocess")
    assert a["key"] == b["key"]
    assert get_ticket(a["key"], jira_path=path, transport="inprocess")["key"] == a["key"]
    assert len(search_tickets("A009", jira_path=path, transport="inprocess")) == 1


@pytest.mark.slow
def test_client_over_stdio_when_mcp_is_installed(tmp_path):
    pytest.importorskip("mcp")
    path = tmp_path / "jira.json"
    try:
        a = file_ticket("A010", "5.2", summary="group exposure", jira_path=path, transport="stdio")
    except Exception as e:  # noqa: BLE001  spawning a child process can fail in locked-down CI
        pytest.skip(f"stdio transport unavailable here: {e}")
    b = file_ticket("A010", "5.2", summary="group exposure", jira_path=path, transport="stdio")
    assert a["key"] == b["key"]

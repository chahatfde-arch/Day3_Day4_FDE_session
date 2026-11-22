"""The local Jira-compatible service (infra/jira_mock/server.py) with the real Jira Cloud client.

Promise: with JIRA_BASE_URL pointing at the local service, the agent's tickets travel over HTTP
with Basic auth exactly as they would to Jira Cloud, stay idempotent, say they came from the
local mock, and fall back to the test project (labelled) when the service is down.
"""
import importlib.util
import socket
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
import requests

from creditlens.mcp_jira.backend import JiraCloud, JiraWithFallback, MockJira, jira_is_local_mock

SERVER = Path(__file__).resolve().parents[1] / "infra" / "jira_mock" / "server.py"


@pytest.fixture()
def local_jira(tmp_path):
    spec = importlib.util.spec_from_file_location("local_jira_mock", SERVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.STORE = tmp_path / "jira_local.json"
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), mod.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_the_cloud_client_files_one_ticket_through_the_local_service_and_says_where(local_jira):
    jira = JiraCloud(local_jira, "trainer@goodbank.example", "any-token")
    first = jira.create_issue("CREDIT", "A901 policy exception on rule 5.2", "DSCR below the minimum", "A901:5.2")
    second = jira.create_issue("CREDIT", "A901 policy exception on rule 5.2", "DSCR below the minimum", "A901:5.2")
    assert first["created"] is True and second["created"] is False
    assert first["key"] == second["key"] == "CREDIT-1"
    assert first["backend"] == "jira-local-mock"
    assert jira.get_issue("CREDIT-1")["summary"] == "A901 policy exception on rule 5.2"
    assert [i["key"] for i in jira.search_issues("DSCR")] == ["CREDIT-1"]
    assert requests.get(f"{local_jira}/browse/CREDIT-1", timeout=5).status_code == 200


def test_the_local_service_wants_basic_auth_like_jira_cloud(local_jira):
    assert requests.get(f"{local_jira}/rest/api/3/myself", timeout=5).status_code == 401
    bad_jql = requests.get(f"{local_jira}/rest/api/3/search", params={"jql": "assignee = me"},
                           auth=("a@b.example", "t"), timeout=5)
    assert bad_jql.status_code == 400, "unknown JQL must be refused, not silently match nothing"


def test_a_stopped_service_falls_back_to_the_test_project_and_says_so(tmp_path):
    with socket.socket() as s:                      # a port nobody listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    jira = JiraWithFallback(JiraCloud(f"http://127.0.0.1:{port}", "a@b.example", "t", timeout=2),
                            MockJira(tmp_path / "jira.json"))
    ticket = jira.create_issue("CREDIT", "A901 policy exception on rule 5.2", "", "A901:5.2")
    assert ticket["backend"] == "mock-fallback" and "not reachable" in ticket["fallback_reason"]


def test_only_local_addresses_count_as_the_local_mock():
    assert jira_is_local_mock("http://localhost:8092")
    assert not jira_is_local_mock("https://goodbank.atlassian.net")
    assert not jira_is_local_mock("")

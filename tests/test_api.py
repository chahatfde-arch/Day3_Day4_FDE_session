"""Milestone 3 and 9: the API answers, and RBAC actually holds.

The fixture below is the interesting part. It writes the session's generated data into
a *temporary* SQLite file, sets DATABASE_URL, and only then calls `create_app()`. That
proves the app reads its configuration at create time rather than at import time, which
is what makes the whole service testable.

Everything runs with LLM_OFFLINE=1 and no network.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("fastapi", reason="pip install -e '.[api]'")
pytest.importorskip("jose", reason="pip install -e '.[api]'")

from fastapi.testclient import TestClient  # noqa: E402

from creditlens.pipeline.load import load_all  # noqa: E402

NORTH_ANALYST = "U01"       # Benjamin Nagarajan, analyst, North
SOUTH_ANALYST = "U02"       # Keya Devi, analyst, South
HEAD_OF_CREDIT = "U05"      # Amaira Dey, head_of_credit, ALL
AUDITOR = "U06"             # Gautami Dora, auditor, ALL


@pytest.fixture(scope="module")
def applications_df(s) -> pd.DataFrame:
    return pd.read_csv(s.paths.raw / "applications" / "applications.csv")


@pytest.fixture(scope="module")
def north_application(applications_df) -> str:
    return applications_df[applications_df.region == "North"].iloc[0]["application_id"]


@pytest.fixture(scope="module")
def south_application(applications_df) -> str:
    return applications_df[applications_df.region == "South"].iloc[0]["application_id"]


@pytest.fixture(scope="module")
def client(s, outputs, tmp_path_factory):
    """A TestClient wired to a throwaway database holding the generated data."""
    monkeypatch = pytest.MonkeyPatch()
    db_file = tmp_path_factory.mktemp("apidb") / "creditlens_test.db"
    url = f"sqlite:///{db_file}"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("LLM_OFFLINE", "1")
    monkeypatch.setenv("JWT_SECRET", "test-secret-not-for-production")
    monkeypatch.setenv("API_DEV_AUTH", "1")

    load_all(url, s.paths.raw, outputs)

    # Imported here, after the environment is set, so nothing can cache the old URL.
    from creditlens.api.app import create_app

    app = create_app()
    assert app.state.database_url == url, "the app must read DATABASE_URL at create time"
    with TestClient(app) as test_client:
        yield test_client
    monkeypatch.undo()


def token_for(client, user_id: str) -> str:
    response = client.post("/auth/token", json={"user_id": user_id})
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


def bearer(client, user_id: str) -> dict:
    return {"Authorization": f"Bearer {token_for(client, user_id)}"}


def dev_header(user_id: str) -> dict:
    return {"X-User-Id": user_id}


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------


def test_health_is_open_and_reports_the_database(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["database_ok"] is True


def test_token_round_trip_identifies_the_user(client):
    response = client.get("/me", headers=bearer(client, HEAD_OF_CREDIT))
    assert response.status_code == 200
    user = response.json()["user"]
    assert user["user_id"] == HEAD_OF_CREDIT
    assert user["role"] == "head_of_credit"
    assert response.json()["can_approve"] is True


def test_unknown_user_cannot_get_a_token(client):
    assert client.post("/auth/token", json={"user_id": "U99"}).status_code == 404


def test_no_credentials_is_401(client):
    assert client.get("/applications").status_code == 401


def test_a_tampered_token_is_401(client):
    good = token_for(client, NORTH_ANALYST)
    tampered = good[:-3] + ("aaa" if not good.endswith("aaa") else "bbb")
    assert client.get("/applications", headers={"Authorization": f"Bearer {tampered}"}).status_code == 401


def test_dev_header_works_when_dev_auth_is_on(client):
    response = client.get("/me", headers=dev_header(NORTH_ANALYST))
    assert response.status_code == 200
    assert response.json()["user"]["region"] == "North"


# ---------------------------------------------------------------------------
# RBAC: the point of milestone 9
# ---------------------------------------------------------------------------


def test_analyst_list_is_filtered_to_their_own_region(client):
    body = client.get("/applications", headers=bearer(client, NORTH_ANALYST)).json()
    assert body["count"] > 0, "the North analyst should see at least one file"
    assert {a["region"] for a in body["applications"]} == {"North"}


def test_analyst_cannot_read_another_region(client, south_application):
    response = client.get(f"/applications/{south_application}", headers=bearer(client, NORTH_ANALYST))
    assert response.status_code == 403
    # The refusal must not confirm which region the file belongs to.
    assert "South" not in response.text


def test_the_same_file_is_invisible_in_the_analyst_list(client, south_application):
    body = client.get("/applications", headers=bearer(client, NORTH_ANALYST)).json()
    assert south_application not in {a["application_id"] for a in body["applications"]}


def test_analyst_can_read_their_own_region(client, north_application):
    response = client.get(f"/applications/{north_application}", headers=bearer(client, NORTH_ANALYST))
    assert response.status_code == 200
    assert response.json()["application"]["region"] == "North"


def test_head_of_credit_sees_every_region(client, applications_df):
    body = client.get("/applications", headers=bearer(client, HEAD_OF_CREDIT), params={"limit": 500}).json()
    assert body["count"] == len(applications_df)
    assert {a["region"] for a in body["applications"]} == set(applications_df.region.unique())


def test_head_of_credit_can_read_a_south_file(client, south_application):
    response = client.get(f"/applications/{south_application}", headers=bearer(client, HEAD_OF_CREDIT))
    assert response.status_code == 200


def test_auditor_reads_everywhere(client, south_application, north_application):
    for application_id in (south_application, north_application):
        assert client.get(f"/applications/{application_id}",
                          headers=bearer(client, AUDITOR)).status_code == 200


def test_missing_application_is_404_for_a_user_who_sees_all(client):
    assert client.get("/applications/A999", headers=bearer(client, HEAD_OF_CREDIT)).status_code == 404


# ---------------------------------------------------------------------------
# Approvals and the audit trail
# ---------------------------------------------------------------------------


def test_auditor_cannot_approve(client, north_application):
    response = client.post(f"/applications/{north_application}/approve",
                           json={"decision": "approve", "comment": "looks fine"},
                           headers=bearer(client, AUDITOR))
    assert response.status_code == 403
    assert "read-only" in response.json()["detail"]


def test_analyst_cannot_approve(client, north_application):
    response = client.post(f"/applications/{north_application}/approve",
                           json={"decision": "approve"}, headers=bearer(client, NORTH_ANALYST))
    assert response.status_code == 403


def test_head_of_credit_approves_and_the_event_is_audited(client, north_application):
    response = client.post(f"/applications/{north_application}/approve",
                           json={"decision": "approve", "comment": "DSCR exception accepted"},
                           headers=bearer(client, HEAD_OF_CREDIT))
    assert response.status_code == 200, response.text
    decision = response.json()["decision"]
    # The server, not the client, decides who the approver was.
    assert decision["decided_by"] == HEAD_OF_CREDIT
    assert decision["decision"] == "approve"

    trail = client.get(f"/applications/{north_application}/audit",
                       headers=bearer(client, AUDITOR)).json()
    assert trail["count"] >= 1
    actions = [e["action"] for e in trail["events"]]
    assert "application_approved" in actions


def test_auditor_cannot_be_spoofed_through_the_body(client, north_application):
    """Even a valid approver cannot claim to be somebody else: there is no such field."""
    response = client.post(f"/applications/{north_application}/approve",
                           json={"decision": "approve", "decided_by": AUDITOR},
                           headers=bearer(client, HEAD_OF_CREDIT))
    assert response.status_code == 200
    assert response.json()["decision"]["decided_by"] == HEAD_OF_CREDIT


def test_analyst_cannot_read_the_audit_trail_of_another_region(client, south_application):
    response = client.get(f"/applications/{south_application}/audit",
                          headers=bearer(client, NORTH_ANALYST))
    assert response.status_code == 403


# ---------------------------------------------------------------------------
# The memo endpoint: present either way, honest when milestone 6 is missing
# ---------------------------------------------------------------------------


def test_memo_endpoint_is_wired_up(client, north_application):
    """The route exists and is honest about what it could not do.

    Milestone 6 (creditlens.agents) is built in parallel, so all three of these are
    correct answers: 200 once it works, 501 while it is not installed, 502 when it is
    installed but still in pieces. What must never happen is an unhandled traceback.
    """
    response = client.post(f"/applications/{north_application}/memo",
                           headers=bearer(client, NORTH_ANALYST))
    assert response.status_code in (200, 202, 501, 502), response.text
    if response.status_code == 501:
        assert "milestone 6" in response.json()["detail"]
    if response.status_code == 502:
        assert "agent graph failed" in response.json()["detail"]


def test_auditor_cannot_draft_a_memo(client, north_application):
    response = client.post(f"/applications/{north_application}/memo", headers=bearer(client, AUDITOR))
    assert response.status_code == 403


def test_analyst_cannot_draft_a_memo_outside_their_region(client, south_application):
    response = client.post(f"/applications/{south_application}/memo",
                           headers=bearer(client, NORTH_ANALYST))
    assert response.status_code == 403

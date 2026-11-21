"""The Good Bank Credit Workbench (app/workbench): the rules live on the server, not in the screen.

The fast tests check who may see what, and that a refusal is written to the audit log. The slow ones load notebook
04b's agent (one to two minutes) and walk Rohan's file (A901) from review to decision: the counter-offer, an analyst
refused, exceptions filed once however often you click, and the Head of Credit's decision recorded.

The audit log, the Jira file and the learned conditions go to a temporary folder, and anything the agent's start-up
writes in data/cache is put back afterwards, so the trainer's demo files are never changed.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SERVER = REPO / "app" / "workbench" / "server.py"
CACHE = REPO / "data" / "cache"
NEEDS = [REPO / "creditlens.db", REPO / "data" / "raw" / "registry" / "users.csv",
         REPO / "data" / "story" / "raw" / "applications" / "applications.csv"]
pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in NEEDS),
    reason="needs the database and the story files: python -m creditlens.pipeline --load, "
           "python -m creditlens.scenarios, python -m creditlens.scenarios.story build")

LUCKY, AMAIRA, AUDITOR = "U04", "U05", "U06"      # credit analyst (West), Head of Credit, auditor


@pytest.fixture(scope="module")
def wb(tmp_path_factory):
    """Import the server without its background loader, the way VS Code would start it: with LLM_OFFLINE=0."""
    saved = {k: os.environ.get(k) for k in ("WORKBENCH_LOAD_AGENT", "LLM_OFFLINE")}
    os.environ["WORKBENCH_LOAD_AGENT"] = "0"
    os.environ["LLM_OFFLINE"] = "0"
    try:
        spec = importlib.util.spec_from_file_location("workbench_server", SERVER)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["workbench_server"] = mod          # pydantic resolves the request models through it
        spec.loader.exec_module(mod)
        offline_after_import = os.environ["LLM_OFFLINE"]
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    tmp = tmp_path_factory.mktemp("workbench")
    mod.AUDIT, mod.JIRA = tmp / "audit.jsonl", tmp / "jira.json"
    mod.offline_after_import = offline_after_import
    from fastapi.testclient import TestClient
    return mod, TestClient(mod.app)


# ----------------------------------------------------------------------------- fast: who may see what

def test_the_workbench_runs_offline_even_when_env_says_live(wb):
    mod, _ = wb
    assert mod.offline_after_import == "1"


def test_an_analyst_sees_only_their_own_region(wb):
    _, c = wb
    apps = c.get("/api/queue", params={"user_id": LUCKY}).json()["applications"]
    assert apps and {a["region"] for a in apps} == {"West"}


def test_the_head_of_credit_and_the_auditor_see_every_region(wb):
    _, c = wb
    for uid in (AMAIRA, AUDITOR):
        regions = {a["region"] for a in c.get("/api/queue", params={"user_id": uid}).json()["applications"]}
        assert regions >= {"North", "South", "East", "West"}


def test_opening_another_regions_file_is_refused_and_audited(wb):
    mod, c = wb
    north = next(a["application_id"] for a in mod.QUEUE.values() if a["region"] == "North")
    r = c.get(f"/api/reviews/{north}", params={"user_id": LUCKY})
    assert r.status_code == 403
    assert any(e["action"] == "access_refused" and e["actor"] == LUCKY for e in mod.audit_events(north))


def test_an_unknown_user_is_refused(wb):
    _, c = wb
    assert c.get("/api/queue", params={"user_id": "U99"}).status_code == 401


def test_an_unknown_application_is_404(wb):
    _, c = wb
    assert c.get("/api/reviews/A999", params={"user_id": AMAIRA}).status_code == 404


# ----------------------------------------------------------------------------- fast: the platform screens

def test_platform_jobs_need_the_head_of_credit_or_the_auditor_and_a_refusal_is_audited(wb):
    mod, c = wb
    r = c.post("/api/platform/jobs/gate", json={"user_id": LUCKY, "option": ""})
    assert r.status_code == 403
    r = c.post("/api/platform/drill", json={"user_id": LUCKY})
    assert r.status_code == 403
    refused = [e for e in mod.audit_events("PLATFORM") if e["action"] == "platform_action_refused"]
    assert len(refused) >= 2 and all(e["actor"] == LUCKY for e in refused)


def test_statement_screening_is_restricted_and_finds_every_planted_signal(wb):
    _, c = wb
    assert c.get("/api/platform/screening", params={"user_id": LUCKY}).status_code == 403
    rows = {a["application_id"]: {f["code"] for f in a["flags"]}
            for a in c.get("/api/platform/screening", params={"user_id": AMAIRA}).json()}
    assert {"RULE_5_2", "ROUND_TRIP"} <= rows["A901"]
    assert "PASS_THROUGH" in rows["A902"] and "RULE_5_2" not in rows["A902"]
    assert {"RULE_4_1", "SEASONAL_PATTERN"} <= rows["A903"]
    assert "WINDOW_DRESSING" in rows["A904"] and "SECTOR_MISMATCH" in rows["A905"] and "FOOTING" in rows["A906"]


def test_the_go_live_checklist_is_computed_with_evidence_for_all_nine_conditions(wb):
    _, c = wb
    items = c.get("/api/platform/release", params={"user_id": AUDITOR, "recheck": True}).json()["checklist"]["items"]
    assert len(items) == 9 and all(i["evidence"] for i in items)
    by = {i["condition"]: i for i in items}
    assert by["Access by role and region"]["ok"] is True          # the North analyst's queue holds only North
    assert by["Guardrails in and out"]["ok"] is True              # one allowed, two blocked
    assert by["Runs outside a laptop"]["ok"] is True              # two stages, a non-root user, a health check


def test_connected_systems_label_every_stand_in(wb):
    _, c = wb
    systems = {s["name"]: s for s in c.get("/api/platform/systems", params={"user_id": AMAIRA}).json()["systems"]}
    assert {"Core banking", "Repayment history", "Exception tickets", "Approver notifications", "Model gateway"} <= set(systems)
    assert systems["Core banking"]["mode"] in {"Live", "Replica"}
    assert systems["Repayment history"]["mode"] in {"Live", "Estimated"}
    assert systems["Model gateway"]["mode"] == "Offline"


# ----------------------------------------------------------------------------- slow: one file, review to decision

@pytest.fixture(scope="module")
def rohan(wb, tmp_path_factory):
    """Load the agent in the foreground, then prepare Rohan's review once for the tests below."""
    mod, c = wb
    before = {p: p.read_bytes() for p in CACHE.glob("*.json*")} if CACHE.exists() else {}
    try:
        if not mod.STATE["ready"]:
            mod.load_agent()
    finally:                                    # the notebook's own demo cells write to data/cache: put it back
        for p in (CACHE.glob("*.json*") if CACHE.exists() else []):
            if p not in before:
                p.unlink()
        for p, data in before.items():
            p.write_bytes(data)
    assert mod.STATE["ready"], mod.STATE["error"]
    mod.STATE["ns"]["FEEDBACK"] = tmp_path_factory.mktemp("memory") / "underwriter_feedback.json"
    body = c.get("/api/reviews/A901/prepare", params={"user_id": AMAIRA}).text
    ready = [json.loads(chunk.split("data: ", 1)[1]) for chunk in body.split("\n\n") if chunk.startswith("event: ready")]
    assert ready, body[-500:]
    return ready[-1]


@pytest.mark.slow
def test_rohans_review_stops_at_the_gate_with_a_counter_offer(rohan):
    assert rohan["status"] == "awaiting_signature"
    assert rohan["note"]["proposal"] == "counter-offer"
    assert rohan["note"]["amount_inr_000"] == 15000          # ₹1.50 Cr: the room left under rule 5.2


@pytest.mark.slow
def test_an_analyst_cannot_submit_a_decision_and_the_attempt_is_audited(wb, rohan):
    mod, c = wb
    r = c.post("/api/reviews/A901/sign", json={"user_id": LUCKY, "decision": "approve", "amount_inr_000": 50000})
    assert r.status_code == 403
    assert mod.REVIEWS["A901"]["status"] == "awaiting_signature"     # the paused run was not touched
    assert any(e["action"] == "signature_refused" and e["actor"] == LUCKY for e in mod.audit_events("A901"))


@pytest.mark.slow
def test_raising_exceptions_twice_files_each_ticket_once(wb, rohan):
    _, c = wb
    first = c.post("/api/reviews/A901/tickets", json={"user_id": AMAIRA}).json()["tickets"]
    second = c.post("/api/reviews/A901/tickets", json={"user_id": AMAIRA}).json()["tickets"]
    assert first and all(t["created"] for t in first)
    assert [t["key"] for t in second] == [t["key"] for t in first]
    assert not any(t["created"] for t in second)


@pytest.mark.slow
def test_the_head_of_credit_decides_and_the_condition_is_learned(wb, rohan):
    mod, c = wb
    condition = {"text": "Personal guarantee from the promoter", "trigger": "always"}
    r = c.post("/api/reviews/A901/sign", json={"user_id": AMAIRA, "decision": "counter-offer",
                                               "amount_inr_000": 15000, "conditions": [condition]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "signed" and body["decision"]["decided_by"] == AMAIRA
    assert any(e["action"] == "signed" for e in body["audit"])
    learned = json.loads(mod.STATE["ns"]["FEEDBACK"].read_text())
    assert learned["always"][condition["text"]] == 1

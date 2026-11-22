"""Milestone 7: the Oracle adapter, the SOAP client with its circuit breaker, and the Slack dry run.
No Docker, no network: every adapter is exercised through its fallback, which is itself an adapter.
"""
from __future__ import annotations

import pandas as pd
import pytest

from creditlens.config import settings
from creditlens.integrations.oracle import COLUMN_MAP, CoreBankingAdapter
from creditlens.integrations.soap import CircuitBreaker, CircuitOpen, RepaymentClient, RepaymentHistory


@pytest.fixture(scope="module")
def existing_loans() -> pd.DataFrame:
    path = settings.paths.raw / "registry" / "existing_loans.csv"
    if not path.exists():
        pytest.skip("generate the data first: python -m creditlens.generators")
    return pd.read_csv(path)


def test_oracle_adapter_maps_legacy_columns_to_our_schema(existing_loans):
    cb = CoreBankingAdapter(dsn="")   # the stand-in on purpose, even when .env points at a running Oracle
    cb.ensure_sqlite(rebuild=True)
    assert cb.backend != "oracle"
    assert COLUMN_MAP["CUST_ID"] == "company_id"
    assert COLUMN_MAP["OUTSTANDING_AMT"] == "outstanding_inr_000"
    with_loan = existing_loans.iloc[0]
    got = cb.get_exposure(with_loan.company_id)
    assert got["company_id"] == with_loan.company_id
    assert got["outstanding_inr_000"] == int(with_loan.outstanding_inr_000)
    companies = pd.read_csv(settings.paths.raw / "registry" / "companies.csv").company_id
    without = next(c for c in companies if c not in set(existing_loans.company_id))
    assert cb.get_exposure(without)["outstanding_inr_000"] == 0


def test_circuit_breaker_opens_after_three_failures_and_half_opens_after_the_reset():
    now = [0.0]
    breaker = CircuitBreaker(failures=3, reset_after=30, clock=lambda: now[0])

    def boom():
        raise RuntimeError("legacy service down")

    for _ in range(3):
        with pytest.raises(RuntimeError):
            breaker.call(boom)
    assert breaker.state == "open"
    with pytest.raises(CircuitOpen):
        breaker.call(lambda: "never runs")
    now[0] = 31.0
    assert breaker.state == "half_open"
    assert breaker.call(lambda: "trial ok") == "trial ok"
    assert breaker.state == "closed"


def test_soap_client_falls_back_to_the_mock_when_no_service_answers():
    client = RepaymentClient(wsdl="http://127.0.0.1:9/?wsdl", timeout=0.3, retries=1, backoff=0.01)
    history = client.get_history("C012")
    assert isinstance(history, RepaymentHistory)
    assert history.source == "mock"
    assert history.customer_id == "C012"
    assert history.entries and all(e.amount > 0 for e in history.entries)
    assert client.get_history("C012").entries == history.entries  # deterministic per customer


def test_slack_blocks_render_without_slack_installed():
    from creditlens.integrations import slack
    from creditlens.schemas import CreditMemo

    memo = CreditMemo(application_id="A001", company_id="C001", applicant_summary="Rohan Packaging",
                      facility_requested="INR 20,000 thousand", financial_analysis="DSCR 1.4x",
                      banking_conduct="clean", related_parties="none", risk_assessment="low",
                      recommendation="refer", recommendation_reasoning="rule 6.1 breached")
    blocks = slack.build_blocks("thread-1", memo)
    text = str(blocks)
    assert "Approve" in text and "Reject" in text and "thread-1" in text


def test_slack_refuses_anyone_but_the_head_of_credit(monkeypatch, tmp_path):
    """A click in the approvals channel is not a signature: unmapped users and analysts are
    refused before the paused run is touched (policy 7.3), exactly like the CLI."""
    from creditlens.agents import run
    from creditlens.integrations import slack

    monkeypatch.setenv("CREDITLENS_AUDIT_PATH", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("SLACK_USER_MAP", "USLACKANALYST:U01,USLACKHEAD:U05")
    touched = []
    monkeypatch.setattr(run, "pending_application_id", lambda *a, **k: touched.append("pending") or None)
    monkeypatch.setattr(run, "resume", lambda *a, **k: touched.append("resume"))

    unmapped = slack.handle_decision("thread-1", "approve", "USTRANGER")
    analyst = slack.handle_decision("thread-1", "approve", "USLACKANALYST")
    assert unmapped["status"] == "refused" and "not mapped" in unmapped["error"]
    assert analyst["status"] == "refused" and "analyst" in analyst["error"]
    assert touched == []                                   # the paused run was never opened

    head = slack.handle_decision("thread-1", "approve", "USLACKHEAD")
    assert head["status"] == "error" and touched == ["pending"]   # allowed through to the lookup

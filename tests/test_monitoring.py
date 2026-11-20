"""Maintain: the drift report flags injected drift, and every incident drill runs with one command."""
from __future__ import annotations

import os

import pytest

from creditlens.config import settings

os.environ.setdefault("LLM_OFFLINE", "1")


@pytest.fixture(scope="module", autouse=True)
def needs_real_data():
    if not (settings.paths.real / "sba_loans.csv").exists():
        pytest.skip("fetch the real data first: python -m creditlens.data_real --only sba")


def test_drift_report_flags_the_injected_drift():
    from monitoring.drift import run

    quiet = run(inject_drift=False, write_html=False, sample=2000, verbose=False)
    drifted = run(inject_drift=True, write_html=False, sample=2000, verbose=False)
    assert drifted["n_drifted"] >= quiet["n_drifted"]
    assert "GrAppv" in drifted["drifted_columns"] or "Term" in drifted["drifted_columns"]


@pytest.mark.parametrize("scenario", ["data_drift", "cost_spike", "policy_update", "prompt_injection"])
def test_each_incident_runs_and_names_where_to_look(scenario):
    from monitoring.incidents import run_scenario

    report = run_scenario(scenario, verbose=False)
    assert report["scenario"] == scenario
    assert report["symptom"] and report["where_to_look"] and report["fix"]
    assert report["triggered"] is True

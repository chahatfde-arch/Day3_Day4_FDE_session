"""Milestone 11: the LLM judge is calibrated against human labels, trajectories are checked, and
the CI gate fails on a regressed prompt. All offline."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("LLM_OFFLINE", "1")

GOOD_MEMO = """# Credit memo: A001

## Applicant summary
Rohan Packaging Pvt Ltd, Manufacturing, North region.

## Facility requested
INR 20,000 thousand term loan over 36 months.

## Financial analysis
DSCR 1.10x against the policy minimum of 1.25x (rule 3.1).

## Banking conduct
One returned cheque in 12 months, within rule 4.1.

## Related parties and group exposure
No related companies found.

## Risk score
Probability of default 9 percent, medium band.

## Policy exceptions
- Rule 3.1: measured 1.10 against 1.25. Referred to the Head of Credit.

## Recommendation: refer
Rule 3.1 is breached, so the file is referred for a decision.
"""

BAD_MEMO = GOOD_MEMO.replace("## Recommendation: refer\nRule 3.1 is breached, so the file is referred for a decision.",
                             "## Recommendation: approve\nThe loan is approved.")


def test_judge_passes_a_faithful_memo_and_fails_approval_language():
    from evals.judge import judge

    good = judge(GOOD_MEMO, checks_json=[{"rule_id": "3.1", "passed": False}])
    bad = judge(BAD_MEMO, checks_json=[{"rule_id": "3.1", "passed": False}])
    assert good.verdict == "pass", good.reasons
    assert bad.verdict == "fail"
    assert bad.no_approval_language is False


def test_judge_calibration_agrees_with_humans_offline():
    from evals.judge import calibrate

    result = calibrate(verbose=False)
    assert result["n"] >= 8
    assert result["agreement"] >= 0.8
    regressed = calibrate(verbose=False, regressed=True)
    assert regressed["agreement"] < result["agreement"]


def test_trajectory_flags_forbidden_and_repeated_calls():
    from evals.trajectory import check_trajectory

    calls = [{"name": "get_application", "args": {"application_id": "A001"}, "order": 0},
             {"name": "check_rule_group", "args": {"rule_group": "financial"}, "order": 1},
             {"name": "memo_writer", "args": {}, "order": 2},
             {"name": "delete_everything", "args": {}, "order": 3},
             {"name": "get_application", "args": {"application_id": "A001"}, "order": 4}]
    report = check_trajectory(calls, {"forbidden_tools": ["delete_everything"],
                                      "required_tools": ["get_application"]})
    assert report["passed"] is False
    checks = {f["check"] for f in report["failures"]}
    assert "forbidden_tools" in checks and "no_repeated_calls" in checks


def test_gate_passes_by_default_and_fails_with_a_regressed_prompt():
    pytest.importorskip("rank_bm25")
    from evals.gate import run_gate

    ok = run_gate(limit=2, verbose=False)
    assert ok["passed"] is True, ok["failed"]
    bad = run_gate(limit=2, regressed=True, verbose=False)
    assert bad["passed"] is False
    assert "judge_agreement" in bad["failed"]

"""Session 3 on the story files: the unchanged agents draft Rohan's and Rahul's memos and stop for a person.

Everything runs offline in a temporary data folder, so the developer's data/story is never touched.
"""
from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("langgraph")

from creditlens.agents.memo import group_headroom  # noqa: E402
from creditlens.scenarios.story import build_story, decide, draft_story_memo, story_env  # noqa: E402


@pytest.fixture(scope="module")
def story(s):
    old = os.environ.get("LLM_OFFLINE")
    os.environ["LLM_OFFLINE"] = "1"
    from creditlens.gateway import llm

    llm._default = None
    root = build_story(s, verbose=False)
    yield root
    if old is None:
        os.environ.pop("LLM_OFFLINE", None)
    else:
        os.environ["LLM_OFFLINE"] = old
    llm._default = None


def _failed(memo):
    return {c.rule_id for c in memo.checks if not c.passed}


def test_on_paper_both_pass_the_financial_rules(story):
    for app in ("A901", "A902"):
        memo = draft_story_memo(app, "U05", root=story)["memo"]
        assert not ({"3.1", "3.2", "3.3"} & _failed(memo)), app   # the documents look healthy


def test_rohan_is_referred_with_one_ticket_and_a_counter_offer(story):
    result = draft_story_memo("A901", "U05", root=story)
    memo = result["memo"]
    assert result["status"] == "awaiting_approval" and memo.recommendation == "refer"
    assert _failed(memo) == {"5.2"}
    assert [e.rule_id for e in memo.policy_exceptions] == ["5.2"] and memo.policy_exceptions[0].jira_key
    assert "15,000 thousand" in memo.recommendation_reasoning
    assert [m.company_id for m in memo.group.members] == ["SC01", "SC02", "SC03", "SC04"]


def test_running_rohan_twice_still_files_one_ticket(story):
    draft_story_memo("A901", "U05", root=story)
    draft_story_memo("A901", "U05", root=story)
    issues = json.loads((story / "cache" / "jira.json").read_text())["issues"]
    assert [i["idempotency_key"] for i in issues].count("A901:5.2") == 1


def test_rahul_is_recommended_for_approval_with_his_group_listed(story):
    memo = draft_story_memo("A902", "U05", root=story)["memo"]
    assert memo.recommendation == "approve" and not memo.policy_exceptions
    assert [m.company_id for m in memo.group.members] == ["SC10", "SC11"]    # rule 5.3: listed even within the limit


def test_only_the_head_of_credit_decides_and_a_counter_offer_is_recorded(story):
    thread = draft_story_memo("A901", "U05", root=story)["thread_id"]
    with pytest.raises(PermissionError):
        decide(thread, "approve", user="U04", root=story)
    result = decide(thread, "approve", user="U05", comment="counter-offer", approved_inr_000=15000, root=story)
    assert result["status"] == "approved" and result["decision"]["approved_inr_000"] == 15000
    from creditlens.agents import adapters

    with story_env(story):
        assert "approved_inr_000=15000" in adapters.audit_events("A901")[-1].detail


def test_headroom_is_limit_minus_what_the_rest_of_the_group_uses():
    facts = {"company_id": "SC01", "group": {"group_id": "G", "limit_inr_000": 120000, "members": [
        {"company_id": "SC01", "existing_outstanding_inr_000": 0, "pending_request_inr_000": 50000},
        {"company_id": "SC02", "existing_outstanding_inr_000": 40000, "pending_request_inr_000": 0},
        {"company_id": "SC03", "existing_outstanding_inr_000": 30000, "pending_request_inr_000": 0},
        {"company_id": "SC04", "existing_outstanding_inr_000": 35000, "pending_request_inr_000": 0}]}}
    assert group_headroom(facts) == 15000
    assert group_headroom({"company_id": "X", "group": {"group_id": None, "members": []}}) is None


def test_the_story_world_never_touches_the_default_dataset(story, s):
    assert story == s.paths.data / "story" and (story / "story.db").exists()
    assert not (s.paths.raw / "financials" / "SC01_financials.pdf").exists()

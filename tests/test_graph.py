"""Graph tests: the related-party groups the registry CSVs never state out loud.

The generator hides a handful of groups in the data (a shared director here, a shared
registered address there, sometimes two hops apart) and writes the truth to
`data/answer_key/related_party_groups.json`. These tests are the Day 3 "did we find them?"
check, and they run on NetworkX, so no Neo4j and no Docker are needed.
"""
from __future__ import annotations

import json

import pytest

from creditlens.generators.policy import RULES
from creditlens.retrieval.graph import CYPHER_RELATED_GROUP, GraphStore, cypher_cheatsheet
from creditlens.schemas import GroupExposure, GroupMember


@pytest.fixture(scope="module")
def store(s) -> GraphStore:
    return GraphStore().load_from_registry(s.paths.raw)


@pytest.fixture(scope="module")
def answer_key(s) -> list[dict]:
    return json.loads((s.paths.answer_key / "related_party_groups.json").read_text())


@pytest.fixture(scope="module")
def expected_flags(s) -> dict:
    return json.loads((s.paths.answer_key / "expected_policy_flags.json").read_text())


def test_graph_has_the_three_node_kinds(store):
    kinds = {data["kind"] for _, data in store.graph.nodes(data=True)}
    assert kinds == {"Company", "Director", "Address"}
    edge_kinds = {data["kind"] for _, _, data in store.graph.edges(data=True)}
    assert edge_kinds == {"DIRECTOR_OF", "REGISTERED_AT"}


def test_all_groups_matches_the_answer_key(store, answer_key):
    found = sorted(sorted(group) for group in store.all_groups())
    expected = sorted(sorted(group["company_ids"]) for group in answer_key)
    assert found == expected


def test_related_group_lists_every_member_with_a_reason(store, answer_key):
    for group in answer_key:
        for company_id in group["company_ids"]:
            members = store.related_group(company_id)
            assert all(isinstance(m, GroupMember) for m in members)
            assert {m.company_id for m in members} == set(group["company_ids"])
            # The applicant is always first, at zero hops, linked to itself.
            assert members[0].company_id == company_id
            assert members[0].link == "self" and members[0].hops == 0
            others = members[1:]
            assert all(m.link in {"shared_director", "shared_address"} for m in others)
            assert all(m.hops >= 1 for m in others)
            assert all(m.legal_name for m in members)


def test_a_company_with_no_related_parties_is_a_group_of_one(store, answer_key):
    grouped = {cid for group in answer_key for cid in group["company_ids"]}
    lonely = next(cid for cid in store.companies if cid not in grouped)
    members = store.related_group(lonely)
    assert [m.company_id for m in members] == [lonely]
    assert store.group_exposure(lonely).group_id is None


def test_group_exposure_sums_outstanding_plus_pending(store, answer_key, s):
    import pandas as pd

    outstanding = pd.read_csv(s.paths.raw / "registry" / "existing_loans.csv") \
        .set_index("company_id")["outstanding_inr_000"].to_dict()
    requested = pd.read_csv(s.paths.raw / "applications" / "applications.csv") \
        .groupby("company_id")["requested_inr_000"].sum().to_dict()

    group = answer_key[0]
    exposure = store.group_exposure(group["company_ids"][0])
    assert isinstance(exposure, GroupExposure)
    assert {m.company_id for m in exposure.members} == set(group["company_ids"])
    assert exposure.limit_inr_000 == RULES["MAX_GROUP_EXPOSURE_INR_000"]
    assert exposure.total_exposure_inr_000 == sum(
        int(outstanding.get(cid, 0)) + int(requested.get(cid, 0)) for cid in group["company_ids"])


def test_breaches_limit_agrees_with_the_expected_policy_flags(store, expected_flags):
    """Rule 5.2 in the answer key and `breaches_limit` must never disagree."""
    for application_id, row in expected_flags.items():
        exposure = store.group_exposure(row["company_id"])
        assert exposure.breaches_limit == ("5.2" in row["flags"]), \
            f"{application_id} ({row['company_id']}): exposure {exposure.total_exposure_inr_000}"


def test_requested_override_changes_the_exposure(store, answer_key):
    group = answer_key[0]["company_ids"]
    base = store.group_exposure(group[0])
    bumped = store.group_exposure(group[0], requested={group[0]: 999_999})
    assert bumped.total_exposure_inr_000 > base.total_exposure_inr_000
    assert bumped.breaches_limit is True


def test_cypher_is_always_recorded(store, answer_key):
    exposure = store.group_exposure(answer_key[0]["company_ids"][0])
    assert exposure.cypher.strip(), "the audit trail needs the query even when NetworkX ran"
    assert "MATCH" in exposure.cypher
    assert "DIRECTOR_OF" in exposure.cypher and "REGISTERED_AT" in exposure.cypher
    assert CYPHER_RELATED_GROUP in exposure.cypher


def test_cypher_cheatsheet_covers_the_queries_used():
    sheet = cypher_cheatsheet()
    assert {"load_companies", "load_directors", "related_group", "group_exposure",
            "all_groups", "constraints", "profile"} <= set(sheet)
    assert all(query.strip() for query in sheet.values())
    assert "PROFILE" in sheet["profile"]
    assert "CREATE CONSTRAINT" in sheet["constraints"]


def test_profile_times_the_traversal_and_explains_neo4j(store, answer_key):
    report = store.profile(answer_key[0]["company_ids"][0], repeats=3)
    assert report["backend"] == "networkx"
    assert report["avg_ms"] >= 0
    assert report["nodes"] > report["companies"]  # directors and addresses are nodes too
    assert any("PROFILE" in note for note in report["neo4j_notes"])
    assert any("CONSTRAINT" in note for note in report["neo4j_notes"])


def test_unknown_company_is_a_clear_error(store):
    with pytest.raises(KeyError):
        store.related_group("C999")


def test_querying_before_loading_is_a_clear_error():
    with pytest.raises(RuntimeError):
        GraphStore().all_groups()

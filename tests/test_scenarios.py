"""The story scenarios: Rohan's linked companies and Rahul's family cash. Every planted fact must be found."""
import json

import pytest

from creditlens.config import Paths
from creditlens.scenarios import case_file, generate_scenarios
from creditlens.scenarios.detect import load_statement, round_trips


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    return generate_scenarios(out_dir=tmp_path_factory.mktemp("scenarios"))


def codes(cf):
    return {f["code"] for f in cf["flags"]}


def test_rohan_group_breaks_rule_5_2_though_he_alone_does_not(pack):
    cf = case_file(pack, "A901")
    group = next(f for f in cf["flags"] if f["code"] == "GROUP")["evidence"]
    assert group["members"] == ["SC01", "SC02", "SC03", "SC04"]            # one hop by director, one by address
    rule = next(f for f in cf["flags"] if f["code"] == "RULE_5_2")["evidence"]
    assert rule["total"] == 155000 > rule["limit"] and rule["alone"] == 50000 < rule["limit"]


def test_rohan_round_trips_and_related_inflow_are_found(pack):
    cf = case_file(pack, "A901")
    assert {"ROUND_TRIP", "RELATED_INFLOW"} <= codes(cf)
    truth = json.loads((pack / "answer_key" / "scenarios.json").read_text())["A901"]["round_trips"]
    assert len(round_trips(load_statement(pack / "statements" / "A901_statement.csv"))) == truth == 24


def test_rahul_is_linked_but_within_the_limit_and_the_flags_are_leads(pack):
    cf = case_file(pack, "A902")
    assert codes(cf) == {"GROUP", "PASS_THROUGH", "REVENUE_GAP"}           # no RULE_5_2: the group is within the limit
    assert "Not a finding" in cf["note"]
    gap = next(f for f in cf["flags"] if f["code"] == "REVENUE_GAP")["evidence"]
    assert 0.55 < gap["visible_over_declared"] < 0.65


def test_the_pack_never_touches_the_default_dataset(pack):
    assert pack != Paths().raw and "raw" not in pack.parts[-1]
    assert Paths().data / "scenarios" != Paths().raw


def test_meena_trips_rule_4_1_but_the_season_explains_it(pack):
    cf = case_file(pack, "A903")
    assert codes(cf) == {"RULE_4_1", "SEASONAL_PATTERN"}
    seasonal = next(f for f in cf["flags"] if f["code"] == "SEASONAL_PATTERN")["evidence"]
    assert seasonal["correlation"] > 0.95 and "2026-07" in seasonal["weakest_months"]


def test_window_dressing_sector_mismatch_and_doctored_financials(pack):
    assert codes(case_file(pack, "A904")) == {"WINDOW_DRESSING"}
    sector = next(f for f in case_file(pack, "A905")["flags"] if f["code"] == "SECTOR_MISMATCH")["evidence"]
    assert sector["declared_sector"] == "Healthcare" and sector["share_of_inflow"] > 0.6
    cf = case_file(pack, "A906")
    assert codes(cf) == {"REVENUE_GAP", "FOOTING"}
    bad = next(f for f in cf["flags"] if f["code"] == "FOOTING")["evidence"]
    assert [(b["fiscal_year"], b["line"]) for b in bad] == [("FY2026", "ebitda")]


def test_bonus_cases_do_not_change_rohan_and_rahul(pack):
    assert codes(case_file(pack, "A901")) == {"GROUP", "RULE_5_2", "ROUND_TRIP", "RELATED_INFLOW"}
    assert codes(case_file(pack, "A902")) == {"GROUP", "PASS_THROUGH", "REVENUE_GAP"}

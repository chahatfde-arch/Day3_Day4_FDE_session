"""Detectors for the story scenarios. They reuse the pipeline and the graph, and they only raise leads.

A flag says "a person should look at this", never "this person did something wrong". There can be an
honest explanation for each one, and the bank's policy keeps the decision with a human underwriter.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from creditlens.pipeline.clean import clean_statement
from creditlens.pipeline.wrangle import normalize_counterparty
from creditlens.scenarios.generate import WINDOW


def load_statement(path: Path) -> pd.DataFrame:
    """Run the Session 1 cleaning on a scenario statement: one row per transaction."""
    clean, _ = clean_statement(Path(path), WINDOW)
    return clean


def top_counterparties(clean: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Share of all money in (credits) by counterparty, after names are normalised."""
    cr = clean[clean["amount"] > 0]
    by = cr.groupby("counterparty")["amount"].sum().sort_values(ascending=False)
    return (by / by.sum()).head(n).rename("share_of_inflow").reset_index()


def round_trips(clean: pd.DataFrame, window_days: int = 5, tolerance: float = 0.05) -> pd.DataFrame:
    """Money out to a party and back from the same party within a few days, for about the same amount."""
    rows = []
    debits = clean[(clean["amount"] < 0) & ~clean["channel"].isin(["CASH", "CHARGES", "ACH"])]
    credits = clean[clean["amount"] > 0]
    used: set[int] = set()
    for _, d in debits.iterrows():
        cand = credits[(credits["counterparty"] == d["counterparty"]) & (~credits["seq"].isin(used))
                       & (credits["txn_date"] >= d["txn_date"])
                       & (credits["txn_date"] <= d["txn_date"] + pd.Timedelta(days=window_days))]
        for _, c in cand.iterrows():
            if abs(c["amount"] + d["amount"]) <= tolerance * abs(d["amount"]):
                used.add(int(c["seq"]))
                rows.append({"counterparty": d["counterparty"], "out_date": d["txn_date"].date(), "out_amount": -d["amount"],
                             "back_date": c["txn_date"].date(), "back_amount": c["amount"]})
                break
    return pd.DataFrame(rows, columns=["counterparty", "out_date", "out_amount", "back_date", "back_amount"])


def related_flows(clean: pd.DataFrame, member_names: set[str]) -> dict:
    """How much of the money in and out is with companies the graph says are in the applicant's group."""
    names = {normalize_counterparty(n) for n in member_names}
    cr, dr = clean[clean["amount"] > 0], clean[clean["amount"] < 0]
    inflow = cr[cr["counterparty"].isin(names)]["amount"].sum()
    outflow = -dr[dr["counterparty"].isin(names)]["amount"].sum()
    return {"inflow_share": float(inflow / cr["amount"].sum()), "outflow_share": float(outflow / -dr["amount"].sum()),
            "inflow": float(inflow), "outflow": float(outflow)}


def pass_through(clean: pd.DataFrame, person_names: set[str], window_days: int = 2) -> pd.DataFrame:
    """Cash arrives, then the same amount goes to a named person the registry ties to the group."""
    people = {normalize_counterparty(n) for n in person_names}
    cash = clean[(clean["channel"] == "CASH") & (clean["amount"] > 0)]
    sent = clean[(clean["amount"] < 0) & clean["counterparty"].isin(people)]
    rows = []
    for _, c in cash.iterrows():
        m = sent[(sent["txn_date"] >= c["txn_date"]) & (sent["txn_date"] <= c["txn_date"] + pd.Timedelta(days=window_days))
                 & ((-sent["amount"] - c["amount"]).abs() <= 0.01 * c["amount"])]
        if len(m):
            rows.append({"cash_date": c["txn_date"].date(), "amount": c["amount"], "sent_to": m.iloc[0]["counterparty"]})
    return pd.DataFrame(rows, columns=["cash_date", "amount", "sent_to"])


def revenue_gap(clean: pd.DataFrame, declared_revenue_inr_000: float) -> dict:
    """Money that reached this account compared with the turnover the borrower declared."""
    visible = float(clean[clean["amount"] > 0]["amount"].sum())
    declared = declared_revenue_inr_000 * 1000.0
    return {"visible_inflow": visible, "declared_revenue": declared, "visible_over_declared": visible / declared}


BETTING_WORDS = ("BETTING", "BETS", "GAMING", "CASINO", "LOTTERY")


def monthly_inflow(clean: pd.DataFrame) -> pd.Series:
    """Money in, per calendar month (index like '2025-10')."""
    cr = clean[clean["amount"] > 0]
    return cr.groupby(cr["txn_date"].dt.strftime("%Y-%m"))["amount"].sum()


def seasonality(clean: pd.DataFrame, declared_monthly: pd.DataFrame) -> dict:
    """Does the shape of the money in match the shape the borrower declared, month by month?"""
    actual = monthly_inflow(clean)
    declared = declared_monthly.set_index("month")["declared_inr"].reindex(actual.index)
    return {"correlation": float(actual.corr(declared)), "weakest_months": list(actual.nsmallest(3).index)}


def window_dressing(clean: pd.DataFrame, days: int = 30) -> dict:
    """How much of the whole year's inflow arrived in the last few weeks before the statement ends."""
    cr = clean[clean["amount"] > 0]
    cutoff = clean["txn_date"].max() - pd.Timedelta(days=days)
    recent = cr[cr["txn_date"] > cutoff]["amount"].sum()
    return {"last_days": days, "share_of_year": float(recent / cr["amount"].sum()), "recent_inflow": float(recent)}


def sector_mismatch(clean: pd.DataFrame, declared_sector: str, words: tuple = BETTING_WORDS) -> dict:
    """Money in from parties whose names say betting, against the sector the borrower declared."""
    cr = clean[clean["amount"] > 0]
    hit = cr[cr["counterparty"].apply(lambda n: any(w in n for w in words))]
    return {"declared_sector": declared_sector, "share_of_inflow": float(hit["amount"].sum() / cr["amount"].sum()),
            "parties": sorted(set(hit["counterparty"]))}


def footing_check(fin: pd.DataFrame, tolerance: float = 0.01) -> pd.DataFrame:
    """Do the printed profit lines add up? EBITDA = revenue - cogs - opex, PAT = EBITDA - depreciation - interest - tax."""
    rows = []
    for _, r in fin.iterrows():
        for line, expected in (("ebitda", r["revenue"] - r["cogs"] - r["opex"]),
                               ("pat", r["ebitda"] - r["depreciation"] - r["interest"] - r["tax"])):
            if abs(r[line] - expected) > tolerance * abs(expected):
                rows.append({"fiscal_year": r["fiscal_year"], "line": line, "printed": float(r[line]), "expected": float(expected)})
    return pd.DataFrame(rows, columns=["fiscal_year", "line", "printed", "expected"])


def case_file(raw_dir: Path, application_id: str) -> dict:
    """Everything a person needs to review one application, as flags with their evidence."""
    from creditlens.retrieval.graph import GraphStore  # Session 2 code; imported here so Session 1 needs only pandas

    raw_dir = Path(raw_dir)
    apps = pd.read_csv(raw_dir / "applications" / "applications.csv")
    app = apps[apps["application_id"] == application_id].iloc[0]
    company = app["company_id"]
    graph = GraphStore(backend="networkx").load_from_registry(raw_dir)
    exposure = graph.group_exposure(company)
    members = {m.company_id: m for m in exposure.members}
    names = {graph.companies[c]["legal_name"] for c in members if c != company}
    directors = pd.read_csv(raw_dir / "registry" / "directors.csv")
    links = pd.read_csv(raw_dir / "registry" / "company_directors.csv")
    in_group = links[links["company_id"].isin(members)]
    people = set(directors[directors["director_id"].isin(in_group["director_id"])]["full_name"])
    clean = load_statement(raw_dir / "statements" / f"{application_id}_statement.csv")

    flags = []
    if len(members) > 1:
        flags.append({"code": "GROUP", "title": f"Linked to {len(members) - 1} other companies",
                      "evidence": {"members": sorted(members), "total_exposure_inr_000": exposure.total_exposure_inr_000,
                                   "limit_inr_000": exposure.limit_inr_000}})
    if exposure.breaches_limit:
        flags.append({"code": "RULE_5_2", "title": "Group exposure is over the policy 5.2 limit",
                      "evidence": {"total": exposure.total_exposure_inr_000, "limit": exposure.limit_inr_000,
                                   "alone": graph.outstanding.get(company, 0) + graph.pending.get(company, 0)}})
    rt = round_trips(clean)
    if len(rt) >= 3:
        flags.append({"code": "ROUND_TRIP", "title": "Money goes out and comes back within days",
                      "evidence": {"count": len(rt), "parties": sorted(set(rt["counterparty"])), "amount": float(rt["out_amount"].sum())}})
    rel = related_flows(clean, names)
    if rel["inflow_share"] > 0.25:
        flags.append({"code": "RELATED_INFLOW", "title": "Much of the money in comes from group companies", "evidence": rel})
    pt = pass_through(clean, people)
    if len(pt) >= 3:
        flags.append({"code": "PASS_THROUGH", "title": "Cash in, then the same amount to a named person tied to the group",
                      "evidence": {"count": len(pt), "to": sorted(set(pt["sent_to"])), "amount": float(pt["amount"].sum())}})
    gap = revenue_gap(clean, float(app["declared_revenue_inr_000"]))
    if gap["visible_over_declared"] < 0.75:
        flags.append({"code": "REVENUE_GAP", "title": "The statement shows much less than the declared turnover", "evidence": gap})
    bounces = int(clean["is_bounce"].sum())
    if bounces > 2:
        flags.append({"code": "RULE_4_1", "title": "More than two cheques returned", "evidence": {"returned": bounces, "limit": 2}})
    dm_path = raw_dir / "applications" / "declared_monthly.csv"
    if dm_path.exists():
        dm = pd.read_csv(dm_path)
        dm = dm[dm["application_id"] == application_id]
        if len(dm):
            s = seasonality(clean, dm)
            if s["correlation"] >= 0.8 and bounces:
                flags.append({"code": "SEASONAL_PATTERN", "title": "The weak months match the seasonal shape the borrower declared",
                              "evidence": {**s, "returned": bounces}})
    wd = window_dressing(clean)
    if wd["share_of_year"] > 0.5:
        flags.append({"code": "WINDOW_DRESSING", "title": "Over half of the year's money in arrived in the last 30 days", "evidence": wd})
    sm = sector_mismatch(clean, str(graph.companies[company]["sector"]))
    if sm["share_of_inflow"] > 0.25:
        flags.append({"code": "SECTOR_MISMATCH", "title": "Much of the money in comes from betting-related names", "evidence": sm})
    fin_path = raw_dir / "financials" / "financials.csv"
    if fin_path.exists():
        fin = pd.read_csv(fin_path)
        fin = fin[fin["application_id"] == application_id]
        bad = footing_check(fin) if len(fin) else pd.DataFrame()
        if len(bad):
            flags.append({"code": "FOOTING", "title": "The printed profit lines do not add up", "evidence": bad.to_dict("records")})
    return {"application_id": application_id, "company_id": company, "flags": flags,
            "note": "Leads for a human to check. Not a finding, and not a decision."}

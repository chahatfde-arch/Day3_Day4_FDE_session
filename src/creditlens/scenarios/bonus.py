"""Four bonus cases, added to the same pack after Rohan and Rahul. They use their own random stream, so
Rohan's and Rahul's files are identical with or without them.

* A903 **Meena**, an honest baker. Her cash dries up every monsoon and three cheques bounce. The policy
  flags it, but her declared monthly turnover has the same shape, so the weak months are explained.
* A904 **Tapti Trading**, window dressing. A quiet account that fills up in the last weeks before the application.
* A905 **Sagar Wellness**, wrong sector. Declared healthcare; most of the money in comes from betting sites.
* A906 **Konark Exports**, doctored numbers. Financials whose profit lines do not add up, and a statement that
  shows about half of the revenue claimed.

Every person and company is invented.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd

from creditlens.generators.common import rng_for
from creditlens.scenarios import generate as g

BONUS_COMPANIES = [
    ("SC20", "Meena Bakery Pvt Ltd", "Food Processing", "2014-05-09", "9, Bakers Lane", "Nashik"),
    ("SC30", "Tapti Trading Pvt Ltd", "Trading", "2017-11-30", "58, Cloth Market", "Surat"),
    ("SC40", "Sagar Wellness Pvt Ltd", "Healthcare", "2020-03-14", "12, Lake View Road", "Pune"),
    ("SC50", "Konark Exports Pvt Ltd", "Trading", "2012-07-21", "30, Port Road", "Mumbai"),
]
BONUS_DIRECTORS = [("SD20", "Meena Kulkarni"), ("SD30", "Vivek Rao"), ("SD40", "Sagar Nair"), ("SD50", "Anil Kapoor")]
BONUS_LINKS = [("SC20", "SD20", "Managing Director"), ("SC30", "SD30", "Managing Director"),
               ("SC40", "SD40", "Managing Director"), ("SC50", "SD50", "Managing Director")]
BONUS_EXISTING = {"SC20": 5_000, "SC30": 0, "SC40": 8_000, "SC50": 20_000}
BONUS_APPS = [("A903", "SC20", 15_000), ("A904", "SC30", 25_000), ("A905", "SC40", 20_000), ("A906", "SC50", 40_000)]
BETTING_NAMES = ["LUCKY STRIKE BETTING", "ROYAL FLUSH GAMING", "SIX SIXES BETS"]
# Monthly strength of Meena's trade, October to September: festive highs, a monsoon trough.
MEENA_SEASON = [1.5, 1.7, 1.6, 1.2, 1.0, 1.1, 1.2, 0.9, 0.5, 0.3, 0.35, 0.7]


def _credits(frame: pd.DataFrame) -> float:
    amt = frame["Amount"].str.replace(",", "").astype(float)
    return float(amt[frame["Dr/Cr"] == "Cr"].sum())


def _meena(rng, out: Path):
    rows, monthly, bounces = [], [], 0
    for k, (y, m) in enumerate(g._months()):
        f = MEENA_SEASON[k]
        month_rows = g._base_month(rng, y, m, ["MERU TRADERS", "ARAVALI RETAIL", "KONARK EXPORTS"], f, out_scale=0.8)
        rows += month_rows
        monthly.append((f"{y}-{m:02d}", sum(r[2] for r in month_rows if r[3] == "Cr")))
        if f <= 0.5:  # the lean months: the occasional cheque comes back
            for _ in range(1 if f < 0.4 else 0):
                rows.append((g._day(y, m, int(rng.integers(12, 26))), "CHQ RTN ARAVALI RETAIL INSUFFICIENT FUNDS", round(float(rng.uniform(0.8, 1.4)) * 100_000, 2), "Dr"))
                rows.append((g._day(y, m, 27), "CHGS CHQ RTN", 590.0, "Dr"))
                bounces += 1
    # one more bounce in the first lean month so the count crosses policy rule 4.1 (more than two)
    y, m = list(g._months())[8]
    rows.append((g._day(y, m, 20), "CHQ RTN MERU TRADERS INSUFFICIENT FUNDS", round(float(rng.uniform(0.8, 1.4)) * 100_000, 2), "Dr"))
    rows.append((g._day(y, m, 21), "CHGS CHQ RTN", 590.0, "Dr")); bounces += 1
    # what she declared each month (as if from GST returns): same seasonal shape, a little noisier
    declared = pd.DataFrame([{"application_id": "A903", "month": mo, "declared_inr": round(v * float(rng.uniform(0.95, 1.05)), 2)} for mo, v in monthly])
    return rows, declared, bounces


def _tapti(rng):
    rows = []
    for k, (y, m) in enumerate(g._months()):
        rows += g._base_month(rng, y, m, ["HAMPI RETAIL", "GODAVARI TEXTILES"], 0.22 if k < 11 else 0.4, out_scale=0.22)
        if k == 11:  # the last weeks before the application: money arrives from everywhere
            for i, cp in enumerate(["SAHYADRI TRADERS", "VINDHYA EXPORTS", "NARMADA FOODS", "GANGA LOGISTICS"]):
                rows.append((g._day(y, m, 3 + i * 5), g._narr(rng, "RTGS", cp), round(float(rng.uniform(18, 26)) * 100_000, 2), "Cr"))
            rows.append((g._day(y, m, 14), "CASH DEPOSIT", round(float(rng.uniform(9, 12)) * 100_000, 2), "Cr"))
    return rows


def _sagar(rng):
    rows = []
    for y, m in g._months():
        rows += g._base_month(rng, y, m, ["ARAVALI RETAIL", "TAPTI FOODS"], 0.5, out_scale=0.5)
        for i in range(4):
            rows.append((g._day(y, m, 2 + i * 6), g._narr(rng, "RTGS", g._pick(rng, BETTING_NAMES)), round(float(rng.uniform(3, 6)) * 100_000, 2), "Cr"))
    return rows


def _konark(rng):
    rows = []
    for y, m in g._months():
        rows += g._base_month(rng, y, m, ["INDUS EXPORTS", "MALABAR TRADERS", "NILGIRI RETAIL"], 1.0)
    return rows


def _financials(app: str, visible_inflow_inr: float) -> pd.DataFrame:
    """Three years in INR thousand. FY2026 EBITDA is printed about 40% above what revenue minus costs gives."""
    revenue26 = round(visible_inflow_inr / 1000 / 0.55)
    rows = []
    for fy, rev in (("FY2024", round(revenue26 * 0.82)), ("FY2025", round(revenue26 * 0.91)), ("FY2026", revenue26)):
        cogs, opex = round(rev * 0.62), round(rev * 0.24)
        true_ebitda = rev - cogs - opex
        ebitda = round(true_ebitda * 1.4) if fy == "FY2026" else true_ebitda
        dep, interest = round(rev * 0.03), round(rev * 0.025)
        tax = round(max(ebitda - dep - interest, 0) * 0.25)
        rows.append({"application_id": app, "fiscal_year": fy, "revenue": rev, "cogs": cogs, "opex": opex, "ebitda": ebitda,
                     "depreciation": dep, "interest": interest, "tax": tax, "pat": ebitda - dep - interest - tax})
    return pd.DataFrame(rows)


def _append(path: Path, new: pd.DataFrame) -> None:
    old = pd.read_csv(path)
    pd.concat([old, new], ignore_index=True).to_csv(path, index=False)


def add_bonus(out: Path, seed: int) -> dict:
    """Append the four bonus cases to a pack written by `generate_scenarios`. Returns their answer-key entries."""
    rng = rng_for(seed, "scenarios_bonus")
    _append(out / "registry" / "companies.csv", pd.DataFrame([{
        "company_id": c, "legal_name": n, "sector": sec, "incorporated_on": inc, "registered_address": addr,
        "city": city, "region": "West", "employees": 40} for c, n, sec, inc, addr, city in BONUS_COMPANIES]))
    _append(out / "registry" / "directors.csv", pd.DataFrame([{
        "director_id": d, "full_name": n, "pan": f"SYNTH{d[2:]}Z", "phone": "+91 00000 00000",
        "email": f"{n.split()[0].lower()}@example.invalid", "residential_address": "Invented address"} for d, n in BONUS_DIRECTORS]))
    _append(out / "registry" / "company_directors.csv", pd.DataFrame([{
        "company_id": c, "director_id": d, "role": r, "appointed_on": "2020-01-01"} for c, d, r in BONUS_LINKS]))
    _append(out / "registry" / "existing_loans.csv", pd.DataFrame([{"company_id": c, "outstanding_inr_000": v} for c, v in BONUS_EXISTING.items()]))

    meena_rows, declared_monthly, bounces = _meena(rng, out)
    stmts = {"A903": meena_rows, "A904": _tapti(rng), "A905": _sagar(rng), "A906": _konark(rng)}
    frames = {a: g._write_statement(out / "statements" / f"{a}_statement.csv", rows) for a, rows in stmts.items()}
    (out / "financials").mkdir(exist_ok=True)
    fin = _financials("A906", _credits(frames["A906"]))
    fin.to_csv(out / "financials" / "financials.csv", index=False)
    declared_monthly.to_csv(out / "applications" / "declared_monthly.csv", index=False)

    declared = {"A903": round(_credits(frames["A903"]) / 1000 * 1.0), "A904": round(_credits(frames["A904"]) / 1000 * 0.95),
                "A905": round(_credits(frames["A905"]) / 1000 * 0.95), "A906": int(fin.iloc[-1]["revenue"])}
    _append(out / "applications" / "applications.csv", pd.DataFrame([{
        "application_id": a, "company_id": c, "requested_inr_000": r, "declared_revenue_inr_000": declared[a]} for a, c, r in BONUS_APPS]))
    return {
        "A903": {"company": "SC20", "story": "honest seasonal baker", "bounces": bounces, "breaches_rule_4_1": True, "explained_by_season": True},
        "A904": {"company": "SC30", "story": "window dressing", "last_30_days_share_min": 0.5},
        "A905": {"company": "SC40", "story": "wrong sector", "declared_sector": "Healthcare", "betting_parties": BETTING_NAMES},
        "A906": {"company": "SC50", "story": "doctored financials", "failing_year": "FY2026", "failing_line": "ebitda"},
    }

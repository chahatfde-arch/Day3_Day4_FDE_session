# %% [markdown]
# # CreditLens AI: the story scenarios, one case at a time
#
# Six borrowers land on Priya's desk. Each one teaches a different lesson about **reading numbers carefully**:
#
# | Case | Who | What it teaches |
# |---|---|---|
# | A901 | **Rohan**, looks safe | A web of linked companies and round-tripped money |
# | A902 | **Rahul**, looks weak | Family cash that explains a thin statement |
# | A903 | **Meena**, honest baker | A rule fires, and the season explains it |
# | A904 | **Tapti Trading** | Window dressing before an application |
# | A905 | **Sagar Wellness** | Declared healthcare, money from betting |
# | A906 | **Konark Exports** | Profit lines that do not add up |
#
# Every person and company is invented. Every section below runs on its own after the setup, so show them in any order.
#
# **One rule for the whole notebook:** a flag is a *lead for a person*, not a verdict. Nothing here says anyone did
# something wrong, and the bank's policy still forbids automated approvals.

# %% [markdown]
# ## 0. Setup (run once, about 2 minutes)
# This clones the pre-work repo, installs it, adds the scenario code and writes the data. Nothing needs an API key.

# %%
import sys
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    !git clone -q https://github.com/chahatfde-arch/Day3_Day4_FDE_session.git creditlens-day1
    %cd creditlens-day1
    !pip install -q -e .
    !mkdir -p src/creditlens/scenarios src/creditlens/retrieval
else:
    print("Not on Colab: skipping the clone. This repository already has the scenario code.")

# %% [markdown]
# **Laptop users:** run every cell. The clone above is skipped, and the next cell finds the repository root, so the notebook
# works whether you opened it from the repo folder or from `notebooks/`. The six cells after it show the scenario code; on a laptop
# they write their copy to a scratch folder, so the repository's own, tested code is never overwritten.

# %%
import os
from pathlib import Path
while not Path("pyproject.toml").exists() and Path.cwd() != Path.cwd().parent:
    os.chdir("..")
assert Path("pyproject.toml").exists(), "Open this notebook from inside the Day3_Day4_FDE_session folder."
print("working folder:", Path.cwd())
import sys, tempfile
IN_COLAB = "google.colab" in sys.modules
CODE = Path("src/creditlens") if IN_COLAB else Path(tempfile.gettempdir()) / "creditlens_nb08_code"
SCENARIOS = Path("/content/scenarios") if IN_COLAB else Path(tempfile.gettempdir()) / "creditlens_nb08_scenarios"
for sub in ("scenarios", "retrieval"):
    (CODE / sub).mkdir(parents=True, exist_ok=True)
print("the next six cells write to:", CODE if IN_COLAB else f"{CODE} (a scratch copy; the repository's own code is used)")

# %%
%%writefile "{CODE}/scenarios/__init__.py"
"""Opt-in story scenarios: Rohan's web of companies and Rahul's family cash. See `generate.py`."""
from creditlens.scenarios.detect import case_file
from creditlens.scenarios.generate import generate_scenarios

__all__ = ["case_file", "generate_scenarios"]

# %%
%%writefile "{CODE}/scenarios/generate.py"
"""Story scenarios: Rohan's web of companies and Rahul's family cash.

An opt-in extra dataset. It is written to `data/scenarios/`, never to `data/raw/`, and it uses
its own random stream, so `generate_all` and every number already shown in the workshop stay
exactly as they were.

Two cases, both in the same file layout the pipeline and the graph already read:

* **Rohan** (company SC01). His own file looks fine. Behind it: he is a director of two more
  firms (SC02, SC03), and SC03 shares a registered address with a fourth (SC04). Money leaves
  his account for SC02 and comes back days later (round trips), and two linked firms pay him
  large sums. Together the group is over the policy 5.2 exposure limit, though SC01 alone is not.
* **Rahul** (company SC10). An honest, open borrower. Part of his takings is paid into his
  mother's account, so his bank statement shows less than his declared turnover. His mother
  (Kamla Devi) is a director of a family firm (SC11) at the same address, so the graph links them.
  The group is within the limit. The lesson is a different one: a flag is a lead for a human, not
  a verdict.

Every person and company is invented.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from creditlens.config import Paths, Settings
from creditlens.generators.common import format_inr, rng_for

WINDOW = (date(2025, 10, 1), date(2026, 9, 30))
START_BALANCE = 9_000_000.0
ROHAN_APP, RAHUL_APP = "A901", "A902"  # the contract wants A plus three digits; 9xx never clashes with the real files

COMPANIES = [
    ("SC01", "Rohan Packaging Pvt Ltd", "Manufacturing", "2018-04-12", "14, Industrial Estate Road", "Pune"),
    ("SC02", "Ravi Containers Pvt Ltd", "Manufacturing", "2021-06-03", "7, MIDC Lane", "Pune"),
    ("SC03", "Ruchi Foam Packs Pvt Ltd", "Manufacturing", "2022-01-20", "22, Sector Five", "Nashik"),
    ("SC04", "RM Logistics Pvt Ltd", "Logistics", "2022-09-09", "22, Sector Five", "Nashik"),
    ("SC05", "Neel Plastics Pvt Ltd", "Manufacturing", "2015-03-18", "3, Station Road", "Surat"),
    ("SC10", "Rahul Textiles Pvt Ltd", "Textiles", "2016-08-25", "41, Market Street", "Surat"),
    ("SC11", "Kamla Fabrics Pvt Ltd", "Textiles", "2019-02-11", "41, Market Street", "Surat"),
]
DIRECTORS = [
    ("SD01", "Rohan Mehta"), ("SD02", "Meena Joshi"), ("SD03", "Arjun Patil"), ("SD04", "Neel Shah"),
    ("SD10", "Rahul Verma"), ("SD11", "Kamla Devi"),
]
LINKS = [  # company, director, role
    ("SC01", "SD01", "Managing Director"), ("SC02", "SD01", "Director"), ("SC02", "SD02", "Director"),
    ("SC03", "SD01", "Director"), ("SC03", "SD03", "Director"), ("SC04", "SD03", "Director"),
    ("SC05", "SD04", "Managing Director"),
    ("SC10", "SD10", "Managing Director"), ("SC11", "SD11", "Managing Director"),
]
EXISTING = {"SC01": 0, "SC02": 40_000, "SC03": 30_000, "SC04": 35_000, "SC05": 12_000, "SC10": 10_000, "SC11": 15_000}
APPLICATIONS = [(ROHAN_APP, "SC01", 50_000), (RAHUL_APP, "SC10", 30_000)]  # requested, INR thousand


def _ref(rng) -> str:
    return "N" + "".join(str(int(x)) for x in rng.integers(0, 10, 10))


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _months():
    y, m = WINDOW[0].year, WINDOW[0].month
    for _ in range(12):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def _day(y, m, d) -> date:
    return date(y, m, min(d, 28))


def _narr(rng, channel: str, cp: str) -> str:
    return {"NEFT": f"NEFT-{cp}-{_ref(rng)}", "RTGS": f"RTGS/{cp}/{_ref(rng)}",
            "IMPS": f"IMPS-{cp}-{int(rng.integers(100000, 999999))}",
            "UPI": f"UPI/{cp}/{int(rng.integers(100000, 999999))}"}[channel]


def _base_month(rng, y, m, customers, scale: float, out_scale: float | None = None):
    """The ordinary business: customers pay in, wages, rent, tax, suppliers and an EMI go out."""
    rows = []
    o = scale if out_scale is None else out_scale  # outflows can scale differently from inflows
    for _ in range(6):
        rows.append((_day(y, m, int(rng.integers(1, 29))), _narr(rng, _pick(rng, ["NEFT", "RTGS", "IMPS"]), _pick(rng, customers)),
                     round(float(rng.uniform(1.2, 3.4)) * 100_000 * scale, 2), "Cr"))
    for label, lo, hi, d in [("SALARY PAYROLL", 6.5, 7.5, 3), ("PREMISES RENT", 1.6, 1.9, 4), ("GST PAYMENT", 1.8, 2.8, 20)]:
        rows.append((_day(y, m, d), f"NEFT-{label}-{_ref(rng)}", round(float(rng.uniform(lo, hi)) * 100_000 * o, 2), "Dr"))
    for _ in range(3):
        rows.append((_day(y, m, int(rng.integers(5, 28))), _narr(rng, "NEFT", "NILGIRI PLASTICS"),
                     round(float(rng.uniform(1.0, 2.6)) * 100_000 * o, 2), "Dr"))
    rows.append((_day(y, m, 5), "ACH D- NORTHSTAR BANK EMI", round(2.0 * 100_000 * o, 2), "Dr"))
    return rows


def _rohan_rows(rng):
    rows, truth = [], {"round_trips": 0}
    variants = ["RAVI CONTAINERS PVT LTD", "Ravi Containers Pvt. Ltd.", "RAVI CONTAINERS"]
    for y, m in _months():
        rows += _base_month(rng, y, m, ["TAPTI FOODS", "HAMPI RETAIL", "GODAVARI TEXTILES", "INDUS EXPORTS"], 1.0)
        for k in range(2):  # money out to SC02 and back within one to three days
            out_day = int(rng.integers(6 + k * 10, 14 + k * 10))
            amount = round(float(rng.uniform(6, 12)) * 100_000, 2)
            back = round(amount * float(rng.uniform(0.97, 1.0)), 2)
            rows.append((_day(y, m, out_day), _narr(rng, "NEFT", _pick(rng, variants)), amount, "Dr"))
            rows.append((_day(y, m, out_day) + timedelta(days=int(rng.integers(1, 4))), _narr(rng, "RTGS", _pick(rng, variants)), back, "Cr"))
            truth["round_trips"] += 1
        rows.append((_day(y, m, 9), _narr(rng, "RTGS", "RUCHI FOAM PACKS PVT LTD"), round(float(rng.uniform(5, 8)) * 100_000, 2), "Cr"))
        rows.append((_day(y, m, 17), _narr(rng, "NEFT", "RM LOGISTICS"), round(float(rng.uniform(4, 7)) * 100_000, 2), "Cr"))
    return rows, truth


def _rahul_rows(rng):
    rows, truth = [], {"pass_through": 0, "hidden_share": 0.4}
    for y, m in _months():
        rows += _base_month(rng, y, m, ["MERU TRADERS", "ARAVALI RETAIL", "KONARK EXPORTS"], 0.55)
        for _ in range(3):  # cash arrives, and goes to his mother's account the next day
            d = _day(y, m, int(rng.integers(2, 26)))
            amount = round(float(rng.uniform(0.4, 0.9)) * 100_000, 2)
            rows.append((d, "CASH DEPOSIT", amount, "Cr"))
            rows.append((d + timedelta(days=1), _narr(rng, "IMPS", "KAMLA DEVI"), amount, "Dr"))
            truth["pass_through"] += 1
    return rows, truth


def _write_statement(path: Path, rows) -> pd.DataFrame:
    rows = sorted(rows, key=lambda r: (r[0], r[3] == "Dr"))
    bal, out = START_BALANCE, []
    for d, narr, amount, drcr in rows:
        bal += amount if drcr == "Cr" else -amount
        out.append({"Txn Date": d.strftime("%d-%b-%Y"), "Narration": narr, "Amount": format_inr(amount),
                    "Dr/Cr": drcr, "Closing Balance": format_inr(bal)})
    frame = pd.DataFrame(out)
    frame.to_csv(path, index=False)
    return frame


def generate_scenarios(settings: Settings | None = None, out_dir: Path | None = None) -> Path:
    """Write the scenario pack and return its folder (a `raw_dir` the pipeline and graph can read)."""
    s = settings or Settings()
    out = Path(out_dir) if out_dir else Paths().data / "scenarios"
    for sub in ("registry", "applications", "statements", "answer_key"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rng = rng_for(s.seed, "scenarios")

    pd.DataFrame([{"company_id": c, "legal_name": n, "sector": sec, "incorporated_on": inc,
                   "registered_address": addr, "city": city, "region": "West", "employees": 40}
                  for c, n, sec, inc, addr, city in COMPANIES]).to_csv(out / "registry" / "companies.csv", index=False)
    pd.DataFrame([{"director_id": d, "full_name": n, "pan": f"SYNTH{d[2:]}Z", "phone": "+91 00000 00000",
                   "email": f"{n.split()[0].lower()}@example.invalid", "residential_address": "Invented address"}
                  for d, n in DIRECTORS]).to_csv(out / "registry" / "directors.csv", index=False)
    pd.DataFrame([{"company_id": c, "director_id": d, "role": r, "appointed_on": "2020-01-01"}
                  for c, d, r in LINKS]).to_csv(out / "registry" / "company_directors.csv", index=False)
    pd.DataFrame([{"company_id": c, "outstanding_inr_000": v} for c, v in EXISTING.items()]).to_csv(
        out / "registry" / "existing_loans.csv", index=False)

    rohan_rows, rohan_truth = _rohan_rows(rng)
    rahul_rows, rahul_truth = _rahul_rows(rng)
    rohan = _write_statement(out / "statements" / f"{ROHAN_APP}_statement.csv", rohan_rows)
    rahul = _write_statement(out / "statements" / f"{RAHUL_APP}_statement.csv", rahul_rows)

    def credits(frame):
        amt = frame["Amount"].str.replace(",", "").astype(float)
        return float(amt[frame["Dr/Cr"] == "Cr"].sum())
    # Declared turnover (as if from the GST returns), in INR thousand.
    rohan_declared = round(credits(rohan) / 1000 * 0.92)
    rahul_declared = round(credits(rahul) / (1 - rahul_truth["hidden_share"]) / 1000)
    pd.DataFrame([{"application_id": a, "company_id": c, "requested_inr_000": r,
                   "declared_revenue_inr_000": rohan_declared if a == ROHAN_APP else rahul_declared}
                  for a, c, r in APPLICATIONS]).to_csv(out / "applications" / "applications.csv", index=False)

    key = {
        "note": "Instructor answer key. Do not give this folder to participants before the reveal.",
        ROHAN_APP: {"company": "SC01", "group": ["SC01", "SC02", "SC03", "SC04"], "links": {
            "SC01-SC02": "shared director SD01", "SC01-SC03": "shared director SD01", "SC03-SC04": "shared address"},
            "group_exposure_inr_000": 155000, "breaches_rule_5_2": True, **rohan_truth},
        RAHUL_APP: {"company": "SC10", "group": ["SC10", "SC11"], "links": {"SC10-SC11": "shared address"},
                    "group_exposure_inr_000": 55000, "breaches_rule_5_2": False, **rahul_truth},
    }
    from creditlens.scenarios.bonus import add_bonus  # four more cases, on their own random stream
    key.update(add_bonus(out, s.seed))
    (out / "answer_key" / "scenarios.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    return out

# %%
%%writefile "{CODE}/scenarios/bonus.py"
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

# %%
%%writefile "{CODE}/scenarios/detect.py"
"""Detectors for the story scenarios. They reuse the pipeline and the graph, and they only raise leads.

A flag says "a person should look at this", never "this person did something wrong". There can be an
honest explanation for each one, and the bank's policy keeps the decision with a human underwriter.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from creditlens.pipeline.clean import clean_statement
from creditlens.pipeline.wrangle import normalize_counterparty
from creditlens.retrieval.graph import GraphStore
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

# %%
%%writefile "{CODE}/retrieval/__init__.py"
"""Colab stand-in package."""

# %%
%%writefile "{CODE}/retrieval/graph.py"
"""A small stand-in for creditlens.retrieval.graph.GraphStore, so this notebook runs on Colab with just the
pre-work repo. Same idea as the real one: companies are linked by a shared director or a shared address,
and a group is everything connected. Session 2 builds the full version with Neo4j and Cypher."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pandas as pd

LIMIT_INR_000 = 120000  # policy rule 5.2


class GraphStore:
    def __init__(self, backend: str = "networkx", limit_inr_000: int = LIMIT_INR_000):
        self.limit_inr_000 = limit_inr_000

    def load_from_registry(self, raw_dir) -> "GraphStore":
        raw = Path(raw_dir); reg = raw / "registry"
        companies = pd.read_csv(reg / "companies.csv")
        links = pd.read_csv(reg / "company_directors.csv")
        loans = pd.read_csv(reg / "existing_loans.csv")
        apps = pd.read_csv(raw / "applications" / "applications.csv")
        self.companies = {r["company_id"]: r for r in companies.to_dict("records")}
        self.outstanding = {r["company_id"]: int(r["outstanding_inr_000"]) for r in loans.to_dict("records")}
        self.pending = {c: int(v) for c, v in apps.groupby("company_id")["requested_inr_000"].sum().items()}
        g = nx.Graph()
        for cid, row in self.companies.items():
            g.add_edge(("C", cid), ("A", str(row["registered_address"]).strip()), kind="REGISTERED_AT")
        for r in links.to_dict("records"):
            if ("C", r["company_id"]) in g:
                g.add_edge(("D", r["director_id"]), ("C", r["company_id"]), kind="DIRECTOR_OF")
        self.graph = g
        return self

    def related_group(self, company_id: str) -> list[str]:
        comp = nx.node_connected_component(self.graph, ("C", company_id))
        return sorted(n[1] for n in comp if n[0] == "C")

    def group_exposure(self, company_id: str):
        members = [SimpleNamespace(company_id=c, existing_outstanding_inr_000=self.outstanding.get(c, 0),
                                   pending_request_inr_000=self.pending.get(c, 0)) for c in self.related_group(company_id)]
        total = sum(m.existing_outstanding_inr_000 + m.pending_request_inr_000 for m in members)
        return SimpleNamespace(company_id=company_id, members=members, total_exposure_inr_000=int(total),
                               limit_inr_000=self.limit_inr_000, breaches_limit=bool(len(members) > 1 and total > self.limit_inr_000))

# %%
import pandas as pd, matplotlib.pyplot as plt, networkx as nx
from pathlib import Path
from creditlens.scenarios import generate_scenarios, case_file
from creditlens.scenarios.detect import (load_statement, top_counterparties, round_trips, pass_through, revenue_gap,
                                         monthly_inflow, seasonality, window_dressing, sector_mismatch, footing_check)
from creditlens.retrieval.graph import GraphStore

OUT = generate_scenarios(out_dir=SCENARIOS)     # six statements, a registry, applications, financials
G = GraphStore().load_from_registry(OUT)
pd.set_option("display.width", 140); pd.set_option("display.max_colwidth", 60)

def statement(app):
    """The Session 1 cleaning step: dates, amounts and counterparty names made consistent."""
    return load_statement(OUT / "statements" / f"{app}_statement.csv")

def show(app):
    """The case file: flags with evidence, every one a lead for a person."""
    cf = case_file(OUT, app)
    print(f"{app} ({cf['company_id']})")
    for f in cf["flags"]:
        print(f"  [{f['code']}] {f['title']}")
    print("  " + cf["note"])
    return cf

print("Ready:", sorted(p.name for p in (OUT / "statements").glob("*.csv")))

# %% [markdown]
# ## A. Rohan (A901): looks safe on Monday
# Rohan Packaging asks for **INR 5.0 crore**. Papers in order, steady money in, no returned cheques, well inside every limit.
# **Vote first:** would you approve? Then run the cells.
#
# ### A1. The raw statement: one company, many spellings

# %%
raw = pd.read_csv(OUT / "statements" / "A901_statement.csv")
ravi = raw[raw["Narration"].str.contains("ravi", case=False)]
spellings = ravi["Narration"].str.extract(r"[-/]([^-/]*ravi[^-/]*)[-/]", expand=False, flags=2)
print(len(ravi), "lines mention Ravi Containers, written", spellings.nunique(), "different ways:")
spellings.value_counts()

# %% [markdown]
# ### A2. After cleaning: the names merge, and one name stands out

# %%
a901 = statement("A901")
top = top_counterparties(a901, 5)
display(top.round(3))
top.set_index("counterparty")["share_of_inflow"].plot.barh(title="Share of money in, by counterparty", figsize=(7, 3)).invert_yaxis()
plt.show()

# %% [markdown]
# ### A3. Round trips: money out, and almost the same money back within days

# %%
rt = round_trips(a901)
print(len(rt), "round trips,", round(rt["out_amount"].sum() / 1e7, 2), "crore out and back")
display(rt.head())
plt.figure(figsize=(8, 3)); plt.vlines(pd.to_datetime(rt["out_date"]), 0, rt["out_amount"] / 1e5, color="#6D28D9")
plt.title("Each line is money sent to RAVI CONTAINERS (lakh), and it came back within days"); plt.show()

# %% [markdown]
# ### A4. The graph: who is behind the names?

# %%
exp = G.group_exposure("SC01")
print("Group:", [m.company_id for m in exp.members])
print(f"Group exposure {exp.total_exposure_inr_000:,} against a limit of {exp.limit_inr_000:,} (INR thousand)")
print("Rule 5.2 breached:", exp.breaches_limit, "| Rohan Packaging alone:", G.outstanding.get("SC01", 0) + G.pending.get("SC01", 0))
start = ("C", "SC01") if ("C", "SC01") in G.graph else "Company:SC01"   # Colab stand-in names, or the repository's
kind = lambda n: n[0] if isinstance(n, tuple) else n[0]                   # C, D or A: company, director, address
ident = lambda n: n[1] if isinstance(n, tuple) else n.split(":", 1)[1]
sub = G.graph.subgraph(nx.node_connected_component(G.graph, start))
labels = {n: (G.companies[ident(n)]["legal_name"].replace(" Pvt Ltd", "") if kind(n) == "C" else ident(n)) for n in sub}
colors = ["#6D28D9" if kind(n) == "C" else "#D98E04" if kind(n) == "D" else "#9CA3AF" for n in sub]
plt.figure(figsize=(9, 5)); nx.draw(sub, labels=labels, node_color=colors, node_size=1800, font_size=8, edge_color="#C4B5FD")
plt.title("Purple: companies. Amber: directors. Grey: addresses"); plt.show()

# %% [markdown]
# ### A5. The case file: leads, not verdicts

# %%
cf = show('A901')

# %% [markdown]
# ### A6. Your turn: the knobs
# `round_trips` has two settings. Which would you defend to an auditor?

# %%
for tol in (0.05, 0.03, 0.01):
    print("tolerance", tol, "->", len(round_trips(a901, tolerance=tol)), "round trips")
for days in (1, 2, 3, 5):
    print("window", days, "days ->", len(round_trips(a901, window_days=days)), "round trips")

# %% [markdown]
# ## B. Rahul (A902): looks weak on Monday
# Rahul Textiles asks for **INR 3.0 crore**. A thin statement, tight cash, less money in than the turnover he declared.
#
# ### B1. What the statement shows

# %%
a902 = statement("A902")
display(top_counterparties(a902, 5).round(3))
cash = a902[a902["channel"] == "CASH"]
print(len(cash), "cash deposits, total", round(cash["amount"].sum() / 1e5, 1), "lakh")

# %% [markdown]
# ### B2. Cash in, then the same amount to one person

# %%
pt = pass_through(a902, {"Kamla Devi"})
print(len(pt), "times: cash in, same amount to", sorted(set(pt["sent_to"])), "the next day")
display(pt.head())
gap = revenue_gap(a902, float(pd.read_csv(OUT / "applications" / "applications.csv").query("application_id == 'A902'")["declared_revenue_inr_000"].iloc[0]))
print("The statement shows", round(gap["visible_over_declared"] * 100), "% of the turnover he declared")

# %% [markdown]
# ### B3. The graph explains it

# %%
exp = G.group_exposure("SC10")
print("Group:", [(m.company_id, G.companies[m.company_id]["legal_name"]) for m in exp.members])
print("Exposure", f"{exp.total_exposure_inr_000:,}", "vs limit", f"{exp.limit_inr_000:,}", "-> breached:", exp.breaches_limit)
print("Kamla Fabrics shares Rahul Textiles' address, and its director is Kamla Devi.")
print("Priya phones Rahul: Kamla Devi is his mother, and the family firm shares the shop.")
cf = show("A902")

# %% [markdown]
# **Notice what is missing:** no rule 5.2 flag. The group is within the limit. The system did not punish Rahul for looking
# weak; it explained why. Whether keeping money in a family account is a tax matter is for the right team to judge, not for this assistant.

# %% [markdown]
# ## C. Meena (A903): an honest baker with a bad monsoon
# Meena Bakery asks for **INR 1.5 crore**. Three cheques bounced, and policy rule 4.1 says more than two is an exception.
# Is she a risk, or just a seasonal business?

# %%
a903 = statement("A903")
actual = monthly_inflow(a903) / 1e5
declared = pd.read_csv(OUT / "applications" / "declared_monthly.csv").set_index("month")["declared_inr"] / 1e5
ax = actual.plot(kind="bar", color="#6D28D9", figsize=(9, 3.5), label="money in (bank)")
declared.plot(ax=ax, color="#D98E04", marker="o", label="declared turnover (GST)")
plt.ylabel("lakh"); plt.legend(); plt.title("Weak months are the months she said would be weak"); plt.show()
print("Returned cheques:", int(a903["is_bounce"].sum()), "(rule 4.1 limit: 2)")
print("Correlation between the two lines:", round(seasonality(a903, pd.read_csv(OUT / 'applications' / 'declared_monthly.csv').query("application_id == 'A903'"))["correlation"], 3))
cf = show("A903")

# %% [markdown]
# **The lesson:** the rule fired, and it should. But the second flag is the explanation: the weak months match the shape
# she declared. A person can now ask a better question than "why did cheques bounce?": "tell me about the monsoon".

# %% [markdown]
# ## D. Tapti Trading (A904): a quiet account that suddenly fills up
# Tapti Trading asks for **INR 2.5 crore**. The year looks small, then the last weeks look wonderful.

# %%
a904 = statement("A904")
cum = a904[a904["amount"] > 0].set_index("txn_date")["amount"].cumsum() / 1e5
cum.plot(figsize=(9, 3.5), color="#6D28D9", title="Cumulative money in (lakh): look at the last month"); plt.show()
w = window_dressing(a904)
print(round(w["share_of_year"] * 100), "% of the whole year's money in arrived in the last", w["last_days"], "days")
cf = show("A904")

# %% [markdown]
# **The lesson:** a late surge is a lead, not a crime. New contracts, an asset sale or a late payment can all explain it.
# The assistant's job is to make sure Priya *sees* it before the committee does.

# %% [markdown]
# ## E. Sagar Wellness (A905): declared healthcare
# Sagar Wellness asks for **INR 2.0 crore** and declares the sector **Healthcare**. Policy rule 2.2 excludes online betting.

# %%
a905 = statement("A905")
sm = sector_mismatch(a905, "Healthcare")
print(round(sm["share_of_inflow"] * 100), "% of the money in comes from", sm["parties"])
top_counterparties(a905, 6).set_index("counterparty")["share_of_inflow"].plot.barh(figsize=(7, 3), title="Money in, by counterparty").invert_yaxis()
plt.show()
cf = show("A905")

# %% [markdown]
# **The lesson:** the sector on the form is what the borrower *says*; the counterparties are what the money *does*.
# A keyword list is a crude detector, so a person confirms who these parties are before anything is decided.

# %% [markdown]
# ## F. Konark Exports (A906): numbers that do not add up
# Konark Exports asks for **INR 4.0 crore**, with three years of financials.

# %%
fin = pd.read_csv(OUT / "financials" / "financials.csv")
display(fin)
bad = footing_check(fin)
print("Lines that do not add up:"); display(bad)
a906 = statement("A906")
cf = show("A906")

# %% [markdown]
# **The lesson:** two independent clues point the same way: FY2026 EBITDA is printed about 40% above revenue minus costs,
# and the bank statement shows only about 55% of the revenue claimed. In Session 1 you built the contracts that make the first
# check automatic; in Session 4 you will see how a check like this runs on every file.

# %% [markdown]
# ## G. All six at a glance

# %%
rows = []
for app in ["A901", "A902", "A903", "A904", "A905", "A906"]:
    cf = case_file(OUT, app)
    rows.append({"application": app, "flags": ", ".join(f["code"] for f in cf["flags"])})
display(pd.DataFrame(rows))

# %% [markdown]
# ### What to remember
# 1. **A flag is a lead, not a verdict.** Every flag here carries its evidence, and none of them says "fraud".
# 2. **Two cases can share a flag and mean opposite things** (Meena and Konark both look weak; only one has numbers that fail to add up).
# 3. **Looking weak is not the same as being weak** (Rahul, Meena), and **looking safe is not the same as being safe** (Rohan).
# 4. The bank's policy forbids automated approvals. The assistant finds, explains and stops. A person decides.

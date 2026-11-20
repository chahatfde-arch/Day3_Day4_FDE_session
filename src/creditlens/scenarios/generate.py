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
        "note": "Instructor answer key. Do not give this folder to pods before the reveal.",
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

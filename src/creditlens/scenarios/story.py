"""The story world: Rohan's and Rahul's files as a complete Good Bank data folder.

Sessions 1 and 2 found the leads with `case_file`. Session 3 hands both files to the same
agents that draft memos for A001 to A030, without changing a line of agent code. The trick is
the one the tests already use: every agent reads `data/` through `CREDITLENS_DATA_ROOT` and
the database through `CREDITLENS_DB_URL`. So we build a second, small data folder,
`data/story/`, in exactly the layout of `data/raw/`, load it into its own SQLite file, and
point those two variables at it while the agents run.

    python -m creditlens.scenarios.story build                      # once, about ten seconds
    python -m creditlens.scenarios.story memo A901 --user U05       # runs to the underwriter, prints the thread id
    python -m creditlens.scenarios.story decide <thread_id> approve --amount 15000 --comment "..." --user U05

What is added on top of `data/scenarios/`:

* full application rows (product, tenure, region, guarantor) for A901 and A902;
* audited financial statement PDFs for SC01 and SC10, rendered by the Session 1 generator,
  with figures chosen so that **on paper both companies pass rules 3.1 to 3.3**. Rohan's
  statement shows exactly the turnover he declared, which the bank statement does not support
  (Session 2, notebook 02 section 6). The ratios are only as true as the documents;
* the Good Bank staff list, so U04 (West analyst) and U05 (Head of Credit) can log in.

Every person and company is invented. Nothing here is written to `data/raw/`.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import replace
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from creditlens.config import Paths, Settings, settings as default_settings  # noqa: F401

STORY_APPS = ("A901", "A902")

#: The application rows the scenario pack leaves out. Region West, so U04 (West analyst) owns them.
APPLICATION_DETAILS = {
    "A901": {"region": "West", "product": "Term Loan", "purpose": "New corrugation line",
             "tenure_months": 60, "submitted_on": "2026-10-20", "analyst_id": "U04",
             "guarantor_director_id": "SD01"},
    "A902": {"region": "West", "product": "Working Capital", "purpose": "Yarn stock for the wedding season",
             "tenure_months": 24, "submitted_on": "2026-10-20", "analyst_id": "U04",
             "guarantor_director_id": "SD10"},
}

#: Latest-year shape of each statement. Revenue in FY2026 equals the declared turnover.
STATEMENT_SHAPE = {
    # company: (gross margin, opex share, depreciation share, total debt, interest, principal,
    #           current assets share, current liabilities share, fixed assets share)
    "SC01": (0.31, 0.11, 0.030, 12_000, 1_320, 2_400, 0.24, 0.13, 0.45),
    "SC10": (0.28, 0.10, 0.028, 6_500, 715, 1_300, 0.25, 0.18, 0.30),
}
GROWTH = {"FY2024": 1 / 1.18, "FY2025": 1 / 1.08, "FY2026": 1.0}


def story_root(paths: Optional[Paths] = None) -> Path:
    return (paths or default_settings.paths).data / "story"


def story_db_url(root: Optional[Path] = None) -> str:
    return f"sqlite:///{(root or story_root()) / 'story.db'}"


def _year(revenue: int, shape: tuple) -> dict[str, int]:
    """One fiscal year that adds up: every subtotal is computed, never typed."""
    gm, opex_s, dep_s, debt, interest, principal, ca_s, cl_s, fa_s = shape
    cogs = round(revenue * (1 - gm))
    gross = revenue - cogs
    opex = round(revenue * opex_s)
    ebitda = gross - opex
    depreciation = round(revenue * dep_s)
    pbt = ebitda - depreciation - interest
    tax = max(0, round(pbt * 0.25))
    current_assets = round(revenue * ca_s)
    cash = round(current_assets * 0.2)
    receivables = round(current_assets * 0.5)
    inventory = current_assets - cash - receivables
    current_liabilities = round(revenue * cl_s)
    fixed_assets = round(revenue * fa_s)
    equity = current_assets + fixed_assets - current_liabilities - debt
    return {"revenue": revenue, "cogs": cogs, "gross_profit": gross, "opex": opex, "ebitda": ebitda,
            "depreciation": depreciation, "interest": interest, "pbt": pbt, "tax": tax, "pat": pbt - tax,
            "principal_repayment": principal, "cash": cash, "receivables": receivables,
            "inventory": inventory, "current_assets": current_assets, "fixed_assets": fixed_assets,
            "total_assets": current_assets + fixed_assets, "current_liabilities": current_liabilities,
            "total_debt": debt, "equity": equity,
            "total_liabilities_and_equity": current_liabilities + debt + equity}


def statement_truth(declared_inr_000: dict[str, int]) -> dict[str, dict]:
    """{company_id: {"unit", "years": {FY: figures}}} for SC01 and SC10."""
    out = {}
    for company, declared in declared_inr_000.items():
        years = {fy: _year(round(declared * g), STATEMENT_SHAPE[company]) for fy, g in GROWTH.items()}
        out[company] = {"unit": "INR '000", "weak": False, "years": years}
    return out


def build_story(s: Settings = default_settings, verbose: bool = True) -> Path:
    """Write data/story (a raw folder in the pipeline's layout), run the pipeline, load story.db."""
    from creditlens.generators.financials import render_financials_pdf
    from creditlens.pipeline.load import load_all
    from creditlens.pipeline.run import run_pipeline
    from creditlens.scenarios.generate import generate_scenarios

    log = print if verbose else (lambda *a, **k: None)
    scenarios = s.paths.data / "scenarios"
    if not (scenarios / "statements" / "A901_statement.csv").exists():
        generate_scenarios(s, out_dir=scenarios)
    root = story_root(s.paths)
    if root.exists():
        shutil.rmtree(root)
    raw = root / "raw"
    for sub in ("registry", "applications", "statements", "financials"):
        (raw / sub).mkdir(parents=True, exist_ok=True)

    for name in ("companies", "directors", "company_directors", "existing_loans"):
        shutil.copy(scenarios / "registry" / f"{name}.csv", raw / "registry" / f"{name}.csv")
    shutil.copy(s.paths.raw / "registry" / "users.csv", raw / "registry" / "users.csv")

    apps = pd.read_csv(scenarios / "applications" / "applications.csv")
    rows = []
    for _, a in apps.iterrows():  # all six, so group exposure counts every pending request
        d = APPLICATION_DETAILS.get(a["application_id"], {
            "region": "West", "product": "Term Loan", "purpose": "Working capital", "tenure_months": 36,
            "submitted_on": "2026-10-20", "analyst_id": "U04", "guarantor_director_id": ""})
        rows.append({"application_id": a["application_id"], "company_id": a["company_id"],
                     "region": d["region"], "product": d["product"], "purpose": d["purpose"],
                     "requested_inr_000": int(a["requested_inr_000"]), "tenure_months": d["tenure_months"],
                     "submitted_on": d["submitted_on"], "analyst_id": d["analyst_id"],
                     "guarantor_director_id": d["guarantor_director_id"]})
    pd.DataFrame(rows).to_csv(raw / "applications" / "applications.csv", index=False)

    for app in STORY_APPS:  # only the two story statements: the bonus cases stay in the notebook 08 pack
        shutil.copy(scenarios / "statements" / f"{app}_statement.csv", raw / "statements" / f"{app}_statement.csv")

    declared = {r["company_id"]: int(r["declared_revenue_inr_000"])
                for _, r in apps.iterrows() if r["application_id"] in STORY_APPS}
    truth = statement_truth(declared)
    names = pd.read_csv(raw / "registry" / "companies.csv").set_index("company_id")["legal_name"]
    for company, data in truth.items():
        render_financials_pdf(raw / "financials" / f"{company}_financials.pdf", names[company], data)
    (root / "answer_key").mkdir(exist_ok=True)
    (root / "answer_key" / "financials_truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")

    story_settings = replace(s, paths=s.paths.with_data_root(root))
    out = run_pipeline(story_settings, load=False, verbose=False)
    load_all(story_db_url(root), raw, out)
    log(f"story world: {root}")
    log(f"  applications: {', '.join(r['application_id'] for r in rows)} (agents run on {', '.join(STORY_APPS)})")
    log(f"  statements cleaned: {len(out['transactions'])} transactions, {len(out['statement_features'])} feature rows")
    log(f"  financial statements: {', '.join(f'{c}_financials.pdf' for c in truth)}")
    log(f"  database: {story_db_url(root)}")
    return root


@contextlib.contextmanager
def story_env(root: Optional[Path] = None) -> Iterator[Path]:
    """Point the agents at data/story for the duration of a `with` block, then put everything back.

    NEO4J_URI is hidden too: the story registry is small and lives in memory, and it must
    never be written into the participant's AuraDB next to the Good Bank graph.
    """
    root = Path(root or story_root())
    if not (root / "story.db").exists():
        raise FileNotFoundError(f"{root / 'story.db'} is missing. Run: python -m creditlens.scenarios.story build")
    saved = {k: os.environ.get(k) for k in ("CREDITLENS_DATA_ROOT", "CREDITLENS_DB_URL", "NEO4J_URI")}
    os.environ["CREDITLENS_DATA_ROOT"] = str(root)
    os.environ["CREDITLENS_DB_URL"] = story_db_url(root)
    os.environ.pop("NEO4J_URI", None)
    try:
        yield root
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def draft_story_memo(application_id: str, user: str = "U05", thread_id: Optional[str] = None,
                     root: Optional[Path] = None) -> dict:
    """Run the Session 3 agents on A901 or A902 until they stop for the underwriter."""
    from creditlens.agents.run import default_checkpoint_path, draft_memo

    with story_env(root):  # a file checkpoint, so `decide` works from another process or a later cell
        return draft_memo(application_id, user, thread_id=thread_id, checkpoint_path=default_checkpoint_path())


def decide(thread_id: str, decision: str, user: str = "U05", comment: str = "",
           approved_inr_000: Optional[int] = None, root: Optional[Path] = None) -> dict:
    """The underwriter's decision on a paused story run. Only the Head of Credit may decide (7.3)."""
    from creditlens.agents import adapters
    from creditlens.agents.run import default_checkpoint_path, pending_application_id, resume
    from creditlens.schemas import ApprovalDecision

    with story_env(root):
        person = adapters.load_user(user)
        if person.role != "head_of_credit":
            raise PermissionError(f"{person.full_name} has the role {person.role}. Policy 7.3: only the Head of Credit decides.")
        path = default_checkpoint_path()
        application_id = pending_application_id(thread_id, checkpoint_path=path)
        if application_id is None:
            raise KeyError(f"no paused story run for thread {thread_id}")
        choice = ApprovalDecision(application_id=application_id, decided_by=person.user_id, decision=decision,
                                  comment=comment, approved_inr_000=approved_inr_000, channel="cli")
        return resume(thread_id, choice, checkpoint_path=path)


def _print_draft(result: dict) -> None:
    memo = result.get("memo")
    print(f"thread id: {result.get('thread_id')}   status: {result.get('status')}")
    if memo is None:
        return
    print(f"recommendation: {memo.recommendation}")
    print(f"why: {getattr(memo, 'recommendation_reason', '') or ''}".rstrip())
    for exc in memo.policy_exceptions or []:
        print(f"exception: rule {exc.rule_id}, measured {exc.measured_value}, ticket {exc.jira_key}")
    group = getattr(memo, "group_exposure", None)
    if group is not None:
        print(f"group: {', '.join(m.company_id for m in group.members)}  exposure {group.total_exposure_inr_000:,} "
              f"vs {group.limit_inr_000:,}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m creditlens.scenarios.story", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="write data/story and load story.db")
    m = sub.add_parser("memo", help="draft a memo for A901 or A902 and stop for the underwriter")
    m.add_argument("application_id", choices=STORY_APPS)
    m.add_argument("--user", default="U05")
    d = sub.add_parser("decide", help="the underwriter's decision on a paused story run")
    d.add_argument("thread_id")
    d.add_argument("decision", choices=["approve", "reject"])
    d.add_argument("--amount", type=int, default=None, help="approved amount, INR thousand (a counter-offer)")
    d.add_argument("--comment", default="")
    d.add_argument("--user", default="U05")
    args = ap.parse_args(argv)

    if args.cmd == "build":
        build_story()
        return 0
    if args.cmd == "memo":
        _print_draft(draft_story_memo(args.application_id, args.user))
        return 0
    try:
        result = decide(args.thread_id, args.decision, args.user, args.comment, args.amount)
    except (PermissionError, KeyError) as e:
        print(e)
        return 2
    amount = f" for INR {args.amount:,} thousand" if args.amount else ""
    print(f"{result.get('application_id', args.thread_id)} is now {result['status']}{amount}.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

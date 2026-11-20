"""CreditLens AI, Session 1 end to end: from a messy bank statement to a scored extractor.

One page per step of notebooks 00, 01 and 02, all computed live from the repository's own code
on the synthetic Good Bank data. Offline: no API key is needed.

    python -m streamlit run app/session1_app.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)
os.environ.setdefault("LLM_OFFLINE", "1")
# One database file, by absolute path, so the pipeline and the API always agree.
os.environ.setdefault("DATABASE_URL", f"sqlite:///{(REPO / 'creditlens.db').as_posix()}")

st.set_page_config(page_title="CreditLens AI, Session 1", layout="wide")
st.markdown("""<style>
h1, h2, h3 { color:#2E1065; }
div[data-testid="stSidebar"] { background:#EDE9FE; }
.say { background:#F3EEFC; border-left:4px solid #6D28D9; padding:10px 14px; border-radius:6px; margin:6px 0 14px; }
</style>""", unsafe_allow_html=True)

RAW, CLEAN, CURATED = REPO / "data/raw", REPO / "data/clean", REPO / "data/curated"
SCN = REPO / "data/scenarios"
WINDOW = (date(2025, 10, 1), date(2026, 9, 30))


def say(text: str) -> None:
    st.markdown(f'<div class="say">{text}</div>', unsafe_allow_html=True)


def run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], cwd=REPO, capture_output=True, text=True,
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"})


@st.cache_resource(show_spinner="Preparing the data (first run only, about a minute)...")
def prepare() -> str:
    steps = []
    if not (RAW / "statements").exists():
        steps.append(["-m", "creditlens.generators"])
    if not (CLEAN / "quality_report.csv").exists() or not (REPO / "creditlens.db").exists():
        steps.append(["-m", "creditlens.pipeline", "--load"])
    if not (SCN / "statements" / "A901_statement.csv").exists():
        steps.append(["-m", "creditlens.scenarios"])
    for s in steps:
        r = run(s)
        if r.returncode:
            return f"`python {' '.join(s)}` failed:\n\n{r.stderr[-800:]}"
    return ""


problem = prepare()
if problem:
    st.error(problem)
    st.stop()

STEPS = ["1. The problem", "2. The data and the repo", "3. Raw statements", "4. Cleaning", "5. The contract",
         "6. Features", "7. The first clue", "8. Ground truth", "9. The API", "10. Reading documents", "11. The score"]
with st.sidebar:
    st.markdown("### CreditLens AI\nSession 1, end to end")
    step = st.radio("Step", STEPS, label_visibility="collapsed")
    st.caption("Notebooks 00, 01 and 02. Synthetic Good Bank data. Offline, no API key.")

# ----------------------------------------------------------------------------- 1
if step == STEPS[0]:
    st.title("Rohan needs a loan. Priya is drowning in paperwork.")
    say("<b>The problem.</b> Rohan runs a small packaging company and applies to Good Bank for a loan. Priya, the credit "
        "analyst, reads twelve months of bank statements, three years of financial statements and the credit policy, "
        "checks related companies, raises a ticket for every broken rule, then writes a memo. Rohan waits weeks, and "
        "mistakes slip through.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Our end goal")
        st.markdown("- Turn a loan application into a **draft credit memo**.\n"
                    "- Read every document, including scans, and check every policy rule.\n"
                    "- Find related companies, score the risk, raise one ticket per exception.\n"
                    "- **Stop for a human.** The underwriter makes every final decision.")
    with c2:
        st.subheader("What Session 1 builds")
        st.markdown("1. **Pre-work** (notebook 00): tools ready, data generated.\n"
                    "2. **Data foundation** (notebook 01): raw to clean to curated, with contracts.\n"
                    "3. **Ground truth and extraction** (notebook 02): an answer key, an API, documents to numbers, a score.")
    st.info("Everything is measured against answers we agree on first. That is why the session ends with a number on the board.")

# ----------------------------------------------------------------------------- 2
elif step == STEPS[1]:
    st.title("Where the data lives, and why the repo looks like this")
    say("Made-up borrowers, real-world documents. The borrowers are generated from one seed, so every laptop has the same "
        "files. The receipts and the rulebook are real public data. No real borrower ever enters the project.")
    rows = []
    for folder, made_by, what in [
        ("data/raw/applications", "python -m creditlens.generators", "30 loan applications"),
        ("data/raw/statements", "python -m creditlens.generators", "12 months of messy bank statements each"),
        ("data/raw/financials", "python -m creditlens.generators", "3 years of financials as PDFs, some scanned"),
        ("data/raw/policy", "python -m creditlens.generators", "the Good Bank credit policy manual"),
        ("data/raw/registry", "python -m creditlens.generators", "companies, directors, existing loans, users"),
        ("data/real/receipts", "python -m creditlens.data_real", "real scanned receipts with human labels (SROIE)"),
        ("data/clean", "python -m creditlens.pipeline", "validated transactions and a quality report"),
        ("data/curated", "python -m creditlens.pipeline", "one row of features per application"),
        ("data/scenarios", "python -m creditlens.scenarios", "Rohan, Rahul and four more story cases"),
        ("ground_truth", "in Git", "the hand-checked answer key"),
    ]:
        p = REPO / folder
        n = len([x for x in p.iterdir() if x.is_file()]) if p.exists() else 0
        rows.append({"folder": folder, "files": n, "what": what, "made by": made_by})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.subheader("The repo, by job")
    st.code("""src/creditlens/      the code that ships (generators, pipeline, extraction, api, gateway)
notebooks/           explore and teach: 00 pre-work, 01 data foundation, 02 ground truth and extraction
data/raw -> clean -> curated   three layers, regenerated from one seed, never in Git
ground_truth/        the answers we agree on first, in Git
evals/               the scoreboards (extraction_accuracy.py, thresholds.json)
tests/               what get_ready.py runs
workshop/            one page per session and topic""", language="text")
    say("<b>Why:</b> notebooks explore, <code>src</code> ships. Data is regenerated, not committed. The answer key is committed, "
        "because it is the ruler everything is measured with.")

# ----------------------------------------------------------------------------- 3
elif step == STEPS[2]:
    st.title("Raw statements: the bank sent this")
    say("Groceries as delivered, mud and all. Before you read the table, guess: how many kinds of mess will you find?")
    files = sorted((RAW / "statements").glob("*.csv"))
    pick = st.selectbox("Statement", [f.name for f in files], index=1)
    raw = pd.read_csv(RAW / "statements" / pick, dtype=str, keep_default_na=False)
    a, b, c, d = st.columns(4)
    a.metric("Rows", len(raw)); b.metric("Exact duplicates", int(raw.duplicated().sum()))
    c.metric("Missing balances", int((raw.iloc[:, -1] == "").sum()))
    d.metric("Distinct date shapes", raw.iloc[:, 0].str.replace(r"\d", "9", regex=True).nunique())
    st.dataframe(raw.head(25), use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------------- 4
elif step == STEPS[3]:
    st.title("Cleaning: one small function per kind of mess")
    from creditlens.pipeline.wrangle import normalize_counterparty, parse_amount, parse_date, parse_narration
    say("Each fix is a tiny tested function. Say the input out loud, let the room guess, then show the answer.")
    demos = []
    for t in ["₹1,23,456.50", "(2,000)", "Rs. 42"]:
        demos.append({"function": "parse_amount", "input": t, "output": str(parse_amount(t))})
    for t in ["03-Apr-2026", "03-10-25", "03.10.2025"]:
        demos.append({"function": "parse_date", "input": t, "output": str(parse_date(t, WINDOW)[0])})
    for t in ["UPI/TATA POWER/753217", "CHQ RTN ARVIND TRADERS INSUFFICIENT FUNDS"]:
        demos.append({"function": "parse_narration", "input": t, "output": str(parse_narration(t))})
    for t in ["Shakti Enterprises Pvt. Ltd.", "SHAKTI ENT PVT LTD", "Amzn Seller Svcs Pvt Ltd."]:
        demos.append({"function": "normalize_counterparty", "input": t, "output": normalize_counterparty(t)})
    if st.toggle("Show the answers", value=False):
        st.dataframe(pd.DataFrame(demos), hide_index=True, use_container_width=True)
    else:
        st.dataframe(pd.DataFrame(demos)[["function", "input"]], hide_index=True, use_container_width=True)
    st.subheader("The report card: what was fixed in each file")
    q = pd.read_csv(CLEAN / "quality_report.csv")
    st.dataframe(q, hide_index=True, use_container_width=True)

# ----------------------------------------------------------------------------- 5
elif step == STEPS[4]:
    st.title("The contract: a promise the table keeps")
    say("Every balance must equal the previous balance plus this row's amount. One wrong number and the whole file is refused, "
        "like airport security: one bad bag and the belt stops.")
    import pandera as pa
    from creditlens.pipeline.contracts import clean_transactions
    tx = pd.read_csv(CLEAN / "transactions.csv", parse_dates=["txn_date"])
    st.success(f"{len(clean_transactions.validate(tx)):,} clean rows pass the contract.")
    row = st.number_input("Flip the sign of row", min_value=0, max_value=len(tx) - 1, value=5)
    if st.button("Break it and validate again", type="primary"):
        broken = tx.copy()
        broken.loc[row, "amount"] = -broken.loc[row, "amount"]
        try:
            clean_transactions.validate(broken, lazy=True)
            st.warning("It passed. Try another row.")
        except pa.errors.SchemaErrors as e:
            st.error("Refused. The file does not move to the next layer.")
            st.dataframe(e.failure_cases[["check", "index", "failure_case"]].head(5), hide_index=True)
    st.caption("Lineage: every clean row carries its source_file, so an auditor can trace any number back.")
    st.dataframe(tx.head(8), hide_index=True, use_container_width=True)

# ----------------------------------------------------------------------------- 6
elif step == STEPS[5]:
    st.title("Features: twelve months become one row")
    say("The plated dish. One row per application, ready for the policy checks and the risk model.")
    f = pd.read_csv(CURATED / "statement_features.csv")
    app_id = st.selectbox("Application", f.application_id.tolist(), index=1)
    st.dataframe(f[f.application_id == app_id].T.rename(columns=lambda _: "value"), use_container_width=True)
    st.bar_chart(f.set_index("application_id")["cheque_returns_12m"])
    st.caption("Returned cheques per application. Policy rule 4.1 allows at most two.")

# ----------------------------------------------------------------------------- 7
elif step == STEPS[6]:
    st.title("The first clue: Rohan and Rahul")
    say("On Monday Rohan looked safe and Rahul looked weak. The same cleaning step, run on their statements, shows a pattern. "
        "Neither is a verdict yet: the graph in Session 2 tells us who these names are.")
    from creditlens.scenarios.detect import load_statement, pass_through, round_trips, top_counterparties
    l, r = st.columns(2)
    with l:
        st.subheader("Rohan, A901")
        raw = pd.read_csv(SCN / "statements" / "A901_statement.csv")
        ravi = raw[raw["Narration"].str.contains("ravi", case=False)]
        spell = ravi["Narration"].str.extract(r"[-/]([^-/]*ravi[^-/]*)[-/]", expand=False, flags=2)
        st.write(f"**{len(ravi)} lines** mention Ravi Containers, written **{spell.nunique()} ways**:")
        st.dataframe(spell.value_counts().rename("lines"), use_container_width=True)
        a901 = load_statement(SCN / "statements" / "A901_statement.csv")
        rt = round_trips(a901)
        st.metric("Round trips", len(rt), help="Money out to one name and back within days")
        st.metric("Money out and back", f"{rt.out_amount.sum() / 1e7:.2f} crore")
        st.dataframe(top_counterparties(a901, 4).round(3), hide_index=True, use_container_width=True)
    with r:
        st.subheader("Rahul, A902")
        a902 = load_statement(SCN / "statements" / "A902_statement.csv")
        pt = pass_through(a902, {"Kamla Devi"})
        st.metric("Cash in, same amount out next day", len(pt))
        st.metric("Sent to", ", ".join(sorted(set(pt.sent_to))) or "-")
        st.dataframe(top_counterparties(a902, 4).round(3), hide_index=True, use_container_width=True)
    st.info("A flag is a lead for a person, not a verdict.")

# ----------------------------------------------------------------------------- 8
elif step == STEPS[7]:
    st.title("Ground truth: write the answers before the exam")
    say("Five humans read the same receipt before any model does. Where careful people disagree, the task is unclear, "
        "and no model will fix that. Their settled answers become the golden set.")
    golden = json.loads((REPO / "ground_truth/golden_set.json").read_text(encoding="utf-8"))
    rec = golden.get("receipts", [])
    names = [x["source_file"] for x in rec][:10]
    pick = st.selectbox("Receipt", names)
    l, r = st.columns([1, 1])
    img = REPO / "data/real/receipts" / pick
    with l:
        if img.exists():
            st.image(str(img), width=320)
        else:
            st.warning("Receipt image missing: run python -m creditlens.data_real --only receipts")
    with r:
        st.subheader("The agreed answer")
        st.json(next(x for x in rec if x["source_file"] == pick).get("entities", {}))
        st.caption(f"Golden set: {len(golden.get('financial_statements', []))} financial statements and {len(rec)} receipts.")

# ----------------------------------------------------------------------------- 9
elif step == STEPS[8]:
    st.title("The API: Priya clicks a button")
    say("Priya never runs Python. The API is the bank counter, and the token is her visitor badge: who she is, and which "
        "region she may see. Same request, different badge, different answer.")
    from fastapi.testclient import TestClient
    from creditlens.api.app import create_app
    client = TestClient(create_app(os.environ["DATABASE_URL"]))
    users = {"U01": "North analyst", "U02": "South analyst", "U05": "Head of Credit", "U06": "Auditor"}
    who = st.radio("Log in as", list(users), format_func=lambda u: f"{u}, {users[u]}", horizontal=True)
    r = client.post("/auth/token", json={"user_id": who})
    if r.status_code != 200:
        st.error(f"Login failed ({r.status_code}): {r.json().get('detail')}")
        st.stop()
    tok = r.json()["access_token"]
    st.code(f"POST /auth/token {{'user_id': '{who}'}}\n-> Bearer {tok[:48]}...", language="text")
    page = client.get("/applications", headers={"Authorization": f"Bearer {tok}"}).json()
    a, b = st.columns(2)
    a.metric("Applications visible", page["count"])
    b.metric("Filter applied", page["region_filter"])
    st.dataframe(pd.DataFrame(page["applications"]), hide_index=True, use_container_width=True)

# ----------------------------------------------------------------------------- 10
elif step == STEPS[9]:
    st.title("Reading documents: text first, squint only when we must")
    say("A text PDF carries its words. A scan is a photo, so we run OCR, and a rotated page is read four ways and the best "
        "one kept. Then the model fills in a strict form, every box the right type, or it is sent back.")
    from creditlens.extraction.extract import extract_financials
    from creditlens.extraction.ocr import pdf_to_text

    @st.cache_data(show_spinner="Reading the PDF...")
    def read(cid: str):
        import time
        t0 = time.time()
        text, ocr = pdf_to_text(RAW / "financials" / f"{cid}_financials.pdf")
        return text, ocr, time.time() - t0

    cid = st.radio("Document", ["C003", "C008"], horizontal=True,
                   format_func=lambda c: {"C003": "C003, a text PDF", "C008": "C008, a rotated scan (about 15 s)"}[c])
    text, ocr, secs = read(cid)
    a, b = st.columns(2)
    a.metric("OCR used", "yes" if ocr else "no"); b.metric("Seconds to read", f"{secs:.1f}")
    with st.expander("The text the model sees"):
        st.text(text[:1500])
    fin = extract_financials(cid)
    st.subheader(f"{fin.company_name}, {fin.unit}")
    st.dataframe(pd.DataFrame([y.model_dump() for y in fin.years]), hide_index=True, use_container_width=True)
    st.caption("Offline mode: the extractor returns the golden values, so the room needs no API key today.")

# ----------------------------------------------------------------------------- 11
elif step == STEPS[10]:
    st.title("The score: the first number on the board")
    say("This is the exam. We mark the extractor against the answer key the room wrote. The pass mark in our CI gate is 0.85.")
    res_path = REPO / "evals/results/extraction_accuracy.json"
    if st.button("Run the evaluation now (5 documents of each kind)", type="primary"):
        with st.spinner("Scoring..."):
            r = run(["-m", "evals.extraction_accuracy", "--limit", "5"])
        if r.returncode:
            st.error(r.stderr[-800:])
    if res_path.exists():
        res = json.loads(res_path.read_text(encoding="utf-8"))
        threshold = json.loads((REPO / "evals/thresholds.json").read_text())["extraction_overall"]
        overall = res["overall"]["accuracy"]
        a, b, c = st.columns(3)
        a.metric("Overall accuracy", f"{overall:.0%}", f"{overall - threshold:+.0%} vs pass mark")
        b.metric("Financial statements", f"{res['financial_statement']['overall']['accuracy']:.0%}")
        c.metric("Receipts", f"{res['receipt']['overall']['accuracy']:.0%}")
        (st.success if overall >= threshold else st.error)(
            f"{'Passes' if overall >= threshold else 'Fails'} the gate ({threshold:.0%}). Generated {res['generated_at']}.")
        for kind in ("financial_statement", "receipt"):
            pf = res[kind].get("per_field") or {}
            if pf:
                st.caption(f"Per-field accuracy, {kind.replace('_', ' ')}")
                st.bar_chart(pd.Series({k: v["accuracy"] for k, v in pf.items()}, name="accuracy"))
        st.caption("Receipts score lower on company and address: careful humans disagree there too. That is the lesson of the labelling party.")
    else:
        st.info("No score yet. Press the button.")

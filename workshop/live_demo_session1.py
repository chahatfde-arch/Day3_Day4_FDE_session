# %% [markdown]
# # Session 1 live demo, in slide order
# Open this file in VS Code and run one cell at a time (Shift+Enter, or "Run Cell" above each `# %%`).
# Each cell names the slide it goes with. Run the whole file once before the session.
#
# Terminal commands are written as comments at the top of the cell that needs them.
# Everything here runs offline. Set `LLM_OFFLINE=1` in your `.env` (or the environment) for the
# extraction cells so no API key is needed; the offline stub returns the golden values.

# %% [markdown]
# ## 0. Before the room fills up
# ```bash
# python workshop/get_ready.py 1          # ends with "Ready"
# python -m creditlens.generators         # data/raw: 30 applications, statements, financials, policy, registry
# python -m creditlens.pipeline --load    # data/clean, data/curated and the SQLite tables the API reads
# python -m creditlens.scenarios          # data/scenarios: Rohan, Rahul and the four bonus files
# ```

# %%
from pathlib import Path
import pandas as pd
from creditlens.config import settings

pd.set_option("display.width", 140); pd.set_option("display.max_colwidth", 70)
RAW, CLEAN, CURATED = settings.paths.raw, settings.paths.clean, settings.paths.curated
SCN = settings.paths.data / "scenarios"
print("raw:", sorted(p.name for p in RAW.iterdir()))
print("scenarios ready:", (SCN / "statements" / "A901_statement.csv").exists())

# %% [markdown]
# ## Slide "Three layers" and "Bingo": the raw statement, as the bank sent it
# Scroll slowly. Ask the room to shout out every kind of mess they see.

# %%
print(Path(RAW / "statements" / "A002_statement.csv").read_text(encoding="utf-8")[:1200])

# %% [markdown]
# ## Slide "Predict each line before it appears": the wrangling functions
# Say the input out loud, let the room predict, then run.

# %%
from datetime import date
from creditlens.pipeline.wrangle import parse_amount, parse_date, parse_narration, normalize_counterparty

window = (date(2025, 10, 1), date(2026, 9, 30))
for text in ["₹1,23,456.00", "(3,400.00)", "Rs. 42", ""]:
    print(f"parse_amount({text!r}) -> {parse_amount(text)}")
for text in ["03/10/2025", "03-Oct-2025", "2025-10-03", "03-10-25", "03.10.2025"]:
    print(f"parse_date({text!r}) -> {parse_date(text, window)}")
for text in ["UPI/TATA POWER/753217", "NEFT-AMAZON SELLER SERVICES-N9225097308", "CHQ RTN ARVIND TRADERS INSUFFICIENT FUNDS", "CASH DEPOSIT", "something odd"]:
    print(f"parse_narration({text!r}) -> {parse_narration(text)}")
for text in ["Amzn Seller Svcs Pvt Ltd.", "SHAKTI ENT PVT LTD", "Shakti Enterprises"]:
    print(f"normalize_counterparty({text!r}) -> {normalize_counterparty(text)!r}")

# %% [markdown]
# ## Slide "One row per file: the answer sheet for bingo"
# ```bash
# python -m creditlens.pipeline
# ```

# %%
q = pd.read_csv(CLEAN / "quality_report.csv")
print(q.head(8).to_string(index=False))
print("\nA002 row:"); print(q[q.source_file.str.startswith("A002")].T)

# %% [markdown]
# ## Slide "A contract is a promise": break one thing, read the refusal

# %%
from creditlens.pipeline.contracts import clean_transactions
import pandera as pa

df = pd.read_csv(CLEAN / "transactions.csv", parse_dates=["txn_date"])
print("valid rows:", len(clean_transactions.validate(df)))
broken = df.copy(); broken.loc[3, "amount"] = -broken.loc[3, "amount"]
try:
    clean_transactions.validate(broken, lazy=True)
except pa.errors.SchemaErrors as e:
    print(e.failure_cases[["check", "index", "failure_case"]].head(5).to_string(index=False))

# %% [markdown]
# ## Slide "Twelve months become one row" (curated features)

# %%
f = pd.read_csv(CURATED / "statement_features.csv")
print(f[f.application_id == "A002"].T)

# %% [markdown]
# ## Slide "The statements tell on themselves": Rohan and Rahul, part one
# First show the raw file and search for RAVI (48 lines, five spellings). Then run this.

# %%
from creditlens.scenarios.detect import load_statement, top_counterparties, round_trips, pass_through

a901 = load_statement(SCN / "statements" / "A901_statement.csv")
print(top_counterparties(a901, 4).round(3).to_string(index=False))
rt = round_trips(a901)
print(len(rt), "round trips, total out", round(rt.out_amount.sum() / 1e7, 2), "crore"); print(rt.head(3).to_string(index=False))

a902 = load_statement(SCN / "statements" / "A902_statement.csv")
print("\nRahul:"); print(top_counterparties(a902, 4).round(3).to_string(index=False))
pt = pass_through(a902, {"Kamla Devi"}); print(len(pt), "times cash in, same amount out to", set(pt.sent_to))

# %% [markdown]
# ## Slide "Waiting versus computing" (optional, 30 seconds)
# ```bash
# python -m creditlens.pipeline.ingest
# ```

# %% [markdown]
# ## Slide "Five documents, every pod member alone, then compare": the labelling party
# Pods label five receipts from data/real/receipts into their own CSVs. Then, per pod:
# ```bash
# python ground_truth/agreement.py labels_a.csv labels_b.csv --key file
# ```
# Then show the golden set entry for the receipt everyone labelled.

# %%
import json
golden = json.loads((Path("ground_truth") / "golden_set.json").read_text(encoding="utf-8"))
receipt = next(r for r in golden["receipts"] if "X00016469670" in r["source_file"])
print(json.dumps(receipt, indent=1))

# %% [markdown]
# ## Slide "Priya clicks a button": the API, RBAC and the audit trail
# Terminal 1 (the API reads the users table, so load the database first):
# ```bash
# python -m creditlens.pipeline --load
# python -m creditlens.api
# ```
# Terminal 2 (U01 sees North only; U05 sees everything; U06 can read but not write):
# ```bash
# python -c "from creditlens.api.auth import create_token; print(create_token('U01'))"
# curl -s -H "Authorization: Bearer <token>" http://127.0.0.1:8000/applications | head -c 400
# ```
# The slide "U01 sees North. U05 sees everything. U06 can read but not write" has the expected output.

# %% [markdown]
# ## Slide "OCR or a vision model?": a clean PDF and a rotated scan

# %%
from creditlens.extraction.ocr import pdf_to_text
import time
for cid in ("C003", "C008"):
    t0 = time.time(); text, used_ocr = pdf_to_text(RAW / "financials" / f"{cid}_financials.pdf")
    print(f"{cid}: ocr_used={used_ocr}, {time.time()-t0:.1f}s, {len(text)} chars"); print(text[:300].replace("\n", " | ")); print()

# %% [markdown]
# ## Slide "One function for every kind of document": extraction (offline stub)

# %%
import os; os.environ.setdefault("LLM_OFFLINE", "1")
from creditlens.extraction.extract import extract_financials, extract_receipt
print(extract_financials("C003").model_dump_json(indent=1)[:700])
print(extract_receipt(settings.paths.real / "receipts" / "X00016469670.jpg").model_dump_json(indent=1)[:500])

# %% [markdown]
# ## Slide "Your first number goes on the board": the evaluation
# ```bash
# LLM_OFFLINE=1 python -m evals.extraction_accuracy --limit 5
# ```
# Write each pod's overall score on the board. The gate threshold is in evals/thresholds.json (0.85).

# %%
print(json.dumps(json.loads(Path("evals/thresholds.json").read_text()), indent=1))

# %% [markdown]
# ## Homework slide
# ```bash
# python -m creditlens.pipeline.ingest      # the async ingestion lab
# python workshop/get_ready.py 2            # tomorrow's check: Neo4j and Langfuse keys optional
# ```

# %% [markdown]
# # Session 3, part two: Enterprise integrations
#
# **Where we are in the story.** The memo is drafted, but Good Bank's truth about Rohan does not live
# in our tidy SQLite file. His existing loans sit in a twenty-year-old Oracle core banking system.
# His repayment history comes from a SOAP web service that another team owns. And the credit manager
# who approves wants to do it from a phone, in Slack, not in a notebook.
#
# **The one idea of today: adapters.** Every legacy system gets a thin adapter that speaks its dialect
# (column names, XML, buttons) and returns our schema. Behind the adapter: timeouts, retries, a circuit
# breaker and idempotency, because other people's systems are slow and flaky on exactly the day you demo.
#
# This is a VS Code and Docker day in the plan. Oracle Database Free and the SOAP mock run in Docker on
# a laptop (`docker compose --profile legacy up`). In Colab, every adapter has a built-in fallback, so
# the lesson still runs: the fallback is itself an adapter.

# %%
import sys
if "google.colab" in sys.modules:   # Colab only. On a laptop this cell does nothing.
    !git clone https://github.com/chahatfde-arch/Day3_Day4_FDE_session.git
    %cd Day3_Day4_FDE_session
    !pip install -q -e ".[dev,llm,agents,integrations]"
    !python -m creditlens.generators --skip-pdfs
    !python -m creditlens.pipeline --load
else:
    print("Not on Colab: skipping the clone. The next cell finds your repo folder.")

# %% [markdown]
# **Laptop users:** skip the clone cell above. The next cell finds the repository root, so the notebook
# works whether you opened it from the repo folder or from `notebooks/`.

# %%
import os
from pathlib import Path
while not Path("pyproject.toml").exists() and Path.cwd() != Path.cwd().parent:
    os.chdir("..")
assert Path("pyproject.toml").exists(), "Open this notebook from inside the Day3_Day4_FDE_session folder."
print("working folder:", Path.cwd())

# %%
import os
os.environ.setdefault("LLM_OFFLINE", "1")

# %% [markdown]
# ## 1. Yesterday's bug
# Open yesterday's memo trace. The group-exposure tool was called with a `company_id` instead of an
# `application_id` in one of the earlier runs. Which schema should have caught it, and did it?

# %% [markdown]
# ## 2. Integration architecture (no code)
# Map every Good Bank system CreditLens must talk to, and for each: read or write? sync or async?
# what happens when it is down? Your map should have at least: SQL Server (applications), Oracle (core
# banking), the SOAP repayment service, Jira, Slack, the LLM providers, Langfuse.
#
# Why do banks keep legacy systems? Because they work, they are audited, and replacing them is a
# multi-year programme. An FDE integrates with them; an FDE does not replace them.

# %% [markdown]
# ## 3. Oracle core banking
# `CoreBankingAdapter` reads existing exposure. With `ORACLE_DSN` set it uses python-oracledb against
# Oracle Database Free (Docker). Without it, a SQLite table with **Oracle's column names** stands in.
#
# **Curveball card: Oracle's column names differ.** The core system calls things `CUST_ID`,
# `OUTSTANDING_AMT`, `PRODUCT_CD`. Our schema says `company_id`, `outstanding_inr_000`, `product`.
# The adapter holds the mapping in one dictionary. Change the mapping, not the agent.

# %%
from creditlens.integrations.oracle import COLUMN_MAP, CoreBankingAdapter
cb = CoreBankingAdapter()
print("backend:", cb.backend)
print("column map:", COLUMN_MAP)
print(cb.get_exposure("C012"))
print(cb.get_exposure("C001"))

# %% [markdown]
# The agent tool `get_core_banking_exposure` wraps the adapter and is read-only: the policy says the
# agent may never write to core banking, and the database user it connects with cannot either.

# %% [markdown]
# ## 4. Legacy SOAP service
# SOAP is XML over HTTP with a contract file (WSDL). `zeep` reads the WSDL and generates Python calls.
# Start the mock service in a terminal (laptop) or in the background here:
#
# ```bash
# python -m creditlens.integrations.soap_server --port 8001
# ```
#
# Then call it and parse the XML into Pydantic models.

# %%
import subprocess, time
soap = subprocess.Popen(["python", "-m", "creditlens.integrations.soap_server", "--port", "8001"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(3)
from creditlens.integrations.soap import RepaymentClient
client = RepaymentClient(wsdl="http://127.0.0.1:8001/?wsdl", timeout=5)
hist = client.get_history("C012")
print("source:", hist.source, "| instalments:", len(hist.entries))
for i in hist.entries[:3]:
    print(" ", i.model_dump())
soap.terminate()

# %% [markdown]
# **Curveball card: the service is slow and flaky today.** Restart it with `--slow 3 --flaky 0.5`
# (3-second responses, half of them failing). Predict what happens with a 2-second timeout and no retries.
# Then watch the client's retries with backoff and the circuit breaker.
#
# A circuit breaker has three states. **Closed**: calls go through. After N failures it **opens**:
# calls fail instantly without touching the sick service, which gives it time to recover. After a
# cool-down it goes **half-open**: one trial call decides whether to close again.

# %%
soap = subprocess.Popen(["python", "-m", "creditlens.integrations.soap_server", "--port", "8001", "--slow", "3", "--flaky", "0.5"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(3)
client = RepaymentClient(wsdl="http://127.0.0.1:8001/?wsdl", timeout=2, retries=3)
for n in range(5):
    t0 = time.time()
    try:
        h = client.get_history("C012")
        print(f"call {n}: ok ({h.source}) in {time.time() - t0:.1f}s, breaker={client.breaker.state}")
    except Exception as e:
        print(f"call {n}: {type(e).__name__} in {time.time() - t0:.1f}s, breaker={client.breaker.state}")
soap.terminate()

# %% [markdown]
# ## 5. Slack approvals
# The credit manager approves from a phone. A Slack app built with Bolt for Python runs in **Socket Mode**,
# which means Slack connects out to your laptop over a websocket: no public URL, no tunnel. The app posts
# each draft memo to `#credit-approvals` with Approve and Reject buttons. A click calls the same
# `resume()` you used yesterday and lands in the audit trail with `channel="slack"`.
#
# Setup (one time, free workspace): create an app at api.slack.com, enable Socket Mode, add the
# `chat:write` and `channels:read` scopes, install it to the workspace, and put the bot token
# (`xoxb-...`) and app token (`xapp-...`) in `.env`. Full steps in `docs/integrations.md`.
#
# Without Slack, `--dry-run` prints the exact Block Kit message, and the CLI approval covers the flow.

# %%
!python -m creditlens.integrations.slack --dry-run --application A001

# %% [markdown]
# ## 6. End-to-end run
# One application flows through Oracle, SOAP, the agents and the approval. Follow it in Langfuse:
# one trace, with a span per tool. Where did the time go? Which call would you cache?

# %%
from creditlens.agents.run import draft_memo, resume
from creditlens.schemas import User, ApprovalDecision
head = User(user_id="U05", full_name="Head of Credit", role="head_of_credit", region="ALL")
r = draft_memo("A012", head)
print(r["memo"].related_parties)
print([t["name"] for t in r["tool_calls"]])
print(resume(r["thread_id"], ApprovalDecision(application_id="A012", decided_by="U05", decision="reject",
                                               comment="Exposure above limit", channel="cli"))["status"])

# %%
from creditlens.api.audit import list_events
for e in list_events("A012")[-5:]:
    print(e.timestamp, e.actor, e.action, e.detail[:60])

# %% [markdown]
# ## Exit ticket
# One thing learned, one open question. Tomorrow: the risk model gets real, and we deploy.

# %% [markdown]
# # Session 4, part one: connect the Underwriter Agent to the bank
#
# **The story so far.** Yesterday we built Amaira's Underwriter Agent (notebook 04b). It prepares every
# review and stops for her signature. She likes it, and asks the question every client asks next:
# *"When can it use our real systems?"* Because so far its sensors read a tidy copy of the data.
#
# Good Bank's truth lives elsewhere. Existing loans sit in a twenty-year-old **Oracle core banking**
# system. Repayment history comes from a **SOAP service** another team owns. Exceptions must become
# **Jira** tickets. And Amaira wants to sign from her phone, in **Slack**.
#
# **The forward deployed engineer's job today: connect, without breaking anything.** Other people's systems
# are slow, flaky, and named differently from ours, on exactly the day you demo. We go from zero to hero:
#
# | Level | What we add | Why |
# |---|---|---|
# | 0 | An integration map | Know every system before touching one |
# | 1 | A naive call | See what goes wrong |
# | 2 | A timeout | Never wait forever |
# | 3 | Retries with backoff | Ride out a blip |
# | 4 | A circuit breaker | Stop hammering a sick service |
# | 5 | A labelled fallback | Keep working, and never pretend |
# | 6 | An adapter | Their names, our schema |
# | 7 | Idempotent writes | A retry must not file twice |
# | 8 | A new channel | Amaira signs from Slack, through the same gate |
# | 9 | Two new sensors | The agent reads the real systems, and says where each fact came from |
# | 10 | Integration tests | Prove it, before anyone relies on it |
#
# Runs offline. Every legacy system has a stand-in, so the lessons run on any laptop or in Colab.

# %%
import sys
if "google.colab" in sys.modules:   # Colab only. On a laptop this cell does nothing.
    !git clone https://github.com/chahatfde-arch/Day3_Day4_FDE_session.git
    %cd Day3_Day4_FDE_session
    !pip install -q -e ".[dev,llm,agents,integrations]"
    !python -m creditlens.generators
    !python -m creditlens.pipeline --load
    !python -m creditlens.scenarios
    !python -m creditlens.scenarios.story build
else:
    print("Not on Colab: skipping the clone. You are already in the repo folder.")

# %%
import os
from pathlib import Path
while not Path("pyproject.toml").exists() and Path.cwd() != Path.cwd().parent:
    os.chdir("..")
assert Path("pyproject.toml").exists(), "Open this notebook from inside the repo folder."
print("working folder:", Path.cwd())

# %%
import os
os.environ["LLM_OFFLINE"] = "1"   # today we run offline: same numbers for everyone, no API key needed
os.environ["CREDITLENS_JIRA_PATH"] = str(Path("data/cache/jira_session4.json"))   # a practice Jira, not the real one
os.environ["JIRA_BASE_URL"] = ""   # always the practice Jira here, even if .env points at a Jira service
print("offline mode:", os.environ["LLM_OFFLINE"] == "1")

# %% [markdown]
# ## Level 0. The integration map, before any code
# For every system: what do we do with it (read or write), who owns it, and **what must happen when it is
# down**? Read-only systems get a fallback that says it is a fallback. Systems we write to get idempotency.
# Nothing optional may break the product.

# %%
INTEGRATION_MAP = [
    # system,                   read/write, owner,                 when it is down
    ("Applications database",   "read",  "our team",             "no review; fail loudly"),
    ("Oracle core banking",     "read",  "core banking team",    "use the stand-in, and say so in the note"),
    ("SOAP repayment service",  "read",  "payments team",        "retries, breaker, then a labelled mock"),
    ("Jira (through MCP)",      "write", "compliance",           "queue the ticket; never file twice"),
    ("Slack",                   "write", "Amaira's channel",     "she signs from the command line instead"),
    ("LLM provider",            "call",  "an outside company",   "next provider, then the cache, then offline"),
]
print(f"{'system':26s}{'r/w':7s}{'owner':22s}when it is down")
for row in INTEGRATION_MAP:
    print(f"{row[0]:26s}{row[1]:7s}{row[2]:22s}{row[3]}")

# %% [markdown]
# ## Levels 1 to 3. A naive call, a timeout, retries
# We need a slow, flaky service to practise on. Good Bank's SOAP service is exactly that, so here is a
# small stand-in that behaves the same way: it answers in half a second, and fails half the time. Its dice
# are seeded, so everyone in the room sees the same run.

# %%
import random
import time


class FlakyService:
    """A stand-in for another team's service: sometimes slow, often failing."""
    def __init__(self, delay=0.5, fail_rate=0.5, seed=7):
        self.delay, self.fail_rate, self.rng, self.calls = delay, fail_rate, random.Random(seed), 0

    def history(self, customer_id, timeout=None):
        self.calls += 1
        if timeout is not None and self.delay > timeout:
            time.sleep(timeout)
            raise TimeoutError(f"no answer within {timeout}s")
        time.sleep(self.delay)
        if self.rng.random() < self.fail_rate:
            raise ConnectionError("service error 503")
        return {"customer_id": customer_id, "instalments_paid_on_time": 11, "of": 12, "source": "live"}


service = FlakyService()
for attempt in range(3):                       # Level 1: a naive call, no protection at all
    try:
        print("naive call:", service.history("C029"))
    except Exception as err:
        print("naive call failed:", type(err).__name__, err)

# %%
slow = FlakyService(delay=3.0, fail_rate=0.0)   # Level 2: a slow day. Without a timeout, we would wait 3 s per call
t0 = time.time()
try:
    slow.history("C029", timeout=1.0)
except TimeoutError as err:
    print(f"timeout after {time.time() - t0:.1f}s:", err)


def with_retries(fn, attempts=3, backoff=0.2):  # Level 3: retries with exponential backoff
    for n in range(1, attempts + 1):
        try:
            return fn(), n
        except (ConnectionError, TimeoutError) as err:
            if n == attempts:
                raise
            time.sleep(backoff * 2 ** (n - 1))  # wait 0.2 s, then 0.4 s: give the service room


service = FlakyService(seed=7)
result, attempts = with_retries(lambda: service.history("C029"))
print(f"with retries: answered on attempt {attempts} ->", result)

# %% [markdown]
# ## Level 4. A circuit breaker
# Retries help with a blip, and make a real outage worse: every caller retries, and the sick service gets
# hammered. A **circuit breaker** stops that. **Closed**: calls go through and failures are counted. After
# three failures it **opens**: calls fail instantly, without touching the service. After a cool-down it
# goes **half-open**: one trial call decides whether to close again. We use the project's own breaker, with
# a pretend clock so we do not have to wait.

# %%
from creditlens.integrations.soap import CircuitBreaker, CircuitOpen

clock = {"now": 0.0}
breaker = CircuitBreaker(failures=3, reset_after=30.0, clock=lambda: clock["now"])
down = FlakyService(delay=0.0, fail_rate=1.0)        # the service is fully down
for n in range(5):
    try:
        breaker.call(down.history, "C029")
    except CircuitOpen as err:
        print(f"call {n}: refused instantly ({breaker.state}) - {err}")
    except ConnectionError:
        print(f"call {n}: failed, breaker now {breaker.state} after {breaker.failure_count} failures")
print("service was actually called", down.calls, "times, not 5")
clock["now"] = 31.0                                   # thirty-one seconds later
print("after the cool-down the breaker is", breaker.state)
up = FlakyService(delay=0.0, fail_rate=0.0)           # the service has recovered
print("trial call:", breaker.call(up.history, "C029")["source"], "-> breaker", breaker.state)

# %% [markdown]
# ## Level 5. A labelled fallback, with the real client
# When everything fails, the review should still go ahead, but the note must **never pretend**. The
# project's `RepaymentClient` does all four levels: timeout, retries, breaker, and a fallback history
# clearly marked `source="mock"`. Point it at a service that is not running and watch.

# %%
from creditlens.integrations.soap import RepaymentClient

client = RepaymentClient(wsdl="http://127.0.0.1:8999/?wsdl", timeout=1, retries=2)   # nothing listens there
for n in range(4):
    t0 = time.time()
    h = client.get_history("C029")
    print(f"call {n}: source={h.source:5s} instalments={len(h.entries)} in {time.time() - t0:.1f}s, breaker={client.breaker.state}")

# %% [markdown]
# The first call spends its retries and falls back. By the second, the breaker is open, so the fallback is
# **instant**. Every answer says `source=mock`. The agent must carry that label into the review note.
#
# Now the same client, pointed at the real address (`SOAP_WSDL_URL`, default `localhost:8001`). With
# Docker running (`docker compose -f infra/docker-compose.yml --profile legacy up -d soap-mock`) it answers
# `source=soap`. Without Docker nothing listens there, so it falls back to `mock`, and says so.

# %%
live = RepaymentClient(timeout=5, retries=1)   # one attempt: without Docker, Windows takes a few seconds to refuse
t0 = time.time()
h = live.get_history("C029")
print(f"live call: source={h.source:5s} instalments={len(h.entries)} late={h.late_count} in {time.time() - t0:.1f}s  ({live.wsdl})")

# %% [markdown]
# ## Level 6. An adapter: their names, our schema
# The core banking table was designed long before CreditLens. It calls things `CUST_ID` and
# `OUTSTANDING_AMT`; our schema says `company_id` and `outstanding_inr_000`. The whole translation lives in
# one dictionary. When their names change, you change the dictionary, not the agent. With `ORACLE_DSN`
# set it talks to a real Oracle database; without it, a SQLite table with **Oracle's own column names**
# stands in, so the same SQL and the same mapping run.

# %%
from creditlens.integrations.oracle import COLUMN_MAP, CoreBankingAdapter

core = CoreBankingAdapter()
print("backend:", core.backend)
print("column map:", COLUMN_MAP)
for cid in ("C001", "C003", "C029"):
    print(core.get_exposure(cid))

if core.backend == "oracle":   # with Docker: does the real system agree with our copy?
    copy = CoreBankingAdapter(dsn="")
    print("\nreal system vs our copy:")
    for cid in ("C001", "C003"):
        a, b = core.get_exposure(cid), copy.get_exposure(cid)
        same = (a["outstanding_inr_000"], a["product"]) == (b["outstanding_inr_000"], b["product"])
        print(f"  {cid}: oracle {a['outstanding_inr_000']:>6} {a['product']!s:5s} | copy {b['outstanding_inr_000']:>6} {b['product']!s:5s}"
              + ("" if same else "   <- they disagree"))

# %% [markdown]
# With Docker, the last lines compare the real Oracle table with the copy we built from the registry. Where
# they disagree, **the system of record wins**, and the difference is a question for the core banking
# team, not something to paper over in the adapter.

# %% [markdown]
# ## Level 7. Idempotent writes, through MCP
# Reading twice is harmless. **Writing twice is not.** Agents retry and graphs resume, so every write needs
# a key that makes the second attempt do nothing. Ours is `application:rule`.
#
# Jira sits behind an **MCP server** (Model Context Protocol). An MCP server is a small program that offers
# typed tools over a standard protocol, so any agent, or Claude Desktop, can use it without custom code.
# Our client starts it as a child process and speaks JSON-RPC over its standard input and output: the
# **stdio** transport. First the handshake: ask the server who it is and which tools it offers.

# %%
from creditlens.mcp_jira.client import file_ticket, idempotency_key, inspect_server

server = inspect_server()      # start the server, shake hands, list its tools, stop it
print(f"server {server['server']} {server['version']} · protocol {server['protocol_version']} · started with: {server['command']}")
print("instructions:", server["instructions"][:96], "...")
for tool in server["tools"]:
    print(f"  tool {tool['name']:<14} {', '.join(tool['arguments'])}")
print("no delete tool: the agent cannot remove a ticket, whatever its prompt says")

# %% [markdown]
# Now file Rohan's exception ticket twice, **through the protocol**, into a practice Jira file. Each call
# starts the server, calls its `create_issue` tool and stops it: about a second each.

# %%
Path(os.environ["CREDITLENS_JIRA_PATH"]).unlink(missing_ok=True)
for attempt in (1, 2):
    t = file_ticket("A901", "5.2", summary="Group exposure over the rule 5.2 limit", transport="stdio")
    print(f"attempt {attempt}: key={t['key']} created={t.get('created')} idempotency_key={idempotency_key('A901', '5.2')} transport={t['transport']}")

# %% [markdown]
# `transport=stdio` means the ticket went through the MCP server. If the server cannot start, the client
# files the ticket in process and says so (`transport=inprocess`, with a `transport_note`): the agent never
# fails because a pipe did. The workbench files its tickets the same way, and Claude Desktop could start
# the same server with `python -m creditlens.mcp_jira`.

# %% [markdown]
# ### Under the hood: MCP end to end
# What `file_ticket` just did, step by step. Nothing here is magic: MCP is **JSON messages, one per line, over a pipe**.
#
# | Step | What happens | Where it lives |
# |---|---|---|
# | 1 | The agent finds a failed rule and calls `file_ticket("A901", "5.2")` | `src/creditlens/mcp_jira/client.py`, `file_ticket` |
# | 2 | The client starts the server as a child process: `python -m creditlens.mcp_jira` | `client.py`, `_server_command`; the server starts in `src/creditlens/mcp_jira/__main__.py` |
# | 3 | `initialize`: the handshake. Both sides agree a protocol version; the server says its name and its instructions | the MCP Python SDK (`pip install mcp`), on both sides |
# | 4 | `tools/list`: the server describes its three tools and the arguments each takes | `src/creditlens/mcp_jira/server.py`, the `@server.tool` functions |
# | 5 | `tools/call create_issue`: the client sends the arguments, the server runs the tool | `server.py`, then `get_backend()` in `backend.py` |
# | 6 | The backend writes the ticket: a practice file here; the local Jira service or Jira Cloud when `.env` says so | `src/creditlens/mcp_jira/backend.py`: `MockJira`, `JiraCloud` |
# | 7 | The reply travels back; the client adds `transport=stdio` and stops the server | `client.py`, `call_tool_over_stdio` |
#
# The next cell plays steps 2 to 7 by hand, with no SDK on our side, so you can see every message.

# %%
import json
import subprocess
import sys
import tempfile

# A fresh practice file for this demo only, so the first call really creates a ticket.
wire_env = dict(os.environ, CREDITLENS_JIRA_PATH=str(Path(tempfile.mkdtemp()) / "jira_wire.json"))
server = subprocess.Popen([sys.executable, "-m", "creditlens.mcp_jira"],      # step 2: exactly what file_ticket starts
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          text=True, encoding="utf-8", env=wire_env)


def send(message, show=None):
    """Write one JSON message to the server's standard input, and read its reply from its standard output."""
    name = message["method"] + (" " + message["params"]["name"] if message["method"] == "tools/call" else "")
    print(f"client -> server   {name}")
    server.stdin.write(json.dumps(message) + "\n")
    server.stdin.flush()
    if "id" not in message:
        print("                   (a notification: no reply)")
        return None
    reply = json.loads(server.stdout.readline())
    print(f"server -> client   {show(reply['result'])}")
    return reply


send({"jsonrpc": "2.0", "id": 1, "method": "initialize",                       # step 3
      "params": {"protocolVersion": "2026-07-28", "capabilities": {}, "clientInfo": {"name": "notebook-05b", "version": "1.0"}}},
     lambda r: f"I am {r['serverInfo']['name']} {r['serverInfo']['version']}; we speak protocol {r['protocolVersion']}")
send({"jsonrpc": "2.0", "method": "notifications/initialized"})
send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"},                      # step 4
     lambda r: "my tools: " + "; ".join(f"{t['name']}({', '.join(t['inputSchema']['properties'])})" for t in r["tools"]))
ticket = {"project_key": "CREDIT", "summary": "Group exposure over the rule 5.2 limit", "idempotency_key": "A901:5.2"}
for n in (3, 4):                                                               # steps 5 to 7, twice
    send({"jsonrpc": "2.0", "id": n, "method": "tools/call", "params": {"name": "create_issue", "arguments": ticket}},
         lambda r: (lambda t: f"{t['key']} created={t['created']} backend={t['backend']}")(json.loads(r["content"][0]["text"])))
server.stdin.close()
server.wait(timeout=10)

print("\nOne message exactly as it travels, a single line of JSON:")
print(json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "create_issue", "arguments": ticket}}))

# %% [markdown]
# **What you saw.** A handshake, a menu of three tools, then the same `create_issue` call twice: `created=True`,
# then `created=False`. The idempotency key is checked in the backend, so it holds whichever client calls. Claude
# Desktop or another team's agent could send these exact lines and get the same answers. One detail: this hand-written client uses the classic handshake, `initialize`, so the server answers with the
# newest classic version, 2025-11-25. The SDK client in the handshake cell above first asks `server/discover`, the newer
# way from the 2026-07-28 revision, which is why it printed 2026-07-28. Same server, two ways in.
#
# **Why MCP, and the other options.** We could call Jira's REST API straight from the agent: simpler at first, but
# every agent then re-implements the login, the retries and the rule that nobody may delete a ticket. With MCP, those
# live once, in one small server, and every agent reuses them. The other transport, **Streamable HTTP**, runs the same
# server somewhere else, so many agents share one copy. stdio is the right choice for one laptop.

# %% [markdown]
# ## Level 8. A new channel: Amaira signs from Slack
# Slack connects out to our laptop over a websocket (Socket Mode), so no public address is needed. The
# message carries the proposal and two buttons, and a click calls **the same signing path** as the
# notebook: one gate, many channels. Without a Slack workspace we print the exact message.

# %%
import json
from creditlens.integrations.slack import build_blocks
from creditlens.scenarios.story import draft_story_memo

rohan = draft_story_memo("A901", "U05")
blocks = build_blocks(rohan["thread_id"], rohan["memo"])
print(json.dumps(blocks, indent=1)[:1100], "\n...")
print("buttons:", [e["text"]["text"] for b in blocks if b["type"] == "actions" for e in b["elements"]])

# %% [markdown]
# ## Level 9. Give the Underwriter Agent two new sensors
# The agent from notebook 04b gains two readings from the real systems: **core banking exposure** and
# **repayment history**. Every fact now carries its **source**, and anything that came from a stand-in or
# a mock becomes a line in the review note. A note that hides where its numbers came from is not
# auditable.

# %%
from creditlens.agents import tools

T = tools.get_tools()


def perceive_live(app):
    form = T["get_application"].invoke({"application_id": app})["application"]
    cid = form["company_id"]
    exposure = core.get_exposure(cid)
    history = live.get_history(cid)          # the real address: soap with Docker, mock without
    facts = {
        "core_banking_outstanding_inr_000": (exposure["outstanding_inr_000"], exposure["source"]),
        "repayment_on_time": (sum(1 for e in history.entries if e.days_late == 0), history.source),
        "repayment_instalments": (len(history.entries), history.source),
    }
    caveats = [f"{name} came from '{src}', not the live system" for name, (_, src) in facts.items()
               if src in ("mock", "sqlite-fallback")]
    return facts, caveats


facts, caveats = perceive_live("A020")
for name, (value, src) in facts.items():
    print(f"{name:34s} {value!s:>8}   source: {src}")
print("\nlines added to the review note:")
for c in caveats:
    print("  -", c)
if not caveats:
    print("  none: every fact came from a live system")

# %% [markdown]
# ## Level 10. Integration tests
# Prove the behaviour, so the next person who changes the code finds out at once if they broke it.

# %%
checks = {
    "the adapter returns our schema, not Oracle's": set(core.get_exposure("C001")) >= {"company_id", "outstanding_inr_000", "source"},
    "a dead service still answers, quickly, labelled as mock": (lambda t0: (client.get_history("C001").source == "mock") and time.time() - t0 < 2)(time.time()),
    "the breaker is open after repeated failures": client.breaker.state == "open",
    "a second ticket write files nothing new": file_ticket("A901", "5.2", transport="inprocess")["key"] == t["key"],
    "every fact carries a source": all(isinstance(v, tuple) and len(v) == 2 for v in facts.values()),
}
for name, ok in checks.items():
    print(("PASS " if ok else "FAIL ") + name)

# %% [markdown]
# ## From zero to hero: connecting to someone else's system
# 1. Draw the integration map first: read or write, owner, and what happens when it is down.
# 2. Never call without a timeout.
# 3. Retry a blip, with backoff. Never retry a write without an idempotency key.
# 4. Put a circuit breaker in front of anything you do not own.
# 5. Fall back, and label the fallback. Carry the label into the output.
# 6. Translate in an adapter, with the mapping as data.
# 7. One gate, many channels: Slack, the command line and the notebook all sign through the same path.
# 8. Test the failure paths, not just the happy path.
#
# ## Exit ticket
# One thing learned, one open question. Next: notebook 06b, securing it and shipping it.

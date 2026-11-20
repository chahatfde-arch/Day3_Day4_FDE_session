# %% [markdown]
# # Session 3: Agents that draft the credit memo
#
# **Where we are in the story.** Priya has the numbers (Session 1) and can find any rule or related company
# (Session 2). Rohan's group of four and Rahul's family firm are on the case board, as leads. Today the
# system does what she does every day: check the application against every rule,
# look up the group, score the risk, write the memo, raise a Jira ticket for each exception, and then
# **stop and wait for a human**. The underwriter decides. Always.
#
# **The one idea of today: tools, not prompts, do the work.** The LLM never computes a ratio and never
# decides approval. Code computes the checks; the LLM turns structured facts into readable prose and
# calls tools with strict schemas. That is what makes the result auditable.
#
# Topics: AG1 to AG12 (agents), L11 to L15 (tool calling), O1, O3 to O6 (checkpoints, interrupts,
# reliability), O9 to O11 (MCP), OB1 (tracing), OG7.
#
# Day 4 is a VS Code day in the workshop plan; this notebook mirrors it so Colab-only participants
# can follow. No GPU needed.

# %%
import sys
if "google.colab" in sys.modules:   # Colab only. On a laptop this cell does nothing.
    !git clone https://github.com/chahatfde-arch/Day3_Day4_FDE_session.git
    %cd Day3_Day4_FDE_session
    !pip install -q -e ".[dev,llm,retrieval,agents,extraction]"
    !python -m creditlens.generators
    !python -m creditlens.pipeline --load
    !python -m creditlens.scenarios.story build
else:
    print("Not on Colab: skipping the clone. You are already in the repo folder.")

# %% [markdown]
# **Laptop users:** skip the clone cell above. The next cell finds the repository root, so the notebook
# works whether you opened it from the repo folder or from `notebooks/`.

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
print("offline mode:", os.environ["LLM_OFFLINE"] == "1")

# %% [markdown]
# ## 1. Last session's bug
# Run `python -m evals.retrieval_eval --k 5 --show-misses` (or open a Langfuse trace from Session 2). The
# three questions hybrid search missed were all Good Bank questions beaten by the SBA rulebook. Which
# missing argument would have fixed them? (Hint: `source`.)

# %% [markdown]
# ## 2. Agent architecture, on paper (AG1 to AG4)
# **On paper, on your own (five minutes).** You are the *supervisor*. Six specialists work for you: financial
# analyst, banking conduct, related party, policy checker, risk scorer, memo writer. You may only pass a
# note to one specialist at a time, and each specialist may only answer the note they get.
#
# Write the notes you would send, in order, to draft a memo for A001. Then answer two questions:
# what did you need to *know* to decide who goes next, and what happens when two specialists need the same
# fact? Your answer to both is **state**, and that is what LangGraph manages for us.

# %% [markdown]
# ## 3. Tool calling (L11 to L15)
# A tool is a Python function with a Pydantic argument schema. The model sees the schema, not the code.
# Three things to notice in `creditlens.agents.tools`:
#
# 1. **Strict schemas.** A hallucinated argument fails validation before any code runs.
# 2. **Safe Text-to-SQL.** `run_sql` accepts SELECT only, from allow-listed tables, with LIMIT enforced,
#    and the current user's region filter injected. The model cannot escape its region even if it tries.
# 3. **Parallel calls.** Independent tools (financials, statement features, group) can run together.

# %%
from creditlens.agents import tools
for t in tools.tool_list():
    print(f"{t.name:28s} {t.description[:80]}")

# %%
import json
run_sql = tools.get_tools()["run_sql"]
print(json.dumps(run_sql.args_schema.model_json_schema(), indent=1))

# %% [markdown]
# **The hallucinated-tool curveball.** Ask for a tool that does not exist and watch the schema reject it.
# Then try an UPDATE through `run_sql`.

# %%
from creditlens.agents.tools import set_current_user
from creditlens.schemas import User
set_current_user(User(user_id="U01", full_name="North analyst", role="analyst", region="North"))
print(run_sql.invoke({"query": "SELECT application_id, region FROM applications LIMIT 5"}))
print(run_sql.invoke({"query": "UPDATE applications SET requested_inr_000 = 1"}))

# %%
from creditlens.agents.tools import text_to_sql
print(text_to_sql("How many applications are there per product?", user=None))

# %% [markdown]
# ## 4. Multi-agent supervisor (AG5 to AG12)
# `build_graph()` returns a LangGraph with:
#
# - a **supervisor** that routes deterministically on what is still missing in the state;
# - **specialists** that each fill one part of the state;
# - a **map-reduce policy checker**: `Send()` fans out one branch per rule group and a **reducer**
#   (`Annotated[list, operator.add]`) merges the results, so adding a rule never changes the graph;
# - a **nested memo subgraph**: draft sections, self-review, final;
# - an **interrupt** before the memo is final, so the underwriter approves or rejects;
# - **loop detection**: a step budget and a repeated-call detector.
#
# Draw it first.

# %%
from creditlens.agents.graph import build_graph
graph = build_graph()

# Picture it: the agent graph, drawn offline (no internet needed)
import matplotlib.pyplot as plt
import networkx as nx

g = graph.get_graph()
G = nx.DiGraph()
G.add_edges_from((e.source, e.target) for e in g.edges)
layers = {n: d for n, d in nx.shortest_path_length(G, "__start__").items()} if "__start__" in G else {}
for n in G:
    G.nodes[n]["layer"] = layers.get(n, max(layers.values(), default=0) + 1)
pos = nx.multipartite_layout(G, subset_key="layer", align="horizontal")
pos = {n: (x, -y) for n, (x, y) in pos.items()}
colour = {n: ("#eb6834" if n == "supervisor" else "#1baf7a" if n.startswith("__") else "#2a78d6") for n in G}
fig, ax = plt.subplots(figsize=(14, 7))
nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#999999", arrows=True, arrowsize=12,
                       connectionstyle="arc3,rad=0.08", node_size=1800)
nx.draw_networkx_nodes(G, pos, ax=ax, node_color=[colour[n] for n in G], node_size=1800, edgecolors="white", linewidths=2)
for n, (x, y) in pos.items():
    ax.text(x, y - 0.06, n.strip("_"), ha="center", va="top", fontsize=9, color="#222222")
ax.set_title("The agent graph: the supervisor (orange) routes work to specialists (blue)", loc="left", fontsize=12)
ax.axis("off"); ax.margins(0.08)
plt.tight_layout()
plt.show()
print("nodes:", sorted(n for n in G if not n.startswith("__")))

# %% [markdown]
# ### Deterministic checks first
# Before any LLM is involved, the policy checks are plain code. Compare with the answer key: every
# failed rule must match, or the agent is building prose on wrong facts.

# %%
from creditlens.agents.policy_checks import run_checks
checks = run_checks("A001")
for c in checks:
    print(f"{'FAIL' if not c.passed else 'ok  '} {c.rule_id:4s} measured={c.measured_value} threshold={c.threshold}  {c.evidence[:60]}")
expected = json.load(open("data/answer_key/expected_policy_flags.json"))["A001"]["flags"]  # instructor only
print("expected failures:", expected)

# %% [markdown]
# ### Run to the approval interrupt (O1, O3 to O6)
# `draft_memo` runs the whole graph until the underwriter interrupt and returns the draft, the thread id
# (which is the checkpoint key) and the tool-call trajectory. Every state change is checkpointed, so a
# crash after the risk score does not redo the extraction.

# %%
from creditlens.agents.run import draft_memo, resume
from creditlens.schemas import User, ApprovalDecision
head = User(user_id="U05", full_name="Head of Credit", role="head_of_credit", region="ALL")
result = draft_memo("A001", head)
print("status:", result["status"], "| thread:", result["thread_id"])
print(result["memo"].to_markdown())

# %%
print("tool calls in order:")
for t in result["tool_calls"]:
    print(" ", t["order"], t["name"])

# %% [markdown]
# **Underwriter role-play.** You are the underwriter. Read the memo slowly. Is every number traceable to a check? Is there approval language the memo must not contain? Then decide.

# %%
final = resume(result["thread_id"], ApprovalDecision(application_id="A001", decided_by="U05", decision="approve",
                                                     comment="Group exposure explained in the field visit."))
print(final["status"])

# %% [markdown]
# ### Kill and resume
# Checkpoints mean we can stop a run mid-way and continue later, even in another process. The CLI uses
# a SQLite checkpointer so two commands share the state:
#
# ```bash
# python -m creditlens.agents.run A002 --user U05        # runs to the interrupt, prints the thread id
# python -m creditlens.agents.approve <thread_id> reject --user U05
# ```

# %% [markdown]
# **Curveball card: an infinite loop.** `inject_loop=True` makes the supervisor misroute. Predict: how
# many steps before the guard trips, and what does the memo say?

# %%
try:
    r = draft_memo("A001", head, inject_loop=True)
    print(r["status"], "|", str(r.get("error", ""))[:200])
except Exception as e:
    print("stopped:", type(e).__name__, str(e)[:200])

# %% [markdown]
# ## 5. MCP and Jira (O9 to O11)
# Policy 7.2: every exception is a Jira ticket *before* the memo reaches the underwriter.
# The Jira integration is an **MCP server**: a small process that exposes `create_issue`, `get_issue`
# and `search_issues` as tools over a standard protocol, so any agent framework can use it, and the
# agent's access is scoped to exactly those three operations. Set `JIRA_BASE_URL`, `JIRA_EMAIL` and
# `JIRA_API_TOKEN` for a real Jira Cloud Free site; without them, a mock Jira persists to
# `data/cache/jira.json`.
#
# **Curveball card: a duplicate ticket.** Run the memo twice. Without an idempotency key, compliance
# gets two tickets for one exception. With it, the second call returns the first ticket.

# %%
from creditlens.mcp_jira.client import search_tickets
before = len(search_tickets("A001"))
r2 = draft_memo("A001", head)
after = len(search_tickets("A001"))
print("tickets for A001 before:", before, "after second run:", after)
for t in search_tickets("A001")[:5]:
    print(t)

# %% [markdown]
# ## 6. Trajectory check (preview of Session 4)
# Was the policy checker called before the memo writer? Was any tool called twice with the same
# arguments? `evals.trajectory` turns those expectations into a test.

# %%
from evals.trajectory import check_trajectory
print(check_trajectory(result["tool_calls"]))

# %% [markdown]
# **Read the result before you trust it.** On A001 the check fails with "never called: search_policy".
# A001 breaks no rule, so the agent never needed to quote one. Is the rule wrong, or the agent? Keep the
# question: Rohan's file in section 7 answers it.

# %% [markdown]
# ## 7. Rohan and Rahul, part three: the memos, and who gets the loan
# Sessions 1 and 2 turned two bank statements and a registry into leads. Now the same agents that drafted
# A001 draft Rohan's (A901) and Rahul's (A902) memos, **without a line of agent code changing**.
#
# `creditlens.scenarios.story` builds `data/story/`: the story registry, full application rows, both
# bank statements, and audited financial statements for both companies, loaded into its own small
# database. While a story memo runs, the agents read `data/story/` instead of `data/raw/`, through the
# same `CREDITLENS_DATA_ROOT` switch the tests use. The setup cell already ran
# `python -m creditlens.scenarios.story build`.
#
# **Predict first.** Rohan's audited statement shows exactly the turnover he declared. Will rules 3.1 to
# 3.3 pass? And what will the recommendation be?

# %%
from creditlens.scenarios.story import draft_story_memo, decide, story_env
rohan = draft_story_memo("A901", "U05")
print(rohan["memo"].to_markdown())
print("thread:", rohan["thread_id"], "| status:", rohan["status"])

# %% [markdown]
# On paper Rohan is a strong borrower: DSCR 2.5x, leverage 1.3x. One rule fails, 5.2, so the memo refers
# the file to the Head of Credit (policy 7.3), files **one** Jira ticket (7.2), lists all four companies
# (5.3), and adds one sentence written by code, not by the model: the most the group can take on without
# an exception. That is a counter-offer for a person to consider, not a decision.
#
# **The duplicate-ticket curveball, on a real exception.** Run Rohan's memo again. How many tickets for
# A901 now?

# %%
import json
draft_story_memo("A901", "U05")
with story_env():
    from creditlens.agents import adapters
    issues = json.loads((adapters.cache_dir() / "jira.json").read_text())["issues"]
print([(i["key"], i["idempotency_key"]) for i in issues if i["idempotency_key"].startswith("A901")])

# %% [markdown]
# One ticket. The idempotency key `A901:5.2` makes the second run find the first ticket instead of
# filing another one. And the trajectory check from section 6 passes on this run, because a failed rule
# made the agent search the policy. So the check was right for files with exceptions, and too strict for
# clean files like A001. Fix the check, not the agent: that is a Session 4 exercise.

# %%
print(check_trajectory(rohan["tool_calls"]))
rahul = draft_story_memo("A902", "U05")
print(rahul["memo"].to_markdown())

# %% [markdown]
# Rahul's memo recommends approval: every rule passes, and his family group (two companies, 55,000
# thousand) is listed even though it is inside the limit, because rule 5.3 says so.
#
# **What did the agents miss?** Compare both memos with Session 2's `case_file`. Rohan's memo says
# "the largest single customer is 40% of collections" but nothing about the 24 round trips. Rahul's memo
# says nothing about the account showing 60% of his declared turnover. The agents only know what their
# tools return, and no tool returns the Session 2 leads. On your own: write the Pydantic argument schema for
# a `get_case_leads(application_id)` tool, and decide which specialist should call it.

# %% [markdown]
# ### Underwriter role-play: who gets the loan?
# Everyone plays Amaira Dey, Head of Credit (U05). Read Rohan's memo, then decide on your own: approve
# the 50,000 as an exception, reject, or approve the 15,000 the group can carry. Then the room votes with a show of hands. Then Rahul.
#
# The cell below records the decisions the bank makes in our story. Change them to your room's.
# First, check that an analyst cannot decide at all.

# %%
try:
    decide(rohan["thread_id"], "approve", user="U04")
except PermissionError as e:
    print("refused:", e)

rohan_final = decide(rohan["thread_id"], "approve", user="U05", approved_inr_000=15000,
                     comment="Counter-offer within rule 5.2. Compliance to review the round trips with "
                             "Ravi Containers before disbursement.")
rahul_final = decide(rahul["thread_id"], "approve", user="U05",
                     comment="Approved in full. Takings to go through the business account; personal "
                             "guarantee from Rahul Verma (rule 6.2).")
print("Rohan:", rohan_final["status"], "for INR", rohan_final["decision"]["approved_inr_000"], "thousand")
print("Rahul:", rahul_final["status"], "in full")
with story_env():
    for app in ("A901", "A902"):
        for e in adapters.audit_events(app)[-2:]:
            print(app, e.actor, e.action, "|", e.detail[:110])

# %% [markdown]
# **How it turns out.** Rahul, who looked weak on Monday, gets his 3 crore. The system explained why his
# account looked thin, and Priya verified the family firm's takings. Rohan, who looked safe, is offered
# 1.5 crore, the most his group can carry under rule 5.2, and compliance reviews the round trips before
# any money moves. Nobody was accused of anything, no system approved or declined on its own (policy
# section 1), and every step is in the audit trail with a name against it.

# %% [markdown]
# ## Exit ticket
# One thing learned, one open question. Next, notebook 05: the bank's legacy systems (Oracle, SOAP) and Slack
# approvals from a phone.

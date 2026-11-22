# %% [markdown]
# # Notebook 09: test everything we built
#
# **The story.** Amaira's condition in notebook 06b was: "Make sure nobody can quietly ship a worse version next
# month." The evaluation gate keeps half of that promise. Tests keep the other half: they are how a team keeps its
# promises after the people who wrote the code have moved on.
#
# Across notebooks 01 to 08 we met many kinds of tests, often without calling them tests: the PASS lines at the end
# of 05b, the gate in 06b, the trajectory check in 07, the incident drills in 07b, the hard cases in 08. This notebook
# gathers them in one place: what each kind protects, where we met it, and the test file that now guards it for good.
#
# | Level | Kind of test | Where we met it | What it protects |
# |---|---|---|---|
# | 1 | The map | every notebook | knowing what is tested at all |
# | 2 | Anatomy of one test | | reading a test as a promise |
# | 3 | Data | 01, 02 | the generated bank data, cleaning, loading |
# | 4 | Extraction, retrieval, graph | 02, 03 | reading documents, finding the rule, the company groups |
# | 5 | The agent | 04, 04b | the memo, the approval gate, read-only SQL |
# | 6 | Integration and MCP | 05b | fallbacks, the circuit breaker, idempotent tickets, the protocol |
# | 7 | Security | 06b | who sees what, personal data, guardrails |
# | 8 | Model and evaluation | 06b, 07 | the risk model, the judge, the gate |
# | 9 | Agent behaviour, by hand | 07 | the path the agent took, not only its answer |
# | 10 | Monitoring and incidents | 07, 07b | drift and the four incident drills |
# | 11 | The hard cases | 08 | Rohan, Rahul and the bonus cases |
# | 12 | The app | the workbench | roles, the audit trail, the go-live checklist |
# | 13 | Safety nets | today | the two real incidents of 22 November |
# | 14 | The scoreboard | | every result in one table |
#
# Everything runs offline, with no API key and no Docker. All levels together take about nine minutes on the
# trainer's laptop; level 8 is the longest (about three minutes), because it runs the evaluation gate twice.

# %%
import sys
if "google.colab" in sys.modules:   # Colab only. On a laptop this cell does nothing.
    !git clone https://github.com/chahatfde-arch/Day3_Day4_FDE_session.git
    %cd Day3_Day4_FDE_session
    !pip install -q -e ".[dev,api,llm,agents,ml,security,evals,realdata,retrieval,integrations,extraction]"
    !python -m creditlens.generators
    !python -m creditlens.pipeline --load
    !python -m creditlens.data_real --only sba receipts sop
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
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))   # so the repo's own folders (tests, evals) import too
print("working folder:", Path.cwd())

# %% [markdown]
# One helper runs a group of test files with **pytest**, the standard Python test runner, in a separate process,
# exactly as you would in a terminal. It prints one line per test and keeps the results for the scoreboard at the end.
# The tests run against the stand-ins, whatever `.env` says, so everyone gets the same answers.

# %%
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET

os.environ["LLM_OFFLINE"] = "1"
for name in ("JIRA_BASE_URL", "JIRA_EMAIL", "JIRA_API_TOKEN", "ORACLE_DSN", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
    os.environ[name] = ""          # stand-ins only: a test must never touch a real system
RESULTS, REPORTS = {}, Path(tempfile.mkdtemp())


def run_tests(level, kind, notebooks, files, slow=False, show=12):
    """Run pytest on some test files; print each result; remember the totals for the scoreboard."""
    report = REPORTS / f"level{level}.xml"
    cmd = [sys.executable, "-m", "pytest", *files, "-q", "-p", "no:cacheprovider", f"--junitxml={report}"]
    if not slow:
        cmd += ["-m", "not slow"]          # the slow tests load a model or take minutes; run them with -m slow
    t0 = time.time()
    run = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if not report.exists():                # pytest never got going: show why, and the usual cure
        print(run.stdout[-1500:], run.stderr[-1500:])
        raise RuntimeError("pytest did not run. Install the test tools: pip install -e \".[dev,api]\"")
    root = ET.parse(report).getroot()
    cases = root.iter("testcase")
    rows = []
    for case in cases:
        status = ("FAIL" if case.find("failure") is not None or case.find("error") is not None
                  else "skip" if case.find("skipped") is not None else "PASS")
        promise = case.get("name").removeprefix("test_").replace("_", " ")
        rows.append((status, case.get("classname").split(".")[-1].removeprefix("test_"), promise))
    for status, file, promise in [r for r in rows if r[0] != "PASS"] + [r for r in rows if r[0] == "PASS"][:show]:
        print(f"{status}  [{file}] {promise}")
    if len([r for r in rows if r[0] == "PASS"]) > show:
        print(f"      ... and {len([r for r in rows if r[0] == 'PASS']) - show} more that passed")
    n = {s: sum(1 for r in rows if r[0] == s) for s in ("PASS", "FAIL", "skip")}
    RESULTS[level] = {"kind": kind, "notebooks": notebooks, "tests": len(rows), **n, "seconds": round(time.time() - t0)}
    print(f"\n{kind}: {len(rows)} tests, {n['PASS']} passed, {n['FAIL']} failed, {n['skip']} skipped, in {RESULTS[level]['seconds']} s")


print("ready: results go to", REPORTS)

# %% [markdown]
# ## Level 1. The map: what is tested at all
# Before running anything, ask pytest to **collect** the tests without running them. Each file belongs to one part of
# the course. A part with no tests is a part nobody can safely change.

# %%
from collections import Counter

PART = {"generators": ("Data", "01, 02"), "wrangle": ("Data", "01, 02"), "pipeline": ("Data", "01, 02"),
        "ingest": ("Data", "01, 02"), "extraction": ("Extraction", "02"), "retrieval": ("Retrieval and graph", "03"),
        "graph": ("Retrieval and graph", "03"), "agents": ("The agent", "04, 04b"),
        "integrations": ("Integration", "05b"), "mcp_jira": ("MCP and Jira", "05b"), "jira_local": ("MCP and Jira", "05b"),
        "guardrails": ("Security", "06b"), "api": ("Security", "06b"), "ml": ("Model", "06b"),
        "evals": ("Evaluation", "06b, 07"), "monitoring": ("Monitoring and incidents", "07, 07b"),
        "scenarios": ("The hard cases", "08"), "story": ("The hard cases", "08"), "workbench": ("The app", "workbench"),
        "safety_nets": ("Safety nets", "today")}
listing = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
ids = [line for line in listing.splitlines() if "::" in line]
per_file = Counter(Path(i.split("::")[0]).stem.removeprefix("test_") for i in ids)
print(f"{'test file':<14} {'tests':>5}  {'part of the course':<26} notebooks")
for file, n in sorted(per_file.items(), key=lambda kv: (PART.get(kv[0], ("?", "?"))[1], kv[0])):
    kind, nbs = PART.get(file, ("not mapped yet", "?"))
    print(f"{file:<14} {n:>5}  {kind:<26} {nbs}")
print(f"\n{len(ids)} tests in {len(per_file)} files. Unmapped files:", [f for f in per_file if f not in PART] or "none")

# %% [markdown]
# ## Level 2. Anatomy of one test
# A good test reads like a promise. Its **name** says what must stay true. Its body does three things:
# **arrange** (set up a small world), **act** (do one thing), **assert** (check what must be true). It runs offline,
# in a fraction of a second, and fails loudly when someone breaks the promise. Here are two, read straight from the
# test files.

# %%
import ast


def show_test(path, name):
    source = Path(path).read_text(encoding="utf-8")
    node = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == name)
    print(f"# {path}, line {node.lineno}")
    print(ast.get_source_segment(source, node), "\n")


show_test("tests/test_mcp_jira.py", "test_idempotency_key_is_application_plus_rule")
show_test("tests/test_safety_nets.py", "test_a_fast_run_refuses_to_delete_pdfs_that_already_exist")

# %% [markdown]
# ## Level 3. Data tests (Build: notebooks 01 and 02)
# The bank data is generated from one seed, so it can be **tested like code**: the same seed gives the same data, the
# planted problems are really planted, the cleaned statements equal the truth, and the database round-trips.

# %%
run_tests(3, "Data", "01, 02", ["tests/test_generators.py", "tests/test_wrangle.py", "tests/test_pipeline.py",
                                "tests/test_ingest.py"])

# %% [markdown]
# ## Level 4. Extraction, retrieval and the company graph (Build: notebooks 02 and 03)
# Reading a scanned statement, finding the right policy rule, and following ownership links between companies. These
# tests compare against the **answer key** the generator wrote, so "correct" is a fact, not an opinion.

# %%
run_tests(4, "Extraction, retrieval, graph", "02, 03", ["tests/test_extraction.py", "tests/test_retrieval.py",
                                                        "tests/test_graph.py"])

# %% [markdown]
# ## Level 5. The agent (Build: notebooks 04 and 04b)
# The memo has every policy section in order, the agent **pauses for approval**, SQL is read-only and limited to the
# user's region, and a loop is stopped by the step limit.

# %%
run_tests(5, "The agent", "04, 04b", ["tests/test_agents.py"])

# %% [markdown]
# ## Level 6. Integration and MCP (Deploy: notebook 05b)
# The promises of 05b, now permanent: a dead service still answers quickly, labelled as a mock; the breaker opens; the
# Oracle adapter maps the bank's column names to ours; Slack refuses anyone but the Head of Credit; a ticket written
# twice is filed once. The MCP tests include the two **slow** ones on purpose: they start the real server over stdio,
# shake hands, and check there are three tools and no delete.

# %%
run_tests(6, "Integration and MCP", "05b", ["tests/test_integrations.py", "tests/test_mcp_jira.py",
                                            "tests/test_jira_local.py"], slow=True)

# %% [markdown]
# ## Level 7. Security (Deploy: notebook 06b)
# The largest group. Personal data is masked before any model sees it, the guardrails block instruction overrides and
# approval language, and through the API an analyst can read only their own region, cannot approve, and a tampered
# token is refused.

# %%
run_tests(7, "Security", "06b", ["tests/test_guardrails.py", "tests/test_api.py"])

# %% [markdown]
# ## Level 8. Model and evaluation (Evaluate: notebooks 06b and 07)
# The risk model scores without crashing even with no model at all (the stub), the judge agrees with the human labels,
# and the evaluation gate **passes by default and fails when the judge's prompt is broken**. This level runs the gate
# twice, so it takes a few minutes: start it and talk.

# %%
run_tests(8, "Model and evaluation", "06b, 07", ["tests/test_ml.py", "tests/test_evals.py"])

# %% [markdown]
# ## Level 9. Agent behaviour, by hand (Evaluate: notebook 07)
# A memo can read well and still be wrong in how it was made. A **trajectory check** looks at the path: which tools,
# in which order, how many times. Here are two made-up runs: a healthy one, and one that wrote the memo before reading
# the policy, asked for the same application twice, and touched a forbidden tool.

# %%
from evals.trajectory import check_trajectory

healthy = [{"name": n, "args": {"application_id": "A901"}} for n in
           ("get_application", "get_statement_features", "search_policy", "score_risk", "memo_writer")]
careless = [{"name": "get_application", "args": {"application_id": "A901"}},
            {"name": "get_application", "args": {"application_id": "A901"}},
            {"name": "memo_writer", "args": {}},
            {"name": "score_risk", "args": {"application_id": "A901"}},
            {"name": "delete_application", "args": {"application_id": "A901"}}]
for label, run in (("healthy", healthy), ("careless", careless)):
    report = check_trajectory(run)
    print(f"{label}: passed={report['passed']}")
    for failure in report["failures"]:
        print(f"   FAIL {failure['check']}: {failure['detail']}")

# %% [markdown]
# ## Level 10. Monitoring and incidents (Maintain: notebooks 07 and 07b)
# The drift report finds the drift we injected, and each of the four incident drills runs and says where to look.
# An incident you have rehearsed is an incident you can handle at three in the morning.

# %%
run_tests(10, "Monitoring and incidents", "07, 07b", ["tests/test_monitoring.py"])

# %% [markdown]
# ## Level 11. The hard cases (Evaluate: notebook 08)
# The story, as tests. On paper Rohan and Rahul both pass the financial rules. Rohan's **group** breaks rule 5.2 and
# he is referred with one ticket and a counter-offer; Rahul is linked but within the limit. Running Rohan twice still
# files one ticket, and the story world never touches the default data.

# %%
run_tests(11, "The hard cases", "08", ["tests/test_scenarios.py", "tests/test_story.py"])

# %% [markdown]
# ## Level 12. The app (Deploy: the workbench)
# The workbench is the agent inside an app, so it gets tested like an app: an analyst sees only their region, opening
# another region's file is refused **and audited**, platform jobs need the Head of Credit or the auditor, the go-live
# checklist is computed with evidence, and every stand-in is labelled.

# %%
run_tests(12, "The app", "workbench", ["tests/test_workbench.py"])

# %% [markdown]
# ## Level 13. Safety nets: tests born from real incidents
# On 22 November two real problems happened on the trainer's laptop. An old notebook cell ran the data generator in
# its fast mode and **deleted the PDFs**. And the "offline" gate quietly graded with **a live model's cached answers**.
# Each was fixed in the code, and each fix got a test, so it can never come back unnoticed. A third test written that
# day, for the local Jira service, **found a real bug on its first run**: the Jira client sent one setting twice, and
# every ticket it tried to create failed. That is what tests are for.

# %%
run_tests(13, "Safety nets", "today", ["tests/test_safety_nets.py"])

# %% [markdown]
# ## Level 14. The scoreboard
# Every level, in one table. This project's continuous integration (notebook 06, `.github/workflows/ci.yml`) runs
# the same fast tests on **every push**. A red line stops the release; nobody lowers a threshold to make it green.
#
# Why fewer tests here than level 1 counted? Level 1 collects **every** test, the slow ones too. The levels leave the
# slow ones out (they load a model or start the whole workbench), except level 6, which keeps its two MCP tests. Run
# them yourself with `python -m pytest -m slow`.

# %%
total = {k: sum(r[k] for r in RESULTS.values()) for k in ("tests", "PASS", "FAIL", "skip", "seconds")}
print(f"{'level':<6} {'kind of test':<30} {'notebooks':<10} {'tests':>5} {'passed':>6} {'failed':>6} {'skipped':>7} {'time':>6}")
for level, r in sorted(RESULTS.items()):
    print(f"{level:<6} {r['kind']:<30} {r['notebooks']:<10} {r['tests']:>5} {r['PASS']:>6} {r['FAIL']:>6} {r['skip']:>7} {r['seconds']:>5}s")
print(f"{'all':<6} {'':<30} {'':<10} {total['tests']:>5} {total['PASS']:>6} {total['FAIL']:>6} {total['skip']:>7} {total['seconds']:>5}s")
print("\nGreen: safe to ship." if total["FAIL"] == 0 else "\nRed: do not ship. Read the FAIL lines above, fix the code, run again.")

# %% [markdown]
# ## From zero to hero: testing an agent a bank will run
# 1. Name every test as a promise, in plain words: the name is the documentation.
# 2. Arrange, act, assert: one small world, one action, one check.
# 3. Test against an answer key wherever you can, so "correct" is a fact.
# 4. Test the failure paths on purpose: dead services, open breakers, refused roles, broken prompts.
# 5. Test the path, not only the answer: trajectories, order, repeats, forbidden tools.
# 6. Run everything offline and fast, so everyone runs it, every time.
# 7. When something breaks for real, fix it and add the test that would have caught it.
# 8. Gate releases on the results: Build, Deploy, Evaluate, Maintain, and test at every step.
#
# ## Exit ticket
# One test you would add to this project tomorrow, and the promise it would keep.

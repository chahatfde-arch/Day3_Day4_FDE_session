# %% [markdown]
# # Session 4, part three: run the Underwriter Agent, and keep it working
#
# **The story.** The agent is live. Three months later, Amaira's queue is shorter: she reads prepared
# reviews and signs, instead of building each one by hand. Then the world changes, as it always does.
# Next quarter's applicants look different. Someone switches the cache off. Compliance raises the DSCR
# minimum. A statement arrives with an instruction hidden in a narration. And one Friday, a model update
# has to be rolled back.
#
# **The forward deployed engineer's job now: you cannot maintain what you cannot measure.** Offline
# evaluation tells you whether a change is safe *before* it ships. Online observability tells you when the
# world changed *after*. We go from zero to hero:
#
# | Level | What we add | The question it answers |
# |---|---|---|
# | 1 | Observe the agent | Where does the time go, and what did it cost? |
# | 2 | Online feedback | How often does Amaira change the proposal? |
# | 3 | A calibrated judge | Can a model mark the notes, and how far can we trust it? |
# | 4 | Resilience drill | What happens when the model provider fails? |
# | 5 | Four incidents | Symptom, where to look, the fix, and what the agent did |
# | 6 | A policy change | Why rules and the answer key must change together |
# | 7 | Rollback | How to undo a bad release, and prove it worked |
# | 8 | Runbook and handoff | Who owns what, when the FDE leaves |
#
# Runs offline, on the data and the agent from the previous notebooks.

# %%
import sys
if "google.colab" in sys.modules:   # Colab only. On a laptop this cell does nothing.
    !git clone https://github.com/chahatfde-arch/Day3_Day4_FDE_session.git
    %cd Day3_Day4_FDE_session
    !pip install -q -e ".[dev,llm,agents,ml,evals,realdata,retrieval]"
    !python -m creditlens.generators
    !python -m creditlens.pipeline --load
    !python -m creditlens.data_real --only sba sop
    !python -m creditlens.scenarios
    !python -m creditlens.scenarios.story build
    !python -m creditlens.ml.train --quick
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
print("offline mode:", os.environ["LLM_OFFLINE"] == "1")

# %% [markdown]
# **Offline evaluation and online observability, side by side.**
#
# | | Offline evaluation | Online observability |
# |---|---|---|
# | When | before a change ships | all the time, in production |
# | Against | the golden set, the exam, human labels | traces, metrics, drift, feedback |
# | Here | `evals/`, the agent's exam | timings, the gateway counters, `monitoring/` |
# | The question | is this version better? | is production still healthy? |

# %% [markdown]
# ## Level 1. Observe the agent: where does the time go?
# Every sensor call is a span in a trace. With Langfuse keys, each one appears in the Langfuse project.
# Without them, we can still time each sensor ourselves and read the gateway's counters.

# %%
import json
import time
from collections import defaultdict
from creditlens.agents import tools
from creditlens.agents.policy_checks import run_checks
from creditlens.gateway.llm import get_gateway

T = tools.get_tools()
timings = defaultdict(list)


def timed(name, fn):
    t0 = time.perf_counter()
    out = fn()
    timings[name].append(time.perf_counter() - t0)
    return out


for app in ("A001", "A005", "A012", "A020", "A022"):
    form = timed("get_application", lambda: T["get_application"].invoke({"application_id": app}))["application"]
    timed("get_statement_features", lambda: T["get_statement_features"].invoke({"application_id": app}))
    timed("get_financials", lambda: T["get_financials"].invoke({"company_id": form["company_id"]}))
    timed("get_group_exposure", lambda: T["get_group_exposure"].invoke({"company_id": form["company_id"]}))
    timed("score_risk", lambda: T["score_risk"].invoke({"application_id": app}))
    timed("run_checks", lambda: run_checks(app))

total = sum(sum(v) for v in timings.values())
print(f"{'step':24s}{'avg ms':>9s}{'share':>8s}")
for name, v in sorted(timings.items(), key=lambda kv: -sum(kv[1])):
    print(f"{name:24s}{1000 * sum(v) / len(v):9.0f}{sum(v) / total:8.0%}")
g = get_gateway()
print(f"\ngateway: provider calls {g.calls}, cache hits {g.cache_hits}, models {g.models}")

# %% [markdown]
# The slowest steps are where caching and parallel calls pay off first. And the gateway line answers the
# cost question: offline, zero provider calls. Live, every call is counted, and the cache keeps repeats free.

# %% [markdown]
# ## Level 2. Online feedback: how often does Amaira change the proposal?
# The most honest production metric for this agent is not a model score. It is **how often the person
# changes what the agent proposed**. If she keeps changing it, the agent is costing her time, not saving it.
# Here are the two decisions from the story, against what the agent proposed.

# %%
FEEDBACK = [
    {"app": "A901", "agent": ("counter-offer", 15000), "amaira": ("counter-offer", 15000),
     "conditions_added": ["Compliance reviews the round trips with Ravi Containers before disbursement"]},
    {"app": "A902", "agent": ("approve", 30000), "amaira": ("approve", 30000),
     "conditions_added": ["Personal guarantee from the managing director (rule 6.2)",
                          "Takings to go through the business account"]},
]
same_outcome = sum(f["agent"] == f["amaira"] for f in FEEDBACK)
with_extra_conditions = sum(bool(f["conditions_added"]) for f in FEEDBACK)
print(f"outcome kept as proposed: {same_outcome}/{len(FEEDBACK)}")
print(f"conditions added by the person: {with_extra_conditions}/{len(FEEDBACK)}  <- the learning stage in 04b exists for this")
print("\nAlert rule for production: if fewer than 8 in 10 outcomes are kept in a week, review the agent before the next release.")

# %% [markdown]
# Two files are an example, not a statistic. In production this table fills itself from the audit trail,
# and the alert threshold is agreed with Amaira, not invented by the engineer.

# %% [markdown]
# ## Level 3. A judge model, and the eval court
# Marking every note by hand does not scale. A **judge** model can mark them against a rubric, but a judge
# is only as good as its agreement with people. So we **calibrate it first**: ten notes that people
# labelled pass or fail, marked again by the judge.

# %%
!python -m evals.judge --calibrate

# %% [markdown]
# **The eval court.** Read the disagreement. Every box on the rubric is ticked, but the prose calls the
# banking conduct clean while the cheque returns breach rule 4.1. A rule-based judge cannot hear tone; a
# person can. For each disagreement, decide: change the rubric, or change the label? Never just accept the
# judge.

# %% [markdown]
# ## Level 4. Resilience: when the model provider fails
# The gateway tries the primary model, then the fallbacks, then the cache, then the offline stub. The drill
# sends a few prompts and reports who answered. Offline, the stub answers every one; with keys, run it again
# with `--kill-primary` and watch the backup answer instead.

# %%
!python -m creditlens.gateway.drill -n 3

# %% [markdown]
# ## Level 5. Four incidents, and what the agent did
# Each incident is one command. For each one, read three lines: the **symptom**, **where to look**, and the
# **fix**. Then ask the FDE question: what did the Underwriter Agent do while it was happening?

# %%
!python -m monitoring.incidents --scenario data_drift

# %% [markdown]
# **The agent during data drift:** its risk scores shifted for a whole batch, while the rule checks looked
# normal. The proposals moved because a sensor moved. Fix: retrain on the new population, compare with the
# champion, and only then move the alias (Level 7).

# %%
!python -m monitoring.incidents --scenario cost_spike

# %% [markdown]
# **The agent during the cost spike:** nothing looked wrong in the notes. Only the counters showed it.
# That is why cost is a metric with an alert, not a monthly surprise.

# %%
!python -m monitoring.incidents --scenario prompt_injection

# %% [markdown]
# **The agent during the injection:** nothing happened, by design. The input rail blocked the narration,
# the memo writer only ever sees aggregated features, and the proposal is decided by code before any model
# is called (notebook 04b, stage 4).

# %%
!python -m monitoring.incidents --scenario policy_update

# %% [markdown]
# ## Level 6. A policy change, replayed on the agent
# Compliance raises the DSCR minimum from 1.25x to 1.40x. The retrieval index is easy to fix: it re-indexes
# when the manual changes. The subtle part is the **exam**. Change the rule, keep the old answer key, and
# watch what the agent's own gate says.

# %%
from creditlens.generators.policy import RULES

KEY = json.load(open("data/answer_key/expected_policy_flags.json"))


def exam_mismatches():
    out = []
    for i in range(1, 31):
        app = f"A{i:03d}"
        failed = sorted(c.rule_id for c in run_checks(app) if not c.passed)
        if failed != sorted(KEY[app]["flags"]):
            out.append((app, failed, KEY[app]["flags"]))
    return out


baseline = exam_mismatches()
old = RULES["MIN_DSCR"]
RULES["MIN_DSCR"] = 1.40                       # compliance's new rule, as code
try:
    after = exam_mismatches()
finally:
    RULES["MIN_DSCR"] = old                    # always put it back in a notebook
new_cases = [m for m in after if m not in baseline]
print(f"mismatches with the old answer key: before {len(baseline)}, after the rule change {len(after)}")
for app, failed, key in new_cases:
    print(f"   {app}: the checks now fail {failed}, the old answer key says {key}")

# %% [markdown]
# The agent is **right** about the new rule, and the exam now calls it **wrong**, because the answer key
# still holds the old policy. A rule change is three changes shipped together: the rule, the answer key,
# and the prompt version. Ship one alone and either the agent or its gate lies to you.

# %% [markdown]
# ## Level 7. Rollback, and prove it worked
# The model registry keeps every version, and an alias called `champion` points at the one in use. Rolling
# back means moving the alias. **But a rollback is only done when you have checked what is actually served.**

# %%
import mlflow
from creditlens.ml.serving import load_champion, score_application
from creditlens.ml.train import default_tracking_uri

uri = default_tracking_uri()          # the same registry training used: the Docker server, or mlruns.db
client = mlflow.MlflowClient(tracking_uri=uri)
mlflow.set_tracking_uri(uri)
print("registry:", uri)


def served():
    load_champion(refresh=True)       # serving loads once per process: reload, as a real redeploy would
    champ = load_champion()
    return score_application("A001").model_version, champ["source"]


versions = sorted(int(v.version) for v in client.search_model_versions("name='creditlens-risk'"))
if len(versions) < 2:                 # a fresh laptop has one version: retrain once, as next month would
    print("only one version registered; training a second one to roll back from (about a minute) ...")
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "creditlens.ml.train", "--quick"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    versions = sorted(int(v.version) for v in client.search_model_versions("name='creditlens-risk'"))
current = int(client.get_model_version_by_alias("creditlens-risk", "champion").version)
print("registered versions:", versions, "| champion:", current)
print("served before: version %s, loaded from %s" % served())

if len(versions) > 1:
    previous = versions[-2]
    client.set_registered_model_alias("creditlens-risk", "champion", str(previous))
    version_now, source_now = served()
    print(f"alias moved to version {previous}; served now: version {version_now}, loaded from {source_now}")
    if str(version_now) == str(previous) and source_now == "mlflow-registry":
        print("  -> rollback verified: the registry now serves the previous version.")
    else:
        print("  -> the rollback did NOT change what is served. Serving fell back to the saved file,"
              "\n     because the registry could not give it a model. Fix the registry, or roll back the file too.")
    client.set_registered_model_alias("creditlens-risk", "champion", str(current))   # put it back for the workshop
    print("alias restored to version %s; served: version %s, loaded from %s" % ((current,) + served()))

# %% [markdown]
# Read the last word of each line: **where the model was loaded from**. With the MLflow server running
# (Docker, `--profile ml`) the registry holds every version, so moving the alias really changes what is
# served, once the service reloads. Without a working registry, serving falls back to
# `data/cache/model.joblib`, and moving the alias changes nothing. Either way the rule is the same:
# **verify a rollback by checking what is served, not by trusting that the command ran.**

# %% [markdown]
# ## Level 8. The runbook and the handoff
# A forward deployed engineer leaves. The system must not depend on them. Every recurring job needs an
# owner, a trigger and a place to look.

# %%
RUNBOOK = [
    ("Golden set and answer key", "credit policy team", "any policy change",           "ground_truth/, data/answer_key/"),
    ("Evaluation gate",           "platform team",      "every pull request",          "python -m evals.gate"),
    ("Agent exam and safety",     "platform team",      "every agent or prompt change", "the agent gate, notebook 06b"),
    ("Prompt versions",           "platform team",      "any prompt edit",             "PROMPT_VERSION, the trace"),
    ("Drift alert",               "risk analytics",     "weekly",                      "monitoring/reports/drift_summary.json"),
    ("Override rate",             "Amaira's office",    "weekly",                      "the audit trail"),
    ("Model retrain and rollback","risk analytics",     "drift, or a worse champion",  "MLflow registry, champion alias"),
    ("Incident response",         "on-call engineer",   "an alert fires",              "monitoring.incidents playbooks"),
]
print(f"{'job':28s}{'owner':22s}{'when':30s}where")
for job, owner, when, where in RUNBOOK:
    print(f"{job:28s}{owner:22s}{when:30s}{where}")

# %% [markdown]
# ## The end of the story
# Rahul received his three crore, with the conditions the agent learned from Amaira. Rohan was offered
# one and a half crore, and compliance reviewed the round trips before any money moved. Amaira's queue is
# shorter: she reads prepared reviews, changes what she disagrees with, and signs. The agent never decided
# anything. Every number points to where it came from, every change passes a gate, and every alert has an
# owner.
#
# ## From zero to hero: keeping an agent working
# 1. Measure before and after shipping: offline evaluation and online observability.
# 2. Time every step, and count every call.
# 3. Track how often the person changes the proposal; agree the alert with them.
# 4. Calibrate any judge against people, and hold an eval court for disagreements.
# 5. Drill failures before they happen: provider down, cache off, injection, drift.
# 6. Ship a rule change as three changes together: rule, answer key, prompt version.
# 7. Verify every rollback by checking what is served.
# 8. Hand over a runbook: every job has an owner, a trigger and a place to look.
#
# ## Demo day, and exit ticket
# Ten minutes to Amaira and the CISO: the story, the architecture, a live review to the gate, the
# scorecard (exam, gate, fairness, masking, judge), and the runbook. Then one thing you learned and one
# open question.

# Run CreditLens AI end to end

**Keys, Docker, every notebook in order, the Credit Workbench, the tests, and hosting on AWS.** CreditLens AI ·
TAN GLOBUS AI · Days 3 and 4.

You work alone, on your own laptop. This guide takes you from an empty folder to everything running. Follow it top
to bottom the first time. Each step says what to type and what you should see.

**The most important sentence in this guide: you need no keys and no Docker.** Everything runs offline by default,
with the same results for everyone. Keys and Docker are upgrades you can add later, one at a time. The guide marks
them *optional*.

The four phases of the course, and where each part of this repository sits:

| Phase | What it means here | Notebooks |
|---|---|---|
| **Build** | build the Underwriter Agent and connect it to the bank's systems | 04, 04b, 05, 05b |
| **Deploy** | secure it, register the risk model, package it in a container, host it | 06, 06b, the workbench, AWS |
| **Evaluate** | measure it: the judge, the gate, the hard cases, the tests | 07, 08, 09 |
| **Maintain** | keep it working: feedback, incidents, policy changes, rollback | 07b |

## Contents

1. [Set up your laptop](#1-set-up-your-laptop)
2. [Working with notebooks in VS Code](#2-working-with-notebooks-in-vs-code)
3. [Keys (optional)](#3-keys-optional)
4. [Docker (optional)](#4-docker-optional)
5. [Every notebook, in order](#5-every-notebook-in-order)
6. [The Credit Workbench](#6-the-credit-workbench)
7. [The tests](#7-the-tests)
8. [Host the workbench on AWS (optional)](#8-host-the-workbench-on-aws-optional)
9. [When something goes wrong](#9-when-something-goes-wrong)
10. [The whole run on one page](#10-the-whole-run-on-one-page)

The times in this guide were measured on the trainer's laptop, from a fresh clone. Yours may differ.

---

## 1. Set up your laptop

### 1.1 What you need

- **Python 3.11 or 3.12.** Check: `py -3.12 --version` (Windows) or `python3 --version` (Mac).
- **Git.** Check: `git --version`.
- **VS Code** with the **Python** and **Jupyter** extensions (Extensions panel, search for each, click Install).
- **Internet** for the setup. After setup, everything except the optional keys works offline.

### 1.2 Get the code and install

Open a terminal (Windows: **PowerShell**; Mac: **Terminal**) in the folder where you keep projects, then:

```bash
git clone https://github.com/chahatfde-arch/Day3_Day4_FDE_session.git
cd Day3_Day4_FDE_session
```

Create a private Python environment for this project and switch it on:

```bash
# Windows (PowerShell)                 # Mac / Linux
py -3.12 -m venv .venv                  # python3 -m venv .venv
.venv\Scripts\Activate.ps1              # source .venv/bin/activate
```

Your prompt now starts with `(.venv)`. If PowerShell refuses to run `Activate.ps1`, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

Install the project and everything the notebooks use (most of the setup time is this download):

```bash
pip install --upgrade pip
pip install -e ".[all]"
```

### 1.3 Make the data

Run these one at a time. All of Good Bank's data is synthetic; the last command downloads small samples of three
real public datasets.

| Command | What it does | What you should see at the end |
|---|---|---|
| `python -m creditlens.generators` | writes Good Bank's applications, statements and policy | `applications: 30`, `financial statements: 30 PDFs, 8 scanned`, then `done in ...` |
| `python -m creditlens.pipeline --load` | cleans the statements and loads the database | `loaded 9 tables into sqlite:///...creditlens.db` |
| `python -m creditlens.scenarios` | writes the story cases | a block of numbers ending with `"REVENUE_GAP"` |
| `python -m creditlens.scenarios.story build` | builds the story world (Rohan and five more) | `applications: A901, A902, A903, A904, A905, A906` |
| `python -m creditlens.data_real --only sba receipts sop` | downloads the SBA loan book, scanned receipts and the SBA rulebook (needs internet) | the paths of `receipts` and `sba_loans.csv` |

A line saying *You are sending unauthenticated requests to the HF Hub* is a harmless warning (see
[`HF_TOKEN`](#38-the-other-lines-in-envexample) if you want it gone).

### 1.4 Check it

```bash
python -m creditlens.setup_check
```

Every line that matters says **OK**: the Python packages, `synthetic data generated`, `real-data samples fetched`,
`database loaded`. **WARN** lines for `GEMINI_API_KEY`, `GROQ_API_KEY`, `NEO4J_URI`, `LANGFUSE_PUBLIC_KEY` and the
`tesseract` program are fine: those are optional.

### 1.5 Already set up before? Update

If you cloned on Day 3, bring your copy up to date. **Close every open notebook in VS Code first**, then:

```bash
git pull
python -m creditlens.data_real --only sba receipts sop
```

The second command is new: without it, notebooks 06, 06b and 07b stop with
`FileNotFoundError: data/real/sba_loans.csv`.

---

## 2. Working with notebooks in VS Code

1. **Open the folder.** VS Code → **File** → **Open Folder** → choose `Day3_Day4_FDE_session`.
2. **Choose the environment.** Press **Ctrl+Shift+P** (Mac: **Cmd+Shift+P**), type *Python: Select Interpreter*,
   choose the one that says `.venv`.
3. **Open a notebook** from the `notebooks` folder. At the top right, click **Select Kernel** → **Python
   Environments** → `.venv`. If VS Code offers to install `ipykernel`, click **Install**.
4. **Run it.** Click **Restart**, then **Run All**, in the toolbar at the top. Wait until every cell has a number or a
   tick. Scroll from the top and read as it runs.
5. **Check the first two outputs.** Cell 1 prints *Not on Colab: skipping the clone*. Cell 2 prints
   `working folder:` followed by your repository folder. Every notebook finds the folder itself; you never edit a
   path.

Three habits that prevent most problems:

- **Restart the kernel after you change `.env`.** Settings are read once, when a notebook starts.
- **After `git pull`, revert any notebook that was open.** Ctrl+Shift+P → *File: Revert File*, then Restart. VS Code
  otherwise keeps running its old copy from memory, and may save it over the new one.
- **One notebook at a time,** in the order of [section 5](#5-every-notebook-in-order). Some notebooks use what an
  earlier one made (07b needs the model that 06b registers).

---

## 3. Keys (optional)

### 3.1 Do you need any?

No. Without keys, every notebook, the workbench and the tests run offline and give the same results for everyone.
Here is what each key adds:

| Key | Adds | Used by | Cost |
|---|---|---|---|
| `GEMINI_API_KEY` | a real model writes answers | notebook 07 (automatically), the workbench and 04b (if you switch them to live) | Google AI Studio has a free tier |
| `GROQ_API_KEY` | a backup model when Gemini fails or is busy | same as above | Groq has a free tier |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` | every agent run appears as a trace you can open | every notebook that runs the agent, and the workbench when live | Langfuse Cloud has a free plan |
| `JIRA_BASE_URL`, `JIRA_EMAIL`, `JIRA_API_TOKEN`, `JIRA_PROJECT_KEY` | exception tickets go to a real Jira | the workbench's *Raise exceptions* | Jira has a free plan |
| `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`, ... | Amaira approves from Slack | the Slack app in `creditlens.integrations.slack` | Slack has a free plan |
| `ORACLE_*`, `MLFLOW_TRACKING_URI`, `SOAP_WSDL_URL` | the real bank systems and the model registry server | notebooks 05b, 06b, 07b and the workbench | free: they run in Docker on your laptop ([section 4](#4-docker-optional)) |

Free plans and their limits change: check each provider's page when you sign up.

### 3.2 Create your `.env` file

`.env` is a private text file in the repository folder that holds your keys. Git ignores it, so it never goes to
GitHub.

```bash
# Windows (PowerShell)                 # Mac / Linux
Copy-Item .env.example .env             # cp .env.example .env
```

Open `.env` in VS Code. The rules:

- One setting per line: `NAME=value`.
- **No quotes, no spaces** around `=`. Write `GEMINI_API_KEY=AIza...`, not `GEMINI_API_KEY = "AIza..."`.
- A comment goes on **its own line**, starting with `#`. Text after a value on the same line becomes part of the
  value.
- Save the file, then **restart the kernel** of any open notebook.
- **Never** paste a key into a notebook cell, a chat or a screenshot. If a key leaks, delete it on the provider's page
  and create a new one.

**If you have no model key, set `LLM_OFFLINE=1` in `.env`.** The notebooks decide this for themselves, but command-line
tools (for example the Slack dry run) otherwise keep trying providers that will never answer.

### 3.3 Gemini (the main model)

1. Open [Google AI Studio's API keys page](https://aistudio.google.com/apikey) and sign in with a Google account.
2. Click **Create API key**. Copy it.
3. In `.env`:

```
GEMINI_API_KEY=paste-your-key-here
LLM_PRIMARY=gemini/gemini-3.5-flash-lite
LLM_OFFLINE=0
```

`LLM_RPM=10` (already in the file) keeps you under the free tier's requests per minute.

### 3.4 Groq (the backup model)

1. Open [the Groq console's API keys page](https://console.groq.com/keys) and sign in.
2. Click **Create API Key**. Copy it (it is shown once).
3. In `.env`: `GROQ_API_KEY=paste-your-key-here`. `LLM_FALLBACKS` already names the Groq model.

### 3.5 What goes live with a key

| Where | Without a key | With a key |
|---|---|---|
| Notebook 07 | prints `model calls: offline stub (no API key found)` | prints `model calls: live (an API key was found)` and a real answer |
| Notebooks 04b, 05b, 06b, 07b | offline | **still offline, on purpose**: the same numbers for everyone. To try 04b live, change its setup line `os.environ["LLM_OFFLINE"] = "1"` to `"0"` and restart |
| The workbench | offline | offline unless you start it with `WORKBENCH_LIVE=1` (PowerShell: `$env:WORKBENCH_LIVE="1"; python app/workbench/server.py`; Mac: `WORKBENCH_LIVE=1 python app/workbench/server.py`) |
| Notebook 09 and `pytest` | offline | **always offline**: tests must not depend on a model's mood |

### 3.6 Langfuse (traces)

1. Choose a region and sign up: [cloud.langfuse.com](https://cloud.langfuse.com) (Europe) or
   [us.cloud.langfuse.com](https://us.cloud.langfuse.com) (United States).
2. Create an organization and a project.
3. In the project: **Settings** → **API Keys** → **Create new API keys**. Copy the **Public Key** (starts `pk-lf-`)
   and the **Secret Key** (starts `sk-lf-`, shown once).
4. In `.env`, with `LANGFUSE_HOST` set to the address you signed up on:

```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

5. Restart the kernel, run notebook 04b or 07, then open your project → **Tracing**. Each agent run is a trace with a
   span per step. Traces appear even offline: they show the path the agent took.

*Failed to export spans batch due to timeout, max retries or shutdown* means one batch of traces could not be sent
(usually a network hiccup). It does not affect the notebook.

### 3.7 Jira and Slack

**Jira.** Without any Jira settings, tickets go to a practice Jira kept in a file, and behave the same. Notebook 05b
always uses its own practice Jira, and the tests always use the file, whatever `.env` says. Two upgrades:

- **The local Jira service (Docker),** the one tested in this course: see [section 4.2](#42-legacy-oracle-soap-and-the-local-jira-service).
- **Jira Cloud (free plan):**
  1. Create a Jira site at [atlassian.com](https://www.atlassian.com/software/jira/free) and a project with the key
     `CREDIT`. Its issue types must include **Task**.
  2. Create an API token on your [Atlassian account's API tokens page](https://id.atlassian.com/manage-profile/security/api-tokens).
  3. In `.env`:

```
JIRA_BASE_URL=https://your-site.atlassian.net
JIRA_EMAIL=the-email-you-sign-in-with
JIRA_API_TOKEN=paste-your-token-here
JIRA_PROJECT_KEY=CREDIT
```

  Jira Cloud uses the same three calls as the local service (create, get, search). It was not part of the course run.

**Slack.** Amaira approves from her phone. Without Slack, the code prints the exact message it would post:

```bash
python -m creditlens.integrations.slack --dry-run --application A001
```

To connect a real workspace (optional, beyond the course run), follow the five setup steps under *Slack approvals* in
[docs/integrations.md](docs/integrations.md). They give you two tokens. Then in `.env`:

```
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
SLACK_APPROVALS_CHANNEL=#credit-approvals
SLACK_USER_MAP=your-slack-member-id:U05
```

`U05` is Amaira Dey, the Head of Credit: only she may approve. Run `python -m creditlens.integrations.slack` in one
terminal and leave it running; in a second terminal, `python -m creditlens.agents.run A001 --user U05 --slack`.

### 3.8 The other lines in `.env.example`

| Line | Leave it as it is unless... |
|---|---|
| `DATABASE_URL=sqlite:///creditlens.db` | always leave it |
| `LLM_CACHE=1` | keep it: repeated questions are answered from the cache, which saves your free quota |
| `SOAP_WSDL_URL`, `ORACLE_*`, `MLFLOW_TRACKING_URI` | you start the Docker services ([section 4](#4-docker-optional)) |
| `NEO4J_*`, `QDRANT_*`, `PINECONE_API_KEY` | Day 2 retrieval only; not used on Days 3 and 4 |
| `JWT_SECRET` | you run the API container: change it to any long random text |
| `HF_TOKEN` (not in the file; add it if you like) | you want to silence the *unauthenticated requests to the HF Hub* warning: create a *Read* token on [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

---

## 4. Docker (optional)

### 4.1 What Docker is, and why we use it

A bank's systems (an Oracle database, a SOAP service, a model registry) cannot be installed on every laptop by hand.
Docker runs each one in a **container**: a sealed box with the program and everything it needs, started with one
command, the same on every laptop. When you stop it, your laptop is as it was.

Without Docker, the notebooks use **stand-ins** that say they are stand-ins (`sqlite-fallback`, `mock`). With Docker,
the same code talks to the real thing.

**Install.** Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and start it. Windows needs
WSL 2: run `wsl --install` in an Administrator PowerShell and restart. Check: `docker version` prints a **Client** and
a **Server**. (The Docker extension for VS Code is not Docker: install Docker Desktop itself.)

**Four rules:**

1. Run every `docker compose` command **from the repository folder**.
2. Always **name the services** you want, as below. Without names, Compose also starts a SQL Server you do not need.
3. On a laptop with 8 GB of memory, run **one group at a time**, and stop it before starting the next.
4. Check with `docker ps`: wait until the STATUS column says **(healthy)** before you use a service.

| Group | Services and ports | Used by | Section |
|---|---|---|---|
| legacy | Oracle (1521), SOAP (8001), local Jira (8092) | notebook 05b, the workbench | [4.2](#42-legacy-oracle-soap-and-the-local-jira-service) |
| ml | MLflow (5000) | notebooks 06b and 07b | [4.3](#43-ml-the-mlflow-model-registry) |
| core | the CreditLens API (8000) | the Deploy step of 06 and 06b | [4.4](#44-core-the-api-in-a-container) |
| observability, retrieval, security | Langfuse, Qdrant, Neo4j, Keycloak | not used on Days 3 and 4 | [4.5](#45-the-other-groups) |

### 4.2 legacy: Oracle, SOAP and the local Jira service

**Start** (the first time downloads Oracle, about 3 GB, and builds the SOAP image):

```bash
docker compose -f infra/docker-compose.yml --profile legacy pull oracle
docker compose -f infra/docker-compose.yml --profile legacy build soap-mock jira-mock
docker compose -f infra/docker-compose.yml --profile legacy up -d oracle soap-mock jira-mock
docker ps
```

Wait until all three say **(healthy)**. Oracle takes several minutes the very first time, while it creates the
database; after that it starts in about two minutes.

**Connect.** Add to `.env`, then restart the kernel:

```
ORACLE_DSN=localhost:1521/FREEPDB1
ORACLE_USER=creditlens
ORACLE_PASSWORD=creditlens
```

The SOAP service needs no line: the code already looks at `http://localhost:8001/?wsdl`. For the workbench to use the
local Jira service, also add (any email and token work):

```
JIRA_BASE_URL=http://localhost:8092
JIRA_EMAIL=anything@example.com
JIRA_API_TOKEN=anything
```

**See the difference in notebook 05b:**

| | Without Docker | With Docker |
|---|---|---|
| Level 5, the real client | `live call: source=mock` | `live call: source=soap  instalments=12 late=7` |
| Level 6, the adapter | `backend: sqlite-fallback` | `backend: oracle`, and `C001` and `C003` show Oracle disagreeing with our copy (a lesson, not a bug) |
| Level 9, the sensors | `source: sqlite-fallback` | `source: oracle` and `source: soap` |

The four `source=mock` lines just above it, also in Level 5, stay even with Docker: that cell calls an address where
nothing listens, on purpose, to show the circuit breaker opening. `notebooks/05b_connect_the_agent_docker.ipynb` has the trainer's full run with
Docker, to compare.

In the workbench, **Connected systems** (bottom left) changes from *Replica* and *Estimated* to **Live** for Oracle and
SOAP, and Jira shows **Local**.

**Stop** when you are done, and put a `#` in front of the three `ORACLE_` lines (and the `JIRA_` lines):

```bash
docker compose -f infra/docker-compose.yml --profile legacy stop
```

If you leave the `ORACLE_` lines on while Oracle is stopped, everything still works, but each Oracle call waits about
four seconds before it falls back.

### 4.3 ml: the MLflow model registry

Without this, notebooks 06b and 07b keep the model registry in a file, `mlruns.db`. With it, they use a registry
server, as a bank's platform team would.

**Start:**

```bash
docker compose -f infra/docker-compose.yml --profile ml up -d mlflow
```

Wait about a minute, then open http://localhost:5000: the MLflow page appears.

**Connect.** In `.env`, change the MLflow line to the following, then restart the kernel:

```
MLFLOW_TRACKING_URI=http://localhost:5000
```

**Run notebook 06b, then 07b.** In 06b Level 1 you see lines like *View run challenger_gbm at:
http://localhost:5000/...* and `Registered creditlens-risk version 1 as 'champion'`. On the MLflow page: **Models**
→ `creditlens-risk` → the alias **champion**. In 07b Level 7 you see `registry: http://localhost:5000`, then
`rollback verified: the registry now serves the previous version.`, then the alias restored.

**Stop, in this order:** first set the line back to `MLFLOW_TRACKING_URI=mlruns` and restart the kernel, then

```bash
docker compose -f infra/docker-compose.yml --profile ml stop
```

The order matters: with the line pointing at a stopped server, MLflow keeps retrying for minutes and the notebook
looks frozen.

**Without Docker, to browse the file registry:** `mlflow ui --backend-store-uri sqlite:///mlruns.db`, then open
http://localhost:5000 (Ctrl+C to stop). Not at the same time as the container: both use port 5000.

### 4.4 core: the API in a container

This is the **Deploy** step on your laptop: the CreditLens API, built into an image and run as a container. It needs a
`.env` file to exist ([section 3.2](#32-create-your-env-file)).

```bash
docker compose -f infra/docker-compose.yml --profile core up -d --build api
```

The first build takes several minutes. Then open http://localhost:8000/health (a short status) and
http://localhost:8000/docs (every endpoint, with a *Try it out* button). Stop it:

```bash
docker compose -f infra/docker-compose.yml --profile core stop
```

### 4.5 The other groups

The compose file has more services. They are not used on Days 3 and 4, and they were not part of the course run.

| Group | Start | Then in `.env` | Note |
|---|---|---|---|
| observability (Langfuse, six containers) | `docker compose -f infra/docker-compose.yml --profile observability up -d` | create a project at http://localhost:3000, copy its keys, `LANGFUSE_HOST=http://localhost:3000` | heavy for 8 GB: use Langfuse Cloud ([3.6](#36-langfuse-traces)) instead |
| retrieval (Qdrant, Neo4j) | `docker compose -f infra/docker-compose.yml --profile retrieval up -d qdrant neo4j` | `QDRANT_URL=http://localhost:6333`, `NEO4J_URI=bolt://localhost:7687`, `NEO4J_PASSWORD=creditlens123` | Day 2 retrieval |
| security (Keycloak) | `docker compose -f infra/docker-compose.yml --profile security up -d keycloak` | none | admin page at http://localhost:8080 (admin / admin) |

### 4.6 Stop, clean up, free the disk

| Command | What it does |
|---|---|
| `docker compose -f infra/docker-compose.yml --profile legacy stop` | stops the containers; everything is kept (use the group you started) |
| `docker compose -f infra/docker-compose.yml --profile legacy down` | removes the containers; the data in volumes is kept |
| `docker compose -f infra/docker-compose.yml --profile legacy down -v` | removes the containers **and** their data (Oracle's database, MLflow's runs) |

[infra/teardown.md](infra/teardown.md) explains how to remove everything, images included.

---

## 5. Every notebook, in order

| # | Notebook | Day | Phase | Time | Keys | Docker |
|---|---|---|---|---|---|---|
| 1 | `04_agents` | 3 | Build | under a minute | no | no |
| 2 | `04b_underwriter_agent` | 3 | Build | under a minute | optional | no |
| 3 | `05_integrations` | 3 | Build | about a minute | no | optional |
| 4 | `05b_connect_the_agent` | 3 | Build | under a minute | no | optional: legacy |
| 5 | `05b_connect_the_agent_docker` | 3 | Build | read only | no | no |
| 6 | the Credit Workbench | 3 | Deploy | a minute to start | optional | optional: legacy |
| 7 | `06_mlops_security_deploy` | 4 | Deploy | about eight minutes | no | optional |
| 8 | `06b_secure_and_ship` | 4 | Deploy | about five minutes | no | optional: ml |
| 9 | `07_evaluate_maintain` | 4 | Evaluate | about two minutes | optional: live and traces | no |
| 10 | `07b_run_and_maintain` | 4 | Maintain | about two minutes | no | optional: ml |
| 11 | `08_story_scenarios` | 4 | Evaluate | seconds | no | no |
| 12 | `09_test_everything` | 4 | Evaluate | about nine minutes | no | no |

Notebooks marked *reference* in the [README](README.md) (04, 05, 06, 07) are the earlier, broader versions. The *b*
notebooks continue the Underwriter Agent story. Run both kinds; the order above works.

For every notebook: open it, **Restart**, **Run All**, read from the top. *What you should see at the end* is how you
know it worked.

### 04: Agents that draft the credit memo (reference)

Tools, a supervisor agent, MCP and a Jira ticket, for application A001. It sets offline mode itself. Read section 5
for the first look at MCP.

### 04b: Build the Underwriter Agent

The story: Amaira Dey, Head of Credit at Good Bank, asks for help with her queue: *"Do the first four jobs for me.
Never the fifth."* You build the agent from zero: PEAS and the architecture (stage 0), then a reflex agent, a
model-based agent, a goal-based agent, prompting, a utility-based agent, a learning agent, a multi-agent review
(stages 1 to 7), one LangGraph with a human gate (stage 8), and an evaluation (stage 9).

**What you should see at the end:**

```
Facts: the final agent's decisions on the 30 applications: right decision on 30/30 · rule breaches found 33/33 · wrong: []
PASS the note always waits for a signature
PASS an analyst cannot sign
PASS injected text cannot move the proposal
PASS no approval language in any note
PASS every proposal cites a reason
```

### 05: The bank's legacy systems (reference)

Oracle, SOAP, Slack and the end-to-end run, in their first form. Every adapter falls back when its system is missing,
so it runs without Docker.

### 05b: Connect the agent to the bank

Levels 0 to 10: the integration map, a naive call, a timeout and retries, a circuit breaker, a labelled fallback, an
adapter (Oracle's names to ours), idempotent ticket writes through MCP, Slack as a new channel, two new sensors for
the agent, and integration tests.

Expected along the way, on purpose: `naive call failed: ConnectionError service error 503` (the service is made
flaky), four `source=mock` lines in Level 5 (a dead address, so the circuit breaker opens), and `CREDIT-1 created=True`, then
`created=False` (a second write files nothing new).

**What you should see at the end:**

```
PASS the adapter returns our schema, not Oracle's
PASS a dead service still answers, quickly, labelled as mock
PASS the breaker is open after repeated failures
PASS a second ticket write files nothing new
PASS every fact carries a source
```

With Docker: [section 4.2](#42-legacy-oracle-soap-and-the-local-jira-service) shows what changes.

### 05b with Docker (read only)

`05b_connect_the_agent_docker.ipynb` keeps the outputs of the trainer's run against the real Oracle and SOAP
services. Open it and compare with your own 05b. Do not run it unless you have started the legacy group.

### The Credit Workbench

After 04b and 05b, see the agent inside the web application a credit team would use: [section 6](#6-the-credit-workbench).

### 06: MLOps, security, Docker, CI and the cloud (reference)

A model trained on the real SBA loan book and tracked in MLflow, the fairness check, personal-data masking,
guardrails, a mock sign-in with Keycloak, the container, continuous integration with an evaluation gate, and the AWS
deploy. It needs the real data from setup step 1.3 (`data_real`). The
training is why it takes about eight minutes.

### 06b: Secure the agent and ship it

Levels 1 to 9, each ticking one condition of a go-live checklist: a real risk model (stub, baseline and challenger,
registered in MLflow), its assumptions written down, fairness, access by role and region, no personal data to a
model, guardrails, secrets out of code, the evaluation gate, and the container.

Level 1 prints a comparison like this (your version number may differ):

```
stub_rules         AUC 0.4621  [rules, no training: the floor]
baseline_logreg    AUC 0.8318  [sklearn.LogisticRegression]
challenger_gbm     AUC 0.9653  [sklearn.HistGradientBoostingClassifier]
Registered creditlens-risk version 1 as 'champion'
```

**What you should see at the end:** nine `[x]` lines, then `9 of 9 conditions met, each with printed evidence.`

With Docker: [section 4.3](#43-ml-the-mlflow-model-registry) (the registry server).

### 07: Evaluate, maintain and hand off

Langfuse, the eval court (a judge model calibrated against people), gateway resilience, and four scripted incidents.
Cell 2 tells you whether model calls are live or offline ([section 3.5](#35-what-goes-live-with-a-key)). With
Langfuse keys, open your project's traces while you read section 2.

**What you should see at the end:** four incident reports, the last one ending `Written to :
...monitoring/reports/incident_prompt_injection.json`.

### 07b: Run the agent, and keep it working

Levels 1 to 8: where the time goes, Amaira's feedback, a calibrated judge, resilience when the model provider fails,
four incidents, a policy change replayed on the agent, a rollback proved, and the runbook. **Run 06b first:** Level 7
rolls back to the model version before the champion (if there is only one, it trains a second, about a minute).

Level 7 prints `rollback verified: the registry now serves the previous version.`

**What you should see at the end:** the runbook table, from *Golden set and answer key* to *Incident response*.

### 08: The hard cases, one at a time

Six borrowers: Rohan (looks safe on Monday), Rahul (looks weak on Monday), Meena (an honest baker with a bad monsoon),
Tapti Trading (a quiet account that suddenly fills up), Sagar Wellness (declared healthcare) and Konark Exports
(numbers that do not add up).

**What you should see at the end:**

```
  application                                        flags
0        A901  GROUP, RULE_5_2, ROUND_TRIP, RELATED_INFLOW
1        A902             GROUP, PASS_THROUGH, REVENUE_GAP
2        A903                   RULE_4_1, SEASONAL_PATTERN
3        A904                              WINDOW_DRESSING
4        A905                              SECTOR_MISMATCH
5        A906                         REVENUE_GAP, FOOTING
```

### 09: Test everything we built

Every kind of test from the course, level by level, then a scoreboard: [section 7](#7-the-tests).

---

## 6. The Credit Workbench

The Underwriter Agent from 04b, with the connections from 05b, inside a web application: a work queue by region, a
credit review you run on demand, a decision only the Head of Credit can submit, exception tickets, an audit log, and
platform screens (statement screening, connected systems, models, release readiness, monitoring).

**Start:** in a terminal, in the repository folder, with `.venv` active:

```bash
python app/workbench/server.py
```

It prints `Good Bank Credit Workbench on http://localhost:8090`. Leave the terminal open. Open http://localhost:8090
and wait until the top bar says **Review assistant online** (a minute or two).

**Try:**

| Do | You should see |
|---|---|
| User menu (top right) → *Lucky Grewal · Credit Analyst, West* | only the six West applications |
| Open *A901, Rohan Packaging Pvt Ltd* → **Run credit review** | seven steps complete; the recommendation is *Counter-offer · ₹1.50 Cr* |
| **Submit decision** as Lucky | refused: an analyst has no approval authority; the attempt is in the **Audit log** |
| Switch to *Amaira Dey · Head of Credit*, open A901, add a condition, **Submit decision** | the decision is recorded; the condition appears under **Learned conditions** |
| **Raise exceptions** twice | *CREDIT-1 raised*, then *CREDIT-1 already exists* |

**Stop:** Ctrl+C in the terminal. **Start clean:** delete `data/cache/workbench_audit.jsonl` and
`data/cache/jira_workbench.json` before starting. More: [app/workbench/README.md](app/workbench/README.md) and the
[walkthrough with screenshots](docs/workbench_walkthrough.html).

---

## 7. The tests

**In a notebook:** open `notebooks/09_test_everything.ipynb`, Restart, Run All. About nine minutes. It runs each kind
of test (data, extraction, the agent, integration and MCP, security, the model and the gate, agent behaviour,
monitoring, the hard cases, the app, safety nets) and ends with a scoreboard. Its last row looks like this (the
number of tests grows as the course adds tests):

```
all                                                208    208      0       0
Green: safe to ship.
```

The columns are tests, passed, failed and skipped. What matters: *passed* equals *tests*, and *failed* and
*skipped* are 0. If your laptop is short of memory (8 GB, with Docker and other notebooks open), close them first:
tests can fail with `MemoryError` when the laptop runs out. Level 9 prints `FAIL` lines on purpose: it grades a made-up bad
run (memo written before reading the policy, a forbidden tool), and the check must fail it.

**In a terminal:**

```bash
# Windows (PowerShell)
$env:LLM_OFFLINE="1"; python -m pytest -m "not slow" -q

# Mac / Linux
LLM_OFFLINE=1 python -m pytest -m "not slow" -q
```

The slow tests load a model or start the whole workbench: run them with `-m slow` in place of `-m "not slow"`.

---

## 8. Host the workbench on AWS (optional)

The last **Deploy** step: the workbench, built into a container image, pushed to **Amazon ECR**, and run on an
**Amazon EC2** server you open in your browser. Then you delete it all.

Every command is in [infra/aws/workbench_on_aws.md](infra/aws/workbench_on_aws.md), with what you should see after
each one. In short:

1. A budget alarm, then a deploy user with [infra/aws/policy.json](infra/aws/policy.json) ([infra/aws/README.md](infra/aws/README.md) steps 1 and 2).
2. `docker build -f infra/Dockerfile.workbench -t creditlens-workbench:latest .`, and try it on your laptop.
3. Push the image to ECR.
4. A role that lets the server pull the image.
5. A `t3.small` server, reachable only from your own address.
6. Open `http://<server address>:8090`.
7. `bash infra/aws/teardown.sh --yes`.

**About cost.** On an AWS account created on or after 15 July 2025 with the **Free plan**, this runs on the account's
credits, and the account is not charged unless you upgrade. On an older account, `t3.small` is not in the free tier:
it costs a small amount for every hour it exists, so delete it the same day. The guide explains both, with links to
AWS's own pages.

---

## 9. When something goes wrong

| What you see | What to do |
|---|---|
| `ModuleNotFoundError: No module named 'creditlens'` (or any module) | The environment is not active, or the notebook uses another kernel. Activate `.venv` in the terminal; in the notebook, **Select Kernel** → `.venv`. Then `pip install -e ".[all]"` again |
| Windows 11: `DLL load failed ... An Application Control policy has blocked this file` | Windows' **Smart App Control** blocked a library file that `pip` just downloaded. It is a Windows setting, not a fault in the project: Windows Security → **App & browser control** → **Smart App Control settings**. Read Windows' warning before you change it, and on a work laptop ask your IT team. Google Colab is the alternative |
| `FileNotFoundError: data/real/sba_loans.csv` (06, 06b, 07b) | `python -m creditlens.data_real --only sba receipts sop`, then Restart and Run All |
| `FileNotFoundError` for anything under `data/raw` or `data/story` | Run the setup commands in 1.3 again, in order |
| A notebook runs old code after `git pull` | Ctrl+Shift+P → *File: Revert File*, then Restart |
| A notebook seems frozen at the MLflow step | `MLFLOW_TRACKING_URI` points at a server that is not running. Set it to `mlruns`, restart the kernel |
| Every Oracle call takes about four seconds | The `ORACLE_` lines are on in `.env` but Oracle is not running. Put `#` in front of them, or start the legacy group |
| You changed `.env` and nothing changed | Restart the kernel (or the workbench) |
| The workbench says the port is in use | Another workbench is running. Close it, or start on another port: PowerShell `$env:WORKBENCH_PORT=8091`, Mac `export WORKBENCH_PORT=8091`, then open http://localhost:8091 |
| *Review assistant starting* for more than two minutes | Read the terminal for an error. Usually the story data is missing: `python -m creditlens.scenarios` and `python -m creditlens.scenarios.story build`, then restart |
| `docker: command not found`, or `docker ps` fails | Docker Desktop is not installed or not started |
| A container is not `(healthy)` | Wait (Oracle takes minutes the first time), then `docker logs creditlens-oracle` (or the container's name) |
| `port is already allocated` when starting a container | Something else uses that port. `docker ps` shows which container; stop it |
| Windows: *WSL installation appears to be corrupted* | `wsl --install` in an Administrator PowerShell, let it repair, restart |
| Windows: *Virtual Machine Platform not enabled* | `dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart` as Administrator, then restart |
| *Failed to export spans batch* | A batch of Langfuse traces was not sent. Harmless |
| *unauthenticated requests to the HF Hub* | Harmless. Add `HF_TOKEN` to `.env` to silence it ([3.8](#38-the-other-lines-in-envexample)) |
| A model call fails with a rate-limit or quota message | The free tier's limit. Wait a minute, or set `LLM_OFFLINE=1` |

---

## 10. The whole run on one page

**Once:**

- [ ] Python 3.11 or 3.12, Git, VS Code with the Python and Jupyter extensions
- [ ] `git clone`, `.venv`, `pip install -e ".[all]"`
- [ ] the five data commands in 1.3, including `data_real`
- [ ] `python -m creditlens.setup_check` says OK for the data and the packages

**Day 3 (Build):**

- [ ] 04 → 04b (five PASS lines) → 05 → 05b (five PASS lines) → read 05b with Docker
- [ ] the workbench: Rohan's review shows *Counter-offer · ₹1.50 Cr*
- [ ] optional: the legacy Docker group, then 05b and the workbench again

**Day 4 (Deploy, Evaluate, Maintain):**

- [ ] 06 → 06b (`9 of 9 conditions met`) → 07 → 07b (`rollback verified`) → 08 (six cases) → 09 (`Green: safe to ship.`)
- [ ] optional: keys in `.env`, then 07 live and traces in Langfuse
- [ ] optional: the ml Docker group, then 06b and 07b against the registry server
- [ ] optional: the core Docker group (the API container)
- [ ] optional: the workbench on AWS, then `teardown.sh --yes`

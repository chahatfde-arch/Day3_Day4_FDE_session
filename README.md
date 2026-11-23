# CreditLens AI · Days 3 and 4 · TAN GLOBUS AI

Days 1 and 2 made Good Bank's data trustworthy and searchable. Day 3 builds an AI agent on top of it, from zero, and connects it to the bank's systems. Day 4 takes it live: secure it, ship it, run it, keep it working, and test it against six hard cases.

**The story.** Amaira Dey, Head of Credit at Good Bank, asks for help with her queue: *"Do the first four jobs for me. Never the fifth."* The Underwriter Agent gathers the evidence, checks every policy rule, weighs the options and writes a review note. Then it stops, because only Amaira signs.

**Start here: [RUN_END_TO_END.md](RUN_END_TO_END.md).** Step by step: setup, every key (all optional), every Docker service, each notebook in order with what you should see, the Credit Workbench, the tests, and hosting the workbench on AWS.

## What is here

| | |
|---|---|
| `notebooks/04_agents.ipynb` | Agents that draft the credit memo: tools, a supervisor, MCP and Jira (reference) |
| `notebooks/04b_underwriter_agent.ipynb` | **Day 3, build:** the Underwriter Agent from zero, through seven agent types, prompting, and a LangGraph with a human gate |
| `notebooks/05_integrations.ipynb` | The bank's legacy systems (reference) |
| `notebooks/05b_connect_the_agent.ipynb` | **Day 3, connect:** Oracle, SOAP, Jira through MCP and Slack, with timeouts, retries, a circuit breaker, labelled fallbacks and idempotent writes |
| `notebooks/05b_connect_the_agent_docker.ipynb` | Notebook 05b with the outputs saved from the trainer's run against the real Oracle and SOAP systems in Docker (read it; no Docker needed) |
| `notebooks/06_mlops_security_deploy.ipynb` | MLOps, security, Docker, continuous integration and the cloud (reference) |
| `notebooks/06b_secure_and_ship.ipynb` | **Day 4, secure and ship:** a real risk model, its assumptions, fairness, access by region, masking, guardrails, secrets, the evaluation gate and the container |
| `notebooks/07_evaluate_maintain.ipynb` | Evaluate and maintain: tracing, the eval court, gateway resilience, four incidents (reference) |
| `notebooks/07b_run_and_maintain.ipynb` | **Day 4, run and maintain:** where the time goes, approver feedback, a calibrated judge, resilience, incidents, a policy change, rollback and the runbook |
| `notebooks/08_story_scenarios.ipynb` | **Day 4, the hard cases:** six borrowers, one case at a time |
| `notebooks/09_test_everything.ipynb` | **Day 4, test everything:** every kind of test we met, level by level, then a scoreboard ([how to run the tests](#run-the-tests)) |
| `app/workbench/` | **The Good Bank Credit Workbench:** the agent inside the credit team's web application, with its platform screens: statement screening, connected systems, model governance, release readiness and monitoring ([how to run it](#run-the-credit-workbench), [walkthrough with screenshots](docs/workbench_walkthrough.html), [the screens](app/workbench/README.md)) |
| `infra/docker-compose.yml` | The bank's systems as Docker services (Oracle, the SOAP service, MLflow, ...) |
| `infra/Dockerfile.workbench`, `infra/aws/` | The Credit Workbench as one container image, and how to push it to Amazon ECR and host it on EC2 ([workbench on AWS](infra/aws/workbench_on_aws.md)) |
| `src/creditlens/` | The project's code |
| `tests/` | The automated tests that guard it |
| `docs/session3_architecture.html` | **The big picture:** the pipeline, the architecture, the flow chart of one credit review, and why the agent is built this way (open in a browser) |
| `docs/agents.md`, `docs/integrations.md` | Notes for Day 3 |
| `docs/mlops.md`, `docs/evaluate_and_maintain.md` | Notes for Day 4 |

Everything about Good Bank and its borrowers is synthetic. No API key is needed: everything runs offline by default.

## Setup (about 45 minutes, most of it downloads)

You need Python 3.11 or 3.12, Git, and VS Code with the Python and Jupyter extensions.

```bash
git clone https://github.com/chahatfde-arch/Day3_Day4_FDE_session.git
cd Day3_Day4_FDE_session

# Windows (PowerShell)              # Mac / Linux
py -3.12 -m venv .venv               # python3 -m venv .venv
.venv\Scripts\Activate.ps1           # source .venv/bin/activate

pip install --upgrade pip
pip install -e ".[all]"

python -m creditlens.generators
python -m creditlens.pipeline --load
python -m creditlens.scenarios
python -m creditlens.scenarios.story build
python -m creditlens.data_real --only sba receipts sop

python -m creditlens.setup_check
```

If PowerShell refuses to run `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

`creditlens.data_real` needs internet. It downloads small samples of three real public datasets: the SBA loan book that trains the risk model in notebook 06b, scanned receipts for the extraction check, and the SBA rulebook for the policy search. Without it, notebooks 06, 06b and 07b stop with `FileNotFoundError: data/real/sba_loans.csv`, and the evaluation gate in notebook 09 fails. **Already set up before this line was added?** Run `git pull`, then just this one command.

**Check it works:** start the Credit Workbench (next section). If Rohan's credit review shows a counter-offer of ₹1.50 Cr, your setup is complete.

**On Google Colab:** open a notebook from this repository and run its first cell. It clones the repository, installs what the notebook needs and generates the data. Docker does not run on Colab.

## Run the Credit Workbench

The Credit Workbench is the Underwriter Agent from notebook 04b, with the connections from notebook 05b, inside the web application a credit team would use: a work queue by region, a credit review you run on demand, a decision only the Head of Credit can submit, exception tickets in Jira, and an audit log.

**1. Start it.** From the repository folder, with your environment activated (setup steps above):

```bash
python app/workbench/server.py
```

The terminal prints `Good Bank Credit Workbench on http://localhost:8090`. Leave this terminal open: the workbench runs as long as it does.

**2. Open it.** Go to http://localhost:8090 in your browser. The top bar says *Review assistant starting* for about a minute, because the server runs notebook 04b's code once at start-up. Wait until it says *Review assistant online*.

**3. Use it.**

| Try | What you should see |
|---|---|
| Open the user menu (top right) and choose *Lucky Grewal · Credit Analyst, West* | the work queue shows only the six West applications |
| Open *A901, Rohan Packaging Pvt Ltd* and click **Run credit review** | seven steps complete in a few seconds; the recommendation is *Counter-offer · ₹1.50 Cr* |
| Click **Submit decision** as Lucky | refused: an analyst has no approval authority; the attempt is in the **Audit log** |
| Switch to *Amaira Dey · Head of Credit*, open A901, add a condition, **Submit decision** | the decision is recorded; the condition appears under **Learned conditions** |
| Click **Raise exceptions** twice | *CREDIT-1 raised*, then *CREDIT-1 already exists*: nothing is filed twice |
| Type `#/application/A001` at the end of the address as Lucky | *Access denied*: A001 is a North application |

The [walkthrough with screenshots](docs/workbench_walkthrough.html) goes through every screen. To read it with its pictures, open the file from your clone in a browser (on GitHub, use *Download raw file*).

**4. Stop it.** Press **Ctrl+C** in the terminal.

**Start again from a clean state.** Reviews live in the server's memory, so restarting clears them. To also clear the audit log and the test Jira tickets, delete `data/cache/workbench_audit.jsonl` and `data/cache/jira_workbench.json` before starting.

**With the real bank systems.** Start Oracle and the SOAP service in Docker (next section), add the three Oracle lines to `.env`, then restart the workbench. Under *Connected systems* (bottom left), core banking and repayments change from *Replica* and *Estimated* to *Live*, and the data-source notices disappear.

**If it does not start**

| Problem | What to do |
|---|---|
| `ModuleNotFoundError` | Your environment is not active, or the install did not finish: activate `.venv` and run `pip install -e ".[all]"` again |
| *Review assistant starting* for more than two minutes | Look at the terminal for an error. Usually the data is missing: run `python -m creditlens.scenarios` and `python -m creditlens.scenarios.story build`, then restart |
| The terminal says the port is in use (on Windows: *only one usage of each socket address*) | Another program, or an earlier workbench, is using port 8090. Close it, or: PowerShell: `$env:WORKBENCH_PORT=8091`; Mac: `export WORKBENCH_PORT=8091`. Start again and open http://localhost:8091 |
| *Run credit review* is greyed out | The application has already been decided, or the review assistant is still starting |

More detail: [app/workbench/README.md](app/workbench/README.md).

## Run the tests

Open `notebooks/09_test_everything.ipynb` and run all. It runs every kind of test from Days 1 to 4, level by level, and ends with a scoreboard. Everything is offline: no API key, no Docker. It takes about nine minutes. When the last line says *Green: safe to ship*, every test passed.

From a terminal instead:

```bash
# Windows (PowerShell)
$env:LLM_OFFLINE="1"; python -m pytest -m "not slow" -q

# Mac / Linux
LLM_OFFLINE=1 python -m pytest -m "not slow" -q
```

`LLM_OFFLINE=1` keeps the tests offline even if you have put an API key in `.env`. The slow tests load a model or start the whole workbench. Run them with `-m slow` in place of `-m "not slow"`.

## Docker (optional)

Notebook 05b runs without Docker, using stand-ins that say they are stand-ins. With Docker, a real Oracle database and a real SOAP service run on your laptop.

1. Install Docker Desktop (Windows needs WSL 2: `wsl --install` in an Administrator PowerShell, then restart). Check that `docker version` prints a Client and a Server.
2. From this folder:

```bash
docker compose -f infra/docker-compose.yml --profile legacy pull oracle
docker compose -f infra/docker-compose.yml --profile legacy build soap-mock
docker compose -f infra/docker-compose.yml --profile legacy up -d oracle soap-mock
docker ps                      # wait until both say (healthy), about two minutes
```

3. To use them, add to `.env` (copy `.env.example` first), then restart the notebook's kernel:

```
ORACLE_DSN=localhost:1521/FREEPDB1
ORACLE_USER=creditlens
ORACLE_PASSWORD=creditlens
```

4. Stop them afterwards: `docker compose -f infra/docker-compose.yml --profile legacy stop`.

On a laptop with 8 GB of RAM, run one group of containers at a time. `infra/teardown.md` explains how to remove everything.

**Windows troubleshooting.** If you see "WSL installation appears to be corrupted", run `wsl --install` in an Administrator PowerShell and let it repair. If the log says "Virtual Machine Platform not enabled", run `dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart` as Administrator, then restart. "Error 740" means the window was not run as Administrator. The Docker extension for VS Code is not Docker: install Docker Desktop itself.

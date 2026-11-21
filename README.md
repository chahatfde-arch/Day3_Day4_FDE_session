# CreditLens AI · Days 3 and 4 · TAN GLOBUS AI

Days 1 and 2 made Good Bank's data trustworthy and searchable. Day 3 builds an AI agent on top of it, from zero, and connects it to the bank's systems.

**The story.** Amaira Dey, Head of Credit at Good Bank, asks for help with her queue: *"Do the first four jobs for me. Never the fifth."* The Underwriter Agent gathers the evidence, checks every policy rule, weighs the options and writes a review note. Then it stops, because only Amaira signs.

## What is here

| | |
|---|---|
| `notebooks/04_agents.ipynb` | Agents that draft the credit memo: tools, a supervisor, MCP and Jira (reference) |
| `notebooks/04b_underwriter_agent.ipynb` | **Day 3, build:** the Underwriter Agent from zero, through seven agent types, prompting, and a LangGraph with a human gate |
| `notebooks/05_integrations.ipynb` | The bank's legacy systems (reference) |
| `notebooks/05b_connect_the_agent.ipynb` | **Day 3, connect:** Oracle, SOAP, Jira through MCP and Slack, with timeouts, retries, a circuit breaker, labelled fallbacks and idempotent writes |
| `app/workbench/` | **The Good Bank Credit Workbench:** the agent inside the credit team's web application ([how to run it](#run-the-credit-workbench), [walkthrough with screenshots](docs/workbench_walkthrough.html)) |
| `infra/docker-compose.yml` | The bank's systems as Docker services (Oracle, the SOAP service, MLflow, ...) |
| `src/creditlens/` | The project's code |
| `docs/session3_architecture.html` | **The big picture:** the pipeline, the architecture, the flow chart of one credit review, and why the agent is built this way (open in a browser) |
| `docs/agents.md`, `docs/integrations.md` | Notes for Day 3 |

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

python -m creditlens.setup_check
```

If PowerShell refuses to run `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

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

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
| `app/workbench/` | **The Good Bank Credit Workbench:** the agent inside the credit team's web application ([README](app/workbench/README.md)) |
| `infra/docker-compose.yml` | The bank's systems as Docker services (Oracle, the SOAP service, MLflow, ...) |
| `src/creditlens/` | The project's code |
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

**Check it works:** `python app/workbench/server.py`, then open http://localhost:8090. After about a minute the top bar says *Review assistant online*. Open *A901, Rohan Packaging Pvt Ltd* and click *Run credit review*: the recommendation is a counter-offer of ₹1.50 Cr.

**On Google Colab:** open a notebook from this repository and run its first cell. It clones the repository, installs what the notebook needs and generates the data. Docker does not run on Colab.

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

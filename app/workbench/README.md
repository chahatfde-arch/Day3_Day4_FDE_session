# Good Bank Credit Workbench

The Underwriter Agent from notebook 04b, connected the way notebook 05b connects it, inside the credit team's workbench.

```
python app/workbench/server.py
```

Then open http://localhost:8090. The top bar says **Review assistant starting** for about a minute (the server is running notebook 04b's own code once), then **Review assistant online**.

## Try this

1. Switch the user (top right) to **Lucky Grewal, Credit Analyst, West**. The work queue shows only the six West applications.
2. Open **A901, Rohan Packaging Pvt Ltd** and click **Run credit review**. Watch the seven steps complete, then read the Summary tab: a counter-offer of ₹1.50 Cr, its rationale, and a point for the approver's attention.
3. Click **Submit decision** as Lucky. The decision is refused: only the Head of Credit has approval authority (credit policy 7.3). The attempt appears in the **Audit log**.
4. Switch to **Amaira Dey, Head of Credit**. Keep the proposed condition, add one of your own, and submit the decision. Open **Learned conditions**.
5. Click **Raise exceptions** twice. The second time the ticket already exists, so nothing new is raised (idempotency key).
6. Bonus: start Docker (`docker compose -f infra/docker-compose.yml --profile legacy up -d oracle soap-mock`), add `ORACLE_DSN=localhost:1521/FREEPDB1`, `ORACLE_USER=creditlens` and `ORACLE_PASSWORD=creditlens` to `.env`, and restart the workbench. Under *Connected systems*, core banking and repayments turn **Live**, and the data-source notices disappear. To see exception tickets go to a Jira-shaped service instead of the test file, also start `jira-mock` (port 8092; the workbench itself uses 8090) and set `JIRA_BASE_URL=http://localhost:8092`, with `JIRA_EMAIL` and `JIRA_API_TOKEN` set to anything: the tickets row turns **Local**, never Live.

## The platform screens

| Screen | What it shows | Notebook |
|---|---|---|
| **Statement screening** (Head of Credit, Auditor) | The six hard cases A901 to A906: each signal with its evidence | 08 |
| **Connected systems** | Each bank system, its protocol, resilience pattern and status: core banking Live or Replica, repayments Live or Estimated, exception tickets Test project, Local (the local Jira service) or Live (a Jira Cloud site), notifications Test mode, the model gateway Offline or Live; **Test connections** probes them. Every Docker service with its profile, image, ports and state | 05b |
| **Model governance** | The champion model, both candidates on the same test set, the top drivers, fairness by segment, the eleven assumptions | 06b levels 1 to 3 |
| **Release readiness** | The nine go-live conditions, re-checked live with evidence; the evaluation gate (run it, or test the candidate judge prompt); the offline test suite (run it); the CI pipeline and the AWS deployment steps | 06b, 06, testing |
| **Monitoring** | Where review time goes and how often the approver keeps the proposal (both from this server's own reviews and audit log); the model provider chain and the resilience drill; drift; the four incident drills | 07b |

Long checks (the gate, the tests, drift, the incident drills) run as separate processes on the server. Starting one needs the Head of Credit or the Auditor; an analyst is refused, and the refusal is in the audit log. The code is `platform_api.py`.

## What is behind the screen

| Piece | Where |
|---|---|
| The review assistant: notebook 04b's code, run once at start-up, its seven nodes wired into one LangGraph | `server.py`, `load_agent` and `build_graph` |
| The credit review, streamed step by step (server-sent events) | `server.py`, `prepare` |
| The approval-authority check before the paused run is resumed, and the audit event | `server.py`, `sign` |
| Core banking (Oracle) and the repayment service (SOAP): live with Docker, a replica or an estimate without | `src/creditlens/integrations/` |
| Exception tickets in Jira through MCP, with an idempotency key (test file `data/cache/jira_workbench.json`) | `src/creditlens/mcp_jira/` |
| The page: one HTML file, no build step | `index.html` |

The audit log is written to `data/cache/workbench_audit.jsonl`. Reviews live in the server's memory: restart it to start clean. It always runs offline, so no API key is needed, even when `.env` says `LLM_OFFLINE=0`. To use the live model, set `WORKBENCH_LIVE=1` before starting (start-up then takes several minutes). Langfuse tracing follows the same switch: offline the workbench sends no traces, even with Langfuse keys in `.env`, because every model call is the stub; with `WORKBENCH_LIVE=1` each call is traced. To use another port, set `WORKBENCH_PORT` before starting.

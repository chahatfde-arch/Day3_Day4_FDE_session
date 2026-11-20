# Good Bank Credit Review Workbench

The Underwriter Agent from notebook 04b, connected the way notebook 05b connects it, inside the app a credit team would actually use.

```
python app/workbench/server.py
```

Then open http://localhost:8090. The header says **Agent starting** for about a minute: the server is running notebook 04b's own code once. When it says **Agent ready**, pick a file and ask the agent to prepare the review.

## Try this

1. Sign in as **Lucky Grewal** (analyst, West). You see only the six West files.
2. Open **Rohan Packaging (A901)** and ask the agent to prepare the review. Watch the eight steps arrive.
3. Try to **sign** it as Lucky. The server refuses: only the Head of Credit decides (policy 7.3). The attempt is in the audit trail.
4. Sign in as **Amaira Dey** (Head of Credit). Choose counter-offer, keep the condition the agent proposes, add one of your own, and sign. Open **Agent memory**.
5. **File the exception tickets** twice. The second time nothing new is filed (idempotency key).
6. Bonus: start Docker (`docker compose -f infra/docker-compose.yml --profile legacy up -d oracle soap-mock`), add `ORACLE_DSN=localhost:1521/FREEPDB1`, `ORACLE_USER=creditlens` and `ORACLE_PASSWORD=creditlens` to `.env`, and restart the workbench. The Oracle and SOAP chips turn green.

## What is behind the screen

| Piece | Where |
|---|---|
| The agent: notebook 04b's code, run once at start-up, its seven nodes wired into one LangGraph | `server.py`, `load_agent` and `build_graph` |
| The live review, streamed step by step (server-sent events) | `server.py`, `prepare` |
| The role check before the paused run is resumed, and the audit event | `server.py`, `sign` |
| Oracle core banking and the SOAP repayment service (live with Docker, labelled stand-ins without) | `src/creditlens/integrations/` |
| Jira through MCP with an idempotency key (practice file `data/cache/jira_workbench.json`) | `src/creditlens/mcp_jira/` |
| The page: one HTML file, no build step | `index.html` |

The audit trail is written to `data/cache/workbench_audit.jsonl`. Reviews live in the server's memory: restart it to start clean. Offline by default, so no API key is needed. Another port: set `WORKBENCH_PORT` before starting.

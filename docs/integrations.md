# Enterprise integrations (Day 5)

*Beginner notes for milestone 7. Code: `src/creditlens/integrations/`.*

## Why banks keep legacy systems

Good Bank's existing loans live in an Oracle core banking system that is older than most of the
team. Repayment history comes from a SOAP web service another department owns. Neither will be
replaced for CreditLens AI, and neither should be: they are audited, they work, and the regulator
knows them. An FDE integrates with them through **adapters**, thin pieces of code that speak the
legacy dialect on one side and our schema on the other.

Every adapter here has a built-in fallback (SQLite for Oracle, a deterministic mock for SOAP, a
dry-run for Slack). The fallback is not a shortcut: it is itself an adapter, and it is what lets
the whole system run in Colab, in CI and in a demo where the real service is down.

## Oracle core banking

`CoreBankingAdapter.get_exposure(company_id)` returns `{"company_id", "outstanding_inr_000", "product", ...}`.

- With `ORACLE_DSN`, `ORACLE_USER`, `ORACLE_PASSWORD` set, it queries Oracle Database Free through
  `python-oracledb` (`docker compose --profile legacy up` starts the container and loads
  `infra/oracle/init.sql`). Oracle Free needs a few GB of memory and starts slowly the first time.
- Otherwise it builds a SQLite table with **Oracle's column names** from `existing_loans.csv`.

The Day 5 curveball: Oracle calls things `CUST_ID`, `OUTSTANDING_AMT`, `PRODUCT_CD`. The translation is
one dictionary, `COLUMN_MAP`. Change the dictionary, not the agent. The database user the adapter
connects with has SELECT only; the agent tool is read-only by construction, not by politeness.

## The SOAP service

SOAP is XML over HTTP with a contract file (a WSDL). `zeep` reads the WSDL and generates the Python
calls. `python -m creditlens.integrations.soap_server --port 8001` runs a mock built with `spyne`;
`--slow 3 --flaky 0.5` makes it answer in three seconds and fail half the time, which is what the real
one does on the day you demo.

`RepaymentClient` wraps the call with three protections every integration needs:

1. **Timeout**: never wait forever. `Transport(timeout=...)`.
2. **Retries with backoff**: try again after 0.25 s, then 0.5 s, then 1 s. Not instantly, or you
   become the outage.
3. **Circuit breaker**: after three failures the breaker **opens** and calls fail instantly for thirty
   seconds, which gives the sick service room to recover. Then it goes **half-open**: one trial call
   decides whether it **closes** again. The breaker's clock is injectable so the test proves the timing
   without sleeping.

Without a reachable service the client returns a deterministic mock history flagged `source="mock"`,
so the memo says where the data came from.

## Slack approvals

The credit manager approves from a phone. A Bolt for Python app in **Socket Mode** connects out to
Slack over a websocket, so your laptop needs no public URL. It posts each draft memo to
`#credit-approvals` with Approve and Reject buttons; a click calls the same `resume()` as the CLI and
lands in the audit trail with `channel="slack"`.

Setup, once, on a free workspace:

1. Create an app at api.slack.com, from scratch.
2. Socket Mode: enable it and create an app-level token with `connections:write` (`xapp-...`).
3. OAuth and Permissions: add bot scopes `chat:write`, `channels:read`, `channels:join`; install to the
   workspace and copy the bot token (`xoxb-...`).
4. Interactivity: enable it (Socket Mode needs no request URL).
5. Create `#credit-approvals` and invite the bot. Put both tokens in `.env`.

```bash
python -m creditlens.integrations.slack --dry-run --application A001   # prints the Block Kit JSON
python -m creditlens.integrations.slack                                # runs the app
python -m creditlens.agents.run A001 --user U05 --slack                # drafts and posts
```

## Jira Cloud Free

1. Create a free Jira Software site and a project with key `CREDIT`.
2. Create an API token from your Atlassian account security page.
3. Set `JIRA_BASE_URL` (`https://<site>.atlassian.net`), `JIRA_EMAIL`, `JIRA_API_TOKEN`.

The MCP server uses the REST v3 API and stores the idempotency key as a label, so a re-run finds the
existing ticket. Without the variables, the mock in `data/cache/jira.json` behaves identically.

## The end-to-end run

One application flows through Oracle, SOAP, the agents and the approval. In Langfuse it is one trace
with a span per tool: the slow span is the SOAP call, the one to cache is the policy search.

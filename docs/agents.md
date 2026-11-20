# Agents that draft the credit memo (Day 4)

*Beginner notes for milestone 6. Code: `src/creditlens/agents/`, `src/creditlens/mcp_jira/`.*

## The rule that shapes everything

The model never computes a ratio and never decides an approval. Code computes the policy checks
(`policy_checks.py`), code decides the recommendation (`memo.recommend`), and the model turns a block of
structured facts into readable prose. That is what makes a memo auditable: every number points to a
check, every check points to a row in a table.

## Tools (L11 to L15)

A tool is a Python function with a Pydantic argument schema. The model sees the schema, never the code.

| Tool | Reads | Guard |
| --- | --- | --- |
| `get_application`, `get_statement_features` | SQLite tables | region filter for analysts |
| `get_financials` | the extraction module (or PDF text) | numbers only, no PII |
| `search_policy` | the retrieval index | `source` filter |
| `get_group_exposure` | the graph | Cypher recorded for the audit trail |
| `score_risk` | the registered model | falls back to a transparent stub |
| `get_core_banking_exposure`, `get_repayment_history` | Oracle adapter, SOAP client (Day 5) | read-only, circuit breaker |
| `file_jira_ticket` | the Jira MCP server | idempotency key `application:rule` |
| `run_sql` | allow-listed tables | SELECT only, no semicolons, LIMIT enforced, region injected |

**Safe Text-to-SQL.** `text_to_sql(question, user)` asks the model for SQL, then puts that SQL through
exactly the same guard a human's query gets. The model cannot escape its region because the WHERE
clause is added after it answers.

**Hallucinated tools and arguments** fail schema validation before any code runs. That is the Day 4
curveball, and the reason schemas are strict rather than "helpful".

## The graph (AG5 to AG12)

`build_graph()` returns a LangGraph `StateGraph` with:

- **State** (`state.py`): a TypedDict holding the application id, the user, the checks, the financials,
  the group, the risk score, the draft memo, a message log, a step counter and the tool-call history.
  Lists that several branches write to are declared `Annotated[list, operator.add]`: that annotation is a
  **reducer**, and it is how parallel branches merge without overwriting each other.
- **Supervisor**: looks at what is still missing and routes to the next specialist. Deterministic, on
  purpose: a routing decision you can read is a routing decision you can test.
- **Specialists**: `financial_analyst`, `banking_conduct`, `related_party`, `risk_scorer`, each fills
  one part of the state through the tools.
- **Map-reduce policy checker**: `Send("check_rule_group", ...)` fans out one branch per rule group
  (eligibility, financial, conduct, group, product). Each branch returns its checks; the reducer
  concatenates them. Adding a rule never changes the graph.
- **Nested memo subgraph**: draft sections, self-review, final. A subgraph has its own state schema,
  which is why its keys are declared on a TypedDict: LangGraph keeps only declared keys between nodes.
- **`file_exceptions`**: one Jira ticket per failed rule, through the MCP client.
- **`underwriter_approval`**: `interrupt()` stops the run before the memo is final and waits for a
  decision. Policy 1 in code.

## Checkpoints and interrupts (O1, O3 to O6)

Every node's output is saved by a checkpointer keyed on a `thread_id`. `draft_memo()` runs to the
interrupt and returns the thread id; `resume(thread_id, decision)` continues from the saved state.
In a notebook both calls share one in-memory saver. The CLI uses a SQLite saver so two processes
share it:

```bash
python -m creditlens.agents.run A001 --user U05
python -m creditlens.agents.approve <thread_id> approve --user U05 --comment "field visit done"
```

Kill the first command half-way and run it again with `--thread-id`: it continues from the last
checkpoint instead of redoing the extraction.

## Loop detection

Two guards: a hard step budget on the state (`MAX_STEPS`) and a detector for identical repeated tool
calls. `build_graph(inject_loop=True)` makes the supervisor misroute so you can watch the budget trip
and the run end with `status="error"` instead of hanging. `evals/trajectory.py` checks the same things
after the fact from the recorded tool calls.

## MCP and Jira (O9 to O11)

The Model Context Protocol is a standard way for a program to expose tools to any agent framework.
`python -m creditlens.mcp_jira` runs a server with three tools: `create_issue`, `get_issue`,
`search_issues`. With `JIRA_BASE_URL`, `JIRA_EMAIL` and `JIRA_API_TOKEN` set it talks to a real Jira
Cloud Free site; otherwise a mock persists tickets to `data/cache/jira.json`.

**Idempotency** means "doing it twice changes nothing". The key is `application_id:rule_id`. The server
checks for an existing ticket with that key before creating one, so re-running a memo never files a
duplicate. That is the Day 4 duplicate-ticket curveball.

## PII

Director names, PAN, phone, email and residential address never reach a prompt. `adapters.mask()`
runs Presidio (Day 6) or a regex fallback over every fact block before the memo writer sees it.

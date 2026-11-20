# Workshop: session by session, topic by topic

This folder organises the CreditLens AI workshop by session and by topic. It moves no code: every topic page points
at the notebook section, the code in `src/`, the docs and the tests that already live in the repository.

## Get your laptop ready before each session

Run this from the repository root a day before the session. It regenerates the data, runs the setup check and
runs that session's tests offline, so you arrive with everything working locally.

```bash
python workshop/get_ready.py 0   # pre-work
python workshop/get_ready.py 1   # before Session 1
python workshop/get_ready.py 2   # before Session 2, and so on up to 4
```

If you fall behind during a session, run the same command for the next session and carry on.

## Sessions

| Session | Folder | Notebooks |
| --- | --- | --- |
| Pre-work | [session-0-prework](session-0-prework/README.md) | 00_prework.ipynb |
| Session 1: Discover, data foundation, ground truth, extraction | [session-1-data-and-extraction](session-1-data-and-extraction/README.md) | 01_data_foundation.ipynb and 02_ground_truth_and_extraction.ipynb |
| Session 2: Retrieval, graph and visual documents | [session-2-retrieval](session-2-retrieval/README.md) | 03_retrieval.ipynb |
| Session 3: Agents that draft the memo, and the bank's legacy systems | [session-3-agents-and-integrations](session-3-agents-and-integrations/README.md) | 04_agents.ipynb and 05_integrations.ipynb |
| Session 4: Deploy, Evaluate, Maintain, demo day | [session-4-deploy-evaluate-maintain](session-4-deploy-evaluate-maintain/README.md) | 06_mlops_security_deploy.ipynb and 07_evaluate_maintain.ipynb |

## Topics A to Z

| Topic | Topic codes | Session | Page |
| --- | --- | --- | --- |
| API and tracing | A1 to A5, OB1 | Session 1 | [open](session-1-data-and-extraction/04-api-and-tracing/README.md) |
| Clone, install and check | · | Pre-work | [open](session-0-prework/01-setup/README.md) |
| Credit policy RAG: chunking, hybrid search, reranking | R5 to R16 | Session 2 | [open](session-2-retrieval/02-policy-rag/README.md) |
| Data foundation: layers, wrangling, contracts | P1 to P5 | Session 1 | [open](session-1-data-and-extraction/02-data-foundation/README.md) |
| Demo day and handoff | OG4, OG5 | Session 4 | [open](session-4-deploy-evaluate-maintain/06-demo-day-and-handoff/README.md) |
| Docker, CI with an evaluation gate, AWS demo | D1 to D4, D9 to D12, C1 to C4, C11 to C13, OB9, OG9 | Session 4 | [open](session-4-deploy-evaluate-maintain/04-docker-ci-and-deploy/README.md) |
| Enterprise integrations: Oracle, SOAP, Slack | Day 5 content | Session 3 | [open](session-3-agents-and-integrations/05-enterprise-integrations/README.md) |
| Foundations, the FDE role and discovery | OG1, OG2 | Session 1 | [open](session-1-data-and-extraction/01-foundations-and-discovery/README.md) |
| Ground truth: the labelling party | evaluation first | Session 1 | [open](session-1-data-and-extraction/03-ground-truth/README.md) |
| Homework: run one memo end to end | · | Session 2 | [open](session-2-retrieval/06-homework/README.md) |
| Homework: SOW, async ingestion lab, SQL Server | OG3, P6 to P10, P15 | Session 1 | [open](session-1-data-and-extraction/06-homework/README.md) |
| Homework: UAT runbook draft | OG5 | Session 3 | [open](session-3-agents-and-integrations/06-homework/README.md) |
| Judge calibration, gateway drill, incident drills | OB2 to OB5, OB10 to OB14 | Session 4 | [open](session-4-deploy-evaluate-maintain/05-evaluate-and-maintain/README.md) |
| MCP and Jira | O9 to O11 | Session 3 | [open](session-3-agents-and-integrations/04-mcp-and-jira/README.md) |
| MLOps and deployment architecture | C5, C9, C10 | Session 4 | [open](session-4-deploy-evaluate-maintain/01-deployment-architecture/README.md) |
| Multi-agent supervisor, checkpoints and approval | AG5 to AG12, O1, O3 to O6 | Session 3 | [open](session-3-agents-and-integrations/03-supervisor-and-approval/README.md) |
| Optional: GraphQL lab | A6 to A10 | Pre-work | [open](session-0-prework/04-graphql-lab/README.md) |
| Optional: Linux and tooling | P11 to P15, A11, A14, A15 | Pre-work | [open](session-0-prework/03-linux-and-tooling/README.md) |
| Python warm-up | P1 to P5 | Pre-work | [open](session-0-prework/02-python-warmup/README.md) |
| Recap quiz, SOW swap and the retriever vote | OG3, R1 to R4 | Session 2 | [open](session-2-retrieval/01-recap-and-retriever-vote/README.md) |
| Related-party graph | G1 to G11 | Session 2 | [open](session-2-retrieval/03-related-party-graph/README.md) |
| Retrieval evaluation | OB6 to OB8 | Session 2 | [open](session-2-retrieval/05-retrieval-evaluation/README.md) |
| Risk model with MLflow and a fairness check | C6 to C8 | Session 4 | [open](session-4-deploy-evaluate-maintain/02-risk-model/README.md) |
| Security: RBAC, PII masking, guardrails | OG6, OG8 | Session 4 | [open](session-4-deploy-evaluate-maintain/03-security/README.md) |
| Structured extraction and the first evaluation | L1 to L10, M1, M2, M8, M10, M11 | Session 1 | [open](session-1-data-and-extraction/05-extraction/README.md) |
| Tool calling and safe Text-to-SQL | L11 to L15, OG7 | Session 3 | [open](session-3-agents-and-integrations/02-tool-calling/README.md) |
| Visual documents with ColPali | M3 to M7, M9, M12 | Session 2 | [open](session-2-retrieval/04-visual-documents/README.md) |
| Yesterday's bug and agent architecture | AG1 to AG4 | Session 3 | [open](session-3-agents-and-integrations/01-agent-architecture/README.md) |

Notebook headings still say Day 1 to Day 7: they follow the original seven-day plan, which the four sessions
compress. [docs/four_session_plan.md](../docs/four_session_plan.md) maps one to the other.

# CreditLens AI in four sessions of three hours

*Trainer plan. The seven-day flow is compressed into four lectures. Each session pairs a Build block with
the evaluation that proves it, so the evaluation spine survives the cut. The pieces that cost the most
time for the least learning (async lab, Oracle and SOAP, Keycloak, AWS, Pinecone, LangSmith) become
homework, short demos or diagram-only.*

## What moves where

| Original day | Kept live | Compressed to | Dropped to homework or demo |
| --- | --- | --- | --- |
| 1 Discover, data | discovery, architecture, dirty-data bingo, contracts | Session 1, first half | async ingestion lab (homework) |
| 2 Ground truth, extraction | labelling party, API and tracing, extraction, first eval | Session 1, second half | SOW swap (5 minutes on Session 2) |
| 3 Retrieval | chunking bake-off, hybrid, graph, retrieval eval | Session 2 | Pinecone (optional), ColPali as a 15-minute demo |
| 4 Agents | tools, supervisor, interrupt, MCP Jira | Session 3, first two hours | the paper agent game (10 minutes) |
| 5 Integrations | Oracle adapter, circuit breaker, Slack | Session 3, last 50 minutes as three short demos | Oracle in Docker (laptop track only) |
| 6 Deploy, MLOps, security | risk model, fairness, RBAC, PII, CI gate | Session 4, first 110 minutes | AWS as a 10-minute demo, Keycloak and Bedrock mentioned |
| 7 Evaluate, Maintain, demo | judge calibration, gateway drill, incident drills, demo day | Session 4, last 70 minutes | LangSmith comparison (one slide) |

Three threads still run through every session: the golden set from Session 1 scores every later block,
Langfuse traces everything from Session 1 onward, and the Session 1 pipeline feeds everything.

## Session 1: Discover, data foundation, ground truth, extraction

Notebooks `01_data_foundation` (sections 1 to 7) and `02_ground_truth_and_extraction`.

| Time | Block | What happens | Topics |
| --- | --- | --- | --- |
| 0:00 to 0:10 | The story and the FDE role | Rohan, Priya, Good Bank. Build → Deploy → Evaluate → Maintain on one slide | |
| 0:10 to 0:35 | Discovery role-play | You play the Head of Credit, CRO and CISO. Pods uncover the four hidden constraint cards, classify data sources by sensitivity | OG1, OG2 |
| 0:35 to 0:50 | Reference architecture | Pods place components on the blank diagram, you reveal the reference and the PII line | |
| 0:50 to 1:20 | Data foundation | Generate the data, dirty-data bingo (points per planted problem), Pandera contracts, break a contract on purpose, run the pipeline and the tests | P1 to P5 |
| 1:20 to 1:30 | Break | | |
| 1:30 to 1:50 | Labelling party | Five real receipts, independent labels, Cohen's kappa, build the golden set | evaluation first |
| 1:50 to 2:10 | API and tracing | Start the API, mint tokens for U01 and U05, predict the counts, see the trace | A1 to A5, OB1 |
| 2:10 to 2:50 | Structured extraction | Text PDF, then the rotated scan curveball, then a receipt. Discriminated union, retries, the gateway cache counter. First evaluation on the scoreboard, find the worst failure | L1 to L10, M1, M2, M8, M10, M11 |
| 2:50 to 3:00 | Exit ticket | Homework: SOW draft (OG3), the async ingestion lab (P6 to P10), get Neo4j AuraDB and Langfuse keys | |

Prerequisite: the pre-work notebook done 48 hours before, confirmed by the setup check.

## Session 2: Retrieval, graph and visual documents

Notebook `03_retrieval`.

| Time | Block | What happens | Topics |
| --- | --- | --- | --- |
| 0:00 to 0:10 | Recap quiz and SOW swap | Three quiz questions on the board; pods mark scope creep in another pod's SOW | OG3 |
| 0:10 to 0:30 | Retrieval architecture | Theory, then the "which retriever?" vote on six analyst questions | R1 to R4 |
| 0:30 to 1:15 | Credit policy RAG | Chunking bake-off on the Good Bank manual and the real SBA rulebook, embeddings, BM25, rank fusion, reranker. Curveball: the `source` filter | R5 to R16 |
| 1:15 to 1:25 | Break | | |
| 1:25 to 2:05 | Related-party graph | Load the registry, hidden-group hunt, exposure with the Cypher recorded, profile and index. The address-only group is the planted bug | G1 to G11 |
| 2:05 to 2:20 | Visual documents | ColPali demo on the real charts from pre-computed embeddings, one vision-model answer | M3 to M7, M9, M12 |
| 2:20 to 2:50 | Retrieval evaluation | Recall and MRR for naive, hybrid and reranked on both corpora; RAGAS if a key is available; check the votes | OB6 to OB8 |
| 2:50 to 3:00 | Exit ticket | Homework: run `python -m creditlens.agents.run A001 --user U05` once and read the memo | |

## Session 3: Agents that draft the memo, and the bank's legacy systems

Notebooks `04_agents` and `05_integrations`.

| Time | Block | What happens | Topics |
| --- | --- | --- | --- |
| 0:00 to 0:10 | Yesterday's bug | The golden question vector search missed and BM25 found, from the eval output or a trace | |
| 0:10 to 0:30 | Agent architecture | Theory and ten minutes of the paper agent game | AG1 to AG4 |
| 0:30 to 0:55 | Tool calling | Strict schemas, the hallucinated-tool curveball, safe Text-to-SQL with the region filter injected | L11 to L15, OG7 |
| 0:55 to 1:35 | Multi-agent supervisor | Draw the graph, deterministic checks against the answer key, run to the interrupt, underwriter role-play, kill and resume, the infinite-loop curveball | AG5 to AG12, O1, O3 to O6 |
| 1:35 to 1:45 | Break | | |
| 1:45 to 2:05 | MCP and Jira | The MCP server, scoped access, the duplicate-ticket curveball fixed by the idempotency key | O9 to O11 |
| 2:05 to 2:45 | Enterprise integrations, three demos | Oracle adapter and the column-name curveball (15 minutes); SOAP client against the slow and flaky mock, watch the circuit breaker (15 minutes); Slack dry run, then a live click if the workspace is ready (10 minutes) | Day 5 content |
| 2:45 to 2:55 | End-to-end trace | One application through Oracle, SOAP, the agents and the approval, followed in Langfuse; trajectory check | OB1 |
| 2:55 to 3:00 | Exit ticket | Homework: UAT runbook draft (OG5), read `docs/mlops.md` | |

## Session 4: Deploy, Evaluate, Maintain, demo

Notebooks `06_mlops_security_deploy` and `07_evaluate_maintain`.

| Time | Block | What happens | Topics |
| --- | --- | --- | --- |
| 0:00 to 0:15 | MLOps and deployment architecture | Network puzzle: public or private subnet per component; NAT gateway on the diagram only | C5, C9, C10 |
| 0:15 to 0:50 | Risk model | Features from the real loan book, the adapter's assumptions read aloud, train with MLflow, beat the baseline, real versus synthetic labels, fairness report, swap the stub for the champion | C6 to C8 |
| 0:50 to 1:15 | Security | RBAC at the SQL and retrieval layers, Presidio masking with the PAN recogniser, guardrails on prompts; Keycloak and Bedrock Guardrails as one slide each | OG6, OG8 |
| 1:15 to 1:25 | Break | | |
| 1:25 to 1:50 | Docker, CI and the eval gate | The multi-stage Dockerfile and Compose profiles in five minutes, then the gate, then the regressed-prompt curveball that fails it. AWS push and run as a ten-minute instructor demo, budget alarm first | D1 to D4, D9 to D12, C1 to C4, C11 to C13, OB9, OG9 |
| 1:50 to 2:20 | Evaluate and Maintain | Judge calibration and a fifteen-minute eval court; gateway drill and cut the primary live; two of the four incident drills live, the other two as homework | OB2 to OB5, OB10 to OB14 |
| 2:20 to 2:50 | Demo day | Five minutes per pod to the Head of Credit and CISO panel: ROI from the brief, architecture, live memo run, eval scorecard, UAT runbook | OG4, OG5 |
| 2:50 to 3:00 | Close | Pod scoreboard, handoff checklist | |

## Checkpoint branches for four sessions

The code is never shortened or stubbed: every branch carries the complete reference solution, and the
four sessions simply cover it in order. A branch is a catch-up point, not a different code base. What
differs per session is which notebooks have been run and which caches exist.

| Branch | Session | Notebooks covered by then | Pods should already have |
| --- | --- | --- | --- |
| `day1-start` | 1 | `00_prework` | setup check green, data generated |
| `day2-start` | 2 | `01_data_foundation`, `02_ground_truth_and_extraction` | golden set, extraction eval result, Neo4j and Langfuse keys |
| `day3-start` | 3 | `03_retrieval` | embedding cache, retrieval eval result |
| `day4-start` | 4 | `04_agents`, `05_integrations` | one memo run with its thread id, Jira tickets filed |

Create them once `main` is committed, all pointing at the same commit:

```bash
for n in 1 2 3 4; do git branch day${n}-start main; done
git push --all
```

A pod that falls behind runs:

```bash
git checkout dayN-start
python -m creditlens.generators && python -m creditlens.pipeline --load && python -m creditlens.data_real
```

Live coding happens in the notebook cells: you type the key function in front of the room (for example
rank fusion on Session 2, the SQL guard on Session 3), then show the repo version as the answer key.

## Before each session

| Session | Pre-compute and bring |
| --- | --- |
| 1 | the LLM cache for the Day 2 notebook (run it once online), the receipts sample |
| 2 | the embedding cache under `data/cache/retrieval/`, the ColPali embeddings under `data/cache/colpali/`, the SBA rulebook sample |
| 3 | the LLM cache for a memo run on A001 and A012, a Jira Cloud Free site or the mock, the Slack app tokens |
| 4 | the SBA loan sample, a trained `data/cache/model.joblib`, the AWS budget alarm, the teardown script tested |

Zip `data/cache/` after the dry run and share the link: a pod with no quota left finishes every exercise from it.

# CreditLens AI: module contracts and conventions

This file is the agreement between the modules. Every milestone is built against it, and
workshop pods can swap a stub for the real component because the shapes never change.

## The story (use these names everywhere)

Rohan runs a small packaging company and applies to **Good Bank** for a business loan.
**Priya**, a credit analyst at Good Bank, handles Rohan's file. CreditLens AI does Priya's
paperwork so she can focus on the decision. A human underwriter makes every final decision.
Never use "Kestrel Finance" or "Ramesh" (old names).

Lifecycle vocabulary: **Build → Deploy → Evaluate → Maintain**.

## Conventions

- Python 3.10+, `src/` layout, package `creditlens`. Import settings with
  `from creditlens.config import settings`.
- **Every LLM call goes through the gateway**: `from creditlens.gateway.llm import get_gateway` then
  `get_gateway().complete(messages, response_model=SomePydanticModel)` or `.chat(prompt)` or
  `.extract(prompt, Model)`. Never import litellm, google.generativeai or groq directly.
  The gateway caches answers on disk, rate-limits, falls back across providers and has an
  offline stub mode (`LLM_OFFLINE=1`). Register a smarter offline stub for your Pydantic model
  with `OFFLINE_HANDLERS["ModelClassName"] = fn(messages) -> dict` so the whole system runs
  end to end without any API key.
- **Tracing**: decorate meaningful functions with `@observe(name="...")` from
  `creditlens.observability`. For LangGraph or LangChain runs pass
  `config={"callbacks": creditlens.observability.callbacks()}`. It is a no-op without keys.
- **Shared shapes** live in `creditlens.schemas`. Add fields there if you must, never duplicate
  a model elsewhere.
- **Tests must pass with no API keys, no network, no Docker.** Use `LLM_OFFLINE=1` behaviour and
  `pytest.importorskip` / `pytest.mark.skipif` for optional services. Heavy models
  (sentence-transformers, ColPali, OCR engines) are skipped unless installed and small.
  Use the session fixtures in `tests/conftest.py` (`s` = settings pointing at a temp data root
  with generated data, `outputs` = pipeline outputs).
- **Beginner-first code**: a module docstring that explains *why* the module exists and the
  one idea it teaches, small named functions, type hints, and comments only where a beginner
  would ask "why". No clever one-liners.
- **Optional dependencies**: wrap imports of heavy or service-specific libraries inside functions
  or `try/except ImportError`, and always provide a lightweight fallback that reads from `data/`
  (see each module's "fallback" line below). Declare the extra in `pyproject.toml`.
- **PII**: director PAN, phone, email and residential address never reach an LLM unmasked.
  Use `creditlens.guardrails.pii.mask_text(text) -> str` (milestone 9) before any prompt that
  could carry borrower data. Until that module exists, strip those columns before prompting.
- **Audit trail**: anything that changes state (memo drafted, ticket filed, approval) appends an
  `AuditEvent` through `creditlens.api.audit.record(event)` (SQLite table `audit_events`).
- No real borrower data, ever. All applications, directors and statements are synthetic. The
  real datasets (SROIE receipts, ChartQA, SBA loan book, SBA SOP) contain no Good Bank PII.

## Data on disk (after `python -m creditlens.generators`, `python -m creditlens.pipeline --load`, `python -m creditlens.data_real`)

| Path | Contents |
| --- | --- |
| `data/raw/registry/companies.csv` | company_id, legal_name, sector, incorporated_on, registered_address, city, region, employees |
| `data/raw/registry/directors.csv` | director_id, full_name, pan, phone, email, residential_address (PII) |
| `data/raw/registry/company_directors.csv` | company_id, director_id, role, appointed_on |
| `data/raw/registry/users.csv` | user_id, full_name, role (analyst, head_of_credit, auditor), region |
| `data/raw/registry/existing_loans.csv` | company_id, outstanding_inr_000 (Good Bank's existing exposure) |
| `data/raw/applications/applications.csv` | application_id, company_id, region, product, purpose, requested_inr_000, tenure_months, submitted_on, analyst_id, guarantor_director_id |
| `data/raw/financials/{company_id}_financials.pdf` | 3 fiscal years; a quarter are scans; one has page 2 rotated |
| `data/raw/policy/credit_policy_manual.md` and `.pdf` | Good Bank policy, sections 1 to 9, rules like 3.1 |
| `data/raw/history/loan_history.csv` | synthetic loans with `defaulted`; kept as the "beat the baseline" comparison |
| `data/clean/transactions.csv` | application_id, seq, txn_date, channel, counterparty, amount, balance, is_bounce, ref_no, source_file |
| `data/curated/statement_features.csv` | one row per application: avg_monthly_inflow, inflow_volatility, cheque_returns_12m, cash_credit_share, monthly_emi_outflow, top_customer_share, unknown_txn_share, min_balance, avg_balance, ... |
| `data/answer_key/` | instructor only: transactions_truth.csv, financials_truth.json, financial_doc_kinds.json, related_party_groups.json, expected_policy_flags.json, statement_issues.json |
| `data/real/receipts/*.jpg` + `labels.json` | SROIE receipts; labels: {file: {entities: {company, date, address, total}, words, image_size}} |
| `data/real/charts/*.png` + `questions.json` | ChartQA: [{image, question, answer}] |
| `data/real/sba_loans.csv` | SBA 7(a) loans: loan_id, State, BankState, NAICS, naics_sector, ApprovalFY, Term, NoEmp, NewExist, CreateJob, RetainedJob, FranchiseCode, UrbanRural, RevLineCr, LowDoc, DisbursementGross, GrAppv, SBA_Appv, defaulted |
| `data/real/sba_sop_50_10.md` | SBA lending rulebook as Markdown with `#` headings |
| `creditlens.db` | SQLite with tables companies, directors, company_directors, users, existing_loans, applications, transactions, quality_report, statement_features |

Policy thresholds are in `creditlens.generators.policy.RULES`. Section numbers in the manual map
to rule ids: 2.1 years in business, 2.2 excluded sectors, 3.1 DSCR, 3.2 debt to EBITDA,
3.3 current ratio, 4.1 cheque returns, 4.2 cash credit share, 5.2 group exposure, 6.1 tenure.

## Module contracts

### `creditlens.api` (milestone 3, also 9)
- `creditlens.api.app:create_app() -> FastAPI` with CORS, dependency injection for the DB engine,
  the current `User` (from a bearer token: mock JWT signed with `JWT_SECRET`, or `X-User-Id` header
  in dev), and the gateway.
- Routes: `GET /health`, `POST /auth/token` (dev login by user_id), `GET /applications` (filtered by
  the caller's region), `GET /applications/{id}`, `POST /applications/{id}/memo` (runs the agent
  graph, returns `CreditMemo` or a "paused for approval" status), `POST /applications/{id}/approve`
  (`ApprovalDecision`, head_of_credit only), `GET /applications/{id}/audit`.
- `creditlens.api.audit.record(event: AuditEvent)` and `.list_events(application_id) -> list[AuditEvent]`.
- `creditlens.api.auth.create_token(user_id) -> str`, `current_user()` dependency.
- RBAC rule: analysts see their region only; head_of_credit sees all and can approve; auditors
  read-only everywhere (any POST except /auth/token returns 403).

### `creditlens.extraction` (milestone 4)
- `creditlens.extraction.ocr.pdf_to_text(path) -> tuple[str, bool]` returns (text, ocr_used).
  Text PDFs via pypdf or pdfplumber; image PDFs via OCR (pytesseract if the binary exists, else
  rapidocr or easyocr if installed, else raise `OCRUnavailable`). Detect and fix a rotated page
  (try 0/90/180/270 and keep the orientation with the most dictionary words).
- `creditlens.extraction.ocr.image_to_text(path) -> str` for receipts.
- `creditlens.extraction.extract.extract_document(path, doc_type: Literal["financial_statement","receipt"]) -> DocumentExtraction`
  builds the prompt, calls the gateway with `ExtractionEnvelope`, and sets `source_file`, `ocr_used`.
- `creditlens.extraction.extract.extract_financials(company_id) -> FinancialStatementExtraction`
  (looks up `data/raw/financials/{company_id}_financials.pdf`).
- Offline stub: register `OFFLINE_HANDLERS["ExtractionEnvelope"]` that parses numbers from the
  text with regexes well enough to pass most golden-set fields.
- `creditlens.extraction.golden.build_golden_set(s) -> Path` writes `ground_truth/golden_set.json`:
  financial statements from the answer key (first 10 companies) plus receipts from SROIE labels.
- `evals/extraction_accuracy.py`: `score(predictions, golden) -> dict` with per-field accuracy
  (exact match for strings after normalisation, within 1 percent for numbers) and a CLI that
  prints a table and writes `evals/results/extraction_accuracy.json`.
- `ground_truth/LABELING_GUIDELINES.md`: how to label receipts and statements, tie-break rules,
  how agreement is measured (`ground_truth/agreement.py`: Cohen's kappa for two labelers).

### `creditlens.retrieval` (milestone 5)
- `creditlens.retrieval.chunking.chunk_markdown(text, source, strategy="section"|"fixed"|"sentence", size=400, overlap=50) -> list[Chunk]`
  where `Chunk` has chunk_id, source, section, text, metadata.
- `creditlens.retrieval.index.PolicyIndex`: `.build(chunks)`, `.search_vector(q, k, source=None)`,
  `.search_bm25(q, k, source=None)`, `.search_hybrid(q, k, source=None)` (reciprocal rank fusion),
  `.search(q, k, source=None, rerank=True) -> list[PolicyHit]`. Qdrant in-memory by default
  (`qdrant_client.QdrantClient(":memory:")`), with `location` configurable for Docker or Qdrant Cloud.
  Embeddings via `creditlens.gateway.llm.embed`. Reranker via sentence-transformers CrossEncoder
  (`settings.llm.reranker_model`), skipped if unavailable.
- `creditlens.retrieval.policy.search_policy(query, k=5, source=None) -> list[PolicyHit]`
  builds the index once per process from `data/raw/policy/credit_policy_manual.md` and
  `data/real/sba_sop_50_10.md` (if present) and caches it on disk under `data/cache/retrieval/`.
  Fallback without sentence-transformers: BM25 only.
- `creditlens.retrieval.pinecone_compare.py`: optional, same chunks into a Pinecone free index when
  `PINECONE_API_KEY` is set; prints recall@5 side by side with Qdrant.
- `creditlens.retrieval.graph.GraphStore`: `.load_from_registry(raw_dir)`, `.related_group(company_id) -> list[GroupMember]`
  (variable-length path over SHARES_DIRECTOR and SHARES_ADDRESS edges), `.group_exposure(company_id, requested: dict) -> GroupExposure`,
  `.all_groups() -> list[set[str]]`. Backend Neo4j (AuraDB or Docker) when `NEO4J_URI` is set,
  else a NetworkX in-memory graph. The Cypher text is stored in `GroupExposure.cypher` even when
  NetworkX runs, so the audit trail is identical.
- `creditlens.retrieval.colpali.py`: Colab-GPU-only visual retrieval over `data/real/charts`
  (byaldi or colpali-engine), with `precompute_embeddings()` and `search(question, k)`; loads
  pre-computed embeddings from `data/cache/colpali/` when present. Must import cleanly without torch.
- `evals/retrieval_eval.py`: golden questions for both corpora in `ground_truth/retrieval_questions.json`
  (question, expected_section, expected_source), recall@k and MRR for vector, bm25, hybrid, reranked;
  RAGAS (faithfulness, answer relevancy, context precision) when `ragas` is installed and an LLM key exists.

### `creditlens.agents` (milestone 6) and `creditlens.integrations` (milestone 7)
- `creditlens.agents.tools`: LangChain tools with strict Pydantic arg schemas:
  `get_application(application_id)`, `get_statement_features(application_id)`,
  `get_financials(company_id)` (extraction with answer-key-free fallback to the PDF text),
  `search_policy(query, source)`, `get_group_exposure(company_id)`, `score_risk(application_id)`,
  `run_sql(query)` (safe Text-to-SQL: SELECT only, allow-listed tables, LIMIT enforced, region
  filter injected from the current user), `get_core_banking_exposure(company_id)` (Oracle adapter),
  `get_repayment_history(company_id)` (SOAP client), `file_jira_ticket(application_id, rule_id, ...)`
  (MCP client to the Jira server, idempotency key = f"{application_id}:{rule_id}").
- `creditlens.agents.graph.build_graph(checkpointer=None)`: LangGraph supervisor with specialists
  `financial_analyst`, `banking_conduct`, `related_party`, `policy_checker` (map-reduce: one
  branch per rule, merged by a reducer), `risk_scorer`, `memo_writer` (nested subgraph), and an
  `underwriter_approval` interrupt before the memo is final. Loop detection: a step counter in the
  state with a hard limit, and a repeated-tool-call detector.
- `creditlens.agents.run.draft_memo(application_id, user: User, thread_id=None) -> dict` runs to the
  interrupt and returns `{"status": "awaiting_approval", "memo": CreditMemo, "thread_id": ...}`;
  `resume(thread_id, decision: ApprovalDecision) -> dict` continues and records the audit event.
- Deterministic policy checks are code, not LLM: `creditlens.agents.policy_checks.run_checks(application_id) -> list[PolicyCheck]`
  using RULES. The LLM only writes prose and chooses the recommendation with the checks as input.
  Offline mode must still produce a complete `CreditMemo` (template prose).
- `creditlens.mcp_jira.server`: an MCP server (MCP Python SDK, stdio) exposing `create_issue`,
  `get_issue`, `search_issues` against the real Jira Cloud REST API when `JIRA_BASE_URL`,
  `JIRA_EMAIL`, `JIRA_API_TOKEN` are set, else an in-memory mock persisted to `data/cache/jira.json`.
  Idempotency: same key returns the existing issue. `creditlens.mcp_jira.client.file_ticket(...)`.
- `creditlens.integrations.oracle`: `CoreBankingAdapter.get_exposure(company_id) -> dict` reading
  Oracle (python-oracledb, `ORACLE_DSN`) with column-name mapping (Oracle uses CUST_ID, OUTSTANDING_AMT)
  and a SQLite fallback built from `existing_loans.csv`. Include `infra/oracle/init.sql`.
- `creditlens.integrations.soap`: spyne mock service `python -m creditlens.integrations.soap_server`
  (`--slow`, `--flaky` flags) and `RepaymentClient` (zeep) with timeout, retries with backoff, and a
  circuit breaker (`CircuitBreaker(failures=3, reset_after=30)`). Fallback without zeep: HTTP mock.
- `creditlens.integrations.slack`: Bolt app in Socket Mode that posts a memo summary with Approve and
  Reject buttons to `#credit-approvals` and resumes the graph on click; `python -m creditlens.integrations.slack`.
  CLI fallback `python -m creditlens.agents.approve <thread_id> approve|reject` when Slack is not configured.

### `creditlens.ml` (milestone 8)
- `creditlens.ml.features.sba_features(df) -> (X, y)` from `data/real/sba_loans.csv`;
  `application_to_sba_features(application_row, company_row, statement_features_row) -> dict`
  (the adapter that maps a Good Bank application onto the model's schema; document every assumption,
  for example INR thousands to USD, sector to NAICS 2-digit).
- `creditlens.ml.train.train(experiment="creditlens-risk") -> dict` trains a logistic-regression
  baseline and a gradient-boosting challenger with MLflow tracking (`MLFLOW_TRACKING_URI`, default
  local `mlruns/`), logs AUC, Brier score, calibration, registers the better model as
  `creditlens-risk` (alias `champion`), and writes `monitoring/fairness_report.json`
  (default rate, AUC and approval-rate parity by State and UrbanRural).
- `creditlens.ml.serving.score(features: dict) -> RiskScore` loads the champion from MLflow, or
  from `data/cache/model.joblib`, or falls back to a transparent rule-based stub.
- `creditlens.ml.serving` also exposes a FastAPI router `/risk/score` mounted by the API.

### `creditlens.guardrails` (milestone 9)
- `creditlens.guardrails.pii.mask_text(text) -> str` and `mask_record(dict) -> dict` using Presidio
  with custom recognizers for Indian PAN (`[A-Z]{5}[0-9]{4}[A-Z]`), Indian phone numbers and the
  director names from the registry; regex fallback when Presidio is not installed.
- `creditlens.guardrails.rails/` NeMo Guardrails config (input rail: prompt injection and off-topic;
  output rail: no PII, no approval language) and `creditlens.guardrails.check.guard(prompt) -> tuple[bool, str]`
  with a regex fallback. `docs/bedrock_guardrails.md` documents the AWS comparison (instructor demo, not free).
- `creditlens.guardrails.rbac.region_filter(user) -> str` (SQL WHERE fragment) used by `run_sql` and
  the API; `retrieval_filter(user) -> dict` for Qdrant metadata filters.

### Deploy and Maintain (milestones 9 to 11)
- `infra/Dockerfile` (multi-stage, non-root), `infra/docker-compose.yml` with profiles: `core`
  (api, sqlserver), `observability` (langfuse + postgres + clickhouse per Langfuse docs), `ml`
  (mlflow), `retrieval` (qdrant, neo4j), `security` (keycloak), `legacy` (oracle-free, soap mock).
- `.github/workflows/ci.yml`: lint, tests, then an **eval gate** job that runs
  `python -m evals.gate` (extraction accuracy and retrieval recall must not drop below
  `evals/thresholds.json`; offline mode so CI needs no keys) and fails the build otherwise.
- `infra/aws/README.md`: ECR push, free-tier EC2 or container Lambda with function URL, S3 lifecycle,
  least-privilege IAM, CloudWatch logs, budget alarm first; `infra/aws/teardown.sh`.
- `creditlens.gateway.litellm_config.yaml` + `python -m creditlens.gateway.proxy` for the LiteLLM
  proxy with rate limits and fallback routing; `evals/judge.py` (LLM judge calibrated against
  `ground_truth/judge_calibration.json`, reports agreement with human labels), `evals/trajectory.py`
  (checks tool-call order and counts from a LangGraph run), `monitoring/drift.py` (Evidently report:
  SBA training features vs current applications), `monitoring/incidents.py` with
  `--scenario data_drift|cost_spike|policy_update|prompt_injection` each triggering one incident
  and printing where to look (trace, drift report, eval score).

## Done-when, per milestone

3. A request to the API shows up as a trace in Langfuse (or in the no-op log when keys are missing).
4. `python -m evals.extraction_accuracy` prints per-field accuracy against the golden set.
5. `python -m evals.retrieval_eval` prints recall for naive vs hybrid, and `GraphStore.all_groups()` matches `related_party_groups.json`.
6. `python -m creditlens.agents.run A001` runs to the approval interrupt and files exactly one Jira ticket per exception.
7. Exposure comes from the Oracle adapter, repayment history from the SOAP client, and a Slack or CLI approval resumes the graph.
8. The agent calls the registered model and MLflow shows comparable runs.
9. A deliberately regressed prompt fails CI, and an analyst token cannot read another region.
10. The deployed API answers, and the teardown script removes everything.
11. Each incident runs with one command and points at the trace or report that diagnoses it.

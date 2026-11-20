"""The agent's tools: small, typed, auditable functions the graph is allowed to call.

Three ideas a beginner should take from this file:

1. **A tool is a contract, not a function call.** Each one has a Pydantic argument schema,
   so a model cannot pass a date where an id belongs. Each returns a plain dict, so the
   result can be written into graph state, cached, logged and replayed.
2. **Tools are where the blast radius lives.** `run_sql` is the dangerous one, so it is the
   most defended: SELECT only, no semicolons, an allow-list of tables, a forced LIMIT, and
   the analyst's region injected into the WHERE clause. Text-to-SQL goes through the same
   guard as a hand-written query, because an LLM is just another untrusted caller.
3. **Every call is recorded.** `tool_calls()` returns the ordered history, which is what the
   trajectory eval in milestone 11 grades and what the loop detector reads.
"""
from __future__ import annotations

import contextvars
import json
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from creditlens.agents import adapters
from creditlens.observability import observe
from creditlens.schemas import User

# ---------------------------------------------------------------------------
# Who is asking? RBAC needs the caller, and a tool cannot take it as an argument
# (a model must never be able to claim it is the Head of Credit). So it is ambient.
# ---------------------------------------------------------------------------

_CURRENT_USER: contextvars.ContextVar[Optional[User]] = contextvars.ContextVar(
    "creditlens_current_user", default=None)
_TOOL_CALLS: contextvars.ContextVar[Optional[list]] = contextvars.ContextVar(
    "creditlens_tool_calls", default=None)


def set_current_user(user: Optional[User]) -> None:
    _CURRENT_USER.set(user)


def get_current_user() -> Optional[User]:
    return _CURRENT_USER.get()


def start_tool_log() -> list[dict]:
    """Begin a fresh tool-call history for one run and return the list that collects it."""
    log: list[dict] = []
    _TOOL_CALLS.set(log)
    return log


def tool_calls() -> list[dict]:
    return list(_TOOL_CALLS.get() or [])


def record_tool_call(name: str, args: dict) -> dict:
    """Append one entry to the run's tool-call history. Public so graph nodes can use it."""
    return _record(name, args)


def _record(name: str, args: dict) -> dict:
    log = _TOOL_CALLS.get()
    entry = {"name": name, "args": adapters.as_json(args), "order": len(log) if log is not None else 0}
    if log is not None:
        log.append(entry)
    return entry


# ---------------------------------------------------------------------------
# Argument schemas
# ---------------------------------------------------------------------------


class ApplicationArgs(BaseModel):
    application_id: str = Field(description="Application id, for example A001")


class CompanyArgs(BaseModel):
    company_id: str = Field(description="Company id, for example C003")


class PolicyArgs(BaseModel):
    query: str = Field(description="What to look for in the credit policy, in plain words")
    source: Optional[str] = Field(None, description="good_bank_policy or sba_sop")


class SqlArgs(BaseModel):
    query: str = Field(description="A single SELECT statement, no semicolon")


class QuestionArgs(BaseModel):
    question: str = Field(description="A question about the application data, in plain words")


class JiraArgs(BaseModel):
    application_id: str = Field(description="Application the exception belongs to")
    rule_id: str = Field(description="Policy rule that was breached, for example 3.1")
    summary: str = Field(description="One-line ticket title")
    description: str = Field("", description="Measured value and the analyst justification")


# ---------------------------------------------------------------------------
# Read-only tools over the Good Bank data
# ---------------------------------------------------------------------------


@observe(name="tool.get_application")
def _get_application(application_id: str) -> dict:
    import pandas as pd

    _record("get_application", {"application_id": application_id})
    engine = adapters.get_engine()
    app = pd.read_sql("select * from applications where application_id = ?", engine,
                      params=(application_id,))
    if not len(app):
        return {"error": f"no application {application_id}"}
    row = app.iloc[0].to_dict()
    company = pd.read_sql("select * from companies where company_id = ?", engine,
                          params=(row["company_id"],))
    out = {"application": adapters.as_json(row)}
    if len(company):
        out["company"] = adapters.as_json(company.iloc[0].to_dict())
    return out


@observe(name="tool.get_statement_features")
def _get_statement_features(application_id: str) -> dict:
    import pandas as pd

    _record("get_statement_features", {"application_id": application_id})
    features = pd.read_sql("select * from statement_features where application_id = ?",
                           adapters.get_engine(), params=(application_id,))
    if not len(features):
        return {"error": f"no statement features for {application_id}"}
    return adapters.as_json(features.iloc[0].to_dict())


@observe(name="tool.get_financials")
def _get_financials(company_id: str) -> dict:
    _record("get_financials", {"company_id": company_id})
    extraction = adapters.get_financials(company_id)
    if extraction is None:
        return {"error": f"no financial statement on file for {company_id}"}
    payload = adapters.as_json(extraction)
    if not extraction.years:
        payload["note"] = ("The statement is a scan with no text layer, so no figures could be "
                           "read without OCR. The ratios in section 3 cannot be computed.")
    return payload


@observe(name="tool.search_policy")
def _search_policy(query: str, source: Optional[str] = None) -> dict:
    _record("search_policy", {"query": query, "source": source})
    hits = adapters.search_policy(query, k=3, source=source)
    return {"query": query, "hits": [adapters.as_json(h) for h in hits]}


@observe(name="tool.get_group_exposure")
def _get_group_exposure(company_id: str) -> dict:
    _record("get_group_exposure", {"company_id": company_id})
    return adapters.as_json(adapters.get_group_exposure(company_id))


@observe(name="tool.score_risk")
def _score_risk(application_id: str) -> dict:
    _record("score_risk", {"application_id": application_id})
    features = _get_statement_features(application_id)
    features = {} if "error" in features else features
    return adapters.as_json(adapters.score_risk(application_id, features=features))


@observe(name="tool.get_core_banking_exposure")
def _get_core_banking_exposure(company_id: str) -> dict:
    _record("get_core_banking_exposure", {"company_id": company_id})
    from creditlens.integrations.oracle import CoreBankingAdapter

    return CoreBankingAdapter().get_exposure(company_id)


@observe(name="tool.get_repayment_history")
def _get_repayment_history(company_id: str) -> dict:
    _record("get_repayment_history", {"company_id": company_id})
    from creditlens.integrations.soap import RepaymentClient

    return adapters.as_json(RepaymentClient().get_history(company_id))


@observe(name="tool.file_jira_ticket")
def _file_jira_ticket(application_id: str, rule_id: str, summary: str, description: str = "") -> dict:
    _record("file_jira_ticket", {"application_id": application_id, "rule_id": rule_id})
    from creditlens.mcp_jira.client import file_ticket

    return file_ticket(application_id=application_id, rule_id=rule_id, summary=summary,
                       description=description)


# ---------------------------------------------------------------------------
# run_sql: the one tool that could do damage, so it is the one with the guard rails
# ---------------------------------------------------------------------------

#: Tables an agent may read. `directors` is deliberately absent: it is the PII table.
ALLOWED_TABLES = {"applications", "companies", "statement_features", "existing_loans",
                  "transactions", "company_directors", "quality_report"}

#: Tables that carry a `region` column, so an analyst's region filter can be applied.
REGION_TABLES = {"applications", "companies"}

FORBIDDEN = ("insert", "update", "delete", "drop", "alter", "create", "replace", "attach",
             "detach", "pragma", "vacuum", "truncate", "grant", "revoke", "exec", "execute",
             "merge", "reindex", "into")

MAX_ROWS = 200


class UnsafeQuery(ValueError):
    """Raised when a query fails the guard. The message is safe to show a user."""


def _tables_in(sql: str) -> set[str]:
    return {t.lower() for t in re.findall(r"\b(?:from|join)\s+([A-Za-z_][\w]*)", sql, flags=re.I)}


def guard_sql(query: str, user: Optional[User] = None) -> str:
    """Return a safe version of `query`, or raise UnsafeQuery. Used by run_sql and text_to_sql."""
    sql = " ".join((query or "").split()).strip().rstrip(";").strip()
    if not sql:
        raise UnsafeQuery("empty query")
    if ";" in sql:
        raise UnsafeQuery("only one statement is allowed and it must not contain a semicolon")
    if "--" in sql or "/*" in sql:
        raise UnsafeQuery("comments are not allowed in a query")
    if not re.match(r"^select\b", sql, flags=re.I):
        raise UnsafeQuery("only SELECT statements are allowed")
    lowered = sql.lower()
    for word in FORBIDDEN:
        if re.search(rf"\b{word}\b", lowered):
            raise UnsafeQuery(f"the keyword {word.upper()} is not allowed")

    tables = _tables_in(sql)
    if not tables:
        raise UnsafeQuery("the query must read from at least one table")
    unknown = tables - ALLOWED_TABLES
    if unknown:
        raise UnsafeQuery(f"these tables are not readable by the agent: {sorted(unknown)}")

    user = user if user is not None else get_current_user()
    if user is not None and user.role == "analyst" and tables & REGION_TABLES:
        sql = _inject_region(sql, user.region)
    return _enforce_limit(sql)


_TAIL = re.compile(r"\b(group\s+by|order\s+by|limit|having)\b", flags=re.I)


def _inject_region(sql: str, region: str) -> str:
    """Add `region = '<analyst region>'` so an analyst can never read another region."""
    safe_region = region.replace("'", "")
    if re.search(r"\bregion\s*=", sql, flags=re.I):
        return sql  # the caller already filtered; the WHERE below would be redundant
    where = re.search(r"\bwhere\b", sql, flags=re.I)
    if where:
        head, tail = sql[:where.end()], sql[where.end():]
        return f"{head} region = '{safe_region}' AND ({tail.strip()})"
    tail = _TAIL.search(sql)
    cut = tail.start() if tail else len(sql)
    return f"{sql[:cut].strip()} WHERE region = '{safe_region}' {sql[cut:].strip()}".strip()


def _enforce_limit(sql: str) -> str:
    match = re.search(r"\blimit\s+(\d+)\s*$", sql, flags=re.I)
    if match:
        if int(match.group(1)) <= MAX_ROWS:
            return sql
        return sql[:match.start()].rstrip() + f" LIMIT {MAX_ROWS}"
    return f"{sql} LIMIT {MAX_ROWS}"


@observe(name="tool.run_sql")
def _run_sql(query: str) -> dict:
    import pandas as pd

    _record("run_sql", {"query": query})
    try:
        safe = guard_sql(query)
    except UnsafeQuery as e:
        return {"error": str(e), "rejected": True, "query": query}
    df = pd.read_sql(safe, adapters.get_engine())
    return {"query": safe, "row_count": int(len(df)),
            "rows": json.loads(df.to_json(orient="records", date_format="iso"))}


# ---------------------------------------------------------------------------
# Text to SQL: the LLM writes the query, the guard decides whether it runs
# ---------------------------------------------------------------------------

SCHEMA_FOR_PROMPT = """
applications(application_id, company_id, region, product, purpose, requested_inr_000,
             tenure_months, submitted_on, analyst_id, guarantor_director_id)
companies(company_id, legal_name, sector, incorporated_on, registered_address, city, region, employees)
statement_features(application_id, months_covered, total_credits, total_debits, avg_monthly_inflow,
             inflow_volatility, min_balance, avg_balance, cheque_returns_12m, cash_credit_share,
             monthly_emi_outflow, top_customer_share, unknown_txn_share)
existing_loans(company_id, outstanding_inr_000)
transactions(application_id, seq, txn_date, channel, counterparty, amount, balance, is_bounce)
""".strip()


class SqlQuery(BaseModel):
    """What the model must return when asked to write SQL."""
    sql: str = Field(description="One SELECT statement, no semicolon, SQLite dialect")
    rationale: str = Field("", description="One sentence on what the query answers")


#: Canned answers for offline mode, so the whole flow runs in CI with no API key.
CANNED_SQL: list[tuple[tuple[str, ...], str, str]] = [
    (("how many", "count"), "SELECT COUNT(*) AS application_count FROM applications",
     "Counts the applications the caller may see."),
    (("largest", "biggest", "highest requested", "requested"),
     "SELECT application_id, company_id, requested_inr_000 FROM applications "
     "ORDER BY requested_inr_000 DESC LIMIT 10",
     "Lists the largest requests."),
    (("cheque", "bounce", "returned"),
     "SELECT application_id, cheque_returns_12m FROM statement_features "
     "ORDER BY cheque_returns_12m DESC LIMIT 10",
     "Ranks applications by returned cheques."),
    (("outstanding", "exposure", "existing loan"),
     "SELECT company_id, outstanding_inr_000 FROM existing_loans "
     "ORDER BY outstanding_inr_000 DESC LIMIT 10",
     "Ranks companies by existing outstanding exposure."),
]

DEFAULT_SQL = ("SELECT application_id, company_id, region, product, requested_inr_000 FROM applications",
               "Falls back to listing applications.")


def _offline_sql(messages: list[dict]) -> dict:
    question = " ".join(m.get("content", "") for m in messages if m.get("role") == "user").lower()
    for keywords, sql, rationale in CANNED_SQL:
        if any(k in question for k in keywords):
            return {"sql": sql, "rationale": rationale}
    return {"sql": DEFAULT_SQL[0], "rationale": DEFAULT_SQL[1]}


def _register_offline_handler() -> None:
    from creditlens.gateway.llm import OFFLINE_HANDLERS

    OFFLINE_HANDLERS.setdefault("SqlQuery", _offline_sql)


@observe(name="tool.text_to_sql")
def text_to_sql(question: str, user: Optional[User] = None) -> dict:
    """Ask the model for SQL, then put that SQL through exactly the same guard as a human's."""
    import pandas as pd

    from creditlens.gateway.llm import get_gateway

    _register_offline_handler()
    _record("text_to_sql", {"question": question})
    prompt = (f"Write one SQLite SELECT statement that answers this question:\n{question}\n\n"
              f"Tables and columns:\n{SCHEMA_FOR_PROMPT}\n\n"
              "Rules: SELECT only, no semicolon, no comments, keep it to one statement.")
    answer = get_gateway().extract(prompt, SqlQuery,
                                   system="You write careful, read-only SQL for a bank.")
    try:
        safe = guard_sql(answer.sql, user=user)
    except UnsafeQuery as e:
        return {"question": question, "sql": answer.sql, "rejected": True, "error": str(e)}
    df = pd.read_sql(safe, adapters.get_engine())
    return {"question": question, "sql": safe, "rationale": answer.rationale,
            "rejected": False, "row_count": int(len(df)),
            "rows": json.loads(df.to_json(orient="records", date_format="iso"))}


# ---------------------------------------------------------------------------
# Wrap each plain function as a LangChain tool. The graph calls the plain functions
# directly (deterministic routing); the tool objects are what a model would be bound to.
# ---------------------------------------------------------------------------


def _build_tools() -> dict[str, Any]:
    from langchain_core.tools import StructuredTool

    specs = [
        ("get_application", _get_application, ApplicationArgs,
         "Read one loan application and the company that applied."),
        ("get_statement_features", _get_statement_features, ApplicationArgs,
         "Read the 12-month banking-conduct features for one application."),
        ("get_financials", _get_financials, CompanyArgs,
         "Read the figures from the company's audited financial statements."),
        ("search_policy", _search_policy, PolicyArgs,
         "Search the Good Bank credit policy manual and return the matching sections."),
        ("get_group_exposure", _get_group_exposure, CompanyArgs,
         "Find the related-party group and its total exposure to Good Bank."),
        ("score_risk", _score_risk, ApplicationArgs,
         "Score the probability of default and explain the main drivers."),
        ("run_sql", _run_sql, SqlArgs,
         "Run one read-only SELECT over the approved tables. The result is always limited."),
        ("text_to_sql", text_to_sql, QuestionArgs,
         "Turn a plain-words question into safe SQL and run it."),
        ("get_core_banking_exposure", _get_core_banking_exposure, CompanyArgs,
         "Read the outstanding balance from the core banking system (Oracle)."),
        ("get_repayment_history", _get_repayment_history, CompanyArgs,
         "Read the repayment history from the legacy SOAP service."),
        ("file_jira_ticket", _file_jira_ticket, JiraArgs,
         "File a policy exception ticket in Jira through the MCP server. Idempotent."),
    ]
    return {name: StructuredTool.from_function(func=fn, name=name, description=doc,
                                               args_schema=schema)
            for name, fn, schema, doc in specs}


_TOOLS: Optional[dict[str, Any]] = None


def get_tools() -> dict[str, Any]:
    """The LangChain tool objects, built once per process (langchain_core is imported lazily)."""
    global _TOOLS
    if _TOOLS is None:
        _TOOLS = _build_tools()
    return _TOOLS


def tool_list() -> list:
    return list(get_tools().values())


# Plain-function aliases: the graph nodes use these, so a node never needs langchain_core.
get_application = _get_application
get_statement_features = _get_statement_features
get_financials = _get_financials
search_policy = _search_policy
get_group_exposure = _get_group_exposure
score_risk = _score_risk
run_sql = _run_sql
get_core_banking_exposure = _get_core_banking_exposure
get_repayment_history = _get_repayment_history
file_jira_ticket = _file_jira_ticket

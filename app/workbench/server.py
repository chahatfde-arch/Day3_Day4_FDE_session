"""Good Bank Credit Review Workbench: the Underwriter Agent as a real app.

    python app/workbench/server.py            # then open http://localhost:8090

The agent is notebook 04b's own code: at start-up the server runs the notebook's code cells once (about a
minute, in the background), then wires the same node functions into a LangGraph that works for every file
in the queue. The connections are notebook 05b's: Oracle core banking and the SOAP repayment service
(live with Docker, labelled stand-ins without), Jira through MCP with an idempotency key, and a Slack
message preview. Every decision is checked against the signer's role on the server, and every action is
written to an audit trail. Offline by default: no API key needed.
"""
from __future__ import annotations

import contextlib
import io
import json

import os
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)
os.environ.setdefault("LLM_OFFLINE", "1")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{(REPO / 'creditlens.db').as_posix()}")

import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

HERE = Path(__file__).resolve().parent
NB04B = REPO / "notebooks" / "src" / "04b_underwriter_agent.py"
JIRA = REPO / "data" / "cache" / "jira_workbench.json"
AUDIT = REPO / "data" / "cache" / "workbench_audit.jsonl"
STORY_APPS = ("A901", "A902")

app = FastAPI(title="Good Bank Credit Review Workbench")

# ----------------------------------------------------------------------------- loading the agent (background)
STATE: dict[str, Any] = {"ready": False, "error": None, "started": time.time(), "ns": None, "graph": None}


def code_cells(path: Path) -> list[str]:
    parts = re.split(r"^# %%(.*)$", path.read_text(encoding="utf-8"), flags=re.M)
    return [body for hdr, body in zip(parts[1::2], parts[2::2]) if "[markdown]" not in hdr]


def load_agent() -> None:
    try:
        import types
        import matplotlib
        matplotlib.use("Agg")
        module = types.ModuleType("notebook_04b")          # a real module: @dataclass and pydantic look it up
        sys.modules["notebook_04b"] = module
        ns: dict = module.__dict__
        for i, code in enumerate(code_cells(NB04B), 1):
            if i in (1, 3):                 # the Colab clone, and the offline switch (the server decides that)
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(code, f"notebook 04b, code cell {i}", "exec"), ns)
        STATE["ns"] = ns
        STATE["graph"] = build_graph(ns)
        STATE["ready"] = True
        STATE["ready_after_s"] = round(time.time() - STATE["started"], 1)
    except Exception as err:  # noqa: BLE001
        STATE["error"] = f"{type(err).__name__}: {err}"


def build_graph(ns: dict):
    """The notebook's seven nodes in the notebook's order. Only perceive and consult_leads are rewired, so
    that the Good Bank files read the bank's data and the two story files read the story's."""
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph

    def n_perceive(s):
        ns["WORLD"][s["app"]] = perceive(s["app"])
        return {"path": ["perceive"]}

    def n_leads(s):
        if s["app"] not in STORY_APPS:
            return {"leads": [], "path": ["consult_leads"]}
        with ns["story_env"]():
            return {"leads": ns["missed_leads"](s["app"]), "path": ["consult_leads"]}

    b = StateGraph(ns["ReviewState"])              # the notebook's own state: app, proposal, note, decision, path...
    for name, fn in [("perceive", n_perceive), ("sanity", ns["n_sanity"]), ("consult_leads", n_leads),
                     ("weigh_options", ns["n_options"]), ("write_note", ns["n_note"]),
                     ("underwriter_signs", ns["n_gate"]), ("learn", ns["n_learn"])]:
        b.add_node(name, fn)
    chain = ["perceive", "sanity", "consult_leads", "weigh_options", "write_note", "underwriter_signs", "learn"]
    b.add_edge(START, chain[0])
    for x, y in zip(chain, chain[1:]):
        b.add_edge(x, y)
    b.add_edge(chain[-1], END)
    return b.compile(checkpointer=MemorySaver())


def perceive(app_id: str):
    ns = STATE["ns"]
    if app_id in STORY_APPS:
        with ns["story_env"]():
            return ns["perceive"](app_id)
    return ns["perceive"](app_id)


threading.Thread(target=load_agent, daemon=True).start()


def need_agent():
    if STATE["error"]:
        raise HTTPException(500, f"the agent could not load: {STATE['error']}")
    if not STATE["ready"]:
        raise HTTPException(503, "the agent is still loading (about a minute after start-up)")
    return STATE["ns"]


# ----------------------------------------------------------------------------- people, queue, audit
USERS = pd.read_csv(REPO / "data" / "raw" / "registry" / "users.csv").set_index("user_id")


def user(uid: str) -> dict:
    if uid not in USERS.index:
        raise HTTPException(401, f"unknown user {uid}")
    r = USERS.loc[uid]
    return {"user_id": uid, "full_name": r.full_name, "role": r.role, "region": r.region}


def load_queue() -> list[dict]:
    import sqlite3
    rows = []
    with sqlite3.connect(REPO / "creditlens.db") as c:
        q = ("select a.application_id, a.company_id, c.legal_name, c.sector, a.region, a.product, a.purpose, "
             "a.requested_inr_000, a.tenure_months, a.submitted_on, a.analyst_id from applications a "
             "join companies c on c.company_id = a.company_id order by a.application_id")
        rows += [dict(zip([d[0] for d in c.execute(q).description], r)) for r in c.execute(q).fetchall()]
    sa = pd.read_csv(REPO / "data/story/raw/applications/applications.csv")
    sc = pd.read_csv(REPO / "data/story/raw/registry/companies.csv")
    story = sa[sa.application_id.isin(STORY_APPS)].merge(sc[["company_id", "legal_name", "sector"]], on="company_id")
    for r in story.to_dict("records"):
        rows.insert(0, {k: r.get(k) for k in ["application_id", "company_id", "legal_name", "sector", "region", "product",
                                              "purpose", "requested_inr_000", "tenure_months", "submitted_on", "analyst_id"]})
    for r in rows:
        r["requested_inr_000"] = int(r["requested_inr_000"])
        r["story"] = r["application_id"] in STORY_APPS
    return rows


QUEUE = {r["application_id"]: r for r in load_queue()}
REVIEWS: dict[str, dict] = {}
LOCK = threading.Lock()


def visible(u: dict, a: dict) -> bool:
    return u["region"] == "ALL" or a["region"] == u["region"]


def audit(app_id: str, actor: str, action: str, detail: str = "") -> None:
    ev = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "application_id": app_id,
          "actor": actor, "action": action, "detail": detail}
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(ev) + "\n")


def audit_events(app_id: Optional[str] = None) -> list[dict]:
    if not AUDIT.exists():
        return []
    evs = [json.loads(l) for l in AUDIT.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [e for e in evs if app_id is None or e["application_id"] == app_id][::-1]


def get_app(app_id: str, uid: str) -> tuple[dict, dict]:
    u = user(uid)
    a = QUEUE.get(app_id)
    if a is None:
        raise HTTPException(404, f"no application {app_id}")
    if not visible(u, a):
        audit(app_id, uid, "access_refused", f"{u['full_name']} ({u['region']}) asked for a {a['region']} file")
        raise HTTPException(403, f"{u['full_name']} works in the {u['region']} region; {app_id} is a {a['region']} file.")
    return u, a


# ----------------------------------------------------------------------------- the bank's systems (notebook 05b)
from creditlens.integrations.oracle import CoreBankingAdapter  # noqa: E402
from creditlens.integrations.soap import RepaymentClient  # noqa: E402

SOAP = RepaymentClient(timeout=3, retries=1)        # one client, so its circuit breaker protects every review


def connected_sensors(company_id: str) -> dict:
    t0 = time.time(); exp = CoreBankingAdapter().get_exposure(company_id); t1 = time.time()
    h = SOAP.get_history(company_id); t2 = time.time()
    facts = {"core_banking_outstanding_inr_000": {"value": exp["outstanding_inr_000"], "source": exp["source"]},
             "core_banking_product": {"value": exp["product"], "source": exp["source"]},
             "repayment_instalments": {"value": len(h.entries), "source": h.source},
             "repayment_paid_late": {"value": h.late_count, "source": h.source},
             "repayment_worst_delay_days": {"value": h.worst_delay_days, "source": h.source}}
    caveats = []
    if exp["source"] != "oracle":
        caveats.append("Core banking is unavailable. Exposure figures are from the replicated copy.")
    if h.source != "soap":
        caveats.append("The repayment service is unavailable. Repayment history is estimated.")
    return {"facts": facts, "caveats": caveats, "oracle_s": round(t1 - t0, 2), "soap_s": round(t2 - t1, 2),
            "breaker": SOAP.breaker.state, "entries": [e.model_dump() for e in h.entries]}


def evidence(m) -> dict:
    years = m.financials.get("years") or []
    return {"form": {k: (str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v) for k, v in m.form.items()},
            "conduct": m.conduct, "financials": years, "group": m.group, "risk": m.risk,
            "checks": [c.model_dump() for c in m.checks], "readings": m.readings + ["run_checks"]}


# ----------------------------------------------------------------------------- API
@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/api/status")
def status():
    return {"ready": STATE["ready"], "error": STATE["error"], "loading_for_s": round(time.time() - STATE["started"]),
            "ready_after_s": STATE.get("ready_after_s"),
            "model": "offline (no API key)" if os.environ.get("LLM_OFFLINE") == "1" else "live model",
            "oracle": "configured" if os.environ.get("ORACLE_DSN") else "stand-in", "breaker": SOAP.breaker.state}


@app.get("/api/users")
def users():
    return [user(u) for u in USERS.index]


@app.get("/api/queue")
def queue(user_id: str):
    u = user(user_id)
    out = []
    for a in QUEUE.values():
        if visible(u, a):
            r = REVIEWS.get(a["application_id"], {})
            out.append(a | {"status": r.get("status", "new"), "proposal": (r.get("note") or {}).get("proposal"),
                            "amount": (r.get("note") or {}).get("amount_inr_000"),
                            "decision": (r.get("decision") or {}).get("decision")})
    return {"user": u, "applications": out}


@app.get("/api/reviews/{app_id}")
def review(app_id: str, user_id: str):
    _, a = get_app(app_id, user_id)
    r = REVIEWS.get(app_id)
    return {"application": a, "review": {k: v for k, v in (r or {}).items() if k != "cfg"} or None,
            "audit": audit_events(app_id)}


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@app.get("/api/reviews/{app_id}/prepare")
def prepare(app_id: str, user_id: str):
    ns = need_agent()
    u, a = get_app(app_id, user_id)
    if REVIEWS.get(app_id, {}).get("status") == "signed":
        raise HTTPException(409, f"{app_id} is already signed")

    def run():
        t_all = time.time()
        log: list[dict] = []          # the processing log, kept with the review

        def step(data: dict) -> str:
            if data.get("state") != "running":
                log.append({k: data.get(k) for k in ("step", "state", "detail", "ms")})
            return sse("step", data)

        yield step({"step": "core banking (Oracle) and repayments (SOAP)", "state": "running"})
        conn = connected_sensors(a["company_id"])
        yield step({"step": "core banking (Oracle) and repayments (SOAP)", "state": "done",
                    "ms": int((conn["oracle_s"] + conn["soap_s"]) * 1000),
                    "detail": f"Oracle {conn['facts']['core_banking_product']['source']} in {conn['oracle_s']} s · "
                              f"SOAP {conn['facts']['repayment_instalments']['source']} in {conn['soap_s']} s · "
                              f"breaker {conn['breaker']}"})
        cfg = {"configurable": {"thread_id": f"workbench-{app_id}-{uuid.uuid4().hex[:8]}"}}
        t = time.time()
        note = None
        try:
            for update in STATE["graph"].stream({"app": app_id}, cfg, stream_mode="updates"):
                for node, val in update.items():
                    if node == "__interrupt__":
                        note = val[0].value["note"]
                        yield step({"step": "underwriter_signs", "state": "waiting",
                                    "detail": "the graph is paused: waiting for the Head of Credit"})
                    else:
                        yield step({"step": node, "state": "done", "ms": int((time.time() - t) * 1000),
                                    "detail": describe(node, val)})
                        t = time.time()
        except Exception as err:  # noqa: BLE001
            yield sse("error", {"error": f"{type(err).__name__}: {err}"})
            return
        m = ns["WORLD"][app_id]
        memory = json.loads(ns["FEEDBACK"].read_text()) if ns["FEEDBACK"].exists() else {}
        learned_for = {cond: trig for trig, conds in memory.items() for cond in conds}   # so signing keeps the trigger
        with LOCK:
            REVIEWS[app_id] = {"status": "awaiting_signature", "cfg": cfg, "note": note, "evidence": evidence(m),
                               "connected": conn, "prepared_by": "Underwriter Agent", "requested_by": u["user_id"],
                               "prompt_version": ns["PROMPT_VERSION"], "seconds": round(time.time() - t_all, 1),
                               "tickets": [], "decision": None, "critic": None, "log": log,
                               "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "condition_triggers": {c: learned_for.get(c, "always") for c in note.get("conditions") or []}}
        audit(app_id, "underwriter-agent", "review_prepared",
              f"proposal {note['proposal']} {note.get('amount_inr_000') or ''}; requested by {u['full_name']}")
        yield sse("ready", {k: v for k, v in REVIEWS[app_id].items() if k != "cfg"})

    return StreamingResponse(run(), media_type="text/event-stream")


def describe(node: str, val: dict) -> str:
    if node == "perceive":
        return "six sensors read the file"
    if node == "sanity":
        u = val.get("unreliable") or []
        return "figures look unreliable: " + "; ".join(u) if u else "every accounting check holds"
    if node == "consult_leads":
        leads = val.get("leads") or []
        return f"{len(leads)} lead(s) the memo did not mention" if leads else "nothing missed"
    if node == "weigh_options":
        p = val.get("proposal") or {}
        return f"proposal: {p.get('proposal')} {p.get('amount_inr_000') or ''}".strip()
    if node == "write_note":
        return "the model wrote the note; code set the proposal; the lint passed"
    return ""


class Condition(BaseModel):
    text: str
    trigger: str = "always"


class Decision(BaseModel):
    user_id: str
    decision: str
    amount_inr_000: int = 0
    conditions: list[Condition] = []
    comment: str = ""


@app.post("/api/reviews/{app_id}/sign")
def sign(app_id: str, d: Decision):
    ns = need_agent()
    u, a = get_app(app_id, d.user_id)
    r = REVIEWS.get(app_id)
    if not r or r["status"] != "awaiting_signature":
        raise HTTPException(409, "there is no review waiting for a signature; prepare one first")
    if u["role"] != "head_of_credit":           # checked before the paused run is touched (policy 7.3)
        audit(app_id, u["user_id"], "signature_refused", f"{u['full_name']} has the role {u['role']}")
        raise HTTPException(403, f"{u['full_name']} has the role {u['role']}. Policy 7.3: only the Head of Credit decides.")
    decision = {"decided_by": u["user_id"], "decision": d.decision, "amount_inr_000": int(d.amount_inr_000),
                "conditions": [c.text for c in d.conditions if c.text.strip()],
                "condition_triggers": {c.text: c.trigger for c in d.conditions if c.text.strip()}}
    from langgraph.types import Command
    final = STATE["graph"].invoke(Command(resume=decision), r["cfg"])   # the gate node checks the role again
    with LOCK:
        r.update(status="signed", decision=decision, critic=final.get("critic"), path=final.get("path"), comment=d.comment,
                 decided_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    audit(app_id, u["user_id"], "signed",
          f"{d.decision} {d.amount_inr_000 or ''}; conditions: {'; '.join(decision['conditions']) or 'none'}"
          + (f"; comment: {d.comment}" if d.comment else ""))
    if final.get("critic"):
        audit(app_id, "underwriter-agent", "learned", "; ".join(final["critic"]))
    return {k: v for k, v in r.items() if k != "cfg"} | {"audit": audit_events(app_id)}


class Who(BaseModel):
    user_id: str


@app.post("/api/reviews/{app_id}/tickets")
def tickets(app_id: str, w: Who):
    from creditlens.mcp_jira.client import file_ticket
    u, a = get_app(app_id, w.user_id)
    r = REVIEWS.get(app_id)
    if not r:
        raise HTTPException(409, "prepare the review first")
    failed = sorted(c["rule_id"] for c in r["evidence"]["checks"] if not c["passed"])
    if not failed:
        return {"tickets": [], "note": "no policy exception, so no ticket"}
    out = [file_ticket(app_id, rule, transport="inprocess", jira_path=str(JIRA)) for rule in failed]
    for t in out:
        audit(app_id, u["user_id"], "ticket_filed" if t.get("created") else "ticket_already_filed",
              f"{t.get('key')} (idempotency key {t.get('idempotency_key')})")
    r["tickets"] = [{"key": t.get("key"), "created": t.get("created"), "idempotency_key": t.get("idempotency_key"),
                     "summary": t.get("summary")} for t in out]
    return {"tickets": r["tickets"], "audit": audit_events(app_id)}


def crore(inr_000: Optional[int]) -> str:
    """INR thousand to the way a bank writes it: 15,000 thousand is ₹1.50 Cr."""
    return f"₹{inr_000 / 10000:,.2f} Cr" if inr_000 else ""


@app.post("/api/reviews/{app_id}/notify")
def notify(app_id: str, w: Who):
    u, a = get_app(app_id, w.user_id)
    r = REVIEWS.get(app_id)
    if not r:
        raise HTTPException(409, "prepare the review first")
    n = r["note"]
    audit(app_id, u["user_id"], "slack_notified", f"{os.environ.get('SLACK_APPROVALS_CHANNEL', '#credit-approvals')} (dry run)")
    return {"channel": os.environ.get("SLACK_APPROVALS_CHANNEL", "#credit-approvals"), "dry_run": True,
            "title": f"Review ready: {app_id} · {a['legal_name']}",
            "fields": {"Requested": crore(a["requested_inr_000"]),
                       "Proposal": f"{n['proposal']} {crore(n.get('amount_inr_000'))}".strip(),
                       "Risk": f"{r['evidence']['risk'].get('risk_band')} band", "Questions": str(len(n['questions_for_underwriter']))},
            "reasons": n["reasons"], "buttons": ["Open the review", "Approve", "Reject"],
            "rule": "A click is checked against SLACK_USER_MAP: only a Slack account mapped to the Head of Credit can decide."}


@app.get("/api/audit")
def all_audit(user_id: str):
    u = user(user_id)
    return [e for e in audit_events() if u["region"] == "ALL" or QUEUE.get(e["application_id"], {}).get("region") == u["region"]]


@app.get("/api/memory")
def memory():
    ns = need_agent()
    fb = ns["FEEDBACK"]
    return json.loads(fb.read_text()) if fb.exists() else {}


@app.get("/api/systems")
def systems():
    import shutil
    import subprocess
    out = {"oracle": CoreBankingAdapter().backend, "soap_breaker": SOAP.breaker.state, "containers": []}
    if shutil.which("docker"):
        try:
            p = subprocess.run(["docker", "ps", "--format", "{{.Names}}\t{{.Status}}"], capture_output=True, text=True, timeout=8)
            out["containers"] = [dict(zip(["name", "status"], l.split("\t"))) for l in p.stdout.splitlines() if l.startswith("creditlens")]
        except Exception:  # noqa: BLE001
            pass
    return out


@app.exception_handler(HTTPException)
def http_error(_, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("WORKBENCH_PORT", "8090"))
    print(f"Good Bank Credit Workbench on http://localhost:{port}  (the review assistant needs about a minute to start)")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

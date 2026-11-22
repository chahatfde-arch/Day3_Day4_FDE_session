"""A local Jira-compatible service: the three Jira Cloud REST v3 calls our MCP server makes.

Why it exists: Jira cannot run on a laptop (Jira Data Center needs a licence and several GB of
memory), but the forward deployed engineer still wants the real Jira Cloud client to run end
to end, over HTTP, with Basic auth and Atlassian's JSON shapes. This file speaks just enough
of that API for the agent:

    POST /rest/api/3/issue                 create an issue          (returns 201, id, key, self)
    GET  /rest/api/3/issue/{key}           read one issue           (404 if it does not exist)
    GET  /rest/api/3/search?jql=...        search with simple JQL   (also /rest/api/3/search/jql)
    GET  /rest/api/3/myself                who am I (the health check the workbench uses)
    GET  /browse/{key}                     a small HTML page, so ticket links open

JQL understood: clauses joined by AND, each one of  labels = "x"  project = X  key = X
text ~ "x". Anything else returns 400 and says so, rather than silently matching nothing.

Honest limits: it accepts ANY email and token (it checks that Basic auth is present, nothing
more), keeps issues in one JSON file, and has no workflow beyond "To Do". It is a stand-in,
and the workbench labels it "Local", never "Live".

Run it:
    python infra/jira_mock/server.py --port 8090 --store data/cache/jira_local.json
    docker compose -f infra/docker-compose.yml --profile legacy up -d jira-mock
Then in .env: JIRA_BASE_URL=http://localhost:8090, JIRA_EMAIL and JIRA_API_TOKEN set to anything.

Standard library only, so it runs in any Python 3.9+ container without installing anything.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import re
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

LOCK = threading.Lock()
STORE = Path("jira_local.json")
CLAUSE = re.compile(r'^\s*(labels|project|key|text)\s*(=|~)\s*"?([^"]*?)"?\s*$', re.IGNORECASE)


# ---------------------------------------------------------------------------- storage
def load() -> dict:
    if STORE.exists():
        try:
            return json.loads(STORE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"issues": [], "counters": {}}


def save(store: dict) -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps(store, indent=1), encoding="utf-8")


def adf_text(doc) -> str:
    """Flatten Atlassian Document Format back to plain text, for search and the browse page."""
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        return doc.get("text", "") + " ".join(adf_text(c) for c in doc.get("content", []))
    if isinstance(doc, list):
        return " ".join(adf_text(c) for c in doc)
    return ""


def as_jira(issue: dict, base: str) -> dict:
    """One stored issue in the shape Jira Cloud returns."""
    return {"id": issue["id"], "key": issue["key"], "self": f"{base}/rest/api/3/issue/{issue['id']}",
            "fields": {"summary": issue["summary"], "description": issue["description"],
                       "labels": issue["labels"], "status": {"name": issue["status"]},
                       "project": {"key": issue["project"]}, "issuetype": {"name": issue["issuetype"]},
                       "created": issue["created"]}}


def parse_jql(jql: str) -> list[tuple[str, str]]:
    """Check the whole query first, so a bad clause is refused even when there are no issues yet."""
    jql = re.split(r"\s+ORDER\s+BY\s+", jql, flags=re.IGNORECASE)[0]
    clauses = []
    for clause in re.split(r"\s+AND\s+", jql.strip(), flags=re.IGNORECASE):
        if not clause:
            continue
        m = CLAUSE.match(clause)
        if not m:
            raise ValueError(f"The local Jira mock does not understand this JQL clause: {clause!r}")
        field, op, value = m.group(1).lower(), m.group(2), m.group(3)
        if (field == "text") != (op == "~"):        # as in Jira: text takes ~, the others take =
            raise ValueError(f"Use {'~' if field == 'text' else '='} with {field}: {clause!r}")
        clauses.append((field, value))
    return clauses


def matches(issue: dict, clauses: list[tuple[str, str]]) -> bool:
    for field, value in clauses:
        if field == "labels" and value not in issue["labels"]:
            return False
        if field == "project" and issue["project"].lower() != value.lower():
            return False
        if field == "key" and issue["key"].lower() != value.lower():
            return False
        if field == "text" and value.lower() not in (issue["summary"] + " " + adf_text(issue["description"])).lower():
            return False
    return True


# ---------------------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "local-jira-mock/1.0"

    def base(self) -> str:
        return f"http://{self.headers.get('Host', 'localhost')}"

    def send_json(self, status: int, body) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def user(self):
        """Basic auth must be present, as with Jira Cloud. Any email and token are accepted."""
        header = self.headers.get("Authorization", "")
        if not header.startswith("Basic "):
            return None
        try:
            email, token = base64.b64decode(header[6:]).decode("utf-8").split(":", 1)
        except Exception:  # noqa: BLE001
            return None
        return email if email and token else None

    def refuse(self) -> None:
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="local-jira-mock"')
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"errorMessages":["Send Basic auth (any email and API token) like Jira Cloud."]}')

    def do_GET(self):  # noqa: N802
        url = urlparse(self.path)
        if url.path == "/health":
            return self.send_json(200, {"ok": True})
        if url.path.startswith("/browse/"):
            return self.browse(url.path.rsplit("/", 1)[-1])
        email = self.user()
        if email is None:
            return self.refuse()
        if url.path == "/rest/api/3/myself":
            return self.send_json(200, {"emailAddress": email, "displayName": "Local Jira mock user", "active": True})
        m = re.fullmatch(r"/rest/api/3/issue/([A-Za-z0-9-]+)", url.path)
        if m:
            with LOCK:
                found = next((i for i in load()["issues"] if m.group(1) in (i["key"], i["id"])), None)
            if not found:
                return self.send_json(404, {"errorMessages": ["Issue does not exist or you do not have permission to see it."]})
            return self.send_json(200, as_jira(found, self.base()))
        if url.path in ("/rest/api/3/search", "/rest/api/3/search/jql"):
            q = parse_qs(url.query)
            jql, limit = q.get("jql", [""])[0], int(q.get("maxResults", ["50"])[0])
            with LOCK:
                issues = load()["issues"]
            try:
                clauses = parse_jql(jql)
            except ValueError as err:
                return self.send_json(400, {"errorMessages": [str(err)]})
            hits = [i for i in issues if matches(i, clauses)]
            return self.send_json(200, {"startAt": 0, "maxResults": limit, "total": len(hits),
                                        "issues": [as_jira(i, self.base()) for i in hits[:limit]]})
        self.send_json(404, {"errorMessages": [f"The local Jira mock has no endpoint {url.path}"]})

    def do_POST(self):  # noqa: N802
        if urlparse(self.path).path != "/rest/api/3/issue":
            return self.send_json(404, {"errorMessages": [f"The local Jira mock has no endpoint {self.path}"]})
        if self.user() is None:
            return self.refuse()
        try:
            fields = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")["fields"]
            project, summary = fields["project"]["key"], fields["summary"]
        except (KeyError, TypeError, json.JSONDecodeError):
            return self.send_json(400, {"errorMessages": ["Send {\"fields\": {\"project\": {\"key\": ...}, \"summary\": ...}}"]})
        with LOCK:
            store = load()
            n = store["counters"].get(project, 0) + 1
            store["counters"][project] = n
            issue = {"id": str(10000 + len(store["issues"]) + 1), "key": f"{project}-{n}", "project": project,
                     "summary": summary, "description": fields.get("description", ""),
                     "labels": list(fields.get("labels", [])), "status": "To Do",
                     "issuetype": (fields.get("issuetype") or {}).get("name", "Task"),
                     "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            store["issues"].append(issue)
            save(store)
        self.send_json(201, {"id": issue["id"], "key": issue["key"], "self": f"{self.base()}/rest/api/3/issue/{issue['id']}"})

    def browse(self, key: str) -> None:
        with LOCK:
            found = next((i for i in load()["issues"] if i["key"] == key), None)
        body = (f"<h1>{html.escape(key)}</h1><p>{html.escape(found['summary'])}</p><p>Status: {found['status']}</p>"
                f"<p>Labels: {html.escape(', '.join(found['labels']))}</p><pre>{html.escape(adf_text(found['description']))}</pre>"
                if found else f"<h1>{html.escape(key)}</h1><p>No such issue.</p>")
        data = (f"<!doctype html><meta charset=utf-8><title>{html.escape(key)} · local Jira mock</title>"
                f"<body style='font-family:system-ui;max-width:720px;margin:40px auto'>"
                f"<p style='color:#a15c00'>Local Jira mock, not a real Jira site.</p>{body}</body>").encode("utf-8")
        self.send_response(200 if found else 404)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):  # one short line per request
        print(f"{self.command} {self.path} -> {args[1] if len(args) > 1 else ''}", flush=True)


def main() -> None:
    global STORE
    ap = argparse.ArgumentParser(description="A local Jira-compatible service for the CreditLens MCP server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--store", default="data/cache/jira_local.json")
    a = ap.parse_args()
    STORE = Path(a.store)
    print(f"Local Jira mock on http://{a.host}:{a.port}, issues in {STORE}", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

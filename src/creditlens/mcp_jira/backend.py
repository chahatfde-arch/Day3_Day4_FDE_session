"""Two Jira backends with the same three methods: the real cloud API, and a JSON-file mock.

The one idea this module teaches: **idempotency**. An agent retries. A graph resumes. A
Slack button gets double-clicked. If "file a ticket" is not idempotent, one policy exception
becomes four tickets and the audit trail stops meaning anything.

The trick is an *idempotency key* that the caller derives from the thing itself, here
`f"{application_id}:{rule_id}"`. Before creating anything, the backend searches for that key.
Jira has no idempotency header, so the key is stored as a label on the issue and searched
with JQL. The mock stores it in a dict. Same contract, same guarantee.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Optional

from creditlens.agents.adapters import cache_dir

LABEL_PREFIX = "creditlens-idem-"
DEFAULT_PROJECT = os.environ.get("JIRA_PROJECT_KEY", "CREDIT")


def label_for(idempotency_key: str) -> str:
    """Jira labels may not contain spaces or colons, so the key is flattened."""
    flat = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in idempotency_key)
    return f"{LABEL_PREFIX}{flat}"


def jira_json_path() -> Path:
    override = os.environ.get("CREDITLENS_JIRA_PATH")
    return Path(override) if override else cache_dir() / "jira.json"


# ---------------------------------------------------------------------------
# The mock: a JSON file that behaves like a very small Jira
# ---------------------------------------------------------------------------


class MockJira:
    """A Jira that fits in one file. Persisted so the server and the tests see the same issues."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else jira_json_path()
        self._lock = threading.Lock()

    # --- storage ----------------------------------------------------------------
    def _load(self) -> dict:
        if not self.path.exists():
            return {"issues": [], "counters": {}}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"issues": [], "counters": {}}

    def _save(self, store: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(store, indent=1), encoding="utf-8")

    # --- the three MCP tools ------------------------------------------------------
    def create_issue(self, project_key: str, summary: str, description: str = "",
                     idempotency_key: str = "") -> dict:
        with self._lock:
            store = self._load()
            label = label_for(idempotency_key) if idempotency_key else ""
            if label:
                for issue in store["issues"]:
                    if label in issue.get("labels", []):
                        # Already filed. Return the same ticket and say so.
                        return {**issue, "created": False}
            number = int(store["counters"].get(project_key, 0)) + 1
            store["counters"][project_key] = number
            issue = {
                "key": f"{project_key}-{number}",
                "id": str(10000 + number),
                "project_key": project_key,
                "summary": summary,
                "description": description,
                "labels": [label] if label else [],
                "idempotency_key": idempotency_key,
                "status": "To Do",
                "url": f"mock://jira/{project_key}-{number}",
                "backend": "mock",
            }
            store["issues"].append(issue)
            self._save(store)
            return {**issue, "created": True}

    def get_issue(self, key: str) -> dict:
        for issue in self._load()["issues"]:
            if issue["key"] == key:
                return issue
        return {"error": f"no issue {key}"}

    def search_issues(self, jql_or_text: str) -> list[dict]:
        needle = (jql_or_text or "").lower()
        # Accept either a label JQL clause or free text; the mock treats both as "contains".
        if "labels" in needle and "=" in needle:
            needle = needle.split("=", 1)[1].strip().strip("\"'")
        return [issue for issue in self._load()["issues"]
                if needle in json.dumps(issue).lower()]


# ---------------------------------------------------------------------------
# The real thing: Jira Cloud REST v3 with basic auth (email + API token)
# ---------------------------------------------------------------------------


class JiraCloud:
    """Jira Cloud REST v3. Only the three calls the agent needs."""

    def __init__(self, base_url: str, email: str, api_token: str, timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.auth = (email, api_token)
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs) -> Any:
        import requests

        response = requests.request(method, f"{self.base_url}{path}", auth=self.auth,
                                    headers={"Accept": "application/json"},
                                    timeout=self.timeout, **kwargs)
        response.raise_for_status()
        return response.json() if response.content else {}

    @staticmethod
    def _adf(text: str) -> dict:
        """Jira v3 wants Atlassian Document Format, not a plain string."""
        return {"type": "doc", "version": 1,
                "content": [{"type": "paragraph",
                             "content": [{"type": "text", "text": text or " "}]}]}

    def create_issue(self, project_key: str, summary: str, description: str = "",
                     idempotency_key: str = "") -> dict:
        label = label_for(idempotency_key) if idempotency_key else ""
        if label:
            existing = self.search_issues(f'labels = "{label}"')
            if existing:
                return {**existing[0], "created": False}
        payload = {"fields": {"project": {"key": project_key}, "summary": summary,
                              "description": self._adf(description),
                              "issuetype": {"name": "Task"},
                              "labels": [label] if label else []}}
        created = self._request("POST", "/rest/api/3/issue", json=payload,
                                headers={"Content-Type": "application/json",
                                         "Accept": "application/json"})
        return {"key": created["key"], "id": created.get("id", ""), "project_key": project_key,
                "summary": summary, "description": description,
                "labels": [label] if label else [], "idempotency_key": idempotency_key,
                "status": "To Do", "url": f"{self.base_url}/browse/{created['key']}",
                "backend": "jira-cloud", "created": True}

    def get_issue(self, key: str) -> dict:
        data = self._request("GET", f"/rest/api/3/issue/{key}")
        fields = data.get("fields", {})
        return {"key": data.get("key", key), "id": data.get("id", ""),
                "summary": fields.get("summary", ""), "labels": fields.get("labels", []),
                "status": (fields.get("status") or {}).get("name", ""),
                "url": f"{self.base_url}/browse/{data.get('key', key)}", "backend": "jira-cloud"}

    def search_issues(self, jql_or_text: str) -> list[dict]:
        jql = jql_or_text if "=" in jql_or_text or "~" in jql_or_text else f'text ~ "{jql_or_text}"'
        data = self._request("GET", "/rest/api/3/search", params={"jql": jql, "maxResults": 20})
        out = []
        for issue in data.get("issues", []):
            fields = issue.get("fields", {})
            out.append({"key": issue["key"], "id": issue.get("id", ""),
                        "summary": fields.get("summary", ""), "labels": fields.get("labels", []),
                        "status": (fields.get("status") or {}).get("name", ""),
                        "url": f"{self.base_url}/browse/{issue['key']}", "backend": "jira-cloud"})
        return out


# ---------------------------------------------------------------------------
# Picking one
# ---------------------------------------------------------------------------


def jira_cloud_configured() -> bool:
    return all(os.environ.get(name) for name in ("JIRA_BASE_URL", "JIRA_EMAIL", "JIRA_API_TOKEN"))


def get_backend(path: Optional[Path] = None):
    """Real Jira when it is configured, the mock otherwise. Same three methods either way."""
    if jira_cloud_configured():
        return JiraCloud(os.environ["JIRA_BASE_URL"], os.environ["JIRA_EMAIL"],
                         os.environ["JIRA_API_TOKEN"])
    return MockJira(path)

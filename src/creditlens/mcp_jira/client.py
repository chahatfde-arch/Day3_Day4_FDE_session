"""Talking to the Jira MCP server from the agent.

`file_ticket()` is the only function the graph needs. It has two ways to reach the server:

* **stdio** (`transport="stdio"`): spawn `python -m creditlens.mcp_jira` as a child process
  and speak MCP over the pipe. This is the real thing, and what you would point Claude
  Desktop or any other MCP client at.
* **in-process** (`transport="inprocess"`, the default): call the same backend class
  directly. No subprocess, no event loop, no 1.5 seconds of Python start-up per ticket.

Tests use the in-process path on purpose: a unit test should not need to spawn processes to
prove that idempotency works, and on Windows each spawn costs more than the test does.
Set `CREDITLENS_MCP_TRANSPORT=stdio` to exercise the protocol for real.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Optional

from creditlens.mcp_jira.backend import DEFAULT_PROJECT, get_backend
from creditlens.observability import observe

STDIO_TIMEOUT_S = float(os.environ.get("CREDITLENS_MCP_TIMEOUT", "30"))


def idempotency_key(application_id: str, rule_id: str) -> str:
    """One policy exception, one key, forever. This is what stops duplicate tickets."""
    return f"{application_id}:{rule_id}"


def chosen_transport(transport: Optional[str] = None) -> str:
    if transport and transport != "auto":
        return transport
    return os.environ.get("CREDITLENS_MCP_TRANSPORT", "inprocess")


# ---------------------------------------------------------------------------
# The MCP protocol path
# ---------------------------------------------------------------------------


def _server_command(jira_path: Optional[str]) -> tuple[list[str], dict]:
    env = dict(os.environ)
    if jira_path:
        env["CREDITLENS_JIRA_PATH"] = str(jira_path)
    return [sys.executable, "-m", "creditlens.mcp_jira"], env


def call_tool_over_stdio(tool_name: str, arguments: dict,
                         jira_path: Optional[str] = None) -> dict:
    """Spawn the server, call one tool, shut the server down. Raises if MCP is unavailable."""
    import asyncio

    from mcp import StdioServerParameters
    from mcp.client import Client

    command, env = _server_command(jira_path)
    params = StdioServerParameters(command=command[0], args=command[1:], env=env)

    async def go() -> dict:
        async with Client(params, read_timeout_seconds=STDIO_TIMEOUT_S) as client:
            result = await client.call_tool(tool_name, arguments)
            return _unwrap(result)

    return asyncio.run(asyncio.wait_for(go(), timeout=STDIO_TIMEOUT_S))


def _unwrap(result: Any) -> dict:
    """An MCP result carries structured output when the tool returns a dict."""
    import json

    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured.get("result", structured)
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"text": text}
    return {}


# ---------------------------------------------------------------------------
# The public API
# ---------------------------------------------------------------------------


@observe(name="mcp_jira.file_ticket")
def file_ticket(application_id: str, rule_id: str, summary: str = "", description: str = "",
                project_key: str = DEFAULT_PROJECT, jira_path: Optional[str] = None,
                transport: Optional[str] = None) -> dict:
    """File one policy-exception ticket. Calling it twice for the same rule changes nothing.

    Returns the issue dict, with `created=False` when the ticket already existed.
    """
    key = idempotency_key(application_id, rule_id)
    summary = summary or f"{application_id} policy exception on rule {rule_id}"
    arguments = {"project_key": project_key, "summary": summary,
                 "description": description, "idempotency_key": key}

    if chosen_transport(transport) == "stdio":
        try:
            return call_tool_over_stdio("create_issue", arguments, jira_path)
        except Exception as e:  # noqa: BLE001  the agent must not fail because a pipe did
            fallback = get_backend(jira_path).create_issue(**arguments)
            fallback["transport_note"] = f"stdio failed ({type(e).__name__}), used in-process backend"
            return fallback
    return get_backend(jira_path).create_issue(**arguments)


@observe(name="mcp_jira.get_ticket")
def get_ticket(key: str, jira_path: Optional[str] = None,
               transport: Optional[str] = None) -> dict:
    if chosen_transport(transport) == "stdio":
        try:
            return call_tool_over_stdio("get_issue", {"key": key}, jira_path)
        except Exception:  # noqa: BLE001
            pass
    return get_backend(jira_path).get_issue(key)


@observe(name="mcp_jira.search_tickets")
def search_tickets(jql_or_text: str, jira_path: Optional[str] = None,
                   transport: Optional[str] = None) -> list[dict]:
    if chosen_transport(transport) == "stdio":
        try:
            result = call_tool_over_stdio("search_issues", {"jql_or_text": jql_or_text}, jira_path)
            return result.get("issues", [])
        except Exception:  # noqa: BLE001
            pass
    return get_backend(jira_path).search_issues(jql_or_text)

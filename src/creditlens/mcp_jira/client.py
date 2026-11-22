"""Talking to the Jira MCP server from the agent.

`file_ticket()` is the only function the graph needs. It has two ways to reach the server:

* **stdio** (`transport="stdio"`): spawn `python -m creditlens.mcp_jira` as a child process
  and speak MCP over the pipe. This is the real thing, and what you would point Claude
  Desktop or any other MCP client at.
* **in-process** (`transport="inprocess"`, the default): call the same backend class
  directly. No subprocess, no event loop, no 1.5 seconds of Python start-up per ticket.

Tests use the in-process path on purpose: a unit test should not need to spawn processes to
prove that idempotency works, and on Windows each spawn costs more than the test does.
Set `CREDITLENS_MCP_TRANSPORT=stdio` to exercise the protocol for real. `inspect_server()`
does the handshake and lists the tools, which is the quickest way to see MCP working.
Every ticket `file_ticket()` returns says which transport carried it (`transport`).
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


def _stdio_transport(params: Any) -> Any:
    """The stdio transport, with somewhere real for the server's error output to go.

    The server's stderr is handed to the child process, so it needs a real file handle. Inside
    Jupyter, `sys.stderr` is the notebook's output stream and has none: use the kernel's own
    stderr then, or discard the server's log lines if that has none either.
    """
    from mcp.client.stdio import stdio_client

    errlog = None
    for stream in (sys.stderr, sys.__stderr__):
        try:
            if stream is not None:
                stream.fileno()
                errlog = stream
                break
        except (AttributeError, OSError, ValueError):
            continue
    if errlog is None:
        errlog = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115  lives as long as the process
    return stdio_client(params, errlog=errlog)


def _run_async(make_coro) -> Any:
    """Run one MCP conversation to completion, from plain Python or from inside Jupyter.

    A notebook already has an event loop running, and `asyncio.run` refuses to start a second
    one in the same thread. So when a loop is running, the conversation gets a thread of its own.
    """
    import asyncio
    import concurrent.futures

    def run() -> Any:
        return asyncio.run(asyncio.wait_for(make_coro(), timeout=STDIO_TIMEOUT_S))

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return run()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(run).result()


def call_tool_over_stdio(tool_name: str, arguments: dict,
                         jira_path: Optional[str] = None) -> dict:
    """Spawn the server, call one tool, shut the server down. Raises if MCP is unavailable."""

    from mcp import StdioServerParameters
    from mcp.client import Client

    command, env = _server_command(jira_path)
    params = StdioServerParameters(command=command[0], args=command[1:], env=env)

    async def go() -> dict:
        async with Client(_stdio_transport(params), read_timeout_seconds=STDIO_TIMEOUT_S) as client:
            result = await client.call_tool(tool_name, arguments)
            return _unwrap(result)

    return _run_async(go)


def inspect_server(jira_path: Optional[str] = None) -> dict:
    """Spawn the server, do the MCP handshake, and report what it says about itself.

    This is what any MCP client sees first: the server's name, the protocol version both sides
    agreed, its instructions, and the typed tools it offers. Nothing is created or changed.
    """
    import time

    from mcp import StdioServerParameters
    from mcp.client import Client

    command, env = _server_command(jira_path)
    params = StdioServerParameters(command=command[0], args=command[1:], env=env)

    async def go() -> dict:
        t0 = time.perf_counter()
        async with Client(_stdio_transport(params), read_timeout_seconds=STDIO_TIMEOUT_S) as client:
            listed = await client.list_tools()
            info = client.server_info
            tools = []
            for t in listed.tools:
                schema = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {}
                required = set(schema.get("required", []))
                tools.append({"name": t.name, "description": t.description or "",
                              "arguments": [a + ("" if a in required else " (optional)") for a in schema.get("properties", {})]})
            return {"transport": "stdio", "command": " ".join(["python"] + command[1:]),
                    "server": getattr(info, "name", "") if info else "", "version": getattr(info, "version", "") if info else "",
                    "protocol_version": client.protocol_version, "instructions": client.instructions or "",
                    "tools": tools, "seconds": round(time.perf_counter() - t0, 2)}

    return _run_async(go)


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
            return dict(call_tool_over_stdio("create_issue", arguments, jira_path), transport="stdio")
        except Exception as e:  # noqa: BLE001  the agent must not fail because a pipe did
            fallback = get_backend(jira_path).create_issue(**arguments)
            fallback["transport_note"] = f"stdio failed ({type(e).__name__}), used in-process backend"
            return dict(fallback, transport="inprocess")
    return dict(get_backend(jira_path).create_issue(**arguments), transport="inprocess")


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

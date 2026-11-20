"""The MCP server itself: three tools on stdio.

Run it by hand to see what an MCP server is:

    python -m creditlens.mcp_jira

Nothing happens visibly, because an MCP server talks JSON-RPC over standard input and
output. An agent (or Claude Desktop, or any MCP client) spawns it as a child process and
calls its tools over that pipe. That is the whole protocol: a process, a pipe, and a list
of typed tools.

The SDK renamed its server class in version 2 (`FastMCP` became `MCPServer`), so both names
are tried here and the rest of the file is identical either way.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from creditlens.mcp_jira.backend import DEFAULT_PROJECT, get_backend

INSTRUCTIONS = (
    "File and read Good Bank policy-exception tickets in Jira. Every create_issue call must "
    "carry an idempotency_key of the form '<application_id>:<rule_id>', and calling it twice "
    "with the same key returns the first ticket instead of making a second one."
)


def _server_class():
    try:  # MCP Python SDK v2
        from mcp.server.mcpserver import MCPServer

        return MCPServer
    except ImportError:  # pragma: no cover - SDK v1
        from mcp.server.fastmcp import FastMCP

        return FastMCP


def build_server(jira_path: Optional[str] = None) -> Any:
    """Create the MCP server with its three tools bound to the chosen backend."""
    server_class = _server_class()
    server = server_class(name="creditlens-jira", instructions=INSTRUCTIONS, version="1.0.0")
    backend = get_backend(jira_path)

    @server.tool(name="create_issue",
                 description="Create a Jira issue, or return the existing one for this "
                             "idempotency key. Use '<application_id>:<rule_id>' as the key.")
    def create_issue(summary: str, description: str = "", idempotency_key: str = "",
                     project_key: str = DEFAULT_PROJECT) -> dict:
        return backend.create_issue(project_key=project_key, summary=summary,
                                    description=description, idempotency_key=idempotency_key)

    @server.tool(name="get_issue", description="Read one Jira issue by its key, for example CREDIT-1.")
    def get_issue(key: str) -> dict:
        return backend.get_issue(key)

    @server.tool(name="search_issues",
                 description="Search issues with JQL, or with plain text when JQL is not needed.")
    def search_issues(jql_or_text: str) -> dict:
        return {"issues": backend.search_issues(jql_or_text)}

    return server


def main() -> None:  # pragma: no cover - exercised by hand and by the stdio client
    path = os.environ.get("CREDITLENS_JIRA_PATH")
    build_server(path).run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    main()

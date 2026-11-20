"""An MCP server that files Good Bank policy exceptions as Jira tickets.

MCP, in plain words: a small standard that lets a tool live in its own process and still be
callable by any agent, in any language, over a pipe. The agent speaks MCP; the server knows
Jira. Swap the server and the agent does not change.

    python -m creditlens.mcp_jira          # run the server on stdio
    from creditlens.mcp_jira.client import file_ticket

The backend is real Jira Cloud when JIRA_BASE_URL, JIRA_EMAIL and JIRA_API_TOKEN are set,
and a JSON-file mock otherwise, so the workshop never needs an account.
"""
from __future__ import annotations

__all__ = ["file_ticket", "get_backend"]


def __getattr__(name: str):
    if name == "file_ticket":
        from creditlens.mcp_jira.client import file_ticket

        return file_ticket
    if name == "get_backend":
        from creditlens.mcp_jira.backend import get_backend

        return get_backend
    raise AttributeError(name)

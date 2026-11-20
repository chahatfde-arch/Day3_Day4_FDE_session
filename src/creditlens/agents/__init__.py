"""Milestone 6: the multi-agent credit memo.

A supervisor dispatches specialists (financial analyst, banking conduct, related party,
policy checker, risk scorer, memo writer), the deterministic policy checks are plain
Python, and a human underwriter approves at an `interrupt()` before the memo is final.

Start here:
    from creditlens.agents.run import draft_memo, resume
"""
from __future__ import annotations

__all__ = ["draft_memo", "resume", "build_graph", "run_checks"]


def __getattr__(name: str):
    # Lazy re-exports so `import creditlens.agents` stays cheap and never needs langgraph.
    if name in ("draft_memo", "resume"):
        from creditlens.agents import run

        return getattr(run, name)
    if name == "build_graph":
        from creditlens.agents.graph import build_graph

        return build_graph
    if name == "run_checks":
        from creditlens.agents.policy_checks import run_checks

        return run_checks
    raise AttributeError(name)

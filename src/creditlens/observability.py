"""Tracing with Langfuse, with a no-op fallback when no keys are configured.

Use it like this anywhere in the code base:

    from creditlens.observability import observe, langchain_handler

    @observe(name="extract_financials")
    def extract_financials(path): ...

    graph.invoke(state, config={"callbacks": [langchain_handler()]})

When LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are missing, every helper here
does nothing, so the same code runs in CI and in Colab without a Langfuse account.
Set LANGFUSE_HOST to http://localhost:3000 for the Docker Compose instance, or to
https://cloud.langfuse.com for the free cloud project.
"""
from __future__ import annotations

import functools
from typing import Any, Callable, Optional

from creditlens.config import settings

try:  # Langfuse Python SDK v3
    if settings.langfuse_enabled:
        from langfuse import get_client as _get_client, observe as _observe
    else:
        raise ImportError("langfuse keys not set")
    ENABLED = True
except Exception:  # noqa: BLE001
    ENABLED = False
    _get_client = None
    _observe = None


def observe(name: Optional[str] = None, **kwargs) -> Callable:
    """Decorator that creates a Langfuse span around the function, or does nothing."""
    if ENABLED and _observe is not None:
        return _observe(name=name, **kwargs)

    def passthrough(fn):
        @functools.wraps(fn)
        def wrapper(*a, **k):
            return fn(*a, **k)
        return wrapper
    return passthrough


def get_client():
    return _get_client() if ENABLED and _get_client is not None else None


def record_generation(model: str, input: Any, output: Any, usage: Optional[dict] = None,
                      cached: bool = False) -> None:
    """Attach LLM details to the current span. Safe to call when tracing is off."""
    if not ENABLED:
        return
    try:
        client = get_client()
        if client is None:
            return
        client.update_current_generation(model=model, input=input, output=output,
                                         usage_details=usage or {},
                                         metadata={"cached": cached})
    except Exception:  # noqa: BLE001  tracing must never break the product
        pass


def update_trace(**kwargs) -> None:
    """Set user_id, session_id, tags or metadata on the current trace."""
    if not ENABLED:
        return
    try:
        client = get_client()
        if client is not None:
            client.update_current_trace(**kwargs)
    except Exception:  # noqa: BLE001
        pass


def current_trace_id() -> Optional[str]:
    if not ENABLED:
        return None
    try:
        client = get_client()
        return client.get_current_trace_id() if client is not None else None
    except Exception:  # noqa: BLE001
        return None


def langchain_handler():
    """Callback handler for LangChain and LangGraph runs. Returns None when tracing is off."""
    if not ENABLED:
        return None
    try:
        from langfuse.langchain import CallbackHandler
        return CallbackHandler()
    except Exception:  # noqa: BLE001
        return None


def callbacks() -> list:
    h = langchain_handler()
    return [h] if h else []


def flush() -> None:
    """Send buffered events now. Call before a short script exits."""
    if ENABLED:
        try:
            client = get_client()
            if client is not None:
                client.flush()
        except Exception:  # noqa: BLE001
            pass

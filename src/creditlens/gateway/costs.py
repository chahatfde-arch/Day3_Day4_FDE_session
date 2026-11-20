"""Token cost estimates, with an intentionally empty price table.

The one idea this module teaches: **never hard-code a price**. Provider prices change,
free tiers appear and disappear, and a number baked into a repository in March is wrong by
June and nobody notices. So `PRICES` starts empty and is filled from exactly one of:

    LLM_PRICES_JSON = {"gemini/gemini-3.5-flash-lite": {"input": 0.0, "output": 0.0}}
    LLM_PRICES_JSON = /path/to/prices.json

Prices are US dollars per one million tokens. When a model has no price, `estimate()`
returns None, and every caller must be able to print "unknown" rather than zero. Zero is a
lie that hides a cost spike; None is the truth that prompts somebody to look it up.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PRICES: dict[str, dict[str, float]] = {}  # deliberately empty. See the module docstring.

PER_TOKENS = 1_000_000  # prices are quoted per million tokens


@dataclass(frozen=True)
class CostEstimate:
    model: str
    prompt_tokens: int
    completion_tokens: int
    input_cost_usd: float
    output_cost_usd: float

    @property
    def total_usd(self) -> float:
        return round(self.input_cost_usd + self.output_cost_usd, 6)

    def __str__(self) -> str:
        return (f"{self.model}: {self.prompt_tokens} in + {self.completion_tokens} out "
                f"= {self.total_usd:.6f} USD")


def load_prices(source: Optional[str] = None) -> dict[str, dict[str, float]]:
    """Read prices from LLM_PRICES_JSON (inline JSON or a file path). Empty when unset."""
    raw = source if source is not None else os.environ.get("LLM_PRICES_JSON", "").strip()
    if not raw:
        return {}
    try:
        if raw.lstrip().startswith("{"):
            data = json.loads(raw)
        else:
            path = Path(raw)
            if not path.exists():
                return {}
            data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    clean: dict[str, dict[str, float]] = {}
    for model, entry in (data or {}).items():
        if isinstance(entry, dict) and "input" in entry and "output" in entry:
            try:
                clean[str(model)] = {"input": float(entry["input"]), "output": float(entry["output"])}
            except (TypeError, ValueError):
                continue
    return clean


def refresh() -> dict[str, dict[str, float]]:
    """Reload PRICES from the environment. Call it after changing LLM_PRICES_JSON."""
    PRICES.clear()
    PRICES.update(load_prices())
    return PRICES


def price_for(model: str) -> Optional[dict[str, float]]:
    """Exact match first, then a provider-wide entry such as "gemini/*"."""
    table = PRICES or load_prices()
    if model in table:
        return table[model]
    provider = model.split("/", 1)[0]
    return table.get(f"{provider}/*") or table.get(provider)


def estimate(prompt_tokens: int, completion_tokens: int, model: str) -> Optional[CostEstimate]:
    """Cost of one call, or None when we have no price for the model.

    None is a real answer here: the incident drill prints "cost unknown, add
    LLM_PRICES_JSON" instead of a comforting 0.00 that would hide a spike.
    """
    prices = price_for(model)
    if prices is None:
        return None
    return CostEstimate(
        model=model,
        prompt_tokens=int(prompt_tokens),
        completion_tokens=int(completion_tokens),
        input_cost_usd=round(int(prompt_tokens) * prices["input"] / PER_TOKENS, 8),
        output_cost_usd=round(int(completion_tokens) * prices["output"] / PER_TOKENS, 8),
    )


def estimate_total(calls: list[dict]) -> tuple[Optional[float], list[str]]:
    """Sum a list of {model, prompt_tokens, completion_tokens}. Returns (total, unpriced)."""
    total = 0.0
    priced_any = False
    unpriced: list[str] = []
    for call in calls:
        one = estimate(call.get("prompt_tokens", 0), call.get("completion_tokens", 0),
                       call.get("model", ""))
        if one is None:
            if call.get("model", "") not in unpriced:
                unpriced.append(call.get("model", ""))
            continue
        priced_any = True
        total += one.total_usd
    return (round(total, 6) if priced_any else None), unpriced


def explain_missing_prices() -> str:
    return ("No prices are configured, so cost is reported as unknown. Set LLM_PRICES_JSON to "
            'inline JSON such as {"gemini/gemini-3.5-flash-lite": {"input": 0.0, "output": 0.0}} '
            "or to the path of a JSON file. Prices are US dollars per million tokens and should "
            "be copied from the provider's pricing page on the day of the session.")

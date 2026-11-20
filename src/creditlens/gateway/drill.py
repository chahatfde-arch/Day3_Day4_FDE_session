"""The resilience drill: send a few prompts and watch which model answers.

This is the one-command form of the Day 7 demo "cut the primary live".

    python -m creditlens.gateway.drill                 # 5 prompts through the gateway
    python -m creditlens.gateway.drill -n 10           # more prompts
    python -m creditlens.gateway.drill --no-cache      # force real calls (the cost-spike drill)
    LLM_KILL_PRIMARY=1 python -m creditlens.gateway.drill   # pretend the primary is gone

What to watch for:
  * the `model` column changing from the primary to a fallback after the primary is cut,
  * `cached` turning True on the second identical prompt (quota saved, latency gone),
  * the cost line, which says "unknown" until LLM_PRICES_JSON is set (see costs.py).

In offline mode (LLM_OFFLINE=1) every answer comes from the deterministic stub and the model
column reads `offline-stub`. The drill still runs, so CI exercises exactly this code path.
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Optional

from creditlens.gateway import costs
from creditlens.gateway.llm import get_gateway

PROMPTS = [
    "In one sentence, what does DSCR measure?",
    "In one sentence, what is a revolving line of credit?",
    "In one sentence, why does a bank look at cheque returns?",
    "In one sentence, what does a loan tenure of 48 months mean?",
    "In one sentence, what is a related party exposure?",
]


def primary_is_cut() -> bool:
    """LLM_KILL_PRIMARY=1 simulates the primary provider being unavailable."""
    return os.environ.get("LLM_KILL_PRIMARY", "0") == "1"


def run(n: int = 5, use_cache: bool = True, verbose: bool = True) -> dict:
    """Send n prompts, report which model answered each and how many were cache hits."""
    gateway = get_gateway()
    killed = primary_is_cut()

    # Cutting the primary: we ask the gateway for the first fallback by name. The real
    # outage equivalent is the primary's key being revoked or the provider returning 429;
    # the gateway handles both the same way, by walking down `models`.
    forced_model: Optional[str] = None
    if killed:
        forced_model = gateway.fallbacks[0] if gateway.fallbacks else None
        if verbose:
            print(f"LLM_KILL_PRIMARY=1: skipping {gateway.primary}, "
                  f"routing to {forced_model or 'no fallback configured'}")

    rows = []
    hits_before = gateway.cache_hits
    for i in range(n):
        prompt = PROMPTS[i % len(PROMPTS)]
        started = time.perf_counter()
        error = ""
        try:
            kwargs = {"use_cache": use_cache}
            if forced_model:
                kwargs["model"] = forced_model
            response = gateway.complete([{"role": "user", "content": prompt}],
                                        max_tokens=120, **kwargs)
            model, cached = response.model, response.cached
            prompt_tokens, completion_tokens = response.prompt_tokens, response.completion_tokens
            answer = (response.text or "").strip().replace("\n", " ")[:70]
        except Exception as exc:  # noqa: BLE001  a drill must survive its own incident
            model, cached, prompt_tokens, completion_tokens = "FAILED", False, 0, 0
            answer, error = "", f"{type(exc).__name__}: {exc}"[:90]
        rows.append({"n": i + 1, "model": model, "cached": cached, "latency_s": round(
            time.perf_counter() - started, 3), "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens, "answer": answer, "error": error})

    cache_hits = sum(1 for r in rows if r["cached"]) or (gateway.cache_hits - hits_before)
    total_cost, unpriced = costs.estimate_total(rows)

    summary = {
        "calls": len(rows),
        "cache_hits": int(cache_hits),
        "primary_cut": killed,
        "models_used": sorted({r["model"] for r in rows}),
        "failures": sum(1 for r in rows if r["model"] == "FAILED"),
        "estimated_cost_usd": total_cost,
        "unpriced_models": unpriced,
        "rows": rows,
    }

    if verbose:
        print(f"\n{'#':>2}  {'model':28s} {'cached':6s} {'lat s':>6s}  answer")
        print("-" * 96)
        for r in rows:
            shown = r["error"] or r["answer"] or "offline-stub"
            print(f"{r['n']:>2}  {r['model']:28s} {str(r['cached']):6s} {r['latency_s']:>6.3f}  {shown}")
        print("-" * 96)
        print(f"calls {summary['calls']}  cache hits {summary['cache_hits']}  "
              f"failures {summary['failures']}  models {', '.join(summary['models_used'])}")
        if total_cost is None:
            print("estimated cost: unknown. " + costs.explain_missing_prices())
        else:
            print(f"estimated cost: {total_cost:.6f} USD"
                  + (f" (no price for {', '.join(unpriced)})" if unpriced else ""))
        print("Langfuse: open the latest trace and look at the generation's model field. "
              "Without Langfuse keys the same information is the model column above.")
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Gateway resilience and cost drill")
    parser.add_argument("-n", "--calls", type=int, default=5)
    parser.add_argument("--no-cache", action="store_true", help="bypass the disk cache")
    parser.add_argument("--kill-primary", action="store_true",
                        help="same as LLM_KILL_PRIMARY=1: route to the first fallback")
    args = parser.parse_args(argv)
    if args.kill_primary:
        os.environ["LLM_KILL_PRIMARY"] = "1"
    run(n=args.calls, use_cache=not args.no_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

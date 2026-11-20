"""Trajectory checks: did the agent take a sensible path, not just reach a sensible answer?

The one idea this module teaches: **the final answer is not the whole evaluation**.
Two memos can read identically while one was written after reading the policy and the
other was written first and justified afterwards. Only the trajectory tells them apart.

Give it the tool calls from a LangGraph run as a plain list of dicts:

    [{"name": "get_application", "args": {"application_id": "A001"}, "order": 1},
     {"name": "search_policy",   "args": {"query": "DSCR"},          "order": 2}, ...]

and a set of expectations. It returns a report dict and never raises on bad input, because
an eval that crashes on a weird trace teaches nothing.

    python -m evals.trajectory run.json
    python -m evals.trajectory run.json --expectations expectations.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional

# What a healthy CreditLens memo run looks like. Milestone 6 owns the tool names; these are
# the ones named in docs/CONTRACTS.md, and an unknown tool name simply shows up as missing.
DEFAULT_EXPECTATIONS: dict[str, Any] = {
    "required_tools": ["get_application", "get_statement_features", "search_policy", "score_risk"],
    "forbidden_tools": ["run_sql_unrestricted", "file_jira_ticket_without_rule", "delete_application"],
    "max_calls_per_tool": {"search_policy": 12, "score_risk": 3, "get_application": 5,
                           "file_jira_ticket": 6},
    # "a must happen before b": policy must be read before the memo is written.
    "ordering": [["search_policy", "memo_writer"], ["score_risk", "memo_writer"]],
    "max_total_calls": 40,
    "allow_repeated_identical_calls": False,
}


def _normalise(records: Any) -> list[dict]:
    """Accept a list of dicts, a dict with a "tool_calls" key, or messy input, and survive."""
    if isinstance(records, dict):
        records = records.get("tool_calls") or records.get("trajectory") or []
    if not isinstance(records, list):
        return []
    out = []
    for i, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        out.append({
            "name": str(record.get("name", record.get("tool", "unknown"))),
            "args": record.get("args", record.get("arguments", {})),
            "order": int(record.get("order", i + 1)),
        })
    return sorted(out, key=lambda r: r["order"])


def _args_key(args: Any) -> str:
    try:
        return json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(args)


def _first_index(calls: list[dict], name: str) -> Optional[int]:
    for i, call in enumerate(calls):
        if call["name"] == name:
            return i
    return None


def check_trajectory(records: Any, expectations: Optional[dict] = None) -> dict:
    """Run every expectation over one trajectory and return a report dict."""
    expect = {**DEFAULT_EXPECTATIONS, **(expectations or {})}
    calls = _normalise(records)
    counts = Counter(call["name"] for call in calls)
    failures: list[dict] = []

    # 1. Required tools: at least one call each.
    missing = [t for t in expect.get("required_tools", []) if counts[t] == 0]
    if missing:
        failures.append({"check": "required_tools", "detail": f"never called: {', '.join(missing)}"})

    # 2. Forbidden tools: never called.
    used_forbidden = [t for t in expect.get("forbidden_tools", []) if counts[t] > 0]
    if used_forbidden:
        failures.append({"check": "forbidden_tools",
                         "detail": f"called a forbidden tool: {', '.join(used_forbidden)}"})

    # 3. Call budget per tool. A run-away loop shows up here first.
    over_budget = []
    for tool, limit in (expect.get("max_calls_per_tool") or {}).items():
        if counts[tool] > limit:
            over_budget.append(f"{tool} called {counts[tool]} times, limit {limit}")
    if over_budget:
        failures.append({"check": "max_calls_per_tool", "detail": "; ".join(over_budget)})

    # 4. Ordering. "Policy before memo" is the one that matters for a credit memo: the
    #    prose must follow the rules, not the other way round.
    order_problems = []
    for pair in expect.get("ordering", []):
        before, after = pair[0], pair[1]
        i_before, i_after = _first_index(calls, before), _first_index(calls, after)
        if i_after is None:
            continue  # the later tool was never called: that is the required_tools check's job
        if i_before is None:
            order_problems.append(f"{after} ran but {before} never did")
        elif i_before > i_after:
            order_problems.append(f"{before} ran after {after}")
    if order_problems:
        failures.append({"check": "ordering", "detail": "; ".join(order_problems)})

    # 5. Identical repeated calls: same tool, same arguments. Always a loop or a lost answer.
    seen: set[tuple[str, str]] = set()
    repeated: list[str] = []
    for call in calls:
        key = (call["name"], _args_key(call["args"]))
        if key in seen and key[0] not in (expect.get("repeat_allowed_tools") or []):
            repeated.append(call["name"])
        seen.add(key)
    if repeated and not expect.get("allow_repeated_identical_calls", False):
        failures.append({"check": "no_repeated_calls",
                         "detail": "identical call repeated: " + ", ".join(sorted(set(repeated)))})

    # 6. Total budget: the cheapest loop detector there is.
    max_total = expect.get("max_total_calls")
    if max_total is not None and len(calls) > max_total:
        failures.append({"check": "max_total_calls",
                         "detail": f"{len(calls)} calls, limit {max_total}"})

    return {
        "passed": not failures,
        "n_calls": len(calls),
        "tools_used": dict(sorted(counts.items())),
        "sequence": [call["name"] for call in calls],
        "failures": failures,
        "repeated_calls": sorted(set(repeated)),
        "forbidden_tools_called": used_forbidden,
        "missing_required_tools": missing,
    }


# Kept as an alias because "run" is the name the gate and the other evals use.
def run(records: Any, expectations: Optional[dict] = None) -> dict:
    return check_trajectory(records, expectations)


def print_report(report: dict) -> None:
    status = "PASS" if report["passed"] else "FAIL"
    print(f"Trajectory: {status}  ({report['n_calls']} tool calls)")
    print("  sequence: " + " -> ".join(report["sequence"] or ["(none)"]))
    for tool, count in report["tools_used"].items():
        print(f"    {tool:34s} {count}")
    if report["failures"]:
        print("  problems:")
        for failure in report["failures"]:
            print(f"    - {failure['check']}: {failure['detail']}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Check an agent run's tool-call trajectory")
    parser.add_argument("trace", help="JSON file: a list of {name, args, order} records")
    parser.add_argument("--expectations", default=None, help="JSON file overriding the defaults")
    args = parser.parse_args(argv)

    records = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    expectations = json.loads(Path(args.expectations).read_text(encoding="utf-8")) \
        if args.expectations else None
    report = check_trajectory(records, expectations)
    print_report(report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""The CI eval gate: run the offline evals, compare with thresholds, fail the build on a drop.

The one idea this module teaches: **a prompt change is a code change, so it needs a test**.
Unit tests catch a broken function. Nothing catches a prompt that still runs, still returns
valid JSON, and is now quietly worse, unless you measure it on every commit. That is this file.

    python -m evals.gate                  # what CI runs
    python -m evals.gate --regress-prompt # the Day 6 curveball: a broken judge, gate must fail

It is deliberately forgiving about missing evals and unforgiving about failing ones. An eval
module that has not been written yet prints SKIPPED; a metric below its threshold exits 1.
Everything runs with LLM_OFFLINE=1 so CI needs no API keys.
"""
from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

from creditlens.config import settings

THRESHOLDS_PATH = Path(__file__).with_name("thresholds.json")
RESULTS_PATH = Path(__file__).parent / "results" / "gate.json"

# Where to look inside each eval's result dict. The first key that resolves to a number wins,
# so a sibling module can change its internal shape without breaking the gate.
METRIC_KEYS: dict[str, list[str]] = {
    "extraction_overall": ["overall", "overall_accuracy", "accuracy", "exact_match",
                           "overall_field_accuracy", "mean_accuracy"],
    "retrieval_hybrid_recall_at_5": ["summary.hybrid.recall@5", "summary.hybrid.recall_at_5",
                                     "hybrid.recall@5", "hybrid.recall_at_5",
                                     "hybrid_recall_at_5", "recall_at_5", "recall@5"],
    "judge_agreement": ["agreement"],
}


def load_thresholds(path: Optional[Path] = None) -> dict[str, float]:
    data = json.loads(Path(path or THRESHOLDS_PATH).read_text(encoding="utf-8"))
    return {k: float(v) for k, v in data.items() if not k.startswith("_")}


def _dig(payload: Any, key: str) -> Optional[float]:
    """Find a number under `key`, including dotted paths and one level of nesting."""
    if payload is None:
        return None
    if "." in key:
        current: Any = payload
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return _as_float(current)
    if isinstance(payload, dict):
        if key in payload:
            value = _as_float(payload[key])
            if value is not None:
                return value
        for nested in payload.values():  # one level down: {"hybrid": {"recall_at_5": 0.8}}
            if isinstance(nested, dict) and key in nested:
                value = _as_float(nested[key])
                if value is not None:
                    return value
    return None


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def extract_metric(name: str, payload: Any) -> Optional[float]:
    for key in METRIC_KEYS.get(name, [name]):
        value = _dig(payload, key)
        if value is not None:
            return value
    return None


def _skipped_by_env(module_name: str) -> bool:
    """CREDITLENS_GATE_SKIP=evals.retrieval_eval,evals.extraction_accuracy.

    Useful for a fast pull-request job (or a test) that only wants the judge, and for a
    machine where an optional eval needs a model download it cannot do.
    """
    skip = os.environ.get("CREDITLENS_GATE_SKIP", "")
    wanted = {part.strip() for part in skip.split(",") if part.strip()}
    return module_name in wanted or module_name.split(".")[-1] in wanted


def _run_optional(module_name: str, call: Callable[[Any], Any]) -> tuple[str, Any, str]:
    """Import a sibling eval lazily. Returns (status, payload, note)."""
    if _skipped_by_env(module_name):
        return "SKIPPED", None, f"{module_name} skipped by CREDITLENS_GATE_SKIP"
    try:
        import importlib

        module = importlib.import_module(module_name)
    except ImportError:
        return "SKIPPED", None, f"{module_name} is not in the repository yet"
    runner = getattr(module, "run", None)
    if runner is None:
        return "SKIPPED", None, f"{module_name} has no run() function"
    try:
        return "OK", call(runner), ""
    except Exception as exc:  # noqa: BLE001  a broken sibling must not hide our own result
        return "ERROR", None, f"{type(exc).__name__}: {exc}"[:160]


def collect(limit: int = 10, regressed: bool = False, verbose: bool = True) -> list[dict]:
    """Run every eval we can reach and return one row per metric."""
    rows: list[dict] = []

    status, payload, note = _run_optional("evals.extraction_accuracy", lambda run: run(limit=limit))
    rows.append({"metric": "extraction_overall", "status": status,
                 "value": extract_metric("extraction_overall", payload), "note": note})

    status, payload, note = _run_optional("evals.retrieval_eval", lambda run: run(k=5))
    rows.append({"metric": "retrieval_hybrid_recall_at_5", "status": status,
                 "value": extract_metric("retrieval_hybrid_recall_at_5", payload), "note": note})

    # The judge is ours, so it is never optional.
    try:
        from evals import judge as judge_module

        judge_module.install_offline_handler(regressed=regressed)
        result = judge_module.calibrate(regressed=regressed, verbose=False)
        rows.append({"metric": "judge_agreement", "status": "OK",
                     "value": float(result["agreement"]),
                     "note": f"kappa {result['kappa']}, {len(result['disagreements'])} disagreements"})
    except Exception as exc:  # noqa: BLE001
        rows.append({"metric": "judge_agreement", "status": "ERROR", "value": None,
                     "note": f"{type(exc).__name__}: {exc}"[:160]})
        if verbose:
            traceback.print_exc()
    return rows


def evaluate(rows: list[dict], thresholds: dict[str, float]) -> list[dict]:
    """Attach the threshold and a verdict to every row."""
    out = []
    for row in rows:
        threshold = thresholds.get(row["metric"])
        value, status = row["value"], row["status"]
        if status == "SKIPPED" or (status == "OK" and value is None):
            verdict = "SKIPPED"
            if status == "OK" and value is None:
                row["note"] = (row["note"] or "") + " (ran, but no matching metric in the result)"
        elif status == "ERROR":
            verdict = "ERROR"
        elif threshold is None:
            verdict = "NO THRESHOLD"
        else:
            verdict = "PASS" if value >= threshold else "FAIL"
        out.append({**row, "threshold": threshold, "verdict": verdict})
    return out


def print_table(rows: list[dict]) -> None:
    print(f"{'metric':32s} {'value':>8s} {'min':>8s}  {'verdict':12s} note")
    print("-" * 104)
    for row in rows:
        value = "-" if row["value"] is None else f"{row['value']:.4f}"
        threshold = "-" if row["threshold"] is None else f"{row['threshold']:.2f}"
        print(f"{row['metric']:32s} {value:>8s} {threshold:>8s}  {row['verdict']:12s} {row['note']}")
    print("-" * 104)


def run_gate(limit: int = 10, regressed: bool = False, thresholds_path: Optional[Path] = None,
             verbose: bool = True) -> dict:
    """Everything the CI job does, as one function so tests can call it."""
    os.environ.setdefault("LLM_OFFLINE", "1")
    thresholds = load_thresholds(thresholds_path)
    rows = evaluate(collect(limit=limit, regressed=regressed, verbose=verbose), thresholds)

    failures = [r for r in rows if r["verdict"] in ("FAIL", "ERROR")]
    skipped = [r for r in rows if r["verdict"] == "SKIPPED"]
    result = {
        "passed": not failures,
        "regressed_prompt": regressed,
        "rows": rows,
        "failed": [r["metric"] for r in failures],
        "skipped": [r["metric"] for r in skipped],
        "thresholds": thresholds,
        "bank": settings.bank_name,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")

    if verbose:
        print(f"Eval gate{' (regressed prompt)' if regressed else ''}")
        print_table(rows)
        if skipped:
            print(f"SKIPPED: {', '.join(result['skipped'])}. "
                  "These evals belong to other milestones and are not in the repository yet.")
        if failures:
            print(f"GATE FAILED: {', '.join(result['failed'])}")
            print("What to do: look at evals/results/gate.json, then at the change that moved the "
                  "metric. Do not lower the threshold to make this green.")
        else:
            print("GATE PASSED")
        print(f"Written to {RESULTS_PATH}")
    return result


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="CreditLens eval gate")
    parser.add_argument("--limit", type=int, default=10, help="documents for the extraction eval")
    parser.add_argument("--regress-prompt", action="store_true",
                        help="install a deliberately broken judge, so the gate must fail")
    parser.add_argument("--thresholds", default=None)
    args = parser.parse_args(argv)

    result = run_gate(limit=args.limit, regressed=args.regress_prompt,
                      thresholds_path=Path(args.thresholds) if args.thresholds else None)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluations: the scripts that turn "it looks right" into a number.

Each eval loads a golden set from `ground_truth/`, runs the real component over it,
and writes a JSON result under `evals/results/` so CI can compare today's number with
the thresholds in `evals/thresholds.json`.

Run them from the repository root:

    LLM_OFFLINE=1 python -m evals.extraction_accuracy --limit 5
"""

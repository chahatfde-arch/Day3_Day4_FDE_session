"""Get your laptop ready for a workshop session.

    python workshop/get_ready.py 0    # pre-work
    python workshop/get_ready.py 1    # the day before Session 1
    python workshop/get_ready.py 2    # the day before Session 2 (and so on up to 4)

Run it from the repository root. It regenerates the Good Bank data, loads the database, fetches the
real-data samples, runs the setup check, then runs that session's tests offline (no API keys needed).
Each step prints OK or FAILED; the script carries on after a failure and lists every failed step at
the end, so you can bring one clear question to the trainer instead of a wall of errors.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

COMMON = [
    ("Generate the Good Bank data", ["-m", "creditlens.generators"]),
    ("Run the pipeline and load the database", ["-m", "creditlens.pipeline", "--load"]),
    ("Fetch the real-data samples (needs internet)", ["-m", "creditlens.data_real"]),
    ("Setup check", ["-m", "creditlens.setup_check"]),
]

TESTS = {
    0: ["tests/test_generators.py"],
    1: ["tests/test_wrangle.py", "tests/test_pipeline.py", "tests/test_ingest.py", "tests/test_api.py",
        "tests/test_extraction.py"],
    2: ["tests/test_retrieval.py", "tests/test_graph.py", "tests/test_evals.py"],
    3: ["tests/test_agents.py", "tests/test_mcp_jira.py", "tests/test_integrations.py"],
    4: ["tests/test_ml.py", "tests/test_guardrails.py", "tests/test_monitoring.py", "tests/test_evals.py"],
}

# Shared by the trainer before the session; nice to have locally, never required.
CACHES = {
    2: ["data/cache/retrieval", "data/cache/colpali"],
    4: ["data/cache/model.joblib"],
}

FOLDERS = {0: "session-0-prework", 1: "session-1-data-and-extraction", 2: "session-2-retrieval",
           3: "session-3-agents-and-integrations", 4: "session-4-deploy-evaluate-maintain"}


def run(label: str, args: list[str]) -> bool:
    print(f"\n==> {label}", flush=True)
    env = {**os.environ, "LLM_OFFLINE": "1"}
    ok = subprocess.run([sys.executable, *args], cwd=REPO, env=env).returncode == 0
    print(f"{'OK    ' if ok else 'FAILED'} {label}", flush=True)
    return ok


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in {str(n) for n in TESTS}:
        print(__doc__)
        return 2
    n = int(sys.argv[1])
    failed = [label for label, args in COMMON if not run(label, args)]
    label = f"Session {n} tests, offline" if n else "Pre-work tests, offline"
    if not run(label, ["-m", "pytest", "-q", "-m", "not slow", *TESTS[n]]):
        failed.append(label)

    for path in CACHES.get(n, []):
        found = (REPO / path).exists()
        print(f"{'OK    ' if found else 'INFO  '} {path} {'found' if found else 'not found yet: use the cache the trainer shares'}")

    print(f"\nTopics for this session: workshop/{FOLDERS[n]}/README.md")
    if failed:
        print("\nNot ready yet. These steps failed:")
        for f in failed:
            print(f"  - {f}")
        print("Bring the first error message to the trainer, or ask your pod's Skeptic.")
        return 1
    print("\nReady. Everything for this session works on your laptop.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

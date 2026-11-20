"""Pre-work setup check. Participants run this 48 hours before Day 1.

    python -m creditlens.setup_check

It prints one line per requirement with OK, WARN or MISSING, and never fails hard:
a participant with three WARN lines can still follow Day 1. Only the lines marked
"needed on Day 1" block the first session.
"""
from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path

from creditlens.config import REPO_ROOT, settings

OK, WARN, MISSING = "OK     ", "WARN   ", "MISSING"


def _mod(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:  # noqa: BLE001
        return False


def check_all() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []

    v = sys.version_info
    rows.append((OK if v >= (3, 10) else MISSING, "Python 3.10 or newer (needed on Day 1)", f"{v.major}.{v.minor}"))

    for mod, day in [("pandas", "Day 1"), ("pandera", "Day 1"), ("pypdfium2", "Day 1"), ("fastapi", "Day 2"),
                     ("pydantic", "Day 2"), ("litellm", "Day 2"), ("rapidocr_onnxruntime", "Day 2"),
                     ("sentence_transformers", "Day 3"), ("qdrant_client", "Day 3"), ("rank_bm25", "Day 3"),
                     ("networkx", "Day 3"), ("langgraph", "Day 4"), ("mcp", "Day 4"), ("zeep", "Day 5"),
                     ("sklearn", "Day 6"), ("mlflow", "Day 6"), ("presidio_analyzer", "Day 6"),
                     ("evidently", "Day 7")]:
        rows.append((OK if _mod(mod) else WARN, f"python package {mod} ({day})",
                     "" if _mod(mod) else "pip install -e \".[all]\""))

    rows.append((OK if shutil.which("git") else MISSING, "git on PATH (needed on Day 1)", ""))
    rows.append((OK if shutil.which("docker") else WARN, "docker on PATH (Day 2 onward, laptop track only)",
                 "Colab track does not need Docker"))
    rows.append((OK if shutil.which("tesseract") else WARN, "tesseract OCR binary (optional)",
                 "rapidocr is the pip-only fallback"))

    env_file = REPO_ROOT / ".env"
    rows.append((OK if env_file.exists() else WARN, ".env file exists", "copy .env.example to .env"))
    for key, day in [("GEMINI_API_KEY", "Day 2"), ("GROQ_API_KEY", "backup provider"),
                     ("NEO4J_URI", "Day 3, optional"), ("LANGFUSE_PUBLIC_KEY", "Day 2, optional")]:
        rows.append((OK if os.environ.get(key) else WARN, f"{key} set ({day})", ""))

    data_ok = (settings.paths.raw / "applications" / "applications.csv").exists()
    rows.append((OK if data_ok else WARN, "synthetic data generated", "python -m creditlens.generators"))
    real_ok = (settings.paths.real / "sba_loans.csv").exists()
    rows.append((OK if real_ok else WARN, "real-data samples fetched", "python -m creditlens.data_real"))
    db_ok = Path(settings.database_url.replace("sqlite:///", "")).exists() if settings.database_url.startswith("sqlite") else True
    rows.append((OK if db_ok else WARN, "database loaded", "python -m creditlens.pipeline --load"))
    return rows


def main() -> None:
    rows = check_all()
    width = max(len(r[1]) for r in rows)
    for status, what, hint in rows:
        print(f"{status}  {what.ljust(width)}  {hint}")
    missing = [r for r in rows if r[0] == MISSING]
    print()
    print("Ready for Day 1." if not missing else f"{len(missing)} blocking item(s). Fix the MISSING lines first.")


if __name__ == "__main__":
    main()

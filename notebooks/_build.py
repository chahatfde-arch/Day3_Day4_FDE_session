"""Build the Colab notebooks from the percent-format scripts in notebooks/src/.

    python notebooks/_build.py

Why scripts? A .ipynb file is JSON that is painful to review in a pull request. The
scripts in notebooks/src/ are plain Python with `# %%` cell markers (the same format
VS Code and Jupytext understand), so a trainer can edit the teaching text with any editor
and rebuild. Markdown cells are `# %% [markdown]` blocks whose lines start with `# `.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"


def parse_percent(text: str) -> list[dict]:
    cells: list[dict] = []
    kind, buf = None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if kind == "markdown":
            body = "\n".join(re.sub(r"^# ?", "", line) for line in body.splitlines())
        if body.strip():
            cells.append({"cell_type": kind, "metadata": {}, "source": body.splitlines(keepends=True),
                          **({"outputs": [], "execution_count": None} if kind == "code" else {})})

    for line in text.splitlines():
        m = re.match(r"^# %%(.*)$", line)
        if m:
            flush()
            kind = "markdown" if "[markdown]" in m.group(1) else "code"
            buf = []
        elif kind is not None:
            buf.append(line)
    flush()
    return cells


def build(src: Path, out: Path) -> int:
    cells = parse_percent(src.read_text(encoding="utf-8"))
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": [], "toc_visible": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return len(cells)


def main() -> None:
    files = sorted(SRC.glob("*.py"))
    if not files:
        sys.exit("no scripts in notebooks/src/")
    for f in files:
        out = HERE / (f.stem + ".ipynb")
        n = build(f, out)
        print(f"{out.name}: {n} cells")


if __name__ == "__main__":
    main()

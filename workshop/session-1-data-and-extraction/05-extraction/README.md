# Structured extraction and the first evaluation

*Session 1: Discover, data foundation, ground truth, extraction*

**Topics:** L1 to L10, M1, M2, M8, M10, M11

**Notebook:** 02_ground_truth_and_extraction, sections 4 and 5 (notebooks/01_data_foundation.ipynb and notebooks/02_ground_truth_and_extraction.ipynb)

## Code

- [`src/creditlens/extraction/extract.py`](../../../src/creditlens/extraction/extract.py)
- [`src/creditlens/extraction/ocr.py`](../../../src/creditlens/extraction/ocr.py)
- [`src/creditlens/extraction/prompts.py`](../../../src/creditlens/extraction/prompts.py)
- [`src/creditlens/schemas.py`](../../../src/creditlens/schemas.py)
- [`src/creditlens/gateway/llm.py`](../../../src/creditlens/gateway/llm.py)
- [`evals/extraction_accuracy.py`](../../../evals/extraction_accuracy.py)

## Read

- [`docs/extraction.md`](../../../docs/extraction.md)

## Tests

- [`tests/test_extraction.py`](../../../tests/test_extraction.py)

## Run from the repo root

```bash
LLM_OFFLINE=1 python -m evals.extraction_accuracy --limit 5
LLM_OFFLINE=1 python -m pytest -q tests/test_extraction.py
```

## Done when

Field accuracy against the golden set is on the scoreboard, and you know your worst document.

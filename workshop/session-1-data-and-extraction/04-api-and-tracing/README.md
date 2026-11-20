# API and tracing

*Session 1: Discover, data foundation, ground truth, extraction*

**Topics:** A1 to A5, OB1

**Notebook:** 02_ground_truth_and_extraction, section 3 (notebooks/01_data_foundation.ipynb and notebooks/02_ground_truth_and_extraction.ipynb)

## Code

- [`src/creditlens/api/app.py`](../../../src/creditlens/api/app.py)
- [`src/creditlens/api/auth.py`](../../../src/creditlens/api/auth.py)
- [`src/creditlens/api/deps.py`](../../../src/creditlens/api/deps.py)
- [`src/creditlens/api/rbac.py`](../../../src/creditlens/api/rbac.py)
- [`src/creditlens/observability.py`](../../../src/creditlens/observability.py)

## Read

- [`docs/api.md`](../../../docs/api.md)

## Tests

- [`tests/test_api.py`](../../../tests/test_api.py)

## Run from the repo root

```bash
python -m creditlens.api
LLM_OFFLINE=1 python -m pytest -q tests/test_api.py
```

## Done when

A request shows up as a trace, and U01 and U05 see different application counts.

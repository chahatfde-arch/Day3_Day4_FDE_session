# Data foundation: layers, wrangling, contracts

*Session 1: Discover, data foundation, ground truth, extraction*

**Topics:** P1 to P5

**Notebook:** 01_data_foundation, sections 3 to 7 (notebooks/01_data_foundation.ipynb and notebooks/02_ground_truth_and_extraction.ipynb)

## Code

- [`src/creditlens/pipeline/wrangle.py`](../../../src/creditlens/pipeline/wrangle.py)
- [`src/creditlens/pipeline/clean.py`](../../../src/creditlens/pipeline/clean.py)
- [`src/creditlens/pipeline/contracts.py`](../../../src/creditlens/pipeline/contracts.py)
- [`src/creditlens/pipeline/features.py`](../../../src/creditlens/pipeline/features.py)
- [`src/creditlens/pipeline/run.py`](../../../src/creditlens/pipeline/run.py)

## Read

- [`docs/CONTRACTS.md`](../../../docs/CONTRACTS.md)
- [`docs/data_sources.md`](../../../docs/data_sources.md)

## Tests

- [`tests/test_wrangle.py`](../../../tests/test_wrangle.py)
- [`tests/test_pipeline.py`](../../../tests/test_pipeline.py)

## Run from the repo root

```bash
python -m creditlens.pipeline --load
LLM_OFFLINE=1 python -m pytest -q tests/test_wrangle.py tests/test_pipeline.py
```

## Done when

Every planted data problem is caught by a contract or fixed by the pipeline, and the tests are green.

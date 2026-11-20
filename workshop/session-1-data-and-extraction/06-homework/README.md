# Homework: SOW, async ingestion lab, SQL Server

*Session 1: Discover, data foundation, ground truth, extraction*

**Topics:** OG3, P6 to P10, P15

**Notebook:** 01_data_foundation, sections 8 and 9 (notebooks/01_data_foundation.ipynb and notebooks/02_ground_truth_and_extraction.ipynb)

## Code

- [`src/creditlens/pipeline/ingest.py`](../../../src/creditlens/pipeline/ingest.py)
- [`src/creditlens/pipeline/load.py`](../../../src/creditlens/pipeline/load.py)

## Tests

- [`tests/test_ingest.py`](../../../tests/test_ingest.py)

## Run from the repo root

```bash
python -m creditlens.pipeline.ingest
```

## Done when

A one-page SOW for Good Bank, your measured ranking of the four ingestion strategies, and Neo4j and Langfuse keys in .env.

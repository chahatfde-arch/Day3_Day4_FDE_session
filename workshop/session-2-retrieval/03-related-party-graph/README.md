# Related-party graph

*Session 2: Retrieval, graph and visual documents*

**Topics:** G1 to G11

**Notebook:** 03_retrieval, section 4 (notebooks/03_retrieval.ipynb)

## Code

- [`src/creditlens/retrieval/graph.py`](../../../src/creditlens/retrieval/graph.py)

## Tests

- [`tests/test_graph.py`](../../../tests/test_graph.py)

## Run from the repo root

```bash
LLM_OFFLINE=1 python -m pytest -q tests/test_graph.py
```

## Done when

Every hidden group is found, including the one linked only by an address, with the Cypher recorded.

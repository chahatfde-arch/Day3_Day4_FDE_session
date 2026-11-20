# Credit policy RAG: chunking, hybrid search, reranking

*Session 2: Retrieval, graph and visual documents*

**Topics:** R5 to R16

**Notebook:** 03_retrieval, section 3 (notebooks/03_retrieval.ipynb)

## Code

- [`src/creditlens/retrieval/chunking.py`](../../../src/creditlens/retrieval/chunking.py)
- [`src/creditlens/retrieval/index.py`](../../../src/creditlens/retrieval/index.py)
- [`src/creditlens/retrieval/policy.py`](../../../src/creditlens/retrieval/policy.py)
- [`src/creditlens/retrieval/pinecone_compare.py`](../../../src/creditlens/retrieval/pinecone_compare.py)

## Read

- [`docs/retrieval.md`](../../../docs/retrieval.md)

## Tests

- [`tests/test_retrieval.py`](../../../tests/test_retrieval.py)

## Run from the repo root

```bash
LLM_OFFLINE=1 python -m pytest -q tests/test_retrieval.py
```

## Done when

Hybrid search returns cited chunks, and the source filter keeps Good Bank answers first.

# Retrieval evaluation

*Session 2: Retrieval, graph and visual documents*

**Topics:** OB6 to OB8

**Notebook:** 03_retrieval, section 6 (notebooks/03_retrieval.ipynb)

## Code

- [`evals/retrieval_eval.py`](../../../evals/retrieval_eval.py)

## Read

- [`ground_truth/retrieval_questions.json`](../../../ground_truth/retrieval_questions.json)

## Tests

- [`tests/test_evals.py`](../../../tests/test_evals.py)

## Run from the repo root

```bash
LLM_OFFLINE=1 python -m evals.retrieval_eval --k 5
```

## Done when

Recall@5 and MRR for naive, hybrid and reranked search on both corpora, and every miss explained.

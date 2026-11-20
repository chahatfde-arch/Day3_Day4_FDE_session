# Clone, install and check

*Pre-work*

**Notebook:** 00_prework, sections 1 to 4 (notebooks/00_prework.ipynb)

## Code

- [`src/creditlens/setup_check.py`](../../../src/creditlens/setup_check.py)
- [`src/creditlens/generators/`](../../../src/creditlens/generators/)
- [`src/creditlens/data_real/`](../../../src/creditlens/data_real/)

## Tests

- [`tests/test_generators.py`](../../../tests/test_generators.py)

## Run from the repo root

```bash
python -m creditlens.generators
python -m creditlens.pipeline --load
python -m creditlens.data_real
python -m creditlens.setup_check
```

## Done when

Every line of the setup check reads OK, or WARN for later sessions only.

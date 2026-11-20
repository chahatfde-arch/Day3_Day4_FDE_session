# Ground truth: the labelling party

*Session 1: Discover, data foundation, ground truth, extraction*

**Topics:** evaluation first

**Notebook:** 02_ground_truth_and_extraction, section 2 (notebooks/01_data_foundation.ipynb and notebooks/02_ground_truth_and_extraction.ipynb)

## Code

- [`ground_truth/agreement.py`](../../../ground_truth/agreement.py)

## Read

- [`ground_truth/LABELING_GUIDELINES.md`](../../../ground_truth/LABELING_GUIDELINES.md)
- [`ground_truth/golden_set.json`](../../../ground_truth/golden_set.json)

## Run from the repo root

```bash
python ground_truth/agreement.py labels_a.csv labels_b.csv --key file
```

## Done when

Your pod has settled its disagreements and the golden set is committed.

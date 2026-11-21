# Risk model and MLOps (Day 6)

*Beginner notes for milestone 8. Code: `src/creditlens/ml/`, `monitoring/fairness_report.json`.*

## Good Bank has no history

A default model needs examples of loans that went bad. Good Bank is a new lender with none, so it
bootstraps from a public loan book: 25,000 real SBA 7(a) small-business loans (Kaggle, CC BY-SA 4.0)
with a charged-off flag. This is what an FDE does when the client has no labels, and it comes with
two obligations that the rest of this page is about: an honest adapter, and a fairness check.

## Features

`sba_features(df)` turns the raw columns into model inputs: loan term, employees, jobs created and
retained, new-business flag, franchise flag, revolving-line and low-doc flags, the amounts, the SBA
guarantee share, the approval year, the urban/rural code and the two-digit industry code.

`State` and `BankState` are **not** features. They are kept as *segments* for the fairness check.
A model that learns geography as a proxy has learned something the bank must not act on, and the
cleanest way to stop it is to never show it the column.

## Train, test, metrics

- **Stratified split**: hold out a fifth of the loans with the same default rate as the rest.
- **Baseline**: logistic regression, one-hot encoded categories. Simple, explainable.
- **Challenger**: gradient boosting. Usually better, harder to explain.
- **AUC**: how often the model ranks a bad loan above a good one. 0.5 is a coin, 1.0 is perfect.
- **PR-AUC**: the same idea focused on the rare class (defaults), which is what matters here.
- **Brier score**: how well the probabilities are calibrated. A model that says "9 percent" should be
  wrong about nine times in a hundred.

```bash
python -m creditlens.ml.train --quick                # subsample, finishes in about a minute
python -m creditlens.ml.train --quick --compare-synthetic   # real labels versus Day 1's synthetic history
```

The synthetic history is easier to predict, because it was generated from a formula. That is why the
real loan book is in the project: easy numbers are the wrong numbers to trust.

## MLflow: runs, registry, aliases

- A **run** is one training: parameters, metrics, artifacts (calibration curve, feature importance).
- The **registry** holds named models with numbered versions: `creditlens-risk` version 1, 2, 3.
- An **alias** is a pointer: `champion` points at the version in production. Serving reads the alias,
  so promoting or rolling back a model is one call and no code change:

```python
client.set_registered_model_alias("creditlens-risk", "champion", version="1")
```

The default tracking URI is `sqlite:///mlruns.db` so the registry works without a server; `mlflow ui
--backend-store-uri sqlite:///mlruns.db` shows the runs side by side.

## The fairness check

`monitoring/fairness_report.json` reports, per State (top ten) and per urban/rural code: the number of
loans, the default rate, the AUC and the share the model flags high-risk. A large gap in the flagged
share between segments with similar default rates is the signal to investigate. The training script
prints the largest gap in plain language. Geography is not a feature, so a gap comes from the other
columns: the discussion is whether those columns are fair proxies.

## Train/serve skew and the adapter

A Good Bank application is in INR thousands with Indian sector names, tenure in months and an
incorporation date. The model wants US dollars, NAICS codes, a term and a new-business flag.
`application_to_sba_features()` maps one onto the other, and every assumption is written down in the
module's `ASSUMPTIONS` list: the currency factor (`INR_PER_USD`, default 84), the sector-to-NAICS map,
"new business" meaning under two years, and so on. Read them aloud on Day 6. Each one is a conversation
with the Chief Risk Officer, and each one is a place where drift will show up first (Day 7).

## Serving

`score(features)` loads the champion from the MLflow registry, else from `data/cache/model.joblib`,
else uses a transparent rule-based stub, and never raises. It returns a `RiskScore` with a probability,
a band (low under 8 percent, high from 20 percent) and three plain-language drivers. `score_application(id)`
assembles the features from the database through the adapter. The API mounts it at `POST /risk/score`.

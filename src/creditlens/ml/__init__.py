"""The risk model: features, training with MLflow, and serving a probability of default.

Milestone 8. Three modules, in the order a beginner should read them:

1. `creditlens.ml.features` turns the real SBA 7(a) loan book into a feature table, and
   adapts one Good Bank application onto that same schema (the "adapter").
2. `creditlens.ml.train` trains a baseline and a challenger, tracks both in MLflow and
   promotes the better one to the `champion` alias.
3. `creditlens.ml.serving` loads the champion and returns a `RiskScore` the agent can use.
"""

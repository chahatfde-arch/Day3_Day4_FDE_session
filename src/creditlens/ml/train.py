"""Train the default-risk model on the real SBA loan book, with MLflow tracking.

The one idea this module teaches: **a model is not a file, it is a tracked experiment**.
Two models are trained every run, and both are measured against a floor:

    stub       the rule-based score the agent used before any model (no training)
    baseline   logistic regression  (simple, linear, easy to explain)
    challenger gradient boosting    (non-linear, usually better, harder to explain)

Both are logged to MLflow with the same metrics, so "is the new one actually better?" is
a question you answer by looking at two rows in a table, not by remembering.

Metrics, in plain words:
  AUC          given one defaulter and one payer, how often does the model rank the
               defaulter as riskier? 0.5 is a coin toss, 1.0 is perfect.
  PR-AUC       the same idea but focused on the rare class (defaults are ~18 percent here).
  Brier score  the average squared error of the probability itself. Lower is better.
               AUC says "the order is right", Brier says "the number is right".

Run it:
    python -m creditlens.ml.train --quick             # subsampled, under two minutes
    python -m creditlens.ml.train                     # the full 25,000 rows
    python -m creditlens.ml.train --compare-synthetic # real labels vs the synthetic set

MLflow note: `mlruns/` as a tracking URI gives you runs but **no model registry**, because
the registry needs a database. So when MLFLOW_TRACKING_URI is unset we default to
`sqlite:///mlruns.db`, which gives runs *and* a registry with aliases, with no server to run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from creditlens.config import settings
from creditlens.ml.features import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    BINARY_FEATURES,
    load_sba,
    sba_features,
    sba_segments,
)

REGISTERED_MODEL_NAME = "creditlens-risk"
CHAMPION_ALIAS = "champion"
STUB_RUN_NAME = "stub_rules"
QUICK_ROWS = 6000
HIGH_RISK_THRESHOLD = 0.2  # the "high" band boundary, shared with creditlens.ml.serving
TOP_STATES_FOR_FAIRNESS = 10


def default_tracking_uri() -> str:
    """A local SQLite file by default, so the model registry and aliases work offline."""
    configured = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if configured and configured not in {"mlruns", "./mlruns"}:
        return configured
    return f"sqlite:///{(settings.paths.root / 'mlruns.db').as_posix()}"


# ---------------------------------------------------------------------------
# The two pipelines
# ---------------------------------------------------------------------------


def _column_transformer():
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # handle_unknown="ignore": a NAICS code the model never saw becomes all zeros
        # instead of crashing. Good Bank sectors map onto a handful of codes only.
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("bin", SimpleImputer(strategy="constant", fill_value=0), BINARY_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def build_baseline():
    """Logistic regression: the number to beat. Every project needs one."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return Pipeline([("prep", _column_transformer()),
                     ("model", LogisticRegression(max_iter=1000, C=1.0))])


def build_challenger(quick: bool = False):
    """Gradient boosting. Returns (pipeline, flavour) so the run can say what it used.

    scikit-learn's HistGradientBoostingClassifier is first choice. On some locked-down
    Windows builds its compiled extension cannot load, so we fall back to XGBoost, which
    is the same family of model. The fallback is reported, never silent.
    """
    from sklearn.pipeline import Pipeline

    try:
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(
            max_iter=120 if quick else 300, learning_rate=0.08, max_depth=6,
            early_stopping=True, validation_fraction=0.15, random_state=settings.seed)
        flavour = "sklearn.HistGradientBoostingClassifier"
    except Exception as exc:  # noqa: BLE001  compiled extension unavailable
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=150 if quick else 400, learning_rate=0.08, max_depth=5,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            tree_method="hist", random_state=settings.seed, n_jobs=2)
        flavour = f"xgboost.XGBClassifier (sklearn.ensemble unavailable: {type(exc).__name__})"
    return Pipeline([("prep", _column_transformer()), ("model", model)]), flavour


# ---------------------------------------------------------------------------
# Metrics and artifacts
# ---------------------------------------------------------------------------


def evaluate(y_true: pd.Series, proba: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    return {
        "auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "default_rate": float(np.mean(y_true)),
        "mean_predicted": float(np.mean(proba)),
    }


def _calibration_png(y_true: pd.Series, proba: np.ndarray, path: Path, title: str) -> Path:
    """A calibration curve: of the loans we said were 10 percent risky, how many defaulted?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    observed, predicted = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    ax.plot(predicted, observed, "o-", label=title)
    ax.set_xlabel("predicted probability of default")
    ax.set_ylabel("observed default rate")
    ax.set_title(f"Calibration: {title}")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _feature_names(pipeline) -> list[str]:
    try:
        return list(pipeline.named_steps["prep"].get_feature_names_out())
    except Exception:  # noqa: BLE001
        return list(MODEL_FEATURES)


def feature_importance(pipeline) -> dict[str, float]:
    """Linear models expose coefficients, trees expose importances. Either way, a dict."""
    names = _feature_names(pipeline)
    model = pipeline.named_steps["model"]
    values: Optional[np.ndarray] = None
    if hasattr(model, "coef_"):
        values = np.asarray(model.coef_).ravel()
    elif hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_).ravel()
    if values is None or len(values) != len(names):
        return {}
    pairs = sorted(zip(names, values.tolist()), key=lambda kv: abs(kv[1]), reverse=True)
    return {k: round(float(v), 6) for k, v in pairs[:40]}


# ---------------------------------------------------------------------------
# Fairness
# ---------------------------------------------------------------------------


def fairness_report(y_true: pd.Series, proba: np.ndarray, segments: pd.DataFrame,
                    threshold: float = HIGH_RISK_THRESHOLD) -> dict[str, Any]:
    """Per segment: how many loans, how many defaulted, how well the model ranks, and
    what share we would call high risk. Equal AUC with unequal high-risk share is still
    a business problem: it means one group is pushed to the back of the queue more often.
    """
    from sklearn.metrics import roc_auc_score

    def rows_for(column: str, values) -> list[dict[str, Any]]:
        out = []
        for value in values:
            mask = (segments[column] == value).to_numpy()
            if mask.sum() < 30:
                continue
            group_y, group_p = y_true[mask], proba[mask]
            try:
                auc = float(roc_auc_score(group_y, group_p)) if group_y.nunique() > 1 else None
            except ValueError:
                auc = None
            out.append({
                "segment": column,
                "value": str(value),
                "n": int(mask.sum()),
                "default_rate": round(float(group_y.mean()), 4),
                "auc": round(auc, 4) if auc is not None else None,
                "share_predicted_high_risk": round(float((group_p >= threshold).mean()), 4),
            })
        return out

    top_states = segments["State"].value_counts().head(TOP_STATES_FOR_FAIRNESS).index.tolist()
    rows = rows_for("State", top_states) + rows_for("UrbanRural", sorted(segments["UrbanRural"].unique()))

    gaps = {}
    for column in ("State", "UrbanRural"):
        subset = [r for r in rows if r["segment"] == column]
        if len(subset) < 2:
            continue
        hi = max(subset, key=lambda r: r["share_predicted_high_risk"])
        lo = min(subset, key=lambda r: r["share_predicted_high_risk"])
        gaps[column] = {
            "highest": hi["value"], "highest_share": hi["share_predicted_high_risk"],
            "lowest": lo["value"], "lowest_share": lo["share_predicted_high_risk"],
            "gap": round(hi["share_predicted_high_risk"] - lo["share_predicted_high_risk"], 4),
        }

    return {
        "threshold": threshold,
        "note": ("State and BankState are not model features. Any gap below comes from the "
                 "other features correlating with geography, which is exactly what a "
                 "fairness check is for."),
        "rows": rows,
        "largest_gaps": gaps,
    }


def _print_fairness(report: dict[str, Any]) -> None:
    for column, gap in report.get("largest_gaps", {}).items():
        print(f"Fairness, {column}: the model calls {gap['highest_share']:.0%} of "
              f"{gap['highest']} loans high risk and {gap['lowest_share']:.0%} of "
              f"{gap['lowest']} loans high risk, a gap of {gap['gap']:.0%}. "
              "Geography is not a feature, so this gap comes from the other columns.")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(experiment: str = "creditlens-risk", data_path: Optional[str] = None,
          tracking_uri: Optional[str] = None, output_dir: Optional[str] = None,
          quick: bool = False, verbose: bool = True) -> dict:
    """Train both models, track them, promote the better one, write the fairness report."""
    import joblib
    import mlflow
    from sklearn.model_selection import train_test_split

    warnings.filterwarnings("ignore", category=UserWarning)

    out_dir = Path(output_dir) if output_dir else settings.paths.cache
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_sba(data_path)
    if quick and len(raw) > QUICK_ROWS:
        raw = raw.sample(n=QUICK_ROWS, random_state=settings.seed).reset_index(drop=True)

    X, y = sba_features(raw)
    segments = sba_segments(raw, X.index)

    # Stratified split: the test set must have the same default rate as the training set,
    # otherwise the metrics describe a different portfolio than the one we trained on.
    X_train, X_test, y_train, y_test, _, seg_test = train_test_split(
        X, y, segments, test_size=0.25, random_state=settings.seed, stratify=y)

    mlflow.set_tracking_uri(tracking_uri or default_tracking_uri())
    mlflow.set_experiment(experiment)

    challenger, challenger_flavour = build_challenger(quick=quick)
    candidates = [("baseline_logreg", build_baseline(), "sklearn.LogisticRegression"),
                  ("challenger_gbm", challenger, challenger_flavour)]

    results: dict[str, dict] = {}
    fitted: dict[str, Any] = {}
    run_ids: dict[str, str] = {}

    # The floor: the rule-based stub the agent answers with when no model exists, scored on the
    # same held-out loans. Logged beside the models, never registered: a model must beat it.
    from creditlens.ml.serving import stub_score

    stub_proba = np.array([stub_score(row).probability_of_default for row in X_test.to_dict("records")])
    stub_metrics = evaluate(y_test, stub_proba)
    with mlflow.start_run(run_name=STUB_RUN_NAME):
        mlflow.log_params({"model": STUB_RUN_NAME, "flavour": "rules, no training", "quick": quick,
                           "n_test": len(X_test), "seed": settings.seed})
        mlflow.log_metrics(stub_metrics)
    if verbose:
        print(f"{STUB_RUN_NAME:18s} AUC {stub_metrics['auc']:.4f}  PR-AUC {stub_metrics['pr_auc']:.4f}  "
              f"Brier {stub_metrics['brier']:.4f}  [rules, no training: the floor]")

    for name, pipeline, flavour in candidates:
        with mlflow.start_run(run_name=name) as run:
            pipeline.fit(X_train, y_train)
            proba = pipeline.predict_proba(X_test)[:, 1]
            metrics = evaluate(y_test, proba)

            mlflow.log_params({
                "model": name, "flavour": flavour, "quick": quick,
                "n_train": len(X_train), "n_test": len(X_test),
                "n_features": len(MODEL_FEATURES), "seed": settings.seed,
            })
            mlflow.log_metrics(metrics)

            png = _calibration_png(y_test, proba, out_dir / f"calibration_{name}.png", name)
            mlflow.log_artifact(str(png))
            importance = feature_importance(pipeline)
            imp_path = out_dir / f"feature_importance_{name}.json"
            imp_path.write_text(json.dumps(importance, indent=1), encoding="utf-8")
            mlflow.log_artifact(str(imp_path))
            _log_model(pipeline, X_train.head(5))

            results[name] = {**metrics, "flavour": flavour, "run_id": run.info.run_id}
            fitted[name] = pipeline
            run_ids[name] = run.info.run_id
            if verbose:
                print(f"{name:18s} AUC {metrics['auc']:.4f}  PR-AUC {metrics['pr_auc']:.4f}  "
                      f"Brier {metrics['brier']:.4f}  [{flavour}]")

    champion_name = max(results, key=lambda k: results[k]["auc"])
    champion = fitted[champion_name]
    champion_proba = champion.predict_proba(X_test)[:, 1]

    # Fairness on the held-out set, with the champion.
    fair = fairness_report(y_test, champion_proba, seg_test)
    fair_path = settings.paths.root / "monitoring" / "fairness_report.json"
    fair_path.parent.mkdir(parents=True, exist_ok=True)
    fair_path.write_text(json.dumps(
        {"model": champion_name, "auc": results[champion_name]["auc"], **fair}, indent=1),
        encoding="utf-8")
    if verbose:
        _print_fairness(fair)

    # Register the champion and move the alias. Rolling back on Day 7 is then one line:
    # client.set_registered_model_alias(name, "champion", previous_version).
    version = _register_champion(run_ids[champion_name], verbose=verbose)

    medians = X_train[NUMERIC_FEATURES].median(numeric_only=True).to_dict()
    metadata = {
        "name": REGISTERED_MODEL_NAME,
        "version": version,
        "algorithm": results[champion_name]["flavour"],
        "run_name": champion_name,
        "auc": round(results[champion_name]["auc"], 4),
        "pr_auc": round(results[champion_name]["pr_auc"], 4),
        "brier": round(results[champion_name]["brier"], 4),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": list(MODEL_FEATURES),
        "numeric_medians": {k: float(v) for k, v in medians.items()},
        "n_train": int(len(X_train)),
        "quick": quick,
        # every candidate on the same held-out set, so the comparison survives without the MLflow server
        "candidates": {k: {"algorithm": v["flavour"], "auc": round(v["auc"], 4), "pr_auc": round(v["pr_auc"], 4),
                           "brier": round(v["brier"], 4)} for k, v in results.items()},
        "floor": {"name": STUB_RUN_NAME, "auc": round(stub_metrics["auc"], 4),
                  "pr_auc": round(stub_metrics["pr_auc"], 4), "brier": round(stub_metrics["brier"], 4)},
    }
    model_path = out_dir / "model.joblib"
    joblib.dump(champion, model_path)
    (out_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=1), encoding="utf-8")

    if verbose:
        print(f"Champion: {champion_name} (AUC {metadata['auc']}), saved to {model_path}")
        print(f"Fairness report: {fair_path}")

    return {"champion": champion_name, "metrics": results, "stub": stub_metrics, "metadata": metadata,
            "model_path": str(model_path), "fairness_path": str(fair_path),
            "tracking_uri": tracking_uri or default_tracking_uri()}


def _log_model(pipeline, example: pd.DataFrame) -> None:
    """Store the fitted pipeline inside the MLflow run, under the artifact path 'model'.

    MLflow changed this call's keyword between 2.x (`artifact_path=`) and 3.x (`name=`),
    so we try the new one and fall back. Logging must never break training.

    The format is pickle on purpose. Recent MLflow defaults scikit-learn models to skops,
    which refuses HistGradientBoosting's internal TreePredictor type, so the champion was
    registered with no model inside it and serving silently fell back to the joblib file.
    Pickle is safe here because the registry only ever holds models this code trained.
    """
    import mlflow.sklearn

    fmt = {"serialization_format": getattr(mlflow.sklearn, "SERIALIZATION_FORMAT_PICKLE", "pickle")}
    try:
        mlflow.sklearn.log_model(pipeline, name="model", input_example=example, **fmt)
    except TypeError:
        mlflow.sklearn.log_model(pipeline, artifact_path="model", input_example=example, **fmt)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not log the model artifact ({type(exc).__name__}: {exc}).")


def _register_champion(run_id: str, verbose: bool = True) -> str:
    """Register the run's model and point the `champion` alias at the new version."""
    import mlflow

    try:
        uri = f"runs:/{run_id}/model"
        try:
            mlflow.register_model(uri, REGISTERED_MODEL_NAME)
        except Exception:  # noqa: BLE001  no model artifact logged: log the pipeline now
            raise
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        latest = max(versions, key=lambda v: int(v.version))
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, latest.version)
        if verbose:
            print(f"Registered {REGISTERED_MODEL_NAME} version {latest.version} as '{CHAMPION_ALIAS}'")
        return str(latest.version)
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"Model registry not available ({type(exc).__name__}: {exc}). "
                  "The joblib file below is still the champion, and serving falls back to it.")
        return "0"


# ---------------------------------------------------------------------------
# "Beat the baseline": real labels against the synthetic comparison set
# ---------------------------------------------------------------------------


def compare_synthetic(quick: bool = True, verbose: bool = True) -> dict:
    """Train the same kind of pipeline on the synthetic loan history and print both AUCs.

    The synthetic set was generated from a formula, so a model can recover that formula
    almost perfectly. The real SBA book is messy and the AUC is lower. That difference is
    the whole point of the Day 6 discussion: a high number on synthetic data proves nothing.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    real = train(experiment="creditlens-risk", quick=quick, verbose=False)
    real_auc = real["metrics"][real["champion"]]["auc"]

    history = pd.read_csv(settings.paths.raw / "history" / "loan_history.csv")
    numeric = ["years_in_business", "dscr", "debt_to_ebitda", "current_ratio",
               "cheque_returns_12m", "cash_credit_share", "avg_monthly_inflow",
               "inflow_volatility", "requested_inr_000", "tenure_months"]
    categorical = ["sector", "region"]
    X = history[numeric + categorical]
    y = history["defaulted"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=settings.seed, stratify=y)

    from sklearn.linear_model import LogisticRegression
    prep = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)])
    pipe = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=1000))])
    pipe.fit(X_train, y_train)
    synthetic_metrics = evaluate(y_test, pipe.predict_proba(X_test)[:, 1])

    out = {"real_sba_auc": round(real_auc, 4), "synthetic_auc": round(synthetic_metrics["auc"], 4),
           "real_champion": real["champion"]}
    if verbose:
        print("\nBeat the baseline: real labels against synthetic labels")
        print(f"  real SBA loan book   AUC {out['real_sba_auc']:.4f}  ({real['champion']})")
        print(f"  synthetic history    AUC {out['synthetic_auc']:.4f}  (logistic regression)")
        print("  The synthetic labels came from a formula, so a model can almost invert it. "
              "Real borrowers do not follow a formula. Trust the lower number.")
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train the CreditLens default-risk model")
    parser.add_argument("--quick", action="store_true", help=f"subsample to {QUICK_ROWS} rows")
    parser.add_argument("--experiment", default="creditlens-risk")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--compare-synthetic", action="store_true",
                        help="also train on data/raw/history/loan_history.csv and print both AUCs")
    args = parser.parse_args(argv)
    # With a tracking server, MLflow prints run links with emoji. On Windows, a notebook's
    # `!python` cell gives the child a cp1252 pipe, and one emoji would crash training.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    if args.compare_synthetic:
        compare_synthetic(quick=args.quick or True)
        return 0
    train(experiment=args.experiment, data_path=args.data_path, tracking_uri=args.tracking_uri,
          output_dir=args.output_dir, quick=args.quick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

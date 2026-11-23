"""Serve a probability of default as a `RiskScore`, with three levels of fallback.

The one idea this module teaches: **serving must never be the thing that breaks**.
A credit analyst waiting on a memo does not care that MLflow is down. So `score()`
tries, in order:

    1. the champion from the MLflow model registry  (models:/creditlens-risk@champion)
    2. the champion saved to data/cache/model.joblib (written by creditlens.ml.train)
    3. a transparent rule-based stub                 (no model at all, still explainable)

Every path returns the same `RiskScore`, and `model_name` says which one answered, so a
trace in Langfuse always shows whether the real model was used.

Risk bands are a business decision, not a model output, so they live in one constant.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from creditlens.config import settings
from creditlens.ml.features import (
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    application_to_sba_features,
)
from creditlens.observability import observe
from creditlens.schemas import RiskScore

# probability < 0.08 is low, < 0.2 is medium, the rest is high.
RISK_BANDS = {"low": 0.08, "medium": 0.20}

REGISTERED_MODEL_URI = "models:/creditlens-risk@champion"

_lock = threading.Lock()
_loaded: Optional[dict[str, Any]] = None


def risk_band(probability: float) -> str:
    if probability < RISK_BANDS["low"]:
        return "low"
    if probability < RISK_BANDS["medium"]:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Loading the champion
# ---------------------------------------------------------------------------


def model_dir() -> Path:
    return Path(os.environ.get("CREDITLENS_MODEL_DIR", settings.paths.cache))


def _load_metadata() -> dict:
    path = model_dir() / "model_metadata.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _load_from_registry() -> Optional[dict]:
    """Try the MLflow registry. Any failure at all means "not reachable", and we move on."""
    if os.environ.get("CREDITLENS_SKIP_MLFLOW") == "1":
        return None
    try:
        import mlflow

        from creditlens.ml.train import default_tracking_uri

        mlflow.set_tracking_uri(default_tracking_uri())
        model = mlflow.sklearn.load_model(REGISTERED_MODEL_URI)
        meta = _load_metadata()
        # Report the version the alias points at now, not the one in the metadata file:
        # after a rollback they differ, and the served version is what an auditor checks.
        try:
            version = mlflow.MlflowClient().get_model_version_by_alias("creditlens-risk", "champion").version
        except Exception:  # noqa: BLE001
            version = meta.get("version", "champion")
        return {"model": model, "source": "mlflow-registry",
                "name": "creditlens-risk", "version": str(version),
                "medians": meta.get("numeric_medians", {})}
    except Exception:  # noqa: BLE001  registry missing, no alias, mlflow not installed
        return None


def _load_from_disk() -> Optional[dict]:
    path = model_dir() / "model.joblib"
    if not path.exists():
        return None
    try:
        import joblib

        model = joblib.load(path)
    except Exception:  # noqa: BLE001
        return None
    meta = _load_metadata()
    return {"model": model, "source": "joblib",
            "name": meta.get("name", "creditlens-risk"), "version": str(meta.get("version", "0")),
            "medians": meta.get("numeric_medians", {})}


def load_champion(refresh: bool = False) -> Optional[dict]:
    """Load once per process. `refresh=True` after a retrain or a rollback."""
    global _loaded
    with _lock:
        if _loaded is None or refresh:
            _loaded = _load_from_registry() or _load_from_disk() or {}
        return _loaded or None


def reset_cache() -> None:
    """Forget the loaded model. Tests and the incident drills use this."""
    global _loaded
    with _lock:
        _loaded = None


# ---------------------------------------------------------------------------
# Explanations in plain English
# ---------------------------------------------------------------------------

PLAIN_NAMES = {
    "Term": "the loan term in months",
    "NoEmp": "the number of employees",
    "CreateJob": "the number of jobs the loan would create",
    "RetainedJob": "the number of jobs the loan would retain",
    "GrAppv": "the amount requested",
    "SBA_Appv": "the guaranteed portion of the amount",
    "sba_guarantee_share": "the share of the loan that is guaranteed",
    "ApprovalFY": "the year of approval",
    "RevLineCr": "the facility being a revolving line of credit",
    "LowDoc": "the loan going through the low documentation route",
    "is_new_business": "the business being new",
    "is_franchise": "the business being a franchise",
    "UrbanRural": "the urban or rural classification",
    "naics_sector": "the industry sector",
}


def _plain(name: str) -> str:
    base = name.split("__")[-1]
    for key, text in PLAIN_NAMES.items():
        if base == key or base.startswith(key + "_"):
            suffix = base[len(key) + 1:] if base.startswith(key + "_") else ""
            return f"{text} ({suffix})" if suffix else text
    return base.replace("_", " ")


def _linear_drivers(pipeline, row: pd.DataFrame, k: int = 3) -> list[str]:
    """For a linear model the contribution of a feature is coefficient times value."""
    prep, model = pipeline.named_steps["prep"], pipeline.named_steps["model"]
    transformed = np.asarray(prep.transform(row))
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    contributions = np.asarray(model.coef_).ravel() * transformed.ravel()
    names = list(prep.get_feature_names_out())
    order = np.argsort(-np.abs(contributions))[:k]
    drivers = []
    for i in order:
        direction = "raises" if contributions[i] > 0 else "lowers"
        drivers.append(f"{_plain(names[i]).capitalize()} {direction} the risk")
    return drivers


def _distance_drivers(features: dict, medians: dict, k: int = 3) -> list[str]:
    """For a tree model we cannot read coefficients, so we say which inputs are unusual.

    "Unusual" means: furthest from the median of the training data, in median-relative
    terms. It is not a causal explanation and the memo should not pretend it is; it is a
    pointer at the three numbers an analyst should look at first.
    """
    scored = []
    for name in NUMERIC_FEATURES:
        median = medians.get(name)
        value = features.get(name)
        if median in (None, 0) or value is None:
            continue
        try:
            distance = abs(float(value) - float(median)) / max(abs(float(median)), 1e-9)
        except (TypeError, ValueError):
            continue
        scored.append((distance, name, float(value), float(median)))
    scored.sort(reverse=True)
    drivers = []
    for _, name, value, median in scored[:k]:
        word = "well above" if value > median else "well below"
        drivers.append(f"{_plain(name).capitalize()} is {word} the typical loan "
                       f"({value:,.0f} against a usual {median:,.0f})")
    return drivers


# ---------------------------------------------------------------------------
# The rule-based stub: what we say when there is no model at all
# ---------------------------------------------------------------------------


def stub_score(features: dict) -> RiskScore:
    """A transparent, arithmetic-only score. Every term is visible and arguable.

    It exists so the agent, the API and the tests all work on a clean checkout before
    anybody has trained anything. It is deliberately simple, and it says so in the drivers.
    """
    term = float(features.get("Term", 0) or 0)
    amount = float(features.get("GrAppv", 0) or 0)
    is_new = int(features.get("is_new_business", 0) or 0)

    probability = 0.06
    drivers = []
    if term > 120:
        probability += 0.05
        drivers.append("A term longer than ten years leaves more time for things to go wrong")
    elif term < 36:
        probability += 0.03
        drivers.append("A short term means large instalments against current cash flow")
    if amount > 250_000:
        probability += 0.06
        drivers.append("The amount requested is large for this kind of borrower")
    elif amount < 25_000:
        probability += 0.02
        drivers.append("Very small loans in this book default slightly more often")
    if is_new:
        probability += 0.07
        drivers.append("The business has been trading for less than two years")
    if not drivers:
        drivers.append("Nothing in the term, the amount or the age of the business stands out")
    drivers.append("No trained model was available, so this is the rule-based fallback")

    probability = float(min(max(probability, 0.0), 1.0))
    return RiskScore(probability_of_default=round(probability, 4), risk_band=risk_band(probability),
                     top_drivers=drivers[:3], model_name="rule-based-stub", model_version="0")


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------


def _feature_frame(features: dict) -> pd.DataFrame:
    """One row with exactly the model's columns, in the model's order."""
    row = {name: features.get(name) for name in MODEL_FEATURES}
    frame = pd.DataFrame([row], columns=MODEL_FEATURES)
    for name in ("UrbanRural", "naics_sector"):
        frame[name] = frame[name].astype(str)
    return frame


@observe(name="ml.score")
def score(features: dict) -> RiskScore:
    """Score one row of model features. Never raises: it falls back to the stub."""
    champion = load_champion()
    if not champion:
        return stub_score(features)

    try:
        frame = _feature_frame(features)
        pipeline = champion["model"]
        probability = float(pipeline.predict_proba(frame)[0, 1])
    except Exception:  # noqa: BLE001  a schema change should degrade, not crash
        return stub_score(features)

    model = getattr(pipeline, "named_steps", {}).get("model") if hasattr(pipeline, "named_steps") else None
    try:
        if model is not None and hasattr(model, "coef_"):
            drivers = _linear_drivers(pipeline, frame)
        else:
            drivers = _distance_drivers(features, champion.get("medians") or {})
    except Exception:  # noqa: BLE001
        drivers = []
    if not drivers:
        drivers = ["The model gave a score but could not point at a single driver",
                   "Read the policy checks in the memo before relying on this number",
                   "Ask for a retrain if this keeps happening"]

    return RiskScore(probability_of_default=round(probability, 4),
                     risk_band=risk_band(probability), top_drivers=drivers[:3],
                     model_name=str(champion.get("name", "creditlens-risk")),
                     model_version=str(champion.get("version", "0")))


# ---------------------------------------------------------------------------
# Scoring a whole application out of the database
# ---------------------------------------------------------------------------


def _read_table(engine, table: str, where_column: str, value: str) -> Optional[dict]:
    from sqlalchemy import text

    with engine.connect() as conn:
        rows = conn.execute(text(f"SELECT * FROM {table} WHERE {where_column} = :v LIMIT 1"),
                            {"v": value}).mappings().all()
    return dict(rows[0]) if rows else None


def features_for_application(application_id: str, db_url: Optional[str] = None) -> dict:
    """The one SBA-shaped row the model sees for a Good Bank application.

    This is where train/serve skew becomes concrete: the application, the company and the
    bank-statement features are three different tables in Good Bank's world, and the
    adapter squeezes them into one SBA-shaped row.
    """
    from sqlalchemy import create_engine

    engine = create_engine(db_url or settings.database_url)
    application = _read_table(engine, "applications", "application_id", application_id)
    if application is None:
        raise ValueError(f"No application {application_id}")
    company = _read_table(engine, "companies", "company_id", str(application.get("company_id")))
    statement = _read_table(engine, "statement_features", "application_id", application_id)
    return application_to_sba_features(application, company, statement)


@observe(name="ml.score_application")
def score_application(application_id: str, db_url: Optional[str] = None) -> RiskScore:
    """Assemble the features for one Good Bank application and score it."""
    return score(features_for_application(application_id, db_url=db_url))


# ---------------------------------------------------------------------------
# FastAPI router. Imported lazily so this module works without FastAPI installed.
# ---------------------------------------------------------------------------
from pydantic import BaseModel, Field  # noqa: E402  pydantic is a core dependency; FastAPI is not


class ScoreRequest(BaseModel):
    """The body of POST /risk/score. At module level, not inside build_router: with postponed annotations
    FastAPI resolves the type by name from this module, and a class inside a function cannot be found, which
    broke /openapi.json and left /docs empty."""
    application_id: Optional[str] = Field(
        None, description="Score a Good Bank application straight from the database")
    features: Optional[dict] = Field(
        None, description="Or pass model features directly, for what-if questions")


def build_router():
    """POST /risk/score with either a feature dict or an application_id."""
    from fastapi import APIRouter, HTTPException

    api = APIRouter(prefix="/risk", tags=["risk"])

    @api.post("/score", response_model=RiskScore)
    def score_endpoint(request: ScoreRequest) -> RiskScore:
        if request.application_id:
            try:
                return score_application(request.application_id)
            except ValueError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
        if request.features is not None:
            return score(request.features)
        raise HTTPException(status_code=422, detail="Pass application_id or features")

    return api


try:  # The API mounts `creditlens.ml.serving.router`; without FastAPI it stays None.
    router = build_router()
except Exception:  # noqa: BLE001
    router = None

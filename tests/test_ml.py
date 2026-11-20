"""Milestone 8: features from the real SBA loan book, training with MLflow, scoring with a fallback."""
from __future__ import annotations

import os

import pandas as pd
import pytest

from creditlens.config import settings
from creditlens.schemas import RiskScore

SBA = settings.paths.real / "sba_loans.csv"


@pytest.fixture(scope="module")
def sba() -> pd.DataFrame:
    if not SBA.exists():
        pytest.skip("fetch the real data first: python -m creditlens.data_real --only sba")
    return pd.read_csv(SBA)


def test_sba_features_are_numeric_and_complete(sba):
    from creditlens.ml.features import sba_features

    X, y = sba_features(sba.head(2000))
    assert len(X) == len(y) > 0
    assert not X.isna().any().any()
    assert set(y.unique()) <= {0, 1}
    assert "State" not in X.columns and "BankState" not in X.columns  # geography is a segment, not a feature


def test_adapter_maps_a_good_bank_application_onto_every_model_column(sba):
    from creditlens.ml.features import ASSUMPTIONS, application_to_sba_features, sba_features

    apps = pd.read_csv(settings.paths.raw / "applications" / "applications.csv")
    companies = pd.read_csv(settings.paths.raw / "registry" / "companies.csv").set_index("company_id")
    feats = pd.read_csv(settings.paths.curated / "statement_features.csv").set_index("application_id")
    app = apps.iloc[0]
    row = application_to_sba_features(app.to_dict(), companies.loc[app.company_id].to_dict(),
                                      feats.loc[app.application_id].to_dict())
    X, _ = sba_features(sba.head(500))
    assert set(X.columns) <= set(row)
    assert len(ASSUMPTIONS) >= 5


@pytest.mark.slow
def test_quick_training_registers_a_champion_and_saves_a_joblib(sba, tmp_path):
    from creditlens.ml.train import train

    sample = tmp_path / "sba_small.csv"
    sba.sample(n=3000, random_state=0).to_csv(sample, index=False)
    result = train(data_path=str(sample), tracking_uri=f"sqlite:///{tmp_path / 'mlruns.db'}",
                   output_dir=str(tmp_path), quick=True)
    assert (tmp_path / "model.joblib").exists()
    name = result["champion"]
    assert name in ("baseline_logreg", "challenger_gbm")
    assert 0.5 < result["metrics"][name]["auc"] <= 1.0


def test_score_returns_a_valid_band_and_drivers():
    from creditlens.ml.serving import score_application

    rs = score_application("A001", db_url=settings.database_url if os.path.exists("creditlens.db") else None)
    assert isinstance(rs, RiskScore)
    assert rs.risk_band in ("low", "medium", "high")
    assert 0 <= rs.probability_of_default <= 1
    assert len(rs.top_drivers) >= 1


def test_stub_scores_without_any_model():
    from creditlens.ml.serving import stub_score

    rs = stub_score({"Term": 84, "GrAppv": 50000, "NewExist": 1, "NoEmp": 3})
    assert isinstance(rs, RiskScore) and "stub" in rs.model_name

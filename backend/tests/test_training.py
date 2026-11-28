"""Training uses labels only to choose thresholds; the evaluation report meets the quality bar."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

import ml.training
from backend.tests.conftest import ROOT, STORAGE
from config.settings import load_settings
from ml.evaluation import run_evaluation
from ml.model import BundleError, DbscanContext, cdf, load_bundle
from ml.training import train_models

MODELS = ROOT / "ml" / "models"


@pytest.fixture(scope="module")
def trained(tmp_path_factory, policy) -> dict:
    """Retrain from the committed files into a temporary directory, recording which labels were opened."""
    opened: list[list[Path]] = []
    original = ml.training.load_labels

    def spy(paths):
        opened.append(list(paths))
        return original(paths)

    ml.training.load_labels = spy
    try:
        models = tmp_path_factory.mktemp("models")
        report = train_models(
            STORAGE / "train.json",
            STORAGE / "validation.json",
            models,
            policy,
            ROOT / "datagen" / "manifest.json",
        )
    finally:
        ml.training.load_labels = original
    return {"models": models, "report": report, "opened": opened}


def test_labels_are_opened_only_for_validation(trained) -> None:
    assert trained["opened"] == [[STORAGE / "validation.json"]]


def test_detectors_are_fitted_on_train_rows_only(trained) -> None:
    fit_rows = json.loads((trained["models"] / "metadata.json").read_text())["fit_rows"]
    assert fit_rows["isolation_forest"] == 70_000
    assert fit_rows["score_mapping_reference"] == 15_000


def test_chosen_thresholds_respect_the_validation_budget(trained, policy) -> None:
    chosen = trained["report"]["threshold_selection"]["chosen"]
    assert chosen["verify_threshold"] < chosen["block_threshold"]
    assert chosen["verify_rate"] <= policy.goals["max_verify_rate"] - policy.new_account_verify_reserve
    assert chosen["legitimate_false_block_rate"] <= policy.goals["max_legitimate_false_block_rate"]


def test_retraining_reproduces_the_committed_models(trained) -> None:
    for name in ("operating_point.json", "metadata.json"):
        assert json.loads((trained["models"] / name).read_text()) == json.loads((MODELS / name).read_text())


def test_evaluation_meets_the_quality_bar(tmp_path: Path) -> None:
    settings = dataclasses.replace(load_settings(), artifacts_dir=tmp_path)
    headline = run_evaluation(settings)["headline"]
    assert headline["recall_verify_or_block"] >= 0.767
    assert headline["verify_rate"] <= 0.05
    assert headline["legitimate_false_block_rate"] <= 0.001
    report = json.loads((tmp_path / "evaluation.json").read_text())
    assert len(report["per_fraud_type"]) == 12
    assert report["per_fraud_type"]["qr_code_tampering"]["test_only"] is True
    committed = json.loads((ROOT / "artifacts" / "evaluation.json").read_text())
    assert {k: v for k, v in report["headline"].items() if k != "latency_p95_ms"} == {
        k: v for k, v in committed["headline"].items() if k != "latency_p95_ms"
    }


def test_score_mapping_is_a_bounded_monotonic_cdf() -> None:
    reference = np.sort(np.array([0.1, 0.2, 0.2, 0.5, 0.9]))
    mapped = cdf(reference, np.array([0.0, 0.2, 0.3, 0.9, 5.0]))
    assert mapped.tolist() == [0.0, 0.6, 0.6, 1.0, 1.0]


def test_dbscan_context_scores_distance_from_the_frozen_cores() -> None:
    rng = np.random.default_rng(1)
    context = DbscanContext(eps=0.5, min_samples=5).fit(rng.normal(0, 0.1, (300, 2)))
    near, far, unknown = context.score(
        np.array([[0.0, 0.0], [50.0, 50.0], [0.0, 0.0]]), ["ok", "ok", "insufficient_history"]
    )
    assert (near.status, near.score) == ("within_context", 0.0)
    assert (far.status, far.score) == ("outside_context", 1.0)
    assert (unknown.status, unknown.score) == ("insufficient_history", 0.0)


def test_a_bundle_from_other_feature_code_is_refused(tmp_path: Path) -> None:
    bundle = load_bundle(MODELS)
    bundle.metadata = {**bundle.metadata, "feature_version": "features-0.0.0"}
    joblib.dump(bundle, tmp_path / "secflow_model.joblib")
    with pytest.raises(BundleError):
        load_bundle(tmp_path)

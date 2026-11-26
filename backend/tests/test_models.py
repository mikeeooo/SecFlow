from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from config.settings import load_settings
from ml.algorithms.dbscan_context import DBSCANContext, EmptyCoreError
from ml.algorithms.isolation_forest import IsolationForestDetector
from ml.algorithms.local_outlier_factor import LOF_SUBSPACES, LocalOutlierFactorDetector
from ml.algorithms.preprocessing import EmpiricalScoreMapping, FeaturePreprocessor
from ml.features import PAYMENT_FEATURES
from ml.model import BundleError, load_bundle, load_operating_point

RNG = np.random.default_rng(7)
NORMAL = RNG.normal(0, 1, size=(2_000, 4))
OUTLIER = np.array([[8.0, 8.0, 8.0, 8.0]])


def test_isolation_forest_scores_higher_for_outliers_and_is_reproducible() -> None:
    a = IsolationForestDetector(100, 256, 42).fit(NORMAL)
    b = IsolationForestDetector(100, 256, 42).fit(NORMAL)
    assert a.raw_scores(OUTLIER)[0] > np.quantile(a.raw_scores(NORMAL[:200]), 0.99)
    assert np.array_equal(a.raw_scores(NORMAL[:50]), b.raw_scores(NORMAL[:50]))


def test_lof_novelty_scores_unseen_rows_and_uses_training_diagnostics_separately() -> None:
    detector = LocalOutlierFactorDetector(n_neighbors=20).fit(NORMAL)
    assert detector.model.novelty is True
    unseen = RNG.normal(0, 1, size=(200, 4))
    assert detector.raw_scores(OUTLIER)[0] > np.quantile(detector.raw_scores(unseen), 0.99)
    diagnostics = detector.training_diagnostics()
    assert diagnostics["fit_rows"] == 2_000
    with pytest.raises(AttributeError):
        detector.model.fit_predict(NORMAL)  # novelty mode forbids scoring the fitting rows this way


def test_score_mapping_is_monotonic_bounded_and_deterministic() -> None:
    mapping = EmpiricalScoreMapping().fit(np.array([1.0, 2.0, 2.0, 3.0]))
    out = mapping.transform(np.array([0.0, 1.0, 2.0, 2.5, 3.0, 99.0]))
    assert out.tolist() == [0.0, 0.25, 0.75, 0.75, 1.0, 1.0]
    assert np.all(np.diff(mapping.transform(np.linspace(-5, 5, 50))) >= 0)
    with pytest.raises(ValueError):
        mapping.transform(np.array([np.nan]))


def test_preprocessor_rejects_non_finite_and_uses_training_quantiles() -> None:
    pre = FeaturePreprocessor().fit(NORMAL)
    clipped = pre.transform(np.array([[1e9, 0, 0, 0]]))
    assert np.isfinite(clipped).all()
    with pytest.raises(ValueError):
        pre.transform(np.array([[np.inf, 0, 0, 0]]))


def test_dbscan_context_near_core_outside_insufficient_and_empty() -> None:
    context = DBSCANContext(eps=0.5, min_samples=5).fit(NORMAL)
    results = context.score(np.vstack([NORMAL[:1], OUTLIER, OUTLIER]), ["ok", "ok", "insufficient_history"])
    assert (results[0].score, results[0].status) == (0.0, "within_context")
    assert results[1].status == "outside_context" and results[1].score > 0
    assert (results[2].score, results[2].status) == (0.0, "insufficient_history")
    sparse = np.arange(20, dtype=float).reshape(-1, 1) * 100
    with pytest.raises(EmptyCoreError):
        DBSCANContext(eps=0.01, min_samples=5).fit(sparse)


def test_lof_uses_only_selected_columns() -> None:
    detector = LocalOutlierFactorDetector(n_neighbors=20, columns=(0, 1)).fit(NORMAL)
    probe = RNG.normal(0, 1, size=(50, 4))
    changed = probe.copy()
    changed[:, 2:] += 100.0  # columns the detector does not use
    np.testing.assert_array_equal(detector.raw_scores(probe), detector.raw_scores(changed))
    assert detector.params["columns"] == (0, 1)


def test_lof_fit_rows_subsample_is_seeded() -> None:
    a = LocalOutlierFactorDetector(n_neighbors=20, fit_rows=500).fit(NORMAL)
    b = LocalOutlierFactorDetector(n_neighbors=20, fit_rows=500).fit(NORMAL)
    assert a.model.n_samples_fit_ == 500
    np.testing.assert_array_equal(a.raw_scores(NORMAL[:100]), b.raw_scores(NORMAL[:100]))
    assert a.training_diagnostics()["fit_rows"] == 500
    full = LocalOutlierFactorDetector(n_neighbors=20, fit_rows=5_000).fit(NORMAL)
    assert full.model.n_samples_fit_ == 2_000


def test_lof_subspaces_name_known_payment_features() -> None:
    assert set(LOF_SUBSPACES) == {"amount", "amount_velocity", "amount_velocity_beneficiary", "full"}
    assert LOF_SUBSPACES["full"] == PAYMENT_FEATURES
    assert LOF_SUBSPACES["amount"] == ("log_amount", "amount_zscore_30d", "log_amount_to_prior_max_30d")
    for names in LOF_SUBSPACES.values():
        assert set(names) <= set(PAYMENT_FEATURES)


def test_dbscan_continuous_score() -> None:
    context = DBSCANContext(eps=0.5, min_samples=5).fit(np.zeros((10, 1)))  # all cores at 0, identity scaling
    probes = np.array([[0.2], [0.75], [1.0], [5.0], [5.0]])
    results = context.score(probes, ["ok", "ok", "ok", "ok", "insufficient_history"])
    assert (results[0].score, results[0].status) == (0.0, "within_context")
    assert results[1].status == "outside_context" and results[1].score == pytest.approx(0.5)
    assert (results[2].score, results[2].status) == (1.0, "outside_context")
    assert (results[3].score, results[3].status) == (1.0, "outside_context")
    assert (results[4].score, results[4].status) == (0.0, "insufficient_history")


def _write_point(directory: Path, **overrides) -> None:
    import json

    raw = {"verify_threshold": 55.0, "block_threshold": 92.0, "source": "validated", "version": "op-x",
           "model_version": "m-1", "weights": {"isolation_forest": 0.4, "lof": 0.05, "dbscan": 0.15, "rules": 0.4},
           "segments": {"personal": {"verify": 55.0, "block": 92.0}, "business": {"verify": 60.0, "block": 95.0}}}
    raw.update(overrides)
    raw = {k: v for k, v in raw.items() if v is not None}
    (directory / "operating_point.json").write_text(json.dumps(raw))


def test_load_operating_point_reads_weights_and_segments(tmp_path: Path) -> None:
    _write_point(tmp_path)
    point = load_operating_point(tmp_path, "m-1")
    assert point.weights == {"isolation_forest": 0.4, "lof": 0.05, "dbscan": 0.15, "rules": 0.4}
    assert point.thresholds_for("business") == (60.0, 95.0)
    assert point.thresholds_for("personal") == (55.0, 92.0)


def test_load_operating_point_rejects_file_without_weights(tmp_path: Path) -> None:
    _write_point(tmp_path, weights=None)
    with pytest.raises(BundleError):
        load_operating_point(tmp_path, "m-1")
    _write_point(tmp_path, segments=None)
    with pytest.raises(BundleError):
        load_operating_point(tmp_path, "m-1")


def test_saved_bundle_round_trip_and_schema_metadata() -> None:
    settings = load_settings()
    bundle = load_bundle(settings.models_dir)
    point = load_operating_point(settings.models_dir, bundle.version)
    assert tuple(bundle.metadata["payment_features"]) == PAYMENT_FEATURES
    assert bundle.metadata["fit_rows"]["isolation_forest"] == 60_000
    assert bundle.metadata["fit_rows"]["score_mapping_reference"] == 10_000
    assert bundle.dbscan.core_samples is not None and len(bundle.dbscan.core_samples) > 0
    assert 0 < point.verify < point.block <= 100
    rows = np.zeros((3, len(PAYMENT_FEATURES)))
    out = bundle.score_batch(rows, np.zeros((3, 8)), ["ok", "ok", "insufficient_history"])
    assert all(0 <= c.if_score <= 1 and 0 <= c.lof_score <= 1 for c in out)


def test_bundle_with_mismatched_schema_is_rejected(tmp_path: Path) -> None:
    import joblib

    settings = load_settings()
    payload = joblib.load(settings.models_dir / "secflow_model.joblib")
    payload["metadata"] = {**payload["metadata"], "payment_features": ["something_else"]}
    joblib.dump(payload, tmp_path / "secflow_model.joblib")
    with pytest.raises(BundleError):
        load_bundle(tmp_path)
    with pytest.raises(BundleError):
        load_bundle(tmp_path / "missing")


def test_training_never_fit_on_held_out_partitions() -> None:
    import json

    settings = load_settings()
    report = json.loads((settings.models_dir / "training_report.json").read_text())
    meta = json.loads((settings.models_dir / "metadata.json").read_text())
    assert meta["fit_rows"]["preprocessing"] == 60_000
    assert meta["fit_rows"]["dbscan"] <= 60_000
    assert report["operating_point"]["source"].startswith("validated")
    assert all(r["fit_rows"] == 60_000 for r in report["search"]["lof"])

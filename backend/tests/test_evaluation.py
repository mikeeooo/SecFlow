from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from config.settings import load_settings
from ml.algorithms.isolation_forest import IsolationForestDetector
from ml.algorithms.local_outlier_factor import LocalOutlierFactorDetector
from ml.evaluation import (
    binary_metrics,
    cluster_bootstrap,
    policy_metrics,
    run_evaluation,
    score_band_edges,
    score_bands,
    wilson_interval,
)
from ml.model import load_bundle


def test_undefined_metrics_are_null_not_invented() -> None:
    y = np.array([True, False, False])
    result = binary_metrics(np.zeros(3, dtype=bool), y)
    assert result["precision"] is None and result["f1"] is None
    assert result["recall"] == 0.0
    assert "undefined" in result["note"]
    assert wilson_interval(0, 0) is None
    empty = policy_metrics(np.zeros(3, dtype=int), np.zeros(3, dtype=bool))
    assert empty["verify_or_block"]["recall"] is None


def test_wilson_interval_widens_for_small_samples() -> None:
    small = wilson_interval(5, 5)
    large = wilson_interval(500, 500)
    assert small is not None and large is not None
    assert small[0] < large[0] < 1.0


@pytest.fixture(scope="module")
def evaluations(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, dict, dict[str, int]]:
    settings = load_settings()
    rows = {"isolation_forest": 0, "lof": 0}
    original_if, original_lof = IsolationForestDetector.raw_scores, LocalOutlierFactorDetector.raw_scores

    def count_if(self, matrix):  # type: ignore[no-untyped-def]
        rows["isolation_forest"] += len(matrix)
        return original_if(self, matrix)

    def count_lof(self, matrix):  # type: ignore[no-untyped-def]
        rows["lof"] += len(matrix)
        return original_lof(self, matrix)

    forbid = mock.Mock(side_effect=AssertionError("evaluation must not fit an estimator"))
    with mock.patch.object(IsolationForestDetector, "raw_scores", count_if), mock.patch.object(
        LocalOutlierFactorDetector, "raw_scores", count_lof
    ), mock.patch("sklearn.ensemble.IsolationForest.fit", forbid), mock.patch(
        "sklearn.neighbors.LocalOutlierFactor.fit", forbid
    ), mock.patch("sklearn.cluster.DBSCAN.fit", forbid):
        first = run_evaluation(replace(settings, artifacts_dir=tmp_path_factory.mktemp("eval-a")))
        counted = dict(rows)
    second = run_evaluation(replace(settings, artifacts_dir=tmp_path_factory.mktemp("eval-b")))
    return first, second, counted


def _without_timing(summary: dict) -> dict:
    data = {k: v for k, v in summary.items() if k not in {"generated_at", "latency_ms"}}
    data["headline"] = {k: v for k, v in summary["headline"].items() if k != "latency_p95_ms"}
    return data


def test_repeated_evaluation_is_identical_apart_from_timing(evaluations) -> None:
    first, second, _ = evaluations
    assert _without_timing(first) == _without_timing(second)


def test_each_component_scores_each_row_once(evaluations) -> None:
    first, _, counted = evaluations
    expected = 10_000 + first["test_requests"] + first["latency_ms"]["requests_timed"]  # validation + test + timing
    assert counted == {"isolation_forest": expected, "lof": expected}


def test_evaluation_uses_only_the_chronological_test_period(evaluations) -> None:
    first, _, _ = evaluations
    assert first["test_requests"] == 20_000 and first["test_fraud"] == 240
    assert first["splits"]["test"] == {"requests": 20_000, "fraud": 240, "fraud_share": 0.012}
    assert first["cold_start_accounts"]["accounts"] == 40
    assert first["unusual_legitimate"]["requests"] == 1_000


def test_reports_both_positive_views_and_separate_evidence_channel(evaluations) -> None:
    first, _, _ = evaluations
    for view in (first["combined_policy"], first["combined_policy_without_evidence"]):
        assert {"verify_or_block", "block_only"} <= set(view)
    channel = first["evidence_channel"]
    assert channel["fraud_with_active_evidence"] + channel["fraud_without_evidence"] == 240
    assert set(first["components"]) == {"isolation_forest", "lof", "dbscan", "rules", "combined_risk_score"}


def test_exported_metadata_matches_loaded_artifact() -> None:
    import json

    settings = load_settings()
    exported = json.loads((settings.artifacts_dir / "evaluation.json").read_text())
    assert exported["model_version"] == load_bundle(settings.models_dir).version
    assert Path(settings.artifacts_dir / "evaluation.md").read_text().startswith("# SecFlow evaluation report")


def test_score_bands_cover_every_request_and_rise_with_the_score() -> None:
    rng = np.random.default_rng(0)
    risk = np.concatenate([rng.uniform(0, 60, 900), rng.uniform(60, 100, 100), [100.0, np.nan]])
    y = np.concatenate([np.zeros(900, bool), rng.random(100) < 0.6, [True, False]])
    codes = (risk >= 61).astype(int)
    codes[np.isnan(risk)] = 0
    report = score_bands(risk, y, codes)
    assert report["requests"] == 1001  # the unscored request is excluded
    assert sum(b["requests"] for b in report["bands"]) == 1001
    assert sum(b["fraud"] for b in report["bands"]) == int(y[:-1].sum())
    assert report["bands"][-1]["requests"] >= 1  # a score of exactly 100 lands in the top band
    assert report["bands"][0]["recall_at_or_above"] == 1.0
    assert all(b["fraud"] == 0 for b in report["bands"][:5])  # bands below 50 hold no fraud


def _two_campaigns() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # two fraud campaigns of 5 rows (only the first is caught) plus 90 legitimate rows, 9 of them flagged
    y = np.array([True] * 10 + [False] * 90)
    codes = np.array([1] * 5 + [0] * 5 + [1] * 9 + [0] * 81)
    clusters = np.array(["CMP-A"] * 5 + ["CMP-B"] * 5 + [f"R{i}" for i in range(90)])
    return codes, y, clusters


def test_cluster_bootstrap_resamples_campaigns_as_units() -> None:
    codes, y, clusters = _two_campaigns()
    ci = cluster_bootstrap(codes, y, clusters, n=500, seed=1)
    assert set(ci) == {"precision_verify_or_block", "recall_verify_or_block", "verify_rate"}
    low, high = ci["recall_verify_or_block"]
    assert {low, high} <= {0.0, 0.5, 1.0}  # a campaign is either drawn whole or not at all
    assert low < 0.5 < high
    assert 0.0 <= ci["verify_rate"][0] <= ci["verify_rate"][1] <= 1.0


def test_cluster_bootstrap_is_seeded() -> None:
    codes, y, clusters = _two_campaigns()
    assert cluster_bootstrap(codes, y, clusters, n=200, seed=3) == cluster_bootstrap(codes, y, clusters, n=200, seed=3)


def test_score_band_edges_follow_operating_point() -> None:
    edges = score_band_edges(58.0, 95.0)
    assert 58.0 in edges and 95.0 in edges
    assert list(edges) == sorted(set(edges))
    assert edges[0] == 0.0 and edges[-1] == 100.0
    assert score_band_edges(70.0, 90.0).count(70.0) == 1  # duplicates removed


def test_evaluation_report_has_new_sections(evaluations) -> None:
    first, _, _ = evaluations
    for key in ("confidence_intervals", "holdout_accounts", "unseen_scenario", "drift_billers",
                "segment_thresholds", "weights"):
        assert key in first, key
    assert first["holdout_accounts"]["requests"] > 0
    assert first["unseen_scenario"]["requests"] == 40
    assert first["drift_billers"]["requests"] > 0
    assert set(first["segment_thresholds"]) == {"personal", "business"}
    verify, block = first["operating_point"]["verify"], first["operating_point"]["block"]
    assert verify in first["score_bands"]["edges"] and block in first["score_bands"]["edges"]

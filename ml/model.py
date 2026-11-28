"""The three unsupervised detectors and the saved model bundle.

* IsolationForest scores the scaled payment vector: few random splits isolate an unusual payment.
* LocalOutlierFactor (novelty mode) compares the local density of a payment's amount features with
  that of its 20 nearest training neighbours.
* DBSCAN clusters trailing 24-hour account profiles. DBSCAN cannot score new points, so we keep the
  training core samples and measure how far a new profile lies from the nearest one.

IsolationForest and LOF return raw scores on different scales. Each raw score is mapped to its
empirical CDF over the validation payments, giving a value in [0, 1]: 0.9 means "more unusual than
90% of validation payments". None of this reads a label.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ml.features import CONTEXT_FEATURES, FEATURE_VERSION, PAYMENT_FEATURES

BUNDLE_FILE = "secflow_model.joblib"
METADATA_FILE = "metadata.json"
# LOF scores the three amount features. No other feature set was measured.
LOF_FEATURES = ("log_amount", "amount_zscore_30d", "log_amount_to_prior_max_30d")
LOF_COLUMNS = [PAYMENT_FEATURES.index(name) for name in LOF_FEATURES]


class BundleError(RuntimeError):
    """The model bundle is missing or was trained with other feature code."""


class Preprocessor:
    """Clip each feature to its 0.5-99.5% training range, then standardise it."""

    def fit(self, matrix: np.ndarray) -> Preprocessor:
        self.lower, self.upper = np.quantile(matrix, [0.005, 0.995], axis=0)
        self.scaler = StandardScaler().fit(np.clip(matrix, self.lower, self.upper))
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(matrix, dtype=np.float64), self.lower, self.upper)
        return np.asarray(self.scaler.transform(clipped), dtype=np.float64)


def cdf(reference: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Share of the sorted `reference` scores that are <= each raw score."""
    return np.searchsorted(reference, raw, side="right") / len(reference)


@dataclass(frozen=True)
class ContextResult:
    score: float  # 0 inside a core neighbourhood, up to 1 at twice eps or further
    status: str  # within_context | outside_context | insufficient_history
    distance: float | None
    cluster: int | None  # cluster of the core neighbourhood the profile falls in
    nearest_cluster: int  # closest cluster, also for profiles outside every neighbourhood


class DbscanContext:
    """DBSCAN fitted on scaled 24-hour profiles; new profiles are scored against the frozen core samples."""

    def __init__(self, eps: float, min_samples: int) -> None:
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.metric = "euclidean"

    def fit(self, profiles: np.ndarray) -> DbscanContext:
        self.scaler = StandardScaler().fit(profiles)
        scaled = self.scaler.transform(profiles)
        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(scaled)
        self.core_samples = scaled[labels.core_sample_indices_]
        self.core_labels = labels.labels_[labels.core_sample_indices_]
        self.nn = NearestNeighbors(n_neighbors=1).fit(self.core_samples)
        self.summary = {
            "fit_profiles": float(len(scaled)),
            "core_samples": float(len(self.core_samples)),
            "clusters": float(len(set(labels.labels_.tolist()) - {-1})),
            "noise_fraction": float(np.mean(labels.labels_ == -1)),
        }
        return self

    def score(self, profiles: np.ndarray, statuses: list[str]) -> list[ContextResult]:
        distances, indices = self.nn.kneighbors(self.scaler.transform(profiles))
        results = []
        for distance, index, status in zip(distances[:, 0], indices[:, 0], statuses, strict=True):
            cluster = int(self.core_labels[index])
            if status != "ok":  # fewer than three earlier payments: no profile to compare
                results.append(ContextResult(0.0, "insufficient_history", None, None, cluster))
            elif distance <= self.eps:
                results.append(ContextResult(0.0, "within_context", float(distance), cluster, cluster))
            else:
                score = min(1.0, (distance - self.eps) / self.eps)
                results.append(ContextResult(score, "outside_context", float(distance), None, cluster))
        return results


@dataclass(frozen=True)
class Components:
    if_raw: float
    if_score: float
    lof_raw: float
    lof_score: float
    context: ContextResult


@dataclass
class ModelBundle:
    preprocessor: Preprocessor
    isolation_forest: IsolationForest
    lof: LocalOutlierFactor
    dbscan: DbscanContext
    if_reference: np.ndarray  # sorted raw IF scores of the validation payments
    lof_reference: np.ndarray
    metadata: dict[str, Any]

    @property
    def version(self) -> str:
        return str(self.metadata["model_version"])

    def raw_scores(self, payment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Raw IF and LOF scores, inverted so that higher means more unusual."""
        x = self.preprocessor.transform(payment)
        return -self.isolation_forest.score_samples(x), -self.lof.score_samples(x[:, LOF_COLUMNS])

    def score_batch(self, payment: np.ndarray, context: np.ndarray, statuses: list[str]) -> list[Components]:
        if_raw, lof_raw = self.raw_scores(payment)
        if_score, lof_score = cdf(self.if_reference, if_raw), cdf(self.lof_reference, lof_raw)
        contexts = self.dbscan.score(context, statuses)
        return [
            Components(
                float(if_raw[i]), float(if_score[i]), float(lof_raw[i]), float(lof_score[i]), contexts[i]
            )
            for i in range(len(contexts))
        ]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, directory / BUNDLE_FILE, compress=3)
        (directory / METADATA_FILE).write_text(json.dumps(self.metadata, indent=2) + "\n")


def load_bundle(directory: Path) -> ModelBundle:
    path = directory / BUNDLE_FILE
    if not path.exists():
        raise BundleError(f"Model bundle not found at {path}. Run `python -m backend.cli train`.")
    bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle):
        raise BundleError(f"{path} holds a {type(bundle).__name__}, not a ModelBundle; retrain it.")
    meta = bundle.metadata
    if (
        meta["feature_version"] != FEATURE_VERSION
        or tuple(meta["payment_features"]) != PAYMENT_FEATURES
        or tuple(meta["context_features"]) != CONTEXT_FEATURES
    ):
        raise BundleError("The model bundle was trained with different feature code; retrain it.")
    return bundle

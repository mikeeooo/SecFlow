"""scikit-learn LocalOutlierFactor adapter in novelty mode, fitted on a behaviour subspace.

`raw_scores` must only be called on rows that were not used for fitting (calibration,
validation, test, or live requests). Training-row diagnostics use `negative_outlier_factor_`
through `training_diagnostics`, never the novelty scoring interface.

Local density is most informative in a few related dimensions, so the detector can be fitted on
a subset of the payment columns (`columns`) and on a seeded subsample of the fitting rows
(`fit_rows`), which bounds the memory and the per-request neighbour search.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from ml.features import PAYMENT_FEATURES

_AMOUNT = ("log_amount", "amount_zscore_30d", "log_amount_to_prior_max_30d")
_VELOCITY = ("requests_1h", "requests_24h", "log_sum_24h", "velocity_ratio_24h")
_BENEFICIARY = (
    "new_recipient",
    "recipient_senders_24h",
    "log_recipient_unique_senders_7d",
    "log_recipient_inbound_24h",
    "pair_requests_24h",
    "recipient_new_global",
)

# Candidate behaviour subspaces searched during training (selected on validation AP).
LOF_SUBSPACES: dict[str, tuple[str, ...]] = {
    "amount": _AMOUNT,
    "amount_velocity": _AMOUNT + _VELOCITY,
    "amount_velocity_beneficiary": _AMOUNT + _VELOCITY + _BENEFICIARY,
    "full": PAYMENT_FEATURES,
}


def subspace_columns(name: str) -> tuple[int, ...]:
    """Column indices of a named subspace within the payment feature vector."""
    return tuple(PAYMENT_FEATURES.index(feature) for feature in LOF_SUBSPACES[name])


class LocalOutlierFactorDetector:
    def __init__(
        self,
        n_neighbors: int,
        columns: tuple[int, ...] | None = None,
        fit_rows: int | None = None,
        random_state: int = 42,
    ) -> None:
        self.columns = tuple(columns) if columns is not None else None
        self.fit_rows = fit_rows
        self.random_state = random_state
        self.params = {"n_neighbors": n_neighbors, "novelty": True, "columns": self.columns, "fit_rows": fit_rows}
        self.model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, n_jobs=1)
        self.n_fit_rows = 0

    def _select(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix)
        return matrix[:, list(self.columns)] if self.columns is not None else matrix

    def fit(self, matrix: np.ndarray) -> LocalOutlierFactorDetector:
        rows = self._select(matrix)
        if self.fit_rows is not None and self.fit_rows < len(rows):
            rng = np.random.default_rng(self.random_state)
            rows = rows[np.sort(rng.choice(len(rows), self.fit_rows, replace=False))]
        self.model.fit(rows)
        self.n_fit_rows = int(rows.shape[0])
        return self

    def raw_scores(self, matrix: np.ndarray) -> np.ndarray:
        # score_samples returns the negative LOF of unseen rows; invert so higher = more anomalous.
        return -self.model.score_samples(self._select(matrix))

    def training_diagnostics(self) -> dict[str, float]:
        lof = -self.model.negative_outlier_factor_
        return {
            "fit_rows": float(self.n_fit_rows),
            "training_lof_median": float(np.median(lof)),
            "training_lof_p99": float(np.quantile(lof, 0.99)),
        }

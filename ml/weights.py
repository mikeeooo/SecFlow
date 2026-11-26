"""Learn the combination weights of the risk index by exhaustive search on the simplex.

The formula stays `risk = 100 × Σ wᵢ · componentᵢ` with `wᵢ ≥ 0` and `Σ wᵢ = 1`, so every
assessment still decomposes into per-component contributions. Candidates lie on a lattice with
`step` spacing; each is scored by average precision on labelled rows (the calibration period in
training). Ties go to the largest rules weight, then IsolationForest, then DBSCAN, so retraining
reproduces the same weights.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import product
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

from ml.policy import COMPONENTS

DEFAULT_MINIMUMS: Mapping[str, float] = {"lof": 0.05}


def simplex_grid(step: float = 0.05, minimums: Mapping[str, float] = DEFAULT_MINIMUMS) -> list[dict[str, float]]:
    """Every weight vector on the `step` lattice that sums to 1 and respects per-component minimums."""
    units = round(1 / step)
    floors = [round(minimums.get(name, 0.0) / step) for name in COMPONENTS]
    grid = []
    for head in product(*(range(f, units + 1) for f in floors[:-1])):
        last = units - sum(head)
        if last < floors[-1]:
            continue
        counts = (*head, last)
        grid.append({name: round(c * step, 2) for name, c in zip(COMPONENTS, counts, strict=True)})
    return grid


def search_weights(
    components: np.ndarray,
    y: np.ndarray,
    step: float = 0.05,
    minimums: Mapping[str, float] = DEFAULT_MINIMUMS,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Pick the weights maximising average precision of `components @ w` against `y`.

    `components` has shape (n, 4) in `COMPONENTS` order, each column in [0, 1].
    """
    components = np.asarray(components, dtype=np.float64)
    y = np.asarray(y, dtype=bool)
    if components.ndim != 2 or components.shape[1] != len(COMPONENTS) or len(components) != len(y):
        raise ValueError("components must have shape (len(y), 4)")
    if y.all() or not y.any():
        raise ValueError("Weight search needs both fraud and legitimate rows")
    rows = []
    for weights in simplex_grid(step, minimums):
        vector = np.array([weights[name] for name in COMPONENTS])
        ap = float(average_precision_score(y, components @ vector))
        rows.append({**weights, "average_precision": ap})
    rows.sort(
        key=lambda r: (r["average_precision"], r["rules"], r["isolation_forest"], r["dbscan"]), reverse=True
    )
    best = {name: rows[0][name] for name in COMPONENTS}
    report = {
        "candidates": len(rows),
        "step": step,
        "minimums": dict(minimums),
        "best_average_precision": rows[0]["average_precision"],
        "top": rows[:10],
    }
    return best, report

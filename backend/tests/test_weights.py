from __future__ import annotations

import numpy as np
import pytest

from ml.policy import COMPONENTS
from ml.weights import search_weights, simplex_grid

RNG = np.random.default_rng(11)


def test_simplex_grid_size_and_constraints() -> None:
    grid = simplex_grid()
    assert len(grid) == 1540
    assert all(set(w) == set(COMPONENTS) for w in grid)
    assert all(abs(sum(w.values()) - 1) < 1e-9 and w["lof"] >= 0.05 - 1e-9 for w in grid)
    assert all(min(w.values()) >= 0 for w in grid)
    assert len({tuple(w[c] for c in COMPONENTS) for w in grid}) == 1540


def test_search_weights_prefers_informative_component() -> None:
    y = RNG.random(2_000) < 0.05
    noise = RNG.random((2_000, 4))
    noise[:, COMPONENTS.index("rules")] = y.astype(float)
    weights, report = search_weights(noise, y)
    assert weights["rules"] == pytest.approx(0.95)
    assert weights["lof"] == pytest.approx(0.05)
    assert report["candidates"] == 1540
    assert report["best_average_precision"] == pytest.approx(1.0)
    assert len(report["top"]) == 10


def test_search_weights_tie_break_is_deterministic() -> None:
    y = RNG.random(500) < 0.1
    column = RNG.random(500)
    identical = np.column_stack([column] * 4)
    first, _ = search_weights(identical, y)
    second, _ = search_weights(identical, y)
    assert first == second
    assert first["rules"] == pytest.approx(0.95)  # every candidate ties; largest rules wins


def test_search_weights_rejects_single_class_labels() -> None:
    with pytest.raises(ValueError):
        search_weights(RNG.random((100, 4)), np.zeros(100, dtype=bool))

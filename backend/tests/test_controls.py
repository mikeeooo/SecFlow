"""Controls on the committed model and test.json: shuffled labels score at chance, and the combined risk
score is not significantly worse than its best single component. The second bar replaced "beats every
baseline with non-overlapping intervals", which the combined score fails (docs/AUDIT.md, F01).

The scores come from `scripts/experiments.py`, which runs every payment through `ml.assess.Assessor`
with the committed bundle. Nothing is retrained.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from backend.tests.conftest import load_script

experiments = load_script("experiments")

COMBINED = "combined"


@pytest.fixture(scope="module")
def scores() -> dict[str, Any]:
    data = experiments.assessed_partitions()
    weights = experiments.as_vector(dict(data["policy"].weights))
    out = {}
    for part in ("validation", "test"):
        components = data[part]["components"]
        out[part] = {
            COMBINED: components @ weights,
            **{name: components[:, i] for i, name in enumerate(experiments.COMPONENTS)},
        }
        out[f"y_{part}"] = data[part]["y"]
    return out


def test_shuffled_test_labels_score_at_the_fraud_prevalence(scores) -> None:
    y, combined = scores["y_test"], scores["test"][COMBINED]
    prevalence = y.mean()
    rng = np.random.default_rng(0)
    shuffled = np.array([experiments.Ranked(combined, rng.permutation(y)).ap() for _ in range(50)])

    assert experiments.Ranked(combined, y).ap() > 20 * prevalence
    assert 0.8 * prevalence < shuffled.mean() < 1.25 * prevalence
    assert shuffled.max() < 1.5 * prevalence


def test_combined_score_is_not_significantly_worse_than_the_best_single_component(scores) -> None:
    y_val, y_test = scores["y_validation"], scores["y_test"]
    components = [name for name in scores["test"] if name != COMBINED]
    # The best component is chosen on validation, so the test labels do not pick it.
    best = max(components, key=lambda name: experiments.Ranked(scores["validation"][name], y_val).ap())
    combined = experiments.Ranked(scores["test"][COMBINED], y_test)
    single = experiments.Ranked(scores["test"][best], y_test)
    boot = experiments.bootstrap_weights(len(y_test))
    diff = np.array([combined.ap(w) - single.ap(w) for w in boot])
    low, high = np.percentile(diff, [2.5, 97.5])

    # Pin the reported result: on test, rules alone match the combined score.
    assert best == "rules"
    assert (round(combined.ap(), 3), round(single.ap(), 3)) == (0.668, 0.675)
    assert (round(low, 3), round(high, 3)) == (-0.023, 0.009)
    assert high >= 0, f"combined minus {best}: AP difference 95% CI [{low:.3f}, {high:.3f}]"

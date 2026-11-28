"""Audit experiments on the committed model bundle and storage/*.json. Nothing is retrained or written
outside docs/audit/.

    uv run python scripts/experiments.py            # writes docs/audit/ml_experiments.json, prints markdown
    uv run python scripts/experiments.py --check    # exit 1 unless the results match that file

a. The combined risk score against each component alone on test.json: average precision (PR-AUC) and
   recall at a fixed false-positive rate (1% and 5%, threshold set on validation.json), each with a
   payment-level bootstrap 95% interval (1,000 resamples, seed 0). The combined-against-best comparison
   is repeated with campaign-level resamples, because fraud arrives in campaigns.
b. Leave-one-out ablation: drop a component, re-search the weights on validation (simplex grid, step
   0.05, maximise validation AP), evaluate on test.
c. Shuffled-label controls: (i) shuffled test labels, (ii) shuffled validation labels driving the
   weight and threshold searches, scored against the true test labels.
d. Permutation importance of IsolationForest feature groups on test AP.

Every test payment goes through `ml.assess.Assessor`, as in `ml.evaluation.run_evaluation`, so the
component scores here are the ones the app and the evaluation report use.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["SECFLOW_AUTOSTART_STREAM"] = "false"

from config.settings import load_settings
from datagen.dataset import load_labels, load_partitions
from ml.assess import Assessor, evidence_store
from ml.features import PAYMENT_FEATURES
from ml.history import HistoryIndex
from ml.model import load_bundle
from ml.policy import COMPONENTS, decision_codes, load_operating_point, load_policy
from ml.training import select_thresholds

OUT = ROOT / "docs" / "audit" / "ml_experiments.json"
SEED = 0
BOOTSTRAP = 1_000
FPRS = (0.01, 0.05)
GRID_STEP = 0.05
SHUFFLE_REPEATS_TEST = 200
SHUFFLE_REPEATS_VALIDATION = 10
PERMUTATION_REPEATS = 20
# Feature groups for the IsolationForest permutation test. Every payment feature is in exactly one group.
FEATURE_GROUPS = {
    "amount": ("log_amount", "amount_zscore_30d", "log_amount_to_prior_max_30d"),
    "velocity": (
        "requests_1h",
        "requests_24h",
        "log_sum_24h",
        "velocity_ratio_24h",
        "unique_recipients_24h",
        "log_requests_30d",
    ),
    "beneficiary": (
        "new_recipient",
        "recipient_senders_24h",
        "log_recipient_history_days",
        "log_recipient_unique_senders_7d",
        "log_recipient_inbound_24h",
        "pair_requests_24h",
        "recipient_new_global",
        "log_incoming_24h",
    ),
    "device_auth": ("log_device_age_hours", "auth_failures_24h", "foreign_ip", "weak_auth"),
    "mia_specific": (
        "amount_to_tx_cap_ratio",
        "daily_cumulative_to_cap_ratio",
        "near_cap_count_24h",
        "alias_rebound_age_hours",
        "is_request_to_pay",
        "qr_merchant_mismatch",
        "scheme_novelty",
    ),
    "time": ("hour_deviation", "log_hours_since_last_request", "log_account_age_days"),
}


# --------------------------------------------------------------------------- metrics


def ap_sorted(y: np.ndarray, w: np.ndarray, ends: np.ndarray) -> float:
    """Weighted average precision for rows already sorted by descending score.

    `ends` holds the last index of each run of tied scores, so ties count as one threshold, as in
    sklearn. Bootstrap resamples are integer weights on the same sorted rows, which avoids a sort per
    resample.
    """
    tp = np.cumsum(w * y)[ends]
    fp = np.cumsum(w * (1 - y))[ends]
    if tp[-1] == 0:
        return float("nan")
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    recall = tp / tp[-1]
    return float(np.sum(np.diff(recall, prepend=0.0) * precision))


class Ranked:
    """A score vector sorted once, for fast weighted AP and recall."""

    def __init__(self, scores: np.ndarray, y: np.ndarray) -> None:
        order = np.argsort(-scores, kind="stable")
        self.scores, self.y = scores[order], y[order].astype(float)
        self.order = order
        self.ends = np.flatnonzero(np.r_[self.scores[1:] != self.scores[:-1], True])

    def ap(self, weights: np.ndarray | None = None) -> float:
        w = np.ones_like(self.y) if weights is None else weights[self.order]
        return ap_sorted(self.y, w, self.ends)


def fpr_threshold(scores: np.ndarray, y: np.ndarray, fpr: float) -> float:
    """Lowest cut t with share(negative scores > t) <= fpr. Ties at t stay unflagged."""
    negatives = np.sort(scores[~y])[::-1]
    k = int(np.floor(fpr * len(negatives)))
    return float(negatives[k]) if k < len(negatives) else float("-inf")


def flagged_rates(
    scores: np.ndarray, y: np.ndarray, t: float, w: np.ndarray | None = None
) -> tuple[float, float]:
    w = np.ones(len(y)) if w is None else w
    flag = scores > t
    return float((w * (flag & y)).sum() / (w * y).sum()), float((w * (flag & ~y)).sum() / (w * ~y).sum())


def ci(values: np.ndarray) -> list[float]:
    return [round(float(np.nanpercentile(values, 2.5)), 6), round(float(np.nanpercentile(values, 97.5)), 6)]


def bootstrap_weights(n: int) -> np.ndarray:
    """BOOTSTRAP x n multiplicity matrix of payment-level resamples (seed 0)."""
    rng = np.random.default_rng(SEED)
    return np.stack([np.bincount(rng.integers(0, n, n), minlength=n).astype(float) for _ in range(BOOTSTRAP)])


def campaign_bootstrap_weights(campaigns: np.ndarray) -> np.ndarray:
    """BOOTSTRAP x n multiplicities from resampling whole campaigns (seed 0).

    A payment outside every campaign is a group of its own.
    """
    _, group = np.unique(campaigns, return_inverse=True)
    k = int(group.max()) + 1
    rng = np.random.default_rng(SEED)
    return np.stack(
        [np.bincount(rng.integers(0, k, k), minlength=k)[group].astype(float) for _ in range(BOOTSTRAP)]
    )


def compare(combined: dict[str, Any], best: dict[str, Any]) -> dict[str, Any]:
    """Intervals of two evaluate_score rows and the paired bootstrap difference, combined minus best."""
    comparison = {}
    for metric, key in (("ap", "test_ap_95ci"), *((f"recall@{f}", f"recall_at_fpr_{f}") for f in FPRS)):
        c_ci = combined[key] if metric == "ap" else combined[key]["test_recall_95ci"]
        b_ci = best[key] if metric == "ap" else best[key]["test_recall_95ci"]
        diff = combined["_draws"][metric] - best["_draws"][metric]
        comparison[metric] = {
            "combined_95ci": c_ci,
            "best_single_95ci": b_ci,
            "cis_do_not_overlap_and_combined_higher": bool(c_ci[0] > b_ci[1]),
            "paired_difference_mean": round(float(np.nanmean(diff)), 6),
            "paired_difference_95ci": ci(diff),
        }
    return comparison


def evaluate_score(
    name: str, val: np.ndarray, test: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, boot: np.ndarray
) -> dict[str, Any]:
    """Point estimates and bootstrap draws for AP and recall at each FPR."""
    ranked = Ranked(test, y_test)
    draws = {"ap": np.array([ranked.ap(w) for w in boot])}
    row: dict[str, Any] = {
        "score": name,
        "validation_ap": round(float(average_precision_score(y_val, val)), 6),
        "test_ap": round(ranked.ap(), 6),
        "test_ap_95ci": ci(draws["ap"]),
    }
    for fpr in FPRS:
        t = fpr_threshold(val, y_val, fpr)
        recall, test_fpr = flagged_rates(test, y_test, t)
        draws[f"recall@{fpr}"] = np.array([flagged_rates(test, y_test, t, w)[0] for w in boot])
        row[f"recall_at_fpr_{fpr}"] = {
            "threshold_from_validation": round(t, 6),
            "validation_fpr": round(flagged_rates(val, y_val, t)[1], 6),
            "test_recall": round(recall, 6),
            "test_recall_95ci": ci(draws[f"recall@{fpr}"]),
            "test_fpr": round(test_fpr, 6),
        }
    row["_draws"] = draws
    return row


# --------------------------------------------------------------------------- weight search


def simplex(k: int, step: float = GRID_STEP) -> np.ndarray:
    n = round(1 / step)
    points = [c for c in itertools.product(range(n + 1), repeat=k) if sum(c) == n]
    return np.array(points, dtype=float) / n


def search_weights(
    components: np.ndarray, y: np.ndarray, columns: list[int]
) -> tuple[dict[str, float], float, int]:
    """Weights on the simplex over `columns` with the highest AP on (components, y); first maximum wins."""
    grid = simplex(len(columns))
    scores = [Ranked(components[:, columns] @ w, y).ap() for w in grid]
    best = int(np.argmax(scores))
    weights = {name: 0.0 for name in COMPONENTS}
    for col, value in zip(columns, grid[best], strict=True):
        weights[COMPONENTS[col]] = round(float(value), 2)
    return weights, float(scores[best]), len(grid)


def as_vector(weights: dict[str, float]) -> np.ndarray:
    return np.array([weights[name] for name in COMPONENTS])


# --------------------------------------------------------------------------- data


def assessed_partitions() -> dict[str, Any]:
    settings = load_settings()
    policy = load_policy(settings.policy_path)
    bundle = load_bundle(settings.models_dir)
    point = load_operating_point(settings.models_dir, bundle.version)
    paths = [settings.train_path, settings.validation_path, settings.test_path]
    dataset = load_partitions(paths)
    history, keys = HistoryIndex.from_events(dataset.requests, dataset.auth_events, dataset.payment_events)
    assessor = Assessor(bundle, evidence_store(dataset.security_events, policy), policy, point)
    labels = load_labels(paths)
    out: dict[str, Any] = {"policy": policy, "bundle": bundle, "point": point, "paths": paths}
    for part in ("validation", "test"):
        requests = [r for r in dataset.requests if dataset.partition_of[r.request_id] == part]
        results = assessor.assess_batch(requests, keys, history, dataset.accounts)
        out[part] = {
            "components": np.array(
                [
                    [r.components.if_score, r.components.lof_score, r.components.context.score, r.rules.score]
                    for r in results
                ]
            ),
            "payment": np.array([r.snapshot.payment_vector for r in results]),
            "families": np.array([len(r.rules.corroborating_families) for r in results]),
            "unsafe": np.array([r.policy.assessment_status == "insufficient_history" for r in results]),
            "risk": np.array([r.risk.total for r in results]),
            "y": np.array([bool(labels[r.request.request_id]["is_fraud"]) for r in results]),
            "campaign": np.array(
                [labels[r.request.request_id]["campaign_id"] or r.request.request_id for r in results]
            ),
        }
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- experiments


def main() -> int:
    data = assessed_partitions()
    policy, val, test = data["policy"], data["validation"], data["test"]
    y_val, y_test = val["y"], test["y"]
    fixed = dict(policy.weights)
    boot = bootstrap_weights(len(y_test))

    # Self-checks: the fast AP equals sklearn's, weighted and unweighted, and the Assessor's risk is C @ w.
    probe = test["components"] @ as_vector(fixed)
    assert np.isclose(Ranked(probe, y_test).ap(), average_precision_score(y_test, probe))
    assert np.isclose(
        Ranked(probe, y_test).ap(boot[0]), average_precision_score(y_test, probe, sample_weight=boot[0])
    )
    assert np.allclose(100 * probe, test["risk"])
    # The production threshold search on these validation scores must give the committed operating point.
    point, _ = select_thresholds(val["risk"], val["families"], val["unsafe"], y_val, policy)
    assert (point.verify, point.block) == (data["point"].verify, data["point"].block), point

    # a. Combined vs each component alone.
    singles = {"rules_only": 3, "isolation_forest_only": 0, "lof_only": 1, "dbscan_only": 2}
    val_combined = val["components"] @ as_vector(fixed)
    rows = [evaluate_score("combined_fixed_weights", val_combined, probe, y_val, y_test, boot)]
    rows += [
        evaluate_score(name, val["components"][:, i], test["components"][:, i], y_val, y_test, boot)
        for name, i in singles.items()
    ]
    combined = rows[0]
    best_single = max(rows[1:], key=lambda r: r["validation_ap"])
    comparison = compare(combined, best_single)
    # The same comparison with campaign-level resamples, which keep each campaign's payments together.
    campaign_boot = campaign_bootstrap_weights(test["campaign"])
    best_column = singles[best_single["score"]]
    comparison_campaign = compare(
        evaluate_score("combined_fixed_weights", val_combined, probe, y_val, y_test, campaign_boot),
        evaluate_score(
            best_single["score"],
            val["components"][:, best_column],
            test["components"][:, best_column],
            y_val,
            y_test,
            campaign_boot,
        ),
    )

    # b. Leave-one-out ablation with re-searched weights.
    full_weights, full_val_ap, full_grid = search_weights(val["components"], y_val, [0, 1, 2, 3])
    full = evaluate_score(
        "all_four_researched",
        val["components"] @ as_vector(full_weights),
        test["components"] @ as_vector(full_weights),
        y_val,
        y_test,
        boot,
    )
    ablation = []
    for drop in range(4):
        keep = [i for i in range(4) if i != drop]
        weights, _, grid = search_weights(val["components"], y_val, keep)
        row = evaluate_score(
            f"without_{COMPONENTS[drop]}",
            val["components"] @ as_vector(weights),
            test["components"] @ as_vector(weights),
            y_val,
            y_test,
            boot,
        )
        row["weights"], row["grid_points"] = weights, grid
        for metric in ("ap", "recall@0.05"):
            d = full["_draws"][metric] - row["_draws"][metric]
            point_drop = (
                (full["test_ap"] - row["test_ap"])
                if metric == "ap"
                else (full["recall_at_fpr_0.05"]["test_recall"] - row["recall_at_fpr_0.05"]["test_recall"])
            )
            row[f"drop_{metric}"] = {"point": round(point_drop, 6), "paired_95ci": ci(d)}
        ablation.append(row)

    # c. Shuffled-label controls.
    rng = np.random.default_rng(SEED)
    ranked_test = Ranked(probe, y_test)
    shuffled_test_ap = []
    for _ in range(SHUFFLE_REPEATS_TEST):
        y_perm = rng.permutation(y_test)
        shuffled_test_ap.append(Ranked(probe, y_perm).ap())
    shuffled_val_runs = []
    for repeat in range(SHUFFLE_REPEATS_VALIDATION):
        y_fake = np.random.default_rng(1000 + repeat).permutation(y_val)
        weights, fake_val_ap, _ = search_weights(val["components"], y_fake, [0, 1, 2, 3])
        val_score, test_score = (
            val["components"] @ as_vector(weights),
            test["components"] @ as_vector(weights),
        )
        run: dict[str, Any] = {
            "seed": 1000 + repeat,
            "weights": weights,
            "shuffled_validation_ap": round(fake_val_ap, 6),
            "true_test_ap": round(Ranked(test_score, y_test).ap(), 6),
        }
        for fpr in FPRS:
            run[f"true_test_recall_at_fpr_{fpr}"] = round(
                flagged_rates(test_score, y_test, fpr_threshold(val_score, y_fake, fpr))[0], 6
            )
        try:
            fake_point, _ = select_thresholds(100 * val_score, val["families"], val["unsafe"], y_fake, policy)
            codes = decision_codes(
                100 * test_score,
                test["families"],
                test["unsafe"],
                fake_point.verify,
                fake_point.block,
                policy.min_families,
            )
            run["policy_thresholds"] = {"verify": fake_point.verify, "block": fake_point.block}
            run["true_test_policy_recall_verify_or_block"] = round(
                float(((codes > 0) & y_test).sum() / y_test.sum()), 6
            )
            run["true_test_policy_recall_block_only"] = round(
                float(((codes == 2) & y_test).sum() / y_test.sum()), 6
            )
        except RuntimeError as error:
            run["policy_thresholds"] = f"no feasible pair: {error}"
        shuffled_val_runs.append(run)
    true_codes = decision_codes(
        test["risk"], test["families"], test["unsafe"], point.verify, point.block, policy.min_families
    )
    true_policy = {
        "verify": point.verify,
        "block": point.block,
        "true_test_policy_recall_verify_or_block": round(
            float(((true_codes > 0) & y_test).sum() / y_test.sum()), 6
        ),
        "true_test_policy_recall_block_only": round(
            float(((true_codes == 2) & y_test).sum() / y_test.sum()), 6
        ),
    }

    # d. IsolationForest permutation importance per feature group on test AP (raw IF scores).
    bundle = data["bundle"]
    assert sorted(sum(FEATURE_GROUPS.values(), ())) == sorted(PAYMENT_FEATURES)

    def if_ap(matrix: np.ndarray) -> float:
        raw = -bundle.isolation_forest.score_samples(bundle.preprocessor.transform(matrix))
        return Ranked(raw, y_test).ap()

    base_if_ap = if_ap(test["payment"])
    prng = np.random.default_rng(SEED)
    importance = []
    for group, names in FEATURE_GROUPS.items():
        cols = [PAYMENT_FEATURES.index(n) for n in names]
        drops = []
        for _ in range(PERMUTATION_REPEATS):
            permuted = test["payment"].copy()
            permuted[:, cols] = permuted[prng.permutation(len(permuted))][:, cols]
            drops.append(base_if_ap - if_ap(permuted))
        importance.append(
            {
                "group": group,
                "features": list(names),
                "ap_drop_mean": round(float(np.mean(drops)), 6),
                "ap_drop_std": round(float(np.std(drops)), 6),
                "ap_drop_min": round(float(np.min(drops)), 6),
                "ap_drop_max": round(float(np.max(drops)), 6),
            }
        )

    def public(row: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in row.items() if k != "_draws"}

    report = {
        "inputs": {
            "files_sha256": {p.name: sha256(p) for p in data["paths"]},
            "model_bundle_sha256": sha256(load_settings().models_dir / "secflow_model.joblib"),
            "model_version": bundle.version,
            "policy_version": policy.version,
            "validation": {"payments": len(y_val), "fraud": int(y_val.sum())},
            "test": {
                "payments": len(y_test),
                "fraud": int(y_test.sum()),
                "prevalence": round(float(y_test.mean()), 6),
            },
        },
        "method": {
            "ap": "average precision (sklearn definition, ties grouped), the PR-AUC",
            "recall_at_fpr": "threshold = lowest cut with at most FPR of validation legitimate payments strictly above "
            "it; the same cut is applied to test",
            "bootstrap": f"{BOOTSTRAP} payment-level resamples of test.json with replacement, seed {SEED}, percentile "
            "95% interval. Fraud arrives in campaigns, so payment-level resampling understates the spread.",
            "campaign_bootstrap": f"{BOOTSTRAP} resamples of whole campaigns of test.json, seed {SEED}. A payment "
            f"outside every campaign is its own group. {len(np.unique(test['campaign']))} groups.",
            "best_single": "the component with the highest validation AP",
            "weight_search": f"simplex grid, step {GRID_STEP}, maximise validation AP, first maximum in "
            "lexicographic grid order wins",
            "self_checks": [
                "fast AP == sklearn AP (unweighted and weighted)",
                "Assessor risk == 100 x components @ weights",
                "select_thresholds on these validation scores reproduces ml/models/operating_point.json",
            ],
        },
        "a_combined_vs_single": {
            "rows": [public(r) for r in rows],
            "best_single": best_single["score"],
            "comparison": comparison,
            "comparison_campaign_bootstrap": comparison_campaign,
        },
        "b_ablation": {
            "fixed_weights": fixed,
            "researched_weights": full_weights,
            "researched_validation_ap": round(full_val_ap, 6),
            "grid_points": full_grid,
            "all_four_researched": public(full),
            "rows": [public(r) for r in ablation],
        },
        "c_shuffled_labels": {
            "i_shuffled_test": {
                "repeats": SHUFFLE_REPEATS_TEST,
                "score": "combined_fixed_weights",
                "true_test_ap": round(ranked_test.ap(), 6),
                "prevalence": round(float(y_test.mean()), 6),
                "first_shuffle_ap": round(shuffled_test_ap[0], 6),
                "mean_ap": round(float(np.mean(shuffled_test_ap)), 6),
                "range": [
                    round(float(np.min(shuffled_test_ap)), 6),
                    round(float(np.max(shuffled_test_ap)), 6),
                ],
            },
            "ii_shuffled_validation": {
                "runs": shuffled_val_runs,
                "true_labels_reference": {
                    "weights": full_weights,
                    "true_test_ap": full["test_ap"],
                    **true_policy,
                },
            },
        },
        "d_isolation_forest_permutation": {
            "base_test_ap_raw_scores": round(base_if_ap, 6),
            "repeats": PERMUTATION_REPEATS,
            "groups": importance,
        },
    }
    text = json.dumps(report, indent=2) + "\n"
    print(render_markdown(report))
    if "--check" in sys.argv:
        # Within the tolerance scripts/reproduce.py allows for LOF floats across platforms.
        from reproduce import differences

        diff = differences(json.loads(OUT.read_text()), json.loads(text))
        print("\n".join(diff) or f"Results match {OUT.relative_to(ROOT)}.")
        return 1 if diff else 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    return 0


# --------------------------------------------------------------------------- markdown


def _ci(pair: list[float]) -> str:
    return f"[{pair[0]:.3f}, {pair[1]:.3f}]"


def render_markdown(r: dict[str, Any]) -> str:
    a, b, c, d = (
        r["a_combined_vs_single"],
        r["b_ablation"],
        r["c_shuffled_labels"],
        r["d_isolation_forest_permutation"],
    )
    lines = [
        "### a. Combined risk score against each component alone (test.json)",
        "",
        (
            "| Score | Validation AP | Test AP [95% CI] | Recall @ 1% FPR [95% CI] | Test FPR | "
            "Recall @ 5% FPR [95% CI] | Test FPR |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in a["rows"]:
        r1, r5 = row["recall_at_fpr_0.01"], row["recall_at_fpr_0.05"]
        lines.append(
            f"| {row['score']} | {row['validation_ap']:.3f} | {row['test_ap']:.3f} {_ci(row['test_ap_95ci'])} | "
            f"{r1['test_recall']:.3f} {_ci(r1['test_recall_95ci'])} | {r1['test_fpr']:.4f} | "
            f"{r5['test_recall']:.3f} {_ci(r5['test_recall_95ci'])} | {r5['test_fpr']:.4f} |"
        )
    lines += [
        "",
        f"Best single component by validation AP: `{a['best_single']}`.",
        "",
        (
            "| Metric | Combined 95% CI | Best single 95% CI | Non-overlapping, combined higher | "
            "Paired difference [95% CI] |"
        ),
        "|---|---:|---:|---|---:|",
    ]
    for label, key in (("payment", "comparison"), ("campaign", "comparison_campaign_bootstrap")):
        for metric, row in a[key].items():
            lines.append(
                f"| {metric}, {label} resamples | {_ci(row['combined_95ci'])} | {_ci(row['best_single_95ci'])} | "
                f"{'yes' if row['cis_do_not_overlap_and_combined_higher'] else 'no'} | "
                f"{row['paired_difference_mean']:+.3f} {_ci(row['paired_difference_95ci'])} |"
            )
    w = b["researched_weights"]
    lines += [
        "",
        "### b. Leave-one-out ablation (weights re-searched on validation)",
        "",
        f"Fixed weights in `config/policy.json`: {b['fixed_weights']}.",
        (
            f"Re-searched 4-component weights: {w} (validation AP {b['researched_validation_ap']:.3f}, "
            f"{b['grid_points']} grid points). Test AP {b['all_four_researched']['test_ap']:.3f} "
            f"{_ci(b['all_four_researched']['test_ap_95ci'])}."
        ),
        "",
        (
            "| Dropped | Re-searched weights | Test AP | AP drop [paired 95% CI] | Recall @ 5% FPR | "
            "Recall drop [paired 95% CI] |"
        ),
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in b["rows"]:
        ws = ", ".join(f"{k}={v:.2f}" for k, v in row["weights"].items() if v)
        lines.append(
            f"| {row['score'].removeprefix('without_')} | {ws} | {row['test_ap']:.3f} | "
            f"{row['drop_ap']['point']:+.3f} {_ci(row['drop_ap']['paired_95ci'])} | "
            f"{row['recall_at_fpr_0.05']['test_recall']:.3f} | {row['drop_recall@0.05']['point']:+.3f} "
            f"{_ci(row['drop_recall@0.05']['paired_95ci'])} |"
        )
    i, ii = c["i_shuffled_test"], c["ii_shuffled_validation"]
    lines += [
        "",
        "### c. Shuffled-label controls",
        "",
        (
            f"(i) Combined score, true test AP {i['true_test_ap']:.3f}. Test labels shuffled {i['repeats']} times: "
            f"first shuffle AP {i['first_shuffle_ap']:.4f}, mean {i['mean_ap']:.4f}, range "
            f"[{i['range'][0]:.4f}, {i['range'][1]:.4f}]. Test fraud prevalence {i['prevalence']:.4f}."
        ),
        "",
        "(ii) Validation labels shuffled, weights and thresholds re-searched, scored on the true test labels.",
        "",
        (
            "| Seed | Weights from shuffled validation | True test AP | Recall @ 1% FPR | Recall @ 5% FPR | "
            "Policy Verify/Block | Policy recall (V or B) |"
        ),
        "|---|---|---:|---:|---:|---|---:|",
    ]
    ref = ii["true_labels_reference"]
    lines.append(
        f"| true labels | {', '.join(f'{k}={v:.2f}' for k, v in ref['weights'].items() if v)} | "
        f"{ref['true_test_ap']:.3f} | | | {ref['verify']:g}/{ref['block']:g} (fixed weights) | "
        f"{ref['true_test_policy_recall_verify_or_block']:.3f} |"
    )
    for run in ii["runs"]:
        pt = run["policy_thresholds"]
        pts = f"{pt['verify']:g}/{pt['block']:g}" if isinstance(pt, dict) else "none feasible"
        rec = run.get("true_test_policy_recall_verify_or_block")
        lines.append(
            f"| {run['seed']} | {', '.join(f'{k}={v:.2f}' for k, v in run['weights'].items() if v)} | "
            f"{run['true_test_ap']:.3f} | {run['true_test_recall_at_fpr_0.01']:.3f} | "
            f"{run['true_test_recall_at_fpr_0.05']:.3f} | {pts} | {'n/a' if rec is None else f'{rec:.3f}'} |"
        )
    lines += [
        "",
        (
            f"### d. IsolationForest permutation importance on test AP (base {d['base_test_ap_raw_scores']:.3f}, "
            f"{d['repeats']} permutations per group)"
        ),
        "",
        "| Group | Features | AP drop mean | Std | Min | Max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for g in sorted(d["groups"], key=lambda g: -g["ap_drop_mean"]):
        lines.append(
            f"| {g['group']} | {len(g['features'])} | {g['ap_drop_mean']:+.4f} | {g['ap_drop_std']:.4f} | "
            f"{g['ap_drop_min']:+.4f} | {g['ap_drop_max']:+.4f} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

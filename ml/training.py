"""Training: fit the detectors on train.json, then choose the Verify/Block thresholds on validation.json.

1. Replay train.json and validation.json in time order and build each payment's features from the
   events before it.
2. Fit the preprocessor, IsolationForest, LOF and DBSCAN on train.json. No labels.
3. Score validation.json. Its raw IF and LOF scores become the CDF references. No labels.
4. Open the validation labels and grid-search the Verify/Block pair that flags the most fraud within
   the goals in config/policy.json.

test.json is never read here; `ml.evaluation` opens it once, after everything is frozen.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler

from datagen.dataset import Dataset, file_hashes, load_labels, load_partitions, read_manifest
from ml.features import CONTEXT_FEATURES, FEATURE_VERSION, PAYMENT_FEATURES, build_features
from ml.history import HistoryIndex
from ml.model import LOF_COLUMNS, LOF_FEATURES, DbscanContext, ModelBundle, Preprocessor, cdf
from ml.policy import (
    COMPONENTS,
    OPERATING_POINT_FILE,
    SEGMENTS,
    OperatingPoint,
    Policy,
    assessment_status,
    decision_codes,
    evaluate_rules,
    segment_of,
)

SEED = 42
IF_TREES, IF_SAMPLES = 200, 256  # 256 is the sample size of the original IsolationForest paper
LOF_NEIGHBORS, LOF_FIT_ROWS = 20, 20_000  # a seeded subsample keeps each LOF query fast
DBSCAN_MIN_SAMPLES, DBSCAN_EPS_QUANTILE, DBSCAN_FIT_ROWS = 5, 0.90, 20_000
# A few weightings measured next to the fixed policy weights, for the report and the docs.
WEIGHT_ALTERNATIVES = (
    {"isolation_forest": 0.25, "lof": 0.25, "dbscan": 0.25, "rules": 0.25},
    {"isolation_forest": 0.35, "lof": 0.25, "dbscan": 0.10, "rules": 0.30},
    {"isolation_forest": 0.50, "lof": 0.00, "dbscan": 0.00, "rules": 0.50},
    {"isolation_forest": 1.00, "lof": 0.00, "dbscan": 0.00, "rules": 0.00},
    {"isolation_forest": 0.00, "lof": 0.00, "dbscan": 0.00, "rules": 1.00},
)


def feature_table(dataset: Dataset) -> dict[str, Any]:
    """Features of every payment in time order, each built from the events ordered before it."""
    history, keys = HistoryIndex.from_events(dataset.requests, dataset.auth_events, dataset.payment_events)
    ordered = sorted(dataset.requests, key=lambda r: keys[r.request_id])
    snapshots = [
        build_features(r, keys[r.request_id], history, dataset.accounts.get(r.sender_id)) for r in ordered
    ]
    return {
        "ids": [r.request_id for r in ordered],
        "partition": np.array([dataset.partition_of[r.request_id] for r in ordered]),
        "segment": np.array([segment_of(dataset.accounts.get(r.sender_id)) for r in ordered]),
        "payment": np.array([s.payment_vector for s in snapshots]),
        "context": np.array([s.context_vector for s in snapshots]),
        "snapshots": snapshots,
        "requests": ordered,
    }


def _subsample(rows: np.ndarray, n: int) -> np.ndarray:
    if len(rows) <= n:
        return rows
    return rows[np.sort(np.random.default_rng(SEED).choice(len(rows), n, replace=False))]


def _ap(y: np.ndarray, scores: np.ndarray) -> float:
    return float(average_precision_score(y, scores))


def fit_detectors(payment: np.ndarray, context: np.ndarray, context_ok: np.ndarray) -> dict[str, Any]:
    """Fit the preprocessor and the three detectors on training rows only."""
    preprocessor = Preprocessor().fit(payment)
    x = preprocessor.transform(payment)
    iforest = IsolationForest(n_estimators=IF_TREES, max_samples=IF_SAMPLES, random_state=SEED).fit(x)
    lof_rows = _subsample(x[:, LOF_COLUMNS], LOF_FIT_ROWS)
    lof = LocalOutlierFactor(n_neighbors=LOF_NEIGHBORS, novelty=True).fit(lof_rows)
    # DBSCAN's eps from the k-distance curve: the distance to the 5th neighbour that 90% of profiles stay under.
    profiles = _subsample(context[context_ok], DBSCAN_FIT_ROWS)
    scaled = StandardScaler().fit_transform(profiles)
    k_distance = NearestNeighbors(n_neighbors=DBSCAN_MIN_SAMPLES).fit(scaled).kneighbors(scaled)[0][:, -1]
    eps = float(np.quantile(k_distance, DBSCAN_EPS_QUANTILE))
    dbscan = DbscanContext(eps, DBSCAN_MIN_SAMPLES).fit(profiles)
    return {
        "preprocessor": preprocessor,
        "iforest": iforest,
        "lof": lof,
        "dbscan": dbscan,
        "rows": len(x),
        "lof_rows": len(lof_rows),
        "dbscan_rows": len(profiles),
    }


class ThresholdRow(TypedDict):
    """One Verify/Block pair scored on validation, as the training report stores it."""

    verify_threshold: float
    block_threshold: float
    recall_verify_or_block: float
    recall_block_only: float
    verify_rate: float
    block_rate: float
    legitimate_false_block_rate: float
    block_precision: float | None  # None when the pair blocks nothing
    predicted_blocks: int
    false_blocks: int
    meets_goals: bool


def select_thresholds(
    risk: np.ndarray, families: np.ndarray, unsafe: np.ndarray, y: np.ndarray, policy: Policy
) -> tuple[OperatingPoint, dict[str, Any]]:
    """Grid-search Verify/Block on validation: most fraud flagged within the goals.

    Validation has no brand-new accounts, but the live system does and they are reviewed far more
    often, so part of the Verify budget (`new_account_verify_reserve`) is kept free for them.
    """
    goals = policy.goals
    verify_budget = goals["max_verify_rate"] - policy.new_account_verify_reserve
    search = policy.threshold_search
    grid = {name: np.arange(s["start"], s["stop"] + 1e-9, s["step"]) for name, s in search.items()}
    candidates: list[ThresholdRow] = []
    frontier: dict[float, ThresholdRow] = {}
    for verify in grid["verify"]:
        for block in grid["block"]:
            if block <= verify:
                continue
            codes = decision_codes(risk, families, unsafe, verify, block, policy.min_families)
            blocked = codes == 2
            n_blocks, true_blocks = int(blocked.sum()), int((blocked & y).sum())
            precision = true_blocks / n_blocks if n_blocks else None
            verify_rate = float((codes == 1).mean())
            false_block_rate = (n_blocks - true_blocks) / float((~y).sum())
            row: ThresholdRow = {
                "verify_threshold": float(verify),
                "block_threshold": float(block),
                "recall_verify_or_block": float(((codes > 0) & y).sum() / y.sum()),
                "recall_block_only": true_blocks / float(y.sum()),
                "verify_rate": verify_rate,
                "block_rate": float(blocked.mean()),
                "legitimate_false_block_rate": false_block_rate,
                "block_precision": precision,
                "predicted_blocks": n_blocks,
                "false_blocks": n_blocks - true_blocks,
                "meets_goals": bool(
                    verify_rate <= verify_budget
                    and false_block_rate <= goals["max_legitimate_false_block_rate"]
                    and (precision is None or precision >= goals["min_block_precision"])
                ),
            }
            candidates.append(row)
            best = frontier.get(row["verify_threshold"])
            if row["meets_goals"] and (best is None or row["recall_block_only"] > best["recall_block_only"]):
                frontier[row["verify_threshold"]] = row
    feasible = [c for c in candidates if c["meets_goals"]]
    if not feasible:
        raise RuntimeError("No Verify/Block pair meets the goals on validation")
    chosen = max(
        feasible,
        key=lambda c: (
            c["recall_verify_or_block"],
            c["recall_block_only"],
            -c["verify_rate"],
            -c["block_threshold"],
        ),
    )
    digest = hashlib.sha256(json.dumps(chosen, sort_keys=True).encode()).hexdigest()[:8]
    point = OperatingPoint(chosen["verify_threshold"], chosen["block_threshold"], "validated", f"op-{digest}")
    initial = next(
        (
            c
            for c in candidates
            if (c["verify_threshold"], c["block_threshold"]) == (policy.initial_verify, policy.initial_block)
        ),
        None,
    )
    return point, {
        "chosen": chosen,
        "candidates_evaluated": len(candidates),
        "feasible_candidates": len(feasible),
        "verify_budget": verify_budget,
        "frontier": [frontier[k] for k in sorted(frontier)],
        "initial_candidate": initial,
    }


def train_models(
    train_path: Path,
    validation_path: Path,
    models_dir: Path,
    policy: Policy,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    table = feature_table(load_partitions([train_path, validation_path]))
    is_train, is_val = table["partition"] == "train", table["partition"] == "validation"
    context_ok = np.array([s.context_status == "ok" for s in table["snapshots"]])
    fitted = fit_detectors(table["payment"][is_train], table["context"][is_train], context_ok[is_train])

    # Score validation; its raw IF and LOF scores are the CDF references (no labels).
    val_index = np.flatnonzero(is_val)
    snapshots = [table["snapshots"][i] for i in val_index]
    x_val = fitted["preprocessor"].transform(table["payment"][is_val])
    if_raw = -fitted["iforest"].score_samples(x_val)
    lof_raw = -fitted["lof"].score_samples(x_val[:, LOF_COLUMNS])
    contexts = fitted["dbscan"].score(table["context"][is_val], [s.context_status for s in snapshots])
    rules = [evaluate_rules(s.facts, s.history_status, policy) for s in snapshots]
    components = np.column_stack(
        [
            cdf(np.sort(if_raw), if_raw),
            cdf(np.sort(lof_raw), lof_raw),
            [c.score for c in contexts],
            [r.score for r in rules],
        ]
    )
    families = np.array([len(r.corroborating_families) for r in rules])
    requests = [table["requests"][i] for i in val_index]
    unsafe = np.array(
        [
            assessment_status(s.history_status, r.amount, r.auth_method, policy)[0] == "insufficient_history"
            for s, r in zip(snapshots, requests, strict=True)
        ]
    )

    # Labels are opened only now, for the threshold search and the report.
    labels = load_labels([validation_path])
    y = np.array([bool(labels[r.request_id]["is_fraud"]) for r in requests])
    weights = dict(policy.weights)
    risk = 100.0 * components @ np.array([weights[c] for c in COMPONENTS])
    point, thresholds = select_thresholds(risk, families, unsafe, y, policy)

    hashes = file_hashes([train_path, validation_path])
    manifest = read_manifest(manifest_path) if manifest_path is not None and manifest_path.exists() else {}
    selection = {
        "isolation_forest": {"n_estimators": IF_TREES, "max_samples": IF_SAMPLES},
        "lof": {"subspace": "amount", "n_neighbors": LOF_NEIGHBORS, "fit_rows_requested": LOF_FIT_ROWS},
        "dbscan": {
            "min_samples": DBSCAN_MIN_SAMPLES,
            "eps": round(fitted["dbscan"].eps, 5),
            "eps_quantile": DBSCAN_EPS_QUANTILE,
        },
        "weights": weights,
    }
    material = json.dumps(
        {
            "files": hashes,
            "selection": selection,
            "features": FEATURE_VERSION,
            "sklearn": sklearn.__version__,
        },
        sort_keys=True,
    )
    segment_thresholds = {name: {"verify": point.verify, "block": point.block} for name in SEGMENTS}
    dbscan = fitted["dbscan"]
    metadata = {
        "model_version": f"secflow-sk-3.0-{hashlib.sha256(material.encode()).hexdigest()[:10]}",
        "feature_version": FEATURE_VERSION,
        "payment_features": list(PAYMENT_FEATURES),
        "context_features": list(CONTEXT_FEATURES),
        "lof_features": list(LOF_FEATURES),
        "sklearn_version": sklearn.__version__,
        "random_seed": SEED,
        "training_dataset": {
            "files": ["storage/train.json", "storage/validation.json"],
            "seed": manifest.get("seed"),
            "generator_version": manifest.get("generator_version"),
            "file_hashes": hashes,
            "splits": {
                "train": "train.json, days 1-70: preprocessing, IsolationForest, LOF, DBSCAN (no labels)",
                "validation_mapping": "validation.json, days 71-85: IF and LOF score references (no labels)",
                "validation_selection": "validation.json, days 71-85: Verify/Block threshold search (labels)",
            },
        },
        "fit_rows": {
            "preprocessing": fitted["rows"],
            "isolation_forest": fitted["rows"],
            "lof": fitted["lof_rows"],
            "dbscan": fitted["dbscan_rows"],
            "score_mapping_reference": int(is_val.sum()),
            "weights_reference": int(is_val.sum()),
        },
        "holdout_accounts": 0,
        "selected_parameters": selection,
        "combination_weights": weights,
        "segment_thresholds": segment_thresholds,
        "dbscan_context": {
            **dbscan.summary,
            "eps": dbscan.eps,
            "min_samples": dbscan.min_samples,
            "metric": dbscan.metric,
            "adapter": (
                "frozen-core: within eps of a training core sample -> 0, beyond -> min(1, (d - eps) / eps), "
                "insufficient history -> 0"
            ),
        },
        "score_mapping": "empirical CDF of inverted raw scores over the validation payments (days 71-85)",
        "policy_version": policy.version,
    }
    bundle = ModelBundle(
        fitted["preprocessor"],
        fitted["iforest"],
        fitted["lof"],
        dbscan,
        np.sort(if_raw),
        np.sort(lof_raw),
        metadata,
    )
    bundle.save(models_dir)
    (models_dir / OPERATING_POINT_FILE).write_text(
        json.dumps(
            {
                "verify_threshold": point.verify,
                "block_threshold": point.block,
                "source": point.source,
                "version": point.version,
                "model_version": bundle.version,
                "segments": segment_thresholds,
            },
            indent=2,
        )
        + "\n"
    )

    def ap_of(w: dict[str, float]) -> float:
        return _ap(y, components @ np.array([w[c] for c in COMPONENTS]))

    weightings = sorted(
        ({**w, "average_precision": ap_of(w)} for w in (weights, *WEIGHT_ALTERNATIVES)),
        key=lambda r: -r["average_precision"],
    )
    outside = np.array([c.status == "outside_context" for c in contexts])
    fraud_outside, legit_outside = float(outside[y].mean()), float(outside[~y].mean())
    report = {
        "policy_version": policy.version,
        "operating_point": {
            "verify": point.verify,
            "block": point.block,
            "source": point.source,
            "version": point.version,
            "weights": weights,
            "segments": segment_thresholds,
        },
        "selected_parameters": selection,
        "search": {
            "isolation_forest": [
                {
                    "n_estimators": IF_TREES,
                    "max_samples": IF_SAMPLES,
                    "validation_average_precision": _ap(y, components[:, 0]),
                }
            ],
            "lof": [
                {
                    "subspace": "amount",
                    "n_neighbors": LOF_NEIGHBORS,
                    "fit_rows_requested": LOF_FIT_ROWS,
                    "fit_rows": float(fitted["lof_rows"]),
                    "validation_average_precision": _ap(y, components[:, 1]),
                }
            ],
            "dbscan": [
                {
                    "min_samples": DBSCAN_MIN_SAMPLES,
                    "eps_quantile": DBSCAN_EPS_QUANTILE,
                    "eps": round(dbscan.eps, 5),
                    **dbscan.summary,
                    "validation_fraud_outside_rate": round(fraud_outside, 4),
                    "validation_legitimate_outside_rate": round(legit_outside, 4),
                    "validation_separation": round(fraud_outside - legit_outside, 4),
                    "validation_average_precision": _ap(y, components[:, 2]),
                }
            ],
        },
        "weights_search": {
            "partition": "validation.json, days 71-85: fixed policy weights next to a few alternatives",
            "candidates": len(weightings),
            "best_average_precision": weightings[0]["average_precision"],
            "selected": weights,
            "top": weightings,
            "rows": len(y),
            "fraud": int(y.sum()),
        },
        "threshold_selection": thresholds,
        "segment_thresholds": {
            name: {
                "requests": int((table["segment"][is_val] == name).sum()),
                "fraud": int(y[table["segment"][is_val] == name].sum()),
                "fallback": True,
                "reason": "one threshold pair for every segment",
            }
            for name in SEGMENTS
        },
        "validation_component_average_precision": {
            **{name: _ap(y, components[:, i]) for i, name in enumerate(COMPONENTS)},
            "combined": _ap(y, risk),
        },
    }
    (models_dir / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report

"""Offline training: chronological fitting, bounded parameter search, and operating-point selection.

Data use per partition (labels are only opened where stated):

* train (days 1-60): fit preprocessing, IsolationForest, LOF (novelty) and DBSCAN. No labels.
* calibration (61-70): fit IF/LOF empirical score mappings (no labels), then use the labels to learn
  the combination weights of the risk index.
* validation (71-80): labels used to choose estimator parameters and per-segment policy thresholds.

Forty established holdout accounts (seeded, chosen without labels) are excluded from every fitting and
selection row so the test period can report an account-disjoint subset.
* test (81-100): not touched here. `ml.evaluation` opens it once, after freezing.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ml.algorithms.dbscan_context import DBSCANContext, EmptyCoreError
from ml.algorithms.isolation_forest import IsolationForestDetector
from ml.algorithms.local_outlier_factor import LOF_SUBSPACES, LocalOutlierFactorDetector, subspace_columns
from ml.algorithms.preprocessing import EmpiricalScoreMapping, FeaturePreprocessor
from ml.assess import build_evidence_store
from ml.features import CONTEXT_FEATURES, FEATURE_VERSION, PAYMENT_FEATURES
from ml.model import ModelBundle, sklearn_version
from ml.policy import (
    SEGMENTS,
    OperatingPoint,
    Policy,
    assessment_status,
    combine_arrays,
    decide_arrays,
    evaluate_rules,
    segment_of,
)
from ml.replay import ReplayResult, replay
from ml.weights import search_weights
from storage.dataset import Dataset, load_dataset, load_labels, load_splits

SEED = 42
IF_GRID = ({"n_estimators": 100, "max_samples": 256}, {"n_estimators": 100, "max_samples": 512},
           {"n_estimators": 200, "max_samples": 256}, {"n_estimators": 200, "max_samples": 512})
LOF_GRID = (20, 35, 50)
LOF_FIT_ROWS = (20_000, None)  # seeded subsample or every fitting row
HOLDOUT_ACCOUNTS = 40
HOLDOUT_SEED = 4242
DBSCAN_MIN_SAMPLES = (5, 10, 20)
DBSCAN_EPS_QUANTILES = (0.90, 0.95, 0.98)
DBSCAN_FIT_ROWS = 20_000
MODEL_FAMILY = "secflow-sk"


@dataclass(frozen=True)
class PartitionView:
    rows: np.ndarray  # boolean mask over replay rows

    def take(self, matrix: np.ndarray) -> np.ndarray:
        return matrix[self.rows]


def _partition_masks(result: ReplayResult, splits: dict[str, str]) -> dict[str, PartitionView]:
    part = np.array([splits[r] for r in result.request_ids])
    return {name: PartitionView(part == name) for name in ("train", "calibration", "validation", "test")}


def _ap(y: np.ndarray, scores: np.ndarray) -> float | None:
    return float(average_precision_score(y, scores)) if 0 < y.sum() < len(y) else None


def rule_scores_for(result: ReplayResult, mask: np.ndarray, policy: Policy) -> tuple[np.ndarray, np.ndarray]:
    """Rules score and number of distinct corroborating families for the masked rows."""
    ids = np.array(result.request_ids)[mask]
    statuses = np.array(result.history_status)[mask]
    scores = np.empty(len(ids))
    families = np.empty(len(ids), dtype=np.int64)
    for i, (rid, status) in enumerate(zip(ids, statuses, strict=True)):
        outcome = evaluate_rules(result.facts[rid], status, policy)
        scores[i] = outcome.score
        families[i] = len(outcome.corroborating_families)
    return scores, families


def unsafe_mask_for(result: ReplayResult, mask: np.ndarray, policy: Policy) -> np.ndarray:
    ids = np.array(result.request_ids)[mask]
    statuses = np.array(result.history_status)[mask]
    out = np.zeros(len(ids), dtype=bool)
    for i, (rid, status) in enumerate(zip(ids, statuses, strict=True)):
        facts = result.facts[rid]
        state, _ = assessment_status(status, facts["amount"], facts["auth_method"], True, True, policy)
        out[i] = state in {"insufficient_history", "model_unavailable", "invalid_features"}
    return out


def _search_isolation_forest(x_train, x_cal, x_val, y_val) -> tuple[dict, list[dict]]:
    records, best = [], None
    for params in IF_GRID:
        started = time.perf_counter()
        detector = IsolationForestDetector(random_state=SEED, **params).fit(x_train)
        mapping = EmpiricalScoreMapping().fit(detector.raw_scores(x_cal))
        val = mapping.transform(detector.raw_scores(x_val))
        record = {**params, "validation_average_precision": _ap(y_val, val),
                  "fit_seconds": round(time.perf_counter() - started, 3)}
        records.append(record)
        if best is None or (record["validation_average_precision"] or 0) > (best[0]["validation_average_precision"] or 0):
            best = (record, detector, mapping)
    assert best is not None
    return {"record": best[0], "detector": best[1], "mapping": best[2]}, records


def holdout_accounts(dataset: Dataset, cold_start_ids: set[str]) -> frozenset[str]:
    """Seeded choice of established accounts kept out of every fitting and selection row (no labels read)."""
    candidates = sorted(a for a in dataset.accounts if a not in cold_start_ids)
    rng = np.random.default_rng(HOLDOUT_SEED)
    return frozenset(rng.choice(candidates, size=min(HOLDOUT_ACCOUNTS, len(candidates)), replace=False).tolist())


def _search_lof(x_train, x_cal, x_val, y_val) -> tuple[dict, list[dict]]:
    """Behaviour subspace x n_neighbors x fit rows; best validation AP, ties to fewer fit rows."""
    records, best = [], None
    for name in LOF_SUBSPACES:
        columns = subspace_columns(name)
        for k in LOF_GRID:
            for fit_rows in LOF_FIT_ROWS:
                started = time.perf_counter()
                detector = LocalOutlierFactorDetector(k, columns=columns, fit_rows=fit_rows, random_state=SEED)
                detector.fit(x_train)
                # novelty scoring is only applied to unseen calibration/validation rows
                mapping = EmpiricalScoreMapping().fit(detector.raw_scores(x_cal))
                val = mapping.transform(detector.raw_scores(x_val))
                record = {"subspace": name, "n_neighbors": k, "fit_rows_requested": fit_rows or "all",
                          "validation_average_precision": _ap(y_val, val), **detector.training_diagnostics(),
                          "fit_seconds": round(time.perf_counter() - started, 3)}
                records.append(record)
                rank = (record["validation_average_precision"] or 0.0, -detector.n_fit_rows)
                if best is None or rank > best[3]:
                    best = (record, detector, mapping, rank)
    assert best is not None
    return {"record": best[0], "detector": best[1], "mapping": best[2]}, records


def _search_dbscan(profiles_train, ctx_val, status_val, y_val) -> tuple[dict, list[dict]]:
    rng = np.random.default_rng(SEED)
    rows = profiles_train
    if len(rows) > DBSCAN_FIT_ROWS:
        rows = rows[np.sort(rng.choice(len(rows), DBSCAN_FIT_ROWS, replace=False))]
    scaled = StandardScaler().fit_transform(rows)
    probe = scaled[rng.choice(len(scaled), min(5_000, len(scaled)), replace=False)]
    records, best = [], None
    for min_samples in DBSCAN_MIN_SAMPLES:
        distances, _ = NearestNeighbors(n_neighbors=min_samples).fit(scaled).kneighbors(probe)
        for quantile in DBSCAN_EPS_QUANTILES:
            eps = float(np.quantile(distances[:, -1], quantile))
            record: dict[str, Any] = {"min_samples": min_samples, "eps_quantile": quantile, "eps": round(eps, 5)}
            try:
                context = DBSCANContext(eps=eps, min_samples=min_samples).fit(rows)
            except EmptyCoreError as error:
                record["error"] = str(error)
                records.append(record)
                continue
            scored = np.array([c.score for c in context.score(ctx_val, status_val)])
            outside = scored > 0
            fraud_rate = float(outside[y_val].mean()) if y_val.any() else 0.0
            legit_rate = float(outside[~y_val].mean())
            record.update(context.summary)
            record.update({"validation_fraud_outside_rate": round(fraud_rate, 4),
                           "validation_legitimate_outside_rate": round(legit_rate, 4),
                           "validation_separation": round(fraud_rate - legit_rate, 4),
                           "validation_average_precision": _ap(y_val, scored)})
            records.append(record)
            ap = record["validation_average_precision"] or 0.0
            if legit_rate <= 0.15 and (best is None or ap > (best[0]["validation_average_precision"] or 0.0)):
                best = (record, context)
    if best is None:
        raise EmptyCoreError("No DBSCAN candidate produced core samples with an acceptable outside rate")
    return {"record": best[0], "context": best[1]}, records


def select_operating_point(
    risk: np.ndarray,
    families: np.ndarray,
    unsafe: np.ndarray,
    y: np.ndarray,
    policy: Policy,
) -> tuple[OperatingPoint, dict[str, Any]]:
    """Grid-search Verify/Block thresholds on validation (model + rules only, no evidence channel)."""
    goals = policy.goals
    search = policy.threshold_search
    verify_grid = np.arange(search["verify"]["start"], search["verify"]["stop"] + 1e-9, search["verify"]["step"])
    block_grid = np.arange(search["block"]["start"], search["block"]["stop"] + 1e-9, search["block"]["step"])
    no_evidence = np.zeros(len(risk), dtype=bool)
    n_legit = int((~y).sum())
    candidates, frontier = [], {}
    for verify in verify_grid:
        for block in block_grid:
            if block <= verify:
                continue
            decisions = decide_arrays(risk, families, unsafe, no_evidence, verify, block, policy.min_families)
            flagged = decisions > 0
            blocked = decisions == 2
            verify_rate = float((decisions == 1).mean())
            false_blocks = int((blocked & ~y).sum())
            true_blocks = int((blocked & y).sum())
            n_blocks = int(blocked.sum())
            row = {
                "verify_threshold": float(verify),
                "block_threshold": float(block),
                "recall_verify_or_block": float((flagged & y).sum() / max(1, y.sum())),
                "recall_block_only": float(true_blocks / max(1, y.sum())),
                "verify_rate": verify_rate,
                "block_rate": float(blocked.mean()),
                "legitimate_false_block_rate": false_blocks / max(1, n_legit),
                "block_precision": (true_blocks / n_blocks) if n_blocks else None,
                "predicted_blocks": n_blocks,
                "false_blocks": false_blocks,
            }
            feasible = (
                verify_rate <= goals["max_verify_rate"]
                and row["legitimate_false_block_rate"] <= goals["max_legitimate_false_block_rate"]
                and (n_blocks == 0 or row["block_precision"] >= goals["min_block_precision"])
            )
            row["meets_goals"] = feasible
            candidates.append(row)
            key = float(verify)
            if feasible and (key not in frontier or row["recall_block_only"] > frontier[key]["recall_block_only"]):
                frontier[key] = row
    feasible_rows = [c for c in candidates if c["meets_goals"]]
    if feasible_rows:
        chosen = max(
            feasible_rows,
            key=lambda c: (c["recall_verify_or_block"], c["recall_block_only"], -c["verify_rate"], -c["block_threshold"]),
        )
        source = "validated"
    else:  # keep ambiguous cases in Verify instead of pretending certainty
        chosen = min(candidates, key=lambda c: (c["legitimate_false_block_rate"], -c["recall_verify_or_block"]))
        source = "validated_goals_not_met"
    digest = hashlib.sha256(json.dumps(chosen, sort_keys=True).encode()).hexdigest()[:8]
    point = OperatingPoint(chosen["verify_threshold"], chosen["block_threshold"], source, f"op-{digest}")
    return point, {
        "chosen": chosen,
        "candidates_evaluated": len(candidates),
        "feasible_candidates": len(feasible_rows),
        "frontier": sorted(frontier.values(), key=lambda r: r["verify_threshold"]),
        "initial_candidate": next(
            (c for c in candidates if c["verify_threshold"] == policy.initial_verify and c["block_threshold"] == policy.initial_block),
            None,
        ),
    }


def select_segment_points(
    risk: np.ndarray,
    families: np.ndarray,
    unsafe: np.ndarray,
    y: np.ndarray,
    segments: np.ndarray,
    policy: Policy,
) -> tuple[OperatingPoint, dict[str, Any]]:
    """Global threshold pair plus one per account segment; a segment without fraud or a feasible pair
    on validation falls back to the global pair, and the report says why."""
    global_point, global_report = select_operating_point(risk, families, unsafe, y, policy)
    pairs: dict[str, tuple[float, float]] = {}
    report: dict[str, dict[str, Any]] = {}
    for segment in SEGMENTS:
        mask = segments == segment
        base = {"requests": int(mask.sum()), "fraud": int(y[mask].sum())}
        if not mask.any() or not y[mask].any():
            reason = "no fraud in this segment on validation" if mask.any() else "no validation requests"
            report[segment] = {**base, "fallback": True, "reason": reason}
        else:
            point, detail = select_operating_point(risk[mask], families[mask], unsafe[mask], y[mask], policy)
            if detail["feasible_candidates"] == 0:
                report[segment] = {**base, "fallback": True, "reason": "no threshold pair meets the validation goals"}
            else:
                pairs[segment] = (point.verify, point.block)
                report[segment] = {**base, "fallback": False, "chosen": detail["chosen"],
                                   "feasible_candidates": detail["feasible_candidates"]}
                continue
        pairs[segment] = (global_point.verify, global_point.block)
    digest = hashlib.sha256(
        json.dumps({"global": global_report["chosen"], "segments": pairs}, sort_keys=True).encode()
    ).hexdigest()[:8]
    point = OperatingPoint(global_point.verify, global_point.block, global_point.source, f"op-{digest}",
                           segments=pairs)
    return point, {"global": global_report, "segments": report}


def _model_version(dataset: Dataset, selection: dict[str, Any]) -> str:
    material = json.dumps(
        {"files": {k: v["sha256"] for k, v in dataset.manifest["files"].items()}, "selection": selection,
         "features": FEATURE_VERSION, "sklearn": sklearn_version()},
        sort_keys=True,
    )
    return f"{MODEL_FAMILY}-1.1-{hashlib.sha256(material.encode()).hexdigest()[:10]}"


def train_models(training_dir: Path, models_dir: Path, policy: Policy) -> dict[str, Any]:
    started = time.perf_counter()
    dataset = load_dataset(training_dir)
    splits = load_splits(training_dir)
    labelled_ids = [rid for rid, part in splits.items() if part in {"calibration", "validation"}]
    result = replay(dataset, keep_facts_for=labelled_ids)
    by_id = {r.request_id: r for r in dataset.requests}
    senders = np.array([by_id[rid].sender_id for rid in result.request_ids])
    holdout = holdout_accounts(dataset, set(dataset.manifest["cold_start"]["account_ids"]))
    keep = ~np.isin(senders, sorted(holdout))
    masks = {name: PartitionView(view.rows & keep) for name, view in _partition_masks(result, splits).items()}
    tr, cal, val = masks["train"], masks["calibration"], masks["validation"]

    preprocessor = FeaturePreprocessor().fit(tr.take(result.payment))
    x_train = preprocessor.transform(tr.take(result.payment))
    x_cal = preprocessor.transform(cal.take(result.payment))
    x_val = preprocessor.transform(val.take(result.payment))

    labels = load_labels(training_dir)  # calibration (weights) and validation (selection) labels only
    ids = np.array(result.request_ids)
    y_cal = np.array([bool(labels[rid]["is_fraud"]) for rid in ids[cal.rows]])
    y_val = np.array([bool(labels[rid]["is_fraud"]) for rid in ids[val.rows]])

    best_if, if_records = _search_isolation_forest(x_train, x_cal, x_val, y_val)
    best_lof, lof_records = _search_lof(x_train, x_cal, x_val, y_val)
    statuses = np.array(result.context_status)
    profiles_train = result.context[tr.rows & (statuses == "ok")]
    best_db, db_records = _search_dbscan(
        profiles_train, val.take(result.context), list(statuses[val.rows]), y_val
    )

    def components_for(view: PartitionView, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if_s = best_if["mapping"].transform(best_if["detector"].raw_scores(x))
        lof_s = best_lof["mapping"].transform(best_lof["detector"].raw_scores(x))
        db_s = np.array([c.score for c in best_db["context"].score(view.take(result.context), list(statuses[view.rows]))])
        rules_s, families = rule_scores_for(result, view.rows, policy)
        return np.column_stack([if_s, lof_s, db_s, rules_s]), families

    cal_components, _ = components_for(cal, x_cal)
    weights, weights_report = search_weights(cal_components, y_cal)

    val_components, families_val = components_for(val, x_val)
    unsafe_val = unsafe_mask_for(result, val.rows, policy)
    risk_val = combine_arrays(*val_components.T, weights)
    segments_val = np.array([segment_of(dataset.accounts.get(s)) for s in senders[val.rows]])
    point, threshold_report = select_segment_points(risk_val, families_val, unsafe_val, y_val, segments_val, policy)
    point = replace(point, weights=weights)

    selection = {
        "isolation_forest": {k: best_if["record"][k] for k in ("n_estimators", "max_samples")},
        "lof": {k: best_lof["record"][k] for k in ("subspace", "n_neighbors", "fit_rows_requested")},
        "dbscan": {k: best_db["record"][k] for k in ("min_samples", "eps", "eps_quantile")},
        "weights": weights,
    }
    metadata = {
        "model_version": _model_version(dataset, selection),
        "feature_version": FEATURE_VERSION,
        "payment_features": list(PAYMENT_FEATURES),
        "context_features": list(CONTEXT_FEATURES),
        "lof_features": list(LOF_SUBSPACES[best_lof["record"]["subspace"]]),
        "sklearn_version": sklearn_version(),
        "random_seed": SEED,
        "training_dataset": {
            "path": "storage/data/training",
            "seed": dataset.manifest["seed"],
            "generator_version": dataset.manifest["generator_version"],
            "file_hashes": {k: v["sha256"] for k, v in dataset.manifest["files"].items()},
        },
        "fit_rows": {
            "preprocessing": int(tr.rows.sum()),
            "isolation_forest": int(tr.rows.sum()),
            "lof": int(best_lof["detector"].n_fit_rows),
            "dbscan": int(min(DBSCAN_FIT_ROWS, len(profiles_train))),
            "score_mapping_reference": int(cal.rows.sum()),
            "weights_reference": int(cal.rows.sum()),
        },
        "holdout_accounts": len(holdout),
        "selected_parameters": selection,
        "combination_weights": weights,
        "segment_thresholds": {k: {"verify": v[0], "block": v[1]} for k, v in point.segments.items()},
        "dbscan_context": {
            **best_db["context"].summary,
            "eps": best_db["context"].eps,
            "min_samples": best_db["context"].min_samples,
            "metric": best_db["context"].metric,
            "adapter": ("frozen-core: within eps of a training core sample -> 0, beyond -> min(1, (d - eps) / eps), "
                        "insufficient history -> 0"),
        },
        "score_mapping": "empirical CDF of inverted raw scores over the unlabelled calibration period",
    }
    bundle = ModelBundle(
        preprocessor=preprocessor,
        isolation_forest=best_if["detector"],
        lof=best_lof["detector"],
        dbscan=best_db["context"],
        if_mapping=best_if["mapping"],
        lof_mapping=best_lof["mapping"],
        metadata=metadata,
    )
    operating_point = {
        "verify": point.verify, "block": point.block, "source": point.source, "version": point.version,
        "weights": weights, "segments": metadata["segment_thresholds"],
    }

    def component_ap(y: np.ndarray, matrix: np.ndarray) -> dict[str, float | None]:
        out: dict[str, float | None] = {name: _ap(y, matrix[:, i]) for i, name in enumerate(
            ("isolation_forest", "lof", "dbscan", "rules"))}
        out["combined"] = _ap(y, combine_arrays(*matrix.T, weights))
        return out

    report = {
        "policy_version": policy.version,
        "operating_point": operating_point,
        "selected_parameters": selection,
        "search": {"isolation_forest": if_records, "lof": lof_records, "dbscan": db_records},
        "weights_search": {**weights_report, "partition": "calibration", "selected": weights,
                           "rows": int(cal.rows.sum()), "fraud": int(y_cal.sum())},
        "threshold_selection": threshold_report["global"],
        "segment_thresholds": threshold_report["segments"],
        "holdout_accounts": {"count": len(holdout), "seed": HOLDOUT_SEED, "account_ids": sorted(holdout)},
        "calibration_component_average_precision": component_ap(y_cal, cal_components),
        "validation_component_average_precision": component_ap(y_val, val_components),
        "training_seconds": round(time.perf_counter() - started, 2),
        "evidence_store_size": len(build_evidence_store(dataset.security_events, policy).records),
    }
    bundle.save(models_dir, point, {"policy_version": policy.version, "operating_point": operating_point})
    (models_dir / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report

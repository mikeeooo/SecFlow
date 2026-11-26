"""Frozen evaluation on the chronological final test period (days 81-100).

The model bundle, score mappings, rules, thresholds and evidence policy are loaded as
frozen artifacts; nothing is fitted or tuned here. Component-only flag thresholds are taken
from the validation period (top 5% review budget) before test labels are opened. Every test
request is assessed once through `ml.assess`; the evidence-free view re-applies the pure
action table to the same already-computed components.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from itertools import pairwise
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from config.settings import Settings
from ml.assess import AssessmentResult, assess_batch, assess_one, build_evidence_store
from ml.model import load_bundle, load_operating_point
from ml.policy import Policy, decide_action, load_policy
from ml.training import holdout_accounts
from storage.dataset import load_dataset, load_labels, load_splits
from storage.history import HistoryIndex

REVIEW_BUDGET = 0.05
LATENCY_SAMPLE = 300
COMPONENT_NAMES = ("isolation_forest", "lof", "dbscan", "rules")
BOOTSTRAP_RESAMPLES = 1_000
BOOTSTRAP_SEED = 42
# Fixed risk-score band edges; the frozen Verify / Block thresholds are added so each band sits wholly
# on one side of them. The last edge is open so a score of exactly 100 lands in the top band.
BASE_BAND_EDGES = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 70.0, 80.0, 90.0, 100.0)


def score_band_edges(verify: float, block: float) -> tuple[float, ...]:
    return tuple(sorted({*BASE_BAND_EDGES, float(verify), float(block)}))


SCORE_BAND_EDGES = score_band_edges(61.0, 97.0)  # default for direct callers without an operating point


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if total == 0:
        return None
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)]


def binary_metrics(flagged: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    tp = int((flagged & y).sum())
    fp = int((flagged & ~y).sum())
    fn = int((~flagged & y).sum())
    tn = int((~flagged & ~y).sum())
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = None if not precision or not recall else 2 * precision * recall / (precision + recall)
    return {
        "confusion_matrix": {"true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn},
        "precision": precision,
        "precision_95ci": wilson_interval(tp, tp + fp),
        "recall": recall,
        "f1": f1,
        "false_positive_rate": _ratio(fp, fp + tn),
        "flag_rate": _ratio(tp + fp, len(y)),
        "positives": int(y.sum()),
        "negatives": int((~y).sum()),
        "note": None if tp + fp else "No positive predictions; precision and F1 are undefined.",
    }


def _decision_codes(decisions: Sequence[str]) -> np.ndarray:
    return np.array([{"Permit": 0, "Verify": 1, "Block": 2}[d] for d in decisions])


def policy_metrics(codes: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    legit = ~y
    blocks = codes == 2
    return {
        "verify_or_block": binary_metrics(codes > 0, y),
        "block_only": binary_metrics(blocks, y),
        "verify_rate": float((codes == 1).mean()),
        "block_rate": float(blocks.mean()),
        "permit_rate": float((codes == 0).mean()),
        "legitimate_false_block_rate": _ratio(int((blocks & legit).sum()), int(legit.sum())),
        "legitimate_review_rate": _ratio(int(((codes == 1) & legit).sum()), int(legit.sum())),
        "predicted_blocks": int(blocks.sum()),
        "block_precision_95ci": wilson_interval(int((blocks & y).sum()), int(blocks.sum())),
    }


def _safe_auc(y: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    if y.all() or not y.any():
        return {"average_precision": None, "roc_auc": None}
    return {"average_precision": float(average_precision_score(y, scores)), "roc_auc": float(roc_auc_score(y, scores))}


def _component_matrix(results: Sequence[AssessmentResult]) -> np.ndarray:
    return np.array(
        [
            [r.components.if_score, r.components.lof_score, r.components.context.score, r.rules.score]
            if r.components is not None
            else [np.nan] * 4
            for r in results
        ]
    )


def _decide_without_evidence(results: Sequence[AssessmentResult], policy: Policy, point) -> list[str]:
    return [
        decide_action(r.risk, r.rules, [], r.policy.assessment_status, point, policy, r.segment).decision
        for r in results
    ]


def score_bands(risk: np.ndarray, y: np.ndarray, codes: np.ndarray,
                edges: Sequence[float] = SCORE_BAND_EDGES) -> dict[str, Any]:
    """Empirical fraud rate per risk-score band on labelled requests (the score's real-world yield).

    For each band: how many requests fell in it, how many were labelled fraud, the observed fraud
    rate with a Wilson 95% interval, its lift over the base rate, and the cumulative view "if every
    request scoring at least this band's lower edge were flagged" (recall, precision, flag rate).
    """
    scored = ~np.isnan(risk)
    risk, y, codes = risk[scored], y[scored], codes[scored]
    total, total_fraud = len(y), int(y.sum())
    base_rate = _ratio(total_fraud, total)
    bands = []
    for i, (low, high) in enumerate(pairwise(edges)):
        last = i == len(edges) - 2
        in_band = (risk >= low) & ((risk <= high) if last else (risk < high))
        at_or_above = risk >= low
        n, fraud = int(in_band.sum()), int((in_band & y).sum())
        above_n, above_fraud = int(at_or_above.sum()), int((at_or_above & y).sum())
        rate = _ratio(fraud, n)
        bands.append({
            "low": low,
            "high": high,
            "requests": n,
            "fraud": fraud,
            "legitimate": n - fraud,
            "fraud_rate": rate,
            "fraud_rate_95ci": wilson_interval(fraud, n),
            "lift": None if rate is None or not base_rate else rate / base_rate,
            "share_of_requests": _ratio(n, total),
            "share_of_fraud": _ratio(fraud, total_fraud),
            "recall_at_or_above": _ratio(above_fraud, total_fraud),
            "precision_at_or_above": _ratio(above_fraud, above_n),
            "flag_rate_at_or_above": _ratio(above_n, total),
            "decisions": {d: int((in_band & (codes == c)).sum()) for d, c in (("Permit", 0), ("Verify", 1), ("Block", 2))},
        })
    return {
        "edges": list(edges),
        "requests": total,
        "fraud": total_fraud,
        "base_fraud_rate": base_rate,
        "bands": bands,
        "note": ("Observed on the labelled synthetic final-test period. The fraud rate of a band is how often a "
                 "request with that score really was fraud; it is an empirical rate, not a calibrated probability "
                 "for any single request."),
    }


def cluster_bootstrap(
    codes: np.ndarray, y: np.ndarray, clusters: np.ndarray, n: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED
) -> dict[str, list[float]]:
    """95% percentile intervals that resample whole clusters (a fraud campaign, or one legitimate request).

    Fraud arrives in correlated campaigns, so resampling single requests would make intervals too narrow.
    """
    codes, y, clusters = np.asarray(codes), np.asarray(y, dtype=bool), np.asarray(clusters)
    _, cluster_index = np.unique(clusters, return_inverse=True)
    k = int(cluster_index.max()) + 1
    flagged = codes > 0

    def per_cluster(values: np.ndarray) -> np.ndarray:
        return np.bincount(cluster_index, weights=values.astype(float), minlength=k)

    totals = np.column_stack([
        per_cluster(np.ones(len(y))), per_cluster(y), per_cluster(flagged & y), per_cluster(flagged),
        per_cluster(codes == 1),
    ])
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(k, np.full(k, 1.0 / k), size=n) @ totals  # (n, 5) resampled sums
    rows, positives, true_flags, flags, verifies = draws.T
    with np.errstate(divide="ignore", invalid="ignore"):
        stats = {
            "precision_verify_or_block": np.where(flags > 0, true_flags / flags, np.nan),
            "recall_verify_or_block": np.where(positives > 0, true_flags / positives, np.nan),
            "verify_rate": verifies / rows,
        }
    return {name: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for name, v in stats.items()}


def _subset(codes: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    if not mask.any():
        return {"requests": 0, "note": "No requests in this subset."}
    return {"requests": int(mask.sum()), "fraud": int(y[mask].sum()), **policy_metrics(codes[mask], y[mask])}


def run_evaluation(settings: Settings) -> dict[str, Any]:
    policy = load_policy(settings.policy_path)
    bundle = load_bundle(settings.models_dir)
    point = load_operating_point(settings.models_dir, bundle.version)
    dataset = load_dataset(settings.training_dir)
    expected = bundle.metadata["training_dataset"]["file_hashes"]
    actual = {k: v["sha256"] for k, v in dataset.manifest["files"].items()}
    if expected != actual:
        raise RuntimeError("Training dataset hashes differ from those recorded in the model bundle; retrain first.")
    splits = load_splits(settings.training_dir)
    history, keys = HistoryIndex.from_events(dataset.requests, dataset.auth_events, dataset.payment_events)
    evidence = build_evidence_store(dataset.security_events, policy)

    # 1) Freeze component-only thresholds on validation, before opening any test label.
    validation = [r for r in dataset.requests if splits[r.request_id] == "validation"]
    val_results = assess_batch(validation, keys, history, dataset.accounts, bundle, evidence, policy, point)
    val_components = _component_matrix(val_results)
    component_thresholds = {
        name: float(np.quantile(val_components[:, i], 1 - REVIEW_BUDGET)) for i, name in enumerate(COMPONENT_NAMES)
    }
    component_thresholds["dbscan"] = 0.0  # flag = outside every frozen core neighbourhood (continuous score > 0)

    # 2) Assess every final-test request exactly once with the frozen pipeline.
    test = [r for r in dataset.requests if splits[r.request_id] == "test"]
    results = assess_batch(test, keys, history, dataset.accounts, bundle, evidence, policy, point)
    labels = load_labels(settings.training_dir)  # opened only after everything above is frozen
    campaign = np.array([labels[r.request.request_id]["campaign_id"] or r.request.request_id for r in results])
    y = np.array([bool(labels[r.request.request_id]["is_fraud"]) for r in results])
    scenario = np.array([labels[r.request.request_id]["scenario"] for r in results])
    components = _component_matrix(results)
    risk = np.array([r.risk.total if r.risk else np.nan for r in results])

    full_codes = _decision_codes([r.policy.decision for r in results])
    model_only_codes = _decision_codes(_decide_without_evidence(results, policy, point))
    evidence_blocks = np.array([r.policy.decision_source == "confirmed_fraud_evidence" for r in results])

    component_results = {}
    for i, name in enumerate(COMPONENT_NAMES):
        flagged = components[:, i] > 0 if name == "dbscan" else components[:, i] >= component_thresholds[name]
        component_results[name] = {
            "flag_threshold": component_thresholds[name],
            "threshold_source": "validation top-5% budget" if name != "dbscan" else "outside_context",
            **binary_metrics(flagged, y),
            **_safe_auc(y, components[:, i]),
        }
    component_results["combined_risk_score"] = {**_safe_auc(y, risk)}

    unusual_mask = (~y) & (scenario != "routine_payment")
    routine_mask = scenario == "routine_payment"
    cold_ids = set(dataset.manifest["cold_start"]["account_ids"])
    cold_mask = np.array([r.request.sender_id in cold_ids for r in results])
    holdout = holdout_accounts(dataset, cold_ids)
    holdout_mask = np.array([r.request.sender_id in holdout for r in results])
    drift_ids = list((dataset.manifest.get("drift") or {}).get("biller_ids") or [])
    drift_mask = np.array([r.request.recipient_id in drift_ids for r in results])
    unseen_mask = scenario == "invoice_redirect"
    victims: dict[str, set[str]] = {}
    for r, cmp, hit in zip(results, campaign, unseen_mask, strict=True):
        if hit:
            victims.setdefault(cmp, set()).add(r.request.sender_id)
    multi = np.array([hit and len(victims[cmp]) > 1 for cmp, hit in zip(campaign, unseen_mask, strict=True)])
    unseen = {
        "scenario": "invoice_redirect",
        **_subset(full_codes, y, unseen_mask),
        "campaigns": len(victims),
        "single_victim": _subset(full_codes, y, unseen_mask & ~multi),
        "multi_victim": _subset(full_codes, y, multi),
        "note": "Never present in train, calibration or validation.",
    }

    per_scenario = {}
    for name in sorted(set(scenario)):
        mask = scenario == name
        is_fraud = bool(y[mask].all())
        per_scenario[name] = {
            "kind": "fraud" if is_fraud else "legitimate",
            "requests": int(mask.sum()),
            "verify_or_block_rate": float((full_codes[mask] > 0).mean()),
            "block_rate": float((full_codes[mask] == 2).mean()),
            "evidence_block_rate": float(evidence_blocks[mask].mean()),
            "model_only_verify_or_block_rate": float((model_only_codes[mask] > 0).mean()),
            "model_only_block_rate": float((model_only_codes[mask] == 2).mean()),
        }

    latency = _measure_latency(test, keys, history, dataset, bundle, evidence, policy, point)
    decision_sources = Counter(f"{r.policy.decision}/{r.policy.decision_source}" for r in results)
    statuses = Counter(r.policy.assessment_status for r in results)
    split_counts: dict[str, dict[str, int]] = {}
    for rid, part in splits.items():
        bucket = split_counts.setdefault(part, {"requests": 0, "fraud": 0})
        bucket["requests"] += 1
        bucket["fraud"] += int(bool(labels[rid]["is_fraud"]))
    for bucket in split_counts.values():
        bucket["fraud_share"] = round(bucket["fraud"] / bucket["requests"], 4)

    full = policy_metrics(full_codes, y)
    model_only = policy_metrics(model_only_codes, y)
    summary = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scope": (
            "Synthetic final-test period (days 81-100) of the 100,000-request training corpus. Results describe this "
            "generated data only; they are not evidence of real-world detection or fraud reduction."
        ),
        "model_version": bundle.version,
        "policy_version": policy.version,
        "operating_point": {"verify": point.verify, "block": point.block, "source": point.source, "version": point.version},
        "positive_definitions": {
            "verify_or_block": "A request counts as flagged when the final action is Verify or Block.",
            "block_only": "A request counts as flagged only when the final action is Block.",
        },
        "splits": split_counts,
        "test_requests": len(results),
        "test_fraud": int(y.sum()),
        "components": component_results,
        "combined_policy": full,
        "combined_policy_without_evidence": model_only,
        "evidence_channel": {
            "evidence_blocks": int(evidence_blocks.sum()),
            "evidence_blocks_on_fraud": int((evidence_blocks & y).sum()),
            "evidence_blocks_on_legitimate": int((evidence_blocks & ~y).sum()),
            "fraud_with_active_evidence": int((evidence_blocks & y).sum()),
            "fraud_without_evidence": int((~evidence_blocks & y).sum()),
            "model_only_recall_on_fraud_without_evidence": _ratio(
                int(((model_only_codes > 0) & y & ~evidence_blocks).sum()), int((y & ~evidence_blocks).sum())
            ),
            "note": "All security evidence is simulated. The evidence channel is reported separately so it cannot hide weak model-only detection.",
        },
        "score_bands": score_bands(risk, y, full_codes, score_band_edges(point.verify, point.block)),
        "per_scenario": per_scenario,
        "unusual_legitimate": _subset(full_codes, y, unusual_mask),
        "routine_legitimate": _subset(full_codes, y, routine_mask),
        "cold_start_accounts": {"accounts": len(cold_ids), **_subset(full_codes, y, cold_mask)},
        "holdout_accounts": {"accounts": len(holdout), **_subset(full_codes, y, holdout_mask),
                             "note": "Accounts excluded from every fitting and selection row during training."},
        "unseen_scenario": unseen,
        "drift_billers": {"biller_ids": drift_ids, **_subset(full_codes, y, drift_mask)},
        "confidence_intervals": {
            **cluster_bootstrap(full_codes, y, np.where(y, campaign, [r.request.request_id for r in results])),
            "method": (f"cluster bootstrap, {BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED}; a fraud campaign "
                       "or one legitimate request is the resampling unit; 2.5th-97.5th percentiles"),
        },
        "weights": dict(point.weights or policy.weights),
        "segment_thresholds": {k: {"verify": v[0], "block": v[1]} for k, v in point.segments.items()},
        "existing_accounts": _subset(full_codes, y, ~cold_mask),
        "decision_sources": dict(sorted(decision_sources.items())),
        "assessment_statuses": dict(sorted(statuses.items())),
        "latency_ms": latency,
        "validation_goals": dict(policy.goals),
        "goal_assessment": _goal_assessment(full, model_only, policy),
    }
    summary["headline"] = {
        "test_requests": summary["test_requests"],
        "test_fraud": summary["test_fraud"],
        "recall_verify_or_block": full["verify_or_block"]["recall"],
        "precision_verify_or_block": full["verify_or_block"]["precision"],
        "recall_block_only": full["block_only"]["recall"],
        "block_precision": full["block_only"]["precision"],
        "verify_rate": full["verify_rate"],
        "legitimate_false_block_rate": full["legitimate_false_block_rate"],
        "model_only_recall_verify_or_block": model_only["verify_or_block"]["recall"],
        "latency_p95_ms": latency["p95"],
    }
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (settings.artifacts_dir / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n")
    (settings.artifacts_dir / "evaluation.md").write_text(render_markdown(summary))
    return summary


def _goal_assessment(full: dict[str, Any], model_only: dict[str, Any], policy: Policy) -> list[dict[str, Any]]:
    goals = policy.goals
    blocks = full["predicted_blocks"]
    precision = full["block_only"]["precision"]
    return [
        {
            "goal": f"Legitimate false-block rate <= {goals['max_legitimate_false_block_rate']:.1%}",
            "measured": full["legitimate_false_block_rate"],
            "met": full["legitimate_false_block_rate"] is not None
            and full["legitimate_false_block_rate"] <= goals["max_legitimate_false_block_rate"],
        },
        {
            "goal": f"Verify rate <= {goals['max_verify_rate']:.0%}",
            "measured": full["verify_rate"],
            "met": full["verify_rate"] <= goals["max_verify_rate"],
        },
        {
            "goal": f"Block precision >= {goals['min_block_precision']:.0%}",
            "measured": precision,
            "interval_95": full["block_precision_95ci"],
            "predicted_blocks": blocks,
            "met": None if blocks < 100 else bool(precision is not None and precision >= goals["min_block_precision"]),
            "note": "Fewer than 100 predicted blocks: the sample cannot establish 99% precision either way."
            if blocks < 100
            else None,
        },
    ]


def _measure_latency(test, keys, history, dataset, bundle, evidence, policy, point) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    sample = [test[int(i)] for i in rng.choice(len(test), size=min(LATENCY_SAMPLE, len(test)), replace=False)]
    timings = []
    for request in sample:
        started = time.perf_counter()
        assess_one(request, keys[request.request_id], history, dataset.accounts.get(request.sender_id), bundle,
                   evidence, policy, point)
        timings.append((time.perf_counter() - started) * 1_000)
    timings.sort()
    return {
        "requests_timed": len(timings),
        "scope": "single-request path: features + IF + LOF + DBSCAN context + rules + evidence + policy, in-process",
        "mean": round(statistics.fmean(timings), 3),
        "p50": round(timings[len(timings) // 2], 3),
        "p95": round(timings[int(len(timings) * 0.95) - 1], 3),
        "p99": round(timings[int(len(timings) * 0.99) - 1], 3),
        "max": round(timings[-1], 3),
    }


def _pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def render_markdown(s: dict[str, Any]) -> str:
    full, model_only = s["combined_policy"], s["combined_policy_without_evidence"]
    lines = [
        "# SecFlow evaluation report",
        "",
        f"Generated {s['generated_at']} · model `{s['model_version']}` · policy `{s['policy_version']}`",
        "",
        f"> {s['scope']}",
        "",
        (f"Operating point (selected on validation, frozen before test): **Verify ≥ {s['operating_point']['verify']:g}**, "
        f"**Block ≥ {s['operating_point']['block']:g}** with at least two corroborating behaviour families "
        f"(`{s['operating_point']['source']}`)."),
        "",
        "## Splits",
        "",
        "| Partition | Requests | Fraud | Fraud share |",
        "|---|---:|---:|---:|",
    ]
    for name in ("train", "calibration", "validation", "test"):
        b = s["splits"][name]
        lines.append(f"| {name} | {b['requests']:,} | {b['fraud']} | {b['fraud_share']:.2%} |")
    lines += [
        "",
        "## Combined policy on the final test period",
        "",
        (f"Positive definitions: *Verify or Block* — {s['positive_definitions']['verify_or_block']} "
        f"*Block only* — {s['positive_definitions']['block_only']}"),
        "",
        "| View | Precision | Recall | F1 | FPR | TP | FP | FN | TN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, block in (
        ("With evidence · Verify or Block", full["verify_or_block"]),
        ("With evidence · Block only", full["block_only"]),
        ("Model + rules only · Verify or Block", model_only["verify_or_block"]),
        ("Model + rules only · Block only", model_only["block_only"]),
    ):
        cm = block["confusion_matrix"]
        f1 = "n/a" if block["f1"] is None else f"{block['f1']:.3f}"
        lines.append(
            f"| {label} | {_pct(block['precision'])} | {_pct(block['recall'])} | {f1} | {_pct(block['false_positive_rate'], 2)} "
            f"| {cm['true_positive']} | {cm['false_positive']} | {cm['false_negative']} | {cm['true_negative']} |"
        )
    lines += [
        "",
        (f"Verify rate {_pct(full['verify_rate'], 2)} · Block rate {_pct(full['block_rate'], 2)} · "
        f"legitimate false-block rate {_pct(full['legitimate_false_block_rate'], 3)} · "
        f"block precision 95% CI {full['block_precision_95ci']} over {full['predicted_blocks']} blocks."),
        "",
        "## Components alone",
        "",
        "Flag thresholds come from the validation period (top 5% review budget; DBSCAN flags `outside_context`).",
        "",
        "| Component | Threshold | Precision | Recall | FPR | Average precision | ROC AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in COMPONENT_NAMES:
        c = s["components"][name]
        ap = "n/a" if c["average_precision"] is None else f"{c['average_precision']:.3f}"
        auc = "n/a" if c["roc_auc"] is None else f"{c['roc_auc']:.3f}"
        lines.append(
            f"| {name} | {c['flag_threshold']:.3f} | {_pct(c['precision'])} | {_pct(c['recall'])} | "
            f"{_pct(c['false_positive_rate'], 2)} | {ap} | {auc} |"
        )
    comb = s["components"]["combined_risk_score"]
    lines += [
        f"| combined risk score | — | — | — | — | {comb['average_precision']:.3f} | {comb['roc_auc']:.3f} |",
        "",
        "## Per scenario (final test)",
        "",
        "| Scenario | Kind | Requests | Verify or Block | Block | Evidence block | Model-only Verify or Block |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in s["per_scenario"].items():
        lines.append(
            f"| {name} | {row['kind']} | {row['requests']:,} | {_pct(row['verify_or_block_rate'])} | {_pct(row['block_rate'])} "
            f"| {_pct(row['evidence_block_rate'])} | {_pct(row['model_only_verify_or_block_rate'])} |"
        )
    bands = s.get("score_bands")
    if bands:
        lines += [
            "",
            "## What the risk score is worth: observed fraud rate per score band (final test)",
            "",
            (f"Base fraud rate {_pct(bands['base_fraud_rate'], 2)} over {bands['requests']:,} scored requests. "
             "*Recall ≥ band* and *precision ≥ band* describe flagging every request at or above the band's lower edge."),
            "",
            "| Score band | Requests | Fraud | Fraud rate (95% CI) | Lift | Recall ≥ band | Precision ≥ band | Flag rate ≥ band |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for b in bands["bands"]:
            ci = b["fraud_rate_95ci"]
            ci_text = "" if ci is None else f" ({ci[0] * 100:.1f}–{ci[1] * 100:.1f}%)"
            lift = "n/a" if b["lift"] is None else f"{b['lift']:.1f}×"
            lines.append(
                f"| {b['low']:g}–{b['high']:g} | {b['requests']:,} | {b['fraud']} | {_pct(b['fraud_rate'])}{ci_text} | "
                f"{lift} | {_pct(b['recall_at_or_above'])} | {_pct(b['precision_at_or_above'])} | "
                f"{_pct(b['flag_rate_at_or_above'], 2)} |"
            )
        lines += ["", bands["note"]]
    ev = s["evidence_channel"]
    lines += [
        "",
        "## Evidence channel versus model-only detection",
        "",
        (f"- Confirmed-evidence blocks: {ev['evidence_blocks']} ({ev['evidence_blocks_on_fraud']} fraud, "
        f"{ev['evidence_blocks_on_legitimate']} legitimate)."),
        (f"- Fraud requests without any active evidence: {ev['fraud_without_evidence']}; model + rules flagged "
        f"{_pct(ev['model_only_recall_on_fraud_without_evidence'])} of them (Verify or Block)."),
        f"- {ev['note']}",
        "",
        "## Subsets",
        "",
        "| Subset | Requests | Fraud | Verify or Block recall | Block recall | Legit review rate | Legit false-block rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in (("Unusual legitimate", "unusual_legitimate"), ("Routine legitimate", "routine_legitimate"),
                       ("Cold-start accounts (40)", "cold_start_accounts"), ("Existing accounts", "existing_accounts")):
        b = s[key]
        if not b.get("requests"):
            continue
        lines.append(
            f"| {label} | {b['requests']:,} | {b['fraud']} | {_pct(b['verify_or_block']['recall'])} | "
            f"{_pct(b['block_only']['recall'])} | {_pct(b['legitimate_review_rate'], 2)} | {_pct(b['legitimate_false_block_rate'], 3)} |"
        )
    lat = s["latency_ms"]
    lines += [
        "",
        "## Validation goals (targets to assess, not promises)",
        "",
    ]
    for goal in s["goal_assessment"]:
        status = {True: "met", False: "not met", None: "not established"}[goal["met"]]
        measured = goal["measured"]
        shown = "n/a" if measured is None else (f"{measured:.4f}" if isinstance(measured, float) else str(measured))
        lines.append(f"- {goal['goal']}: measured {shown} — **{status}**" + (f" ({goal['note']})" if goal.get("note") else ""))
    lines += [
        "",
        "## Latency",
        "",
        (f"{lat['requests_timed']} single-request assessments ({lat['scope']}): mean {lat['mean']} ms, "
        f"p50 {lat['p50']} ms, p95 {lat['p95']} ms, p99 {lat['p99']} ms."),
        "",
        "## Decision sources",
        "",
    ]
    lines += [f"- {k}: {v:,}" for k, v in s["decision_sources"].items()]
    ci = s["confidence_intervals"]
    weights = ", ".join(f"{k} {v:.2f}" for k, v in s["weights"].items())
    segments = "; ".join(f"{k}: Verify ≥ {v['verify']:g}, Block ≥ {v['block']:g}"
                         for k, v in s["segment_thresholds"].items())
    unseen, holdout, drift = s["unseen_scenario"], s["holdout_accounts"], s["drift_billers"]

    def flagged(subset: dict[str, Any]) -> str:
        return _pct(subset.get("verify_or_block", {}).get("recall")) if subset.get("fraud") else "n/a"

    lines += [
        "",
        "## Learned weights and segment thresholds",
        "",
        f"- Weights (calibration AP search): {weights}",
        f"- Segment thresholds (validation): {segments}",
        "",
        "## Confidence intervals (95%)",
        "",
        (f"- Verify-or-Block precision: {_pct(ci['precision_verify_or_block'][0])} – "
        f"{_pct(ci['precision_verify_or_block'][1])}"),
        f"- Verify-or-Block recall: {_pct(ci['recall_verify_or_block'][0])} – {_pct(ci['recall_verify_or_block'][1])}",
        f"- Verify rate: {_pct(ci['verify_rate'][0])} – {_pct(ci['verify_rate'][1])}",
        f"- Method: {ci['method']}",
        "",
        "## Unseen scenario, holdout accounts and drift",
        "",
        (f"- invoice_redirect ({unseen.get('requests', 0)} requests, {unseen['campaigns']} campaigns): "
        f"flagged {flagged(unseen)}; single-victim {flagged(unseen['single_victim'])}, "
        f"multi-victim {flagged(unseen['multi_victim'])}"),
        (f"- Holdout accounts ({holdout['accounts']} accounts, {holdout.get('requests', 0)} requests, "
        f"{holdout.get('fraud', 0)} fraud): recall {flagged(holdout)}, "
        f"verify rate {_pct(holdout.get('verify_rate'))}"),
        (f"- Drift billers ({drift.get('requests', 0)} requests): verify rate {_pct(drift.get('verify_rate'))}, "
        f"false-block rate {_pct(drift.get('legitimate_false_block_rate'), 2)}"),
    ]
    lines += [
        "",
        "Scores are anomaly-risk indices, not probabilities of fraud. A high score alone is never treated as confirmed fraud.",
        "",
    ]
    return "\n".join(lines)

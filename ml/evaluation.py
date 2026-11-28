"""Final evaluation on test.json (days 86-100), run once after training has frozen everything.

Every test payment goes through the same `Assessor` as the app, with train.json and validation.json
as earlier history. The labels are opened only after all decisions are made. Nothing is fitted or
tuned here. The report goes to artifacts/evaluation.json and artifacts/evaluation.md.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from datetime import UTC, datetime
from itertools import pairwise
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from config.settings import Settings
from datagen.dataset import file_hashes, load_labels, load_partitions
from datagen.generator import ALL_FRAUD_TYPES, TEST_ONLY_FRAUD_TYPES
from ml.assess import AssessmentResult, Assessor, evidence_store
from ml.history import HistoryIndex
from ml.model import load_bundle
from ml.policy import COMPONENTS, decide_action, load_operating_point, load_policy

REVIEW_BUDGET = 0.05  # a component alone flags its top 5% (threshold taken from validation)
LATENCY_SAMPLE = 300
BAND_EDGES = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 70.0, 80.0, 90.0, 100.0)
CODES = {"Permit": 0, "Verify": 1, "Block": 2}


def ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    """95% Wilson score interval for a proportion (treats payments as independent)."""
    if total == 0:
        return None
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)]


def binary_metrics(flagged: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    tp, fp = int((flagged & y).sum()), int((flagged & ~y).sum())
    fn, tn = int((~flagged & y).sum()), int((~flagged & ~y).sum())
    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    return {
        "confusion_matrix": {"true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn},
        "precision": precision,
        "precision_95ci": wilson_interval(tp, tp + fp),
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision and recall else None,
        "false_positive_rate": ratio(fp, fp + tn),
        "flag_rate": ratio(tp + fp, len(y)),
        "positives": int(y.sum()),
        "negatives": int((~y).sum()),
        "note": None if tp + fp else "No positive predictions; precision and F1 are undefined.",
    }


def policy_metrics(codes: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Metrics of a decision vector, with positive = Verify or Block, and positive = Block only."""
    blocks, legit = codes == 2, ~y
    return {
        "verify_or_block": binary_metrics(codes > 0, y),
        "block_only": binary_metrics(blocks, y),
        "verify_rate": float((codes == 1).mean()),
        "block_rate": float(blocks.mean()),
        "permit_rate": float((codes == 0).mean()),
        "legitimate_false_block_rate": ratio(int((blocks & legit).sum()), int(legit.sum())),
        "legitimate_review_rate": ratio(int(((codes == 1) & legit).sum()), int(legit.sum())),
        "predicted_blocks": int(blocks.sum()),
        "block_precision_95ci": wilson_interval(int((blocks & y).sum()), int(blocks.sum())),
    }


def subset(codes: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    if not mask.any():
        return {"requests": 0, "note": "No requests in this subset."}
    return {"requests": int(mask.sum()), "fraud": int(y[mask].sum()), **policy_metrics(codes[mask], y[mask])}


def ranking(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {"average_precision": float(average_precision_score(y, scores)), "roc_auc": float(roc_auc_score(y, scores))}


def score_bands(risk: np.ndarray, y: np.ndarray, codes: np.ndarray, verify: float, block: float) -> dict[str, Any]:
    """Observed fraud rate per risk-score band: what a score is worth on labelled data.

    The Verify and Block thresholds are band edges, so each band sits on one side of them.
    """
    edges = sorted({*BAND_EDGES, verify, block})
    total, total_fraud = len(y), int(y.sum())
    base = ratio(total_fraud, total)
    bands = []
    for i, (low, high) in enumerate(pairwise(edges)):
        upper = risk <= high if i == len(edges) - 2 else risk < high  # the top band includes 100
        in_band, above = (risk >= low) & upper, risk >= low
        n, fraud = int(in_band.sum()), int((in_band & y).sum())
        rate = ratio(fraud, n)
        bands.append({
            "low": low, "high": high, "requests": n, "fraud": fraud, "legitimate": n - fraud,
            "fraud_rate": rate, "fraud_rate_95ci": wilson_interval(fraud, n),
            "lift": rate / base if rate is not None and base else None,
            "share_of_requests": ratio(n, total), "share_of_fraud": ratio(fraud, total_fraud),
            "recall_at_or_above": ratio(int((above & y).sum()), total_fraud),
            "precision_at_or_above": ratio(int((above & y).sum()), int(above.sum())),
            "flag_rate_at_or_above": ratio(int(above.sum()), total),
            "decisions": {d: int((in_band & (codes == c)).sum()) for d, c in CODES.items()},
        })
    return {"edges": edges, "requests": total, "fraud": total_fraud, "base_fraud_rate": base, "bands": bands,
            "note": ("Observed on the labelled synthetic final-test period. A band's fraud rate is how often a request "
                     "with that score really was fraud; it is an empirical rate, not a calibrated probability.")}


def _components(results: list[AssessmentResult]) -> np.ndarray:
    return np.array([[r.components.if_score, r.components.lof_score, r.components.context.score, r.rules.score]
                     for r in results])


def run_evaluation(settings: Settings) -> dict[str, Any]:
    policy = load_policy(settings.policy_path)
    bundle = load_bundle(settings.models_dir)
    point = load_operating_point(settings.models_dir, bundle.version)
    paths = [settings.train_path, settings.validation_path, settings.test_path]
    if bundle.metadata["training_dataset"]["file_hashes"] != file_hashes(paths[:2]):
        raise RuntimeError("train.json / validation.json changed since training; retrain first.")
    dataset = load_partitions(paths)  # test payments see the earlier files as history, never the reverse
    history, keys = HistoryIndex.from_events(dataset.requests, dataset.auth_events, dataset.payment_events)
    assessor = Assessor(bundle, evidence_store(dataset.security_events, policy), policy, point)

    def assess(part: str) -> list[AssessmentResult]:
        requests = [r for r in dataset.requests if dataset.partition_of[r.request_id] == part]
        return assessor.assess_batch(requests, keys, history, dataset.accounts)

    # Each component alone flags its top 5%; the cut-off comes from validation, before any test label.
    val_components = _components(assess("validation"))
    cutoffs = {name: float(np.quantile(val_components[:, i], 1 - REVIEW_BUDGET)) for i, name in enumerate(COMPONENTS)}
    cutoffs["dbscan"] = 0.0  # DBSCAN flags every profile outside the frozen core neighbourhoods

    results = assess("test")
    labels = load_labels(paths)  # opened only now
    ids = [r.request.request_id for r in results]
    y = np.array([bool(labels[i]["is_fraud"]) for i in ids])
    fraud_type = np.array([labels[i]["fraud_type"] or "" for i in ids])
    anomaly_type = np.array([labels[i]["anomaly_type"] or "" for i in ids])
    risk = np.array([r.risk.total for r in results])
    components = _components(results)
    codes = np.array([CODES[r.policy.decision] for r in results])
    # The same payments without the evidence channel: model and rules alone.
    model_codes = np.array([CODES[decide_action(r.risk, r.rules, (), r.policy.assessment_status, point, policy,
                                                r.segment).decision] for r in results])
    evidence_blocks = np.array([r.policy.decision_source == "confirmed_fraud_evidence" for r in results])

    component_results: dict[str, Any] = {}
    for i, name in enumerate(COMPONENTS):
        flagged = components[:, i] > cutoffs[name] if name == "dbscan" else components[:, i] >= cutoffs[name]
        component_results[name] = {
            "flag_threshold": cutoffs[name],
            "threshold_source": "outside_context" if name == "dbscan" else "validation top-5% budget",
            **binary_metrics(flagged, y), **ranking(y, components[:, i])}
    component_results["combined_risk_score"] = ranking(y, risk)

    def rate(mask: np.ndarray, values: np.ndarray) -> float | None:
        return ratio(int(values[mask].sum()), int(mask.sum()))

    per_fraud_type = {}
    for name in ALL_FRAUD_TYPES:
        mask = fraud_type == name
        per_fraud_type[name] = {
            "test_only": name in TEST_ONLY_FRAUD_TYPES,
            "requests": int(mask.sum()),
            "campaigns": len({labels[i]["campaign_id"] for i, m in zip(ids, mask, strict=True) if m}),
            "recall_verify_or_block": rate(mask, codes > 0),
            "recall_block_only": rate(mask, codes == 2),
            "evidence_block_rate": rate(mask, evidence_blocks),
            "model_only_recall_verify_or_block": rate(mask, model_codes > 0),
            "model_only_recall_block_only": rate(mask, model_codes == 2),
        }
    per_anomaly_type = {name: {"requests": int((anomaly_type == name).sum()),
                               "verify_or_block_rate": rate(anomaly_type == name, codes > 0),
                               "block_rate": rate(anomaly_type == name, codes == 2)}
                        for name in sorted(set(anomaly_type) - {""})}

    earlier = {r.sender_id for r in dataset.requests if dataset.partition_of[r.request_id] != "test"}
    cold = np.array([r.request.sender_id not in earlier for r in results])
    unseen = np.isin(fraud_type, TEST_ONLY_FRAUD_TYPES)
    full, model_only = policy_metrics(codes, y), policy_metrics(model_codes, y)
    summary = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scope": ("Synthetic final-test file storage/test.json (days 86-100, 15,000 assessed MIA instant payments in "
                  "MDL). Results describe this generated data only; they are not evidence of real-world detection."),
        "model_version": bundle.version,
        "policy_version": policy.version,
        "operating_point": {"verify": point.verify, "block": point.block, "source": point.source,
                            "version": point.version},
        "positive_definitions": {
            "verify_or_block": "A payment counts as flagged when the final action is Verify or Block.",
            "block_only": "A payment counts as flagged only when the final action is Block.",
        },
        "splits": _splits(dataset.partition_of, labels),
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
            "model_only_recall_on_fraud_without_evidence": rate(y & ~evidence_blocks, model_codes > 0),
            "note": "All security evidence is simulated. It is reported separately so it cannot hide weak model-only "
                    "detection.",
        },
        "score_bands": score_bands(risk, y, codes, point.verify, point.block),
        "per_fraud_type": per_fraud_type,
        "per_anomaly_type": per_anomaly_type,
        "unusual_legitimate": subset(codes, y, ~y & (anomaly_type != "")),
        "routine_legitimate": subset(codes, y, ~y & (anomaly_type == "")),
        "cold_start_accounts": {"accounts": len({r.request.sender_id for r, c in zip(results, cold, strict=True) if c}),
                                **subset(codes, y, cold)},
        "existing_accounts": subset(codes, y, ~cold),
        "unseen_scenario": {"fraud_types": list(TEST_ONLY_FRAUD_TYPES), **subset(codes, y, unseen),
                            "note": "Never present in train.json or validation.json."},
        "confidence_intervals": {
            "precision_verify_or_block": wilson_interval(int(((codes > 0) & y).sum()), int((codes > 0).sum())),
            "recall_verify_or_block": wilson_interval(int(((codes > 0) & y).sum()), int(y.sum())),
            "verify_rate": wilson_interval(int((codes == 1).sum()), len(codes)),
            "method": ("Wilson score interval; it treats payments as independent, so it is too narrow for fraud "
                       "that arrives in campaigns"),
        },
        "weights": dict(policy.weights),
        "segment_thresholds": {name: {"verify": v, "block": b} for name, (v, b) in point.segments.items()},
        "decision_sources": dict(sorted(Counter(f"{r.policy.decision}/{r.policy.decision_source}"
                                                for r in results).items())),
        "assessment_statuses": dict(sorted(Counter(r.policy.assessment_status for r in results).items())),
        "latency_ms": _latency(assessor, results, keys, history, dataset.accounts),
        "validation_goals": dict(policy.goals),
        "goal_assessment": _goals(full, policy),
    }
    summary["headline"] = {
        "test_requests": len(results),
        "test_fraud": int(y.sum()),
        "recall_verify_or_block": full["verify_or_block"]["recall"],
        "precision_verify_or_block": full["verify_or_block"]["precision"],
        "recall_block_only": full["block_only"]["recall"],
        "block_precision": full["block_only"]["precision"],
        "verify_rate": full["verify_rate"],
        "legitimate_false_block_rate": full["legitimate_false_block_rate"],
        "model_only_recall_verify_or_block": model_only["verify_or_block"]["recall"],
        "test_only_fraud_recall_verify_or_block": summary["unseen_scenario"]["verify_or_block"]["recall"],
        "latency_p95_ms": summary["latency_ms"]["p95"],
    }
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (settings.artifacts_dir / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n")
    (settings.artifacts_dir / "evaluation.md").write_text(render_markdown(summary))
    return summary


def _splits(partition_of: dict[str, str], labels: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for rid, part in partition_of.items():
        if rid in labels:  # assessed payments only, not incoming credits
            bucket = out.setdefault(part, {"requests": 0, "fraud": 0})
            bucket["requests"] += 1
            bucket["fraud"] += int(bool(labels[rid]["is_fraud"]))
    for bucket in out.values():
        bucket["fraud_share"] = round(bucket["fraud"] / bucket["requests"], 4)
    return out


def _goals(full: dict[str, Any], policy) -> list[dict[str, Any]]:
    goals, blocks, precision = policy.goals, full["predicted_blocks"], full["block_only"]["precision"]
    return [
        {"goal": f"Legitimate false-block rate <= {goals['max_legitimate_false_block_rate']:.1%}",
         "measured": full["legitimate_false_block_rate"],
         "met": full["legitimate_false_block_rate"] <= goals["max_legitimate_false_block_rate"]},
        {"goal": f"Verify rate <= {goals['max_verify_rate']:.0%}", "measured": full["verify_rate"],
         "met": full["verify_rate"] <= goals["max_verify_rate"]},
        {"goal": f"Block precision >= {goals['min_block_precision']:.0%}", "measured": precision,
         "interval_95": full["block_precision_95ci"], "predicted_blocks": blocks,
         "met": None if blocks < 100 else bool(precision is not None and precision >= goals["min_block_precision"]),
         "note": "Fewer than 100 predicted blocks: the sample cannot establish 99% precision either way."
         if blocks < 100 else None},
    ]


def _latency(assessor: Assessor, results, keys, history, accounts) -> dict[str, Any]:
    """Time the single-payment path (what one simulation costs) on a seeded sample of test payments."""
    rng = np.random.default_rng(0)
    sample = [results[int(i)].request for i in rng.choice(len(results), min(LATENCY_SAMPLE, len(results)), replace=False)]
    timings = []
    for request in sample:
        started = time.perf_counter()
        assessor.assess_batch([request], keys, history, accounts)
        timings.append((time.perf_counter() - started) * 1_000)
    timings.sort()
    return {"requests_timed": len(timings),
            "scope": "single-request path: features + IF + LOF + DBSCAN context + rules + evidence + policy, in-process",
            "mean": round(float(np.mean(timings)), 3), "p50": round(timings[len(timings) // 2], 3),
            "p95": round(timings[int(len(timings) * 0.95) - 1], 3),
            "p99": round(timings[int(len(timings) * 0.99) - 1], 3), "max": round(timings[-1], 3)}


def _pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def render_markdown(s: dict[str, Any]) -> str:
    full, model_only, op = s["combined_policy"], s["combined_policy_without_evidence"], s["operating_point"]
    lines = [
        "# SecFlow evaluation report", "",
        f"Generated {s['generated_at']} · model `{s['model_version']}` · policy `{s['policy_version']}`", "",
        f"> {s['scope']}", "",
        (f"Thresholds chosen on validation and frozen before test: **Verify ≥ {op['verify']:g}**, "
        f"**Block ≥ {op['block']:g}** with at least two corroborating rule families."), "",
        "## Decisions on the final test period", "",
        "| View | Precision | Recall | F1 | FPR | TP | FP | FN | TN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, block in (("With evidence · Verify or Block", full["verify_or_block"]),
                         ("With evidence · Block only", full["block_only"]),
                         ("Model + rules only · Verify or Block", model_only["verify_or_block"]),
                         ("Model + rules only · Block only", model_only["block_only"])):
        cm = block["confusion_matrix"]
        f1 = "n/a" if block["f1"] is None else f"{block['f1']:.3f}"
        lines.append(f"| {label} | {_pct(block['precision'])} | {_pct(block['recall'])} | {f1} | "
                     f"{_pct(block['false_positive_rate'], 2)} | {cm['true_positive']} | {cm['false_positive']} | "
                     f"{cm['false_negative']} | {cm['true_negative']} |")
    ci = s["confidence_intervals"]
    lines += [
        "",
        (f"Verify rate {_pct(full['verify_rate'], 2)} · block rate {_pct(full['block_rate'], 2)} · legitimate "
        f"false-block rate {_pct(full['legitimate_false_block_rate'], 3)} · {full['predicted_blocks']} blocks."),
        (f"95% intervals ({ci['method']}): recall {ci['recall_verify_or_block']}, precision "
        f"{ci['precision_verify_or_block']}, verify rate {ci['verify_rate']}."),
        "", "## Components alone", "",
        "| Component | Threshold | Precision | Recall | FPR | Average precision | ROC AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in COMPONENTS:
        c = s["components"][name]
        lines.append(f"| {name} | {c['flag_threshold']:.3f} | {_pct(c['precision'])} | {_pct(c['recall'])} | "
                     f"{_pct(c['false_positive_rate'], 2)} | {c['average_precision']:.3f} | {c['roc_auc']:.3f} |")
    comb = s["components"]["combined_risk_score"]
    lines += [f"| combined risk score | — | — | — | — | {comb['average_precision']:.3f} | {comb['roc_auc']:.3f} |",
              "", "## Recall per fraud type", "",
              "| Fraud type | Test only | Payments | Verify or Block | Block | Evidence block | Model + rules only |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for name, row in s["per_fraud_type"].items():
        lines.append(f"| {name} | {'yes' if row['test_only'] else ''} | {row['requests']} | "
                     f"{_pct(row['recall_verify_or_block'])} | {_pct(row['recall_block_only'])} | "
                     f"{_pct(row['evidence_block_rate'])} | {_pct(row['model_only_recall_verify_or_block'])} |")
    lines += ["", "## Unusual but legitimate payments", "", "| Anomaly type | Payments | Verify or Block | Block |",
              "|---|---:|---:|---:|"]
    for name, row in s["per_anomaly_type"].items():
        lines.append(f"| {name} | {row['requests']} | {_pct(row['verify_or_block_rate'])} | {_pct(row['block_rate'])} |")
    lines += ["", "## Subsets", "",
              "| Subset | Requests | Fraud | Recall (V or B) | Legit review rate | Legit false-block rate |",
              "|---|---:|---:|---:|---:|---:|"]
    for label, key in (("Unusual legitimate", "unusual_legitimate"), ("Routine legitimate", "routine_legitimate"),
                       ("Cold-start accounts", "cold_start_accounts"), ("Existing accounts", "existing_accounts"),
                       ("Test-only fraud types", "unseen_scenario")):
        b = s[key]
        recall = _pct(b["verify_or_block"]["recall"]) if b.get("fraud") else "n/a"
        lines.append(f"| {label} | {b['requests']:,} | {b['fraud']} | {recall} | "
                     f"{_pct(b['legitimate_review_rate'], 2)} | {_pct(b['legitimate_false_block_rate'], 3)} |")
    ev, lat = s["evidence_channel"], s["latency_ms"]
    lines += ["", "## Evidence channel", "",
              (f"{ev['evidence_blocks']} confirmed-evidence blocks ({ev['evidence_blocks_on_fraud']} fraud, "
              f"{ev['evidence_blocks_on_legitimate']} legitimate). Model and rules flagged "
              f"{_pct(ev['model_only_recall_on_fraud_without_evidence'])} of the {ev['fraud_without_evidence']} fraud "
              "payments without evidence."), "", "## Goals", ""]
    for goal in s["goal_assessment"]:
        status = {True: "met", False: "not met", None: "not established"}[goal["met"]]
        lines.append(f"- {goal['goal']}: measured {goal['measured']:.4f}, **{status}**"
                     + (f" ({goal['note']})" if goal.get("note") else ""))
    lines += ["", "## Latency", "",
              f"{lat['requests_timed']} single payments: mean {lat['mean']} ms, p95 {lat['p95']} ms, p99 {lat['p99']} ms.",
              "", "Scores are anomaly-risk indices, not probabilities of fraud.", ""]
    return "\n".join(lines)

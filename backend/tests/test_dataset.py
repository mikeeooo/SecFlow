from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pytest

from storage.dataset import (
    DatasetError,
    cross_dataset_fingerprint_overlap,
    load_dataset,
    load_labels,
    load_splits,
    read_jsonl,
    validate_dataset,
)
from storage.evidence import EvidenceStore, evidence_from_row
from storage.generator import (
    DEMO_SPEC,
    FRAUD_CATEGORIES,
    TRAINING_SPEC,
    _build_account_params,
    _build_base_counterparties,
    day_start_ts,
    generate_dataset,
)
from storage.identities import build_accounts
from storage.records import parse_ts

# sha256 of the generator-only account parameters as produced by generator 1.0.0.
ACCOUNT_PARAMS_SNAPSHOT = "bea5ee12d4d55aa37c69d08900bb15fb1bfa3bf66dadb600c77f20bcdc270fda"
DRIFT_BILLERS = {"CP-00651", "CP-00652", "CP-00653"}


def _day(spec, timestamp: str) -> int:
    return (parse_ts(timestamp) - day_start_ts(spec.start, 1)) // 86_400 + 1


def test_training_corpus_has_exact_counts(generated: dict[str, Path]) -> None:
    manifest = json.loads((generated["training"] / "manifest.json").read_text())
    counts = manifest["counts"]
    assert counts["requests"] == 100_000
    assert counts["accounts"] == 400
    assert counts["fraud"] == 1_040
    assert counts["legitimate"] == 98_960
    assert counts["unusual_legitimate"] == 5_000
    assert counts["routine_legitimate"] == 93_960
    for category in FRAUD_CATEGORIES:
        assert counts["by_scenario"][category] == 200
    assert counts["by_scenario"]["invoice_redirect"] == 40
    expected = {
        "train": (60_000, 600, 3_000),
        "calibration": (10_000, 100, 500),
        "validation": (10_000, 100, 500),
        "test": (20_000, 240, 1_000),
    }
    for name, (requests, fraud, unusual) in expected.items():
        bucket = counts["by_partition"][name]
        assert (bucket["requests"], bucket["fraud"], bucket["unusual_legitimate"]) == (requests, fraud, unusual)
        by_category = bucket["fraud_by_category"]
        assert {by_category[c] for c in FRAUD_CATEGORIES} == {TRAINING_SPEC.fraud_per_category[name]}
        assert by_category.get("invoice_redirect", 0) == (40 if name == "test" else 0)


def test_demo_dataset_has_exact_counts(generated: dict[str, Path]) -> None:
    counts = json.loads((generated["demo"] / "manifest.json").read_text())["counts"]
    assert (counts["requests"], counts["accounts"], counts["fraud"], counts["legitimate"]) == (10_000, 400, 120, 9_880)
    assert counts["unusual_legitimate"] == 500
    for category in FRAUD_CATEGORIES:
        assert counts["by_scenario"][category] == 20
    assert counts["by_scenario"]["invoice_redirect"] == 20


def test_validation_passes_for_both_datasets(generated: dict[str, Path]) -> None:
    assert validate_dataset(generated["training"], TRAINING_SPEC) == []
    assert validate_dataset(generated["demo"], DEMO_SPEC) == []


def test_every_request_is_an_instant_transfer(generated: dict[str, Path]) -> None:
    for name in ("training", "demo"):
        rows = read_jsonl(generated[name] / "requests.jsonl")
        assert {row["transfer_type"] for row in rows} == {"instant"}, name
    assert {r.transfer_type for r in load_dataset(generated["demo"]).requests} == {"instant"}


def _rewrite_first_request(source: Path, target: Path, change) -> None:
    import shutil

    shutil.copytree(source, target)
    rows = read_jsonl(target / "requests.jsonl")
    change(rows[0])
    (target / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.mark.parametrize(
    "change",
    [lambda row: row.pop("transfer_type"), lambda row: row.update(transfer_type="standard")],
    ids=["missing", "not_instant"],
)
def test_transfer_type_is_mandatory_and_instant(generated: dict[str, Path], tmp_path: Path, change) -> None:
    broken = tmp_path / "demo"
    _rewrite_first_request(generated["demo"], broken, change)
    problems = validate_dataset(broken, DEMO_SPEC)
    assert any("transfer_type" in p for p in problems), problems
    with pytest.raises(DatasetError, match="transfer_type"):
        load_dataset(broken, check_hashes=False)


def test_same_seed_reproduces_identical_files(generated: dict[str, Path], tmp_path: Path) -> None:
    again = generate_dataset(tmp_path / "demo", DEMO_SPEC)
    original = json.loads((generated["demo"] / "manifest.json").read_text())
    assert {k: v["sha256"] for k, v in again["files"].items()} == {
        k: v["sha256"] for k, v in original["files"].items()
    }


def test_accounts_are_stable_and_uniquely_named_across_datasets(generated: dict[str, Path]) -> None:
    training = read_jsonl(generated["training"] / "accounts.jsonl")
    demo = read_jsonl(generated["demo"] / "accounts.jsonl")
    assert training == demo
    names = [a["display_name"] for a in training]
    assert len(set(names)) == 400
    assert all(" " in name for name in names)


def test_runtime_files_never_contain_labels(generated: dict[str, Path]) -> None:
    forbidden = {"is_fraud", "is_anomaly", "scenario", "campaign_id", "partition", "expected_decision"}
    for name in ("requests.jsonl", "payment_events.jsonl", "authentication_events.jsonl", "accounts.jsonl"):
        first = read_jsonl(generated["training"] / name)[0]
        assert not forbidden & set(first), name
    dataset = load_dataset(generated["demo"])
    assert not hasattr(dataset.requests[0], "is_fraud")


def test_demo_is_independent_and_after_corpus(generated: dict[str, Path]) -> None:
    assert cross_dataset_fingerprint_overlap(generated["training"], generated["demo"]) == 0
    training_last = read_jsonl(generated["training"] / "requests.jsonl")[-1]["timestamp"]
    demo_first = read_jsonl(generated["demo"] / "requests.jsonl")[0]["timestamp"]
    assert parse_ts(demo_first) > parse_ts(training_last)


def test_campaigns_stay_in_one_partition(generated: dict[str, Path]) -> None:
    labels = load_labels(generated["training"])
    splits = load_splits(generated["training"])
    parts: dict[str, set[str]] = {}
    for request_id, label in labels.items():
        if label["campaign_id"]:
            parts.setdefault(label["campaign_id"], set()).add(splits[request_id])
    assert parts and all(len(p) == 1 for p in parts.values())


def test_forty_cold_start_accounts_first_request_in_test(generated: dict[str, Path]) -> None:
    manifest = json.loads((generated["training"] / "manifest.json").read_text())
    cold = set(manifest["cold_start"]["account_ids"])
    assert len(cold) == 40
    splits = load_splits(generated["training"])
    first_partition: dict[str, str] = {}
    for row in read_jsonl(generated["training"] / "requests.jsonl"):
        first_partition.setdefault(row["sender_id"], splits[row["request_id"]])
    assert all(first_partition.get(a) == "test" for a in cold)
    assert sum(1 for a, p in first_partition.items() if p == "test") == 40


def test_fraud_is_not_identified_by_one_field(generated: dict[str, Path]) -> None:
    labels = load_labels(generated["training"])
    dataset = load_dataset(generated["training"])
    fraud_methods = Counter(r.auth_method for r in dataset.requests if labels[r.request_id]["is_fraud"])
    legit_methods = Counter(r.auth_method for r in dataset.requests if not labels[r.request_id]["is_fraud"])
    assert set(fraud_methods) <= set(legit_methods)
    subtle_fraud = sum(1 for l in labels.values() if l["is_fraud"] and not l["is_anomaly"])
    assert subtle_fraud > 50
    failures = {e.session_id for e in dataset.auth_events if e.event_type == "login_failure"}
    legit_with_failures = sum(
        1 for r in dataset.requests if r.session_id in failures and not labels[r.request_id]["is_fraud"]
    )
    assert legit_with_failures > 1_000


def test_security_evidence_is_time_aware(generated: dict[str, Path]) -> None:
    rows = read_jsonl(generated["training"] / "security_events.jsonl")
    store = EvidenceStore(
        [evidence_from_row(r) for r in rows],
        trusted_sources=("bank_security_operations", "fraud_investigations_unit"),
        blocking_types=("compromised_session", "fraud_controlled_beneficiary", "compromised_device"),
    )
    fixtures = [evidence_from_row(r) for r in rows if r["note"].startswith("Fixture")]
    assert len(fixtures) == 10
    dataset = load_dataset(generated["training"])
    fixture_entities = {f.entity_id for f in fixtures}
    paid_to_fixtures = [r for r in dataset.requests if r.recipient_id in fixture_entities]
    assert paid_to_fixtures, "fixtures must reference entities that legitimate requests use"
    assert all(not store.active_for(r) for r in paid_to_fixtures)
    campaign = [evidence_from_row(r) for r in rows if not r["note"].startswith("Fixture")]
    assert all(e.recorded_at >= e.effective_from or e.entity_type != "counterparty" for e in campaign)


def _training_rows(generated):
    path = generated["training"]
    requests = read_jsonl(path / "requests.jsonl")
    labels = load_labels(path)
    return requests, labels, load_splits(path)


def test_invoice_redirect_only_in_test_partition(generated: dict[str, Path]) -> None:
    requests, labels, splits = _training_rows(generated)
    accounts = {a["account_id"]: a for a in read_jsonl(generated["training"] / "accounts.jsonl")}
    first_day: dict[str, int] = {}
    seen_devices: dict[str, list[tuple[int, str]]] = {}
    for row in requests:
        first_day.setdefault(row["recipient_id"], _day(TRAINING_SPEC, row["timestamp"]))
        seen_devices.setdefault(row["sender_id"], []).append((parse_ts(row["timestamp"]), row["device_id"]))
    redirect = [r for r in requests if labels[r["request_id"]]["scenario"] == "invoice_redirect"]
    assert len(redirect) == 40
    campaign_day: dict[str, int] = {}
    for row in redirect:
        campaign = labels[row["request_id"]]["campaign_id"]
        campaign_day[campaign] = min(campaign_day.get(campaign, 999), _day(TRAINING_SPEC, row["timestamp"]))
    for row in redirect:
        label = labels[row["request_id"]]
        assert splits[row["request_id"]] == "test"
        assert label["is_fraud"] and not label["is_anomaly"]
        assert accounts[row["sender_id"]]["account_type"] == "business"
        assert first_day[row["recipient_id"]] == campaign_day[label["campaign_id"]]
        earlier = {d for t, d in seen_devices[row["sender_id"]] if t < parse_ts(row["timestamp"])}
        assert row["device_id"] in earlier
        assert row["auth_method"] in {"biometric", "password_otp"}
    assert not any(
        labels[r["request_id"]]["scenario"] == "invoice_redirect" and splits[r["request_id"]] != "test"
        for r in requests
    )


def test_invoice_redirect_has_multi_victim_campaigns(generated: dict[str, Path]) -> None:
    requests, labels, _ = _training_rows(generated)
    senders: dict[str, set[str]] = {}
    for row in requests:
        label = labels[row["request_id"]]
        if label["scenario"] == "invoice_redirect":
            senders.setdefault(label["campaign_id"], set()).add(row["sender_id"])
    assert senders
    multi = sum(1 for s in senders.values() if len(s) >= 2)
    assert multi / len(senders) >= 0.25


def test_drift_billers_first_paid_in_test_period(generated: dict[str, Path]) -> None:
    requests, _, _ = _training_rows(generated)
    paid = [r for r in requests if r["recipient_id"] in DRIFT_BILLERS]
    assert paid
    days = [_day(TRAINING_SPEC, r["timestamp"]) for r in paid]
    assert min(days) >= 81
    first_by_sender: dict[str, int] = {}
    for row, day in zip(paid, days):
        first_by_sender[row["sender_id"]] = min(first_by_sender.get(row["sender_id"], 999), day)
    assert len(first_by_sender) >= 80
    assert len(set(first_by_sender.values())) >= 15
    counterparties = {c["counterparty_id"] for c in read_jsonl(generated["training"] / "counterparties.jsonl")}
    assert DRIFT_BILLERS <= counterparties


def test_drift_raises_routine_amounts(generated: dict[str, Path]) -> None:
    requests, labels, _ = _training_rows(generated)
    window: dict[str, list[float]] = {"before": [], "after": []}
    for row in requests:
        if labels[row["request_id"]]["scenario"] != "routine_payment":
            continue
        day = _day(TRAINING_SPEC, row["timestamp"])
        if 61 <= day <= 80:
            window["before"].append(row["amount"])
        elif day >= 81:
            window["after"].append(row["amount"])
    ratio = statistics.median(window["after"]) / statistics.median(window["before"])
    assert 1.05 <= ratio <= 1.20


def test_account_params_unchanged_by_drift() -> None:
    params = _build_account_params(build_accounts(), _build_base_counterparties())
    digest = hashlib.sha256(json.dumps([asdict(p) for p in params], sort_keys=True).encode()).hexdigest()
    assert digest == ACCOUNT_PARAMS_SNAPSHOT

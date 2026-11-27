"""The Kaggle-style MIA dataset: structure, counts, limits, leakage guarantees, reproducibility."""

from __future__ import annotations

import json
import re
import shutil
import statistics
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from config.limits import MIA_LIMITS
from datagen.dataset import (
    DatasetError,
    cross_file_fingerprint_overlap,
    load_labels,
    load_partition,
    read_rows,
    reproducibility_problems,
    storage_listing_problems,
    validate_dataset,
)
from datagen.generator import (
    ALL_FRAUD_TYPES,
    ANOMALY_TYPES,
    FRAUD_TYPES,
    LABEL_COLUMNS,
    MIA_SPEC,
    ROW_COLUMNS,
    TEST_ONLY_FRAUD_TYPES,
    DatasetSpec,
    Partition,
    day_start_ts,
    generate_dataset,
    holiday_days,
    write_rows,
)
from datagen.identities import build_accounts
from datagen.records import PAYMENT_SCHEMES, QR_SCHEMES, parse_ts
from runtime.evidence import EvidenceStore, evidence_from_row

PARTS = ("train", "validation", "test")
DRIFT_BILLERS = {"CP-00651", "CP-00652", "CP-00653"}


@pytest.fixture(scope="module")
def rows(generated: dict[str, Path]) -> dict[str, list[dict]]:
    return {name: read_rows(generated[name]) for name in PARTS}


def _day(timestamp: str) -> int:
    return (parse_ts(timestamp) - day_start_ts(MIA_SPEC.start, 1)) // 86_400 + 1


def _assessed(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["assessed"]]


# -- structure -------------------------------------------------------------------------------


def test_storage_contains_exactly_three_partition_files(generated: dict[str, Path], tmp_path: Path) -> None:
    assert sorted(p.name for p in generated["storage"].iterdir()) == ["test.json", "train.json", "validation.json"]
    assert storage_listing_problems(generated["storage"]) == []
    copy = tmp_path / "storage"
    shutil.copytree(generated["storage"], copy)
    (copy / "manifest.json").write_text("{}")
    (copy / "generator.py").write_text("")
    problems = storage_listing_problems(copy)
    assert problems and "manifest.json" in problems[0] and "generator.py" in problems[0]
    (copy / "test.json").unlink()
    assert any("missing test.json" in p for p in storage_listing_problems(copy))


def test_committed_storage_holds_only_the_three_files() -> None:
    from config.settings import load_settings

    storage = load_settings().storage_dir
    assert storage_listing_problems(storage) == []
    assert not list(storage.glob("*.py")) and not (storage / "data").exists()


def test_each_file_is_a_json_array_of_self_contained_rows(rows: dict[str, list[dict]]) -> None:
    for name in PARTS:
        assert isinstance(rows[name], list) and rows[name]
        assert all(tuple(r) == ROW_COLUMNS for r in rows[name]), name
    sample = _assessed(rows["test"])[0]
    assert isinstance(sample["session_events"], list) and isinstance(sample["security_evidence"], list)
    assert set(LABEL_COLUMNS) <= set(sample)


def test_validation_passes(generated: dict[str, Path]) -> None:
    assert validate_dataset(generated["storage"], generated["manifest"]) == []


def test_partitions_are_whole_chronological_days(rows: dict[str, list[dict]]) -> None:
    previous = ("", "")
    for partition in MIA_SPEC.partitions:
        days = {_day(r["timestamp"]) for r in rows[partition.name]}
        assert days == set(range(partition.first_day, partition.last_day + 1))
        for r in rows[partition.name]:
            assert (r["timestamp"], r["transaction_id"]) >= previous
            previous = (r["timestamp"], r["transaction_id"])
    ids = [r["transaction_id"] for name in PARTS for r in rows[name]]
    assert len(ids) == len(set(ids))


def test_counts_per_partition_and_fraud_type(rows: dict[str, list[dict]]) -> None:
    expected_assessed = {"train": 70_000, "validation": 15_000, "test": 15_000}
    per_type = {"train": 150, "validation": 30, "test": 30}
    for name in PARTS:
        assessed = _assessed(rows[name])
        assert len(assessed) == expected_assessed[name]
        fraud = Counter(r["fraud_type"] for r in assessed if r["is_fraud"])
        for fraud_type in FRAUD_TYPES:
            assert fraud[fraud_type] == per_type[name], (name, fraud_type)
        for fraud_type in TEST_ONLY_FRAUD_TYPES:
            assert fraud[fraud_type] == (30 if name == "test" else 0), (name, fraud_type)
        assert sum(1 for r in assessed if r["anomaly_type"]) == {"train": 3_500, "validation": 750, "test": 750}[name]
        assert len(rows[name]) > len(assessed)  # incoming context credits come on top


def test_test_only_fraud_types_absent_from_train_and_validation(rows: dict[str, list[dict]]) -> None:
    for name in ("train", "validation"):
        assert not {r["fraud_type"] for r in rows[name]} & set(TEST_ONLY_FRAUD_TYPES)
    assert {r["fraud_type"] for r in rows["test"] if r["fraud_type"]} == set(ALL_FRAUD_TYPES)


def test_anomaly_taxonomy_is_covered(rows: dict[str, list[dict]]) -> None:
    train = {r["anomaly_type"] for r in rows["train"] if r["anomaly_type"]}
    assert train == set(ANOMALY_TYPES)
    for name in PARTS:
        assert {r["anomaly_type"] for r in rows[name] if r["anomaly_type"]} <= set(ANOMALY_TYPES)
    festive = holiday_days(MIA_SPEC)
    assert all(_day(r["timestamp"]) in festive for name in PARTS for r in rows[name]
               if r["anomaly_type"] == "holiday_spike")


def test_campaigns_stay_inside_one_partition(rows: dict[str, list[dict]]) -> None:
    parts: dict[str, set[str]] = defaultdict(set)
    for name in PARTS:
        for r in rows[name]:
            if r["campaign_id"]:
                parts[r["campaign_id"]].add(name)
    assert parts and all(len(p) == 1 for p in parts.values())


def test_no_duplicate_fingerprints_across_files(generated: dict[str, Path]) -> None:
    assert cross_file_fingerprint_overlap(generated["storage"]) == 0


# -- MIA domain ------------------------------------------------------------------------------


def test_every_row_is_an_instant_mdl_transfer_on_a_known_scheme(rows: dict[str, list[dict]]) -> None:
    for name in PARTS:
        assert {r["transfer_type"] for r in rows[name]} == {"instant"}
        assert {r["currency"] for r in rows[name]} == {"MDL"}
        assert {r["payment_scheme"] for r in rows[name]} <= set(PAYMENT_SCHEMES)
    assert {r["payment_scheme"] for r in rows["train"]} == set(PAYMENT_SCHEMES)
    assert all(not r["assessed"] for name in PARTS for r in rows[name] if r["payment_scheme"] == "remittance_in")


def test_personal_payments_respect_the_mia_limits(rows: dict[str, list[dict]]) -> None:
    personal = MIA_LIMITS.personal
    assert (personal.per_transaction, personal.per_day, personal.per_month) == (5_000, 10_000, 30_000)
    assert MIA_LIMITS.business.per_transaction == 100_000
    for name in PARTS:
        day: dict[tuple[str, str], float] = defaultdict(float)
        month: dict[tuple[str, str], float] = defaultdict(float)
        for r in _assessed(rows[name]):
            if r["sender_type"] != "personal":
                assert r["amount"] <= MIA_LIMITS.business.per_transaction
                continue
            assert r["amount"] <= personal.per_transaction
            day[(r["sender_id"], r["timestamp"][:10])] += r["amount"]
            month[(r["sender_id"], r["timestamp"][:7])] += r["amount"]
        assert max(day.values()) <= personal.per_day + 1e-6
        assert max(month.values()) <= personal.per_month + 1e-6


def test_structuring_stays_just_under_the_cap(rows: dict[str, list[dict]]) -> None:
    structuring = [r for r in _assessed(rows["train"]) if r["fraud_type"] == "limit_structuring"]
    assert all(0.9 * 5_000 <= r["amount"] < 5_000 for r in structuring)
    per_day = Counter((r["sender_id"], r["timestamp"][:10]) for r in structuring)
    assert max(per_day.values()) <= 2


def test_aliases_follow_the_mia_phone_format(rows: dict[str, list[dict]]) -> None:
    pattern = re.compile(r"^\+373[67]\d{7}$")
    aliases = {r["sender_alias"] for r in rows["train"] if r["sender_alias"]} | {
        r["recipient_alias"] for r in rows["train"] if r["recipient_alias"]}
    assert aliases and all(pattern.match(a) for a in aliases)
    assert all(r["sender_alias"] is None for r in _assessed(rows["train"]) if r["sender_type"] == "business")


def test_qr_and_request_to_pay_fields(rows: dict[str, list[dict]]) -> None:
    for name in PARTS:
        for r in _assessed(rows[name]):
            assert (r["request_to_pay_id"] is not None) == (r["payment_scheme"] == "request_to_pay")
            assert (r["qr_payload_merchant_id"] is not None) == (r["payment_scheme"] in QR_SCHEMES)
            mismatch = r["qr_payload_merchant_id"] not in (None, r["recipient_id"])
            assert mismatch == (r["fraud_type"] == "qr_code_tampering")


def test_mule_campaigns_receive_context_credits_first(rows: dict[str, list[dict]]) -> None:
    credits = defaultdict(list)
    for r in rows["train"]:
        if not r["assessed"] and r["campaign_id"]:
            credits[r["campaign_id"]].append(r["timestamp"])
    mule = defaultdict(list)
    for r in _assessed(rows["train"]):
        if r["fraud_type"] == "money_mule":
            mule[r["campaign_id"]].append(r["timestamp"])
    assert mule and all(max(credits[c]) < min(times) for c, times in mule.items())


def test_forty_cold_start_accounts_first_pay_in_test(rows: dict[str, list[dict]], generated: dict[str, Path]) -> None:
    manifest = json.loads(generated["manifest"].read_text())
    cold = set(manifest["cold_start"]["account_ids"])
    assert len(cold) == 40
    earlier = {r["sender_id"] for name in ("train", "validation") for r in _assessed(rows[name])}
    assert not cold & earlier
    test_senders = {r["sender_id"] for r in _assessed(rows["test"])}
    assert cold <= test_senders and len(test_senders - earlier) == 40
    test_start = day_start_ts(MIA_SPEC.start, 86)
    assert all(parse_ts(r["sender_created_at"]) >= test_start - 86_400
               for r in _assessed(rows["test"]) if r["sender_id"] in cold)


def test_fraud_is_not_identified_by_one_field(rows: dict[str, list[dict]]) -> None:
    train = _assessed(rows["train"])
    fraud = [r for r in train if r["is_fraud"]]
    legit = [r for r in train if not r["is_fraud"]]
    assert {r["auth_method"] for r in fraud} <= {r["auth_method"] for r in legit}
    assert {r["payment_scheme"] for r in fraud} <= {r["payment_scheme"] for r in legit}
    assert sum(1 for r in fraud if not r["is_anomaly"]) > 300
    events = lambda r, kind: any(e["event_type"] == kind for e in r["session_events"])
    assert sum(1 for r in legit if events(r, "login_failure")) > 1_000
    assert sum(1 for r in legit if events(r, "alias_rebound")) > 50
    assert sum(1 for r in legit if events(r, "limit_changed")) > 50
    assert sum(1 for r in legit if r["payment_scheme"] == "request_to_pay") > 1_000
    assert sum(1 for r in legit if r["sender_type"] == "personal" and r["amount"] >= 4_500) > 100
    assert sum(1 for r in legit if r["ip_country"] != "MD") > 500


def test_security_evidence_is_time_aware(rows: dict[str, list[dict]]) -> None:
    items = {e["evidence_id"]: e for name in PARTS for r in rows[name] for e in r["security_evidence"]}
    fixtures = [evidence_from_row(e) for k, e in items.items() if k.startswith("SEC-FX")]
    assert len(fixtures) == 10
    store = EvidenceStore([evidence_from_row(e) for e in items.values()],
                          trusted_sources=("bank_security_operations", "fraud_investigations_unit"),
                          blocking_types=("compromised_session", "fraud_controlled_beneficiary", "compromised_device"))
    fixture_ids = {f.entity_id for f in fixtures}
    from datagen.records import request_from_row

    paid = [request_from_row(r) for name in PARTS for r in _assessed(rows[name]) if r["recipient_id"] in fixture_ids]
    assert paid, "fixtures must reference billers that legitimate customers keep paying"
    assert all(not any(e.evidence_id.startswith("SEC-FX") for e in store.active_for(p)) for p in paid)
    assert any(store.active_for(request_from_row(r)) for r in _assessed(rows["test"]) if r["is_fraud"])


# -- runtime view and leakage ----------------------------------------------------------------


def test_runtime_view_never_contains_labels(generated: dict[str, Path]) -> None:
    dataset = load_partition(generated["test"])
    request = dataset.requests[0]
    assert not any(hasattr(request, column) for column in LABEL_COLUMNS + ("scenario",))
    assert len(dataset.requests) == 15_000 and dataset.context_credits > 0
    assert dataset.accounts and dataset.auth_events and dataset.payment_events and dataset.security_events
    labels = load_labels([generated["test"]])
    assert len(labels) == 15_000 and set(next(iter(labels.values()))) == set(LABEL_COLUMNS)


def test_label_changes_do_not_change_the_runtime_view(generated: dict[str, Path], tmp_path: Path) -> None:
    rows = read_rows(generated["test"])
    flipped = [{**r, "is_fraud": not r["is_fraud"], "fraud_type": "x", "campaign_id": "C"} for r in rows]
    path = tmp_path / "test.json"
    write_rows(path, flipped)
    original, changed = load_partition(generated["test"]), load_partition(path)
    assert original.requests == changed.requests
    assert original.auth_events == changed.auth_events and original.payment_events == changed.payment_events


def _rewrite_first(generated: dict[str, Path], tmp_path: Path, change) -> Path:
    storage = tmp_path / "storage"
    shutil.copytree(generated["storage"], storage)
    rows = read_rows(storage / "test.json")
    change(rows[0])
    write_rows(storage / "test.json", rows)
    return storage


@pytest.mark.parametrize(
    "change",
    [lambda row: row.pop("transfer_type"), lambda row: row.update(transfer_type="standard")],
    ids=["missing", "not_instant"],
)
def test_transfer_type_is_mandatory_and_instant(generated: dict[str, Path], tmp_path: Path, change) -> None:
    storage = _rewrite_first(generated, tmp_path, change)
    problems = validate_dataset(storage, generated["manifest"])
    assert any("transfer_type" in p for p in problems), problems
    with pytest.raises(DatasetError, match="transfer_type"):
        load_partition(storage / "test.json")


def test_currency_must_be_mdl(generated: dict[str, Path], tmp_path: Path) -> None:
    storage = _rewrite_first(generated, tmp_path, lambda row: row.update(currency="EUR"))
    assert any("currency" in p for p in validate_dataset(storage, generated["manifest"]))
    with pytest.raises(DatasetError, match="currency"):
        load_partition(storage / "test.json")


def test_personal_cap_breach_is_reported(generated: dict[str, Path], tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    shutil.copytree(generated["storage"], storage)
    rows = read_rows(storage / "test.json")
    target = next(r for r in rows if r["assessed"] and r["sender_type"] == "personal")
    target["amount"] = 5_000.01
    write_rows(storage / "test.json", rows)
    assert any("per-transaction cap" in p for p in validate_dataset(storage, generated["manifest"]))


# -- reproducibility -------------------------------------------------------------------------

SMALL = replace(
    MIA_SPEC, seed=7, days=12, payments_per_day=150,
    partitions=(Partition("train", 1, 8), Partition("validation", 9, 10), Partition("test", 11, 12)),
    fraud_per_type={"train": 6, "validation": 3, "test": 3},
    test_only_fraud={"test": {"qr_code_tampering": 3, "invoice_redirect": 3}},
    anomalies={"train": 40, "validation": 10, "test": 10}, cold_start_accounts=5, drift_first_day=11,
)


def test_same_seed_reproduces_byte_identical_files(tmp_path: Path) -> None:
    assert isinstance(SMALL, DatasetSpec) and SMALL.start == date(2026, 1, 5)
    generate_dataset(tmp_path / "a", tmp_path / "a.json", SMALL)
    generate_dataset(tmp_path / "b", tmp_path / "b.json", SMALL)
    for name in PARTS:
        assert (tmp_path / "a" / f"{name}.json").read_bytes() == (tmp_path / "b" / f"{name}.json").read_bytes()
    assert validate_dataset(tmp_path / "a", tmp_path / "a.json", SMALL) == []


def test_full_dataset_regenerates_byte_identically(generated: dict[str, Path]) -> None:
    assert reproducibility_problems(generated["storage"]) == []


def test_accounts_are_stable_and_uniquely_named() -> None:
    first, second = build_accounts(), build_accounts()
    assert first == second and len({a.display_name for a in first}) == 400
    assert {a.home_country for a in first} == {"MD"}


# -- drift -----------------------------------------------------------------------------------


def test_drift_billers_first_paid_in_test_period(rows: dict[str, list[dict]]) -> None:
    earlier = [r for name in ("train", "validation") for r in rows[name] if r["recipient_id"] in DRIFT_BILLERS]
    assert not earlier
    paid = [r for r in _assessed(rows["test"]) if r["recipient_id"] in DRIFT_BILLERS]
    assert len({r["sender_id"] for r in paid}) >= 50


def test_drift_raises_routine_amounts(rows: dict[str, list[dict]]) -> None:
    def routine(name: str) -> list[float]:
        return [r["amount"] for r in _assessed(rows[name]) if not r["is_fraud"] and not r["anomaly_type"]
                and r["sender_type"] == "personal"]

    ratio = statistics.median(routine("test")) / statistics.median(routine("validation"))
    assert 1.03 <= ratio <= 1.25

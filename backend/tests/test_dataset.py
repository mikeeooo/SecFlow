"""The three dataset files: the checks `validate` runs, and that labels never reach the runtime view."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from backend.tests.conftest import ROOT, STORAGE
from datagen.dataset import DatasetError, load_partitions, read_rows, validate_dataset
from datagen.generator import TEST_ONLY_FRAUD_TYPES, generate_dataset, sha256_file, write_rows

MANIFEST = ROOT / "datagen" / "manifest.json"


@pytest.fixture(scope="module")
def rows() -> dict[str, list[dict]]:
    return {name: read_rows(STORAGE / f"{name}.json") for name in ("train", "validation", "test")}


def copy_storage(tmp_path: Path) -> Path:
    target = tmp_path / "storage"
    shutil.copytree(STORAGE, target, ignore=shutil.ignore_patterns(".*"))
    return target


def test_committed_dataset_is_valid() -> None:
    assert validate_dataset(STORAGE, MANIFEST) == []


def test_generator_reproduces_the_committed_files_byte_for_byte(tmp_path: Path) -> None:
    manifest = generate_dataset(tmp_path / "storage", tmp_path / "manifest.json")
    for name, part in manifest["partitions"].items():
        assert (
            sha256_file(tmp_path / "storage" / f"{name}.json")
            == sha256_file(STORAGE / f"{name}.json")
            == part["sha256"]
        )


def test_partitions_hold_the_documented_counts(rows) -> None:
    for name, days in (("train", 70), ("validation", 15), ("test", 15)):
        assessed = [r for r in rows[name] if r["assessed"]]
        assert len(assessed) == days * 1_000
    assert sum(r["is_fraud"] for r in rows["test"] if r["assessed"]) == 360


def test_test_only_fraud_types_never_appear_before_test(rows) -> None:
    for name in ("train", "validation"):
        assert not {r["fraud_type"] for r in rows[name]} & set(TEST_ONLY_FRAUD_TYPES)
    assert {r["fraud_type"] for r in rows["test"]} >= set(TEST_ONLY_FRAUD_TYPES)


def test_every_row_is_an_instant_mdl_payment_within_the_personal_cap(rows) -> None:
    for part in rows.values():
        assert {r["transfer_type"] for r in part} == {"instant"}
        assert {r["currency"] for r in part} == {"MDL"}
        assert max(r["amount"] for r in part if r["assessed"] and r["sender_type"] == "personal") <= 5_000


def test_runtime_view_ignores_labels(tmp_path: Path) -> None:
    original = load_partitions([STORAGE / "test.json"])
    flipped = tmp_path / "test.json"
    write_rows(
        flipped,
        [
            {**r, "is_fraud": not r["is_fraud"], "fraud_type": "money_mule", "campaign_id": "X"}
            for r in read_rows(STORAGE / "test.json")
        ],
    )
    changed = load_partitions([flipped])
    assert changed.requests == original.requests
    assert changed.auth_events == original.auth_events
    assert not {"is_fraud", "fraud_type", "campaign_id"} & set(type(original.requests[0]).__slots__)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r.update(currency="EUR"), "not instant transfers in MDL"),
        (
            lambda r: r.update(amount=9_000.0) if r["assessed"] and r["sender_type"] == "personal" else None,
            "above 5,000 MDL",
        ),
    ],
)
def test_validation_reports_broken_rows(tmp_path: Path, change, message: str) -> None:
    storage = copy_storage(tmp_path)
    rows = read_rows(storage / "validation.json")
    for row in rows[:50]:
        change(row)
    write_rows(storage / "validation.json", rows)
    problems = validate_dataset(storage, MANIFEST)
    assert any("SHA-256 differs" in p for p in problems)
    assert any(message in p for p in problems)


def test_validation_requires_exactly_three_files(tmp_path: Path) -> None:
    storage = copy_storage(tmp_path)
    (storage / "extra.json").write_text("[]")
    assert "exactly" in validate_dataset(storage, MANIFEST)[0]


def test_validation_reports_copied_rows_across_files(tmp_path: Path) -> None:
    storage = copy_storage(tmp_path)
    test_rows = read_rows(storage / "test.json")
    write_rows(
        storage / "test.json",
        [*test_rows, {**read_rows(storage / "train.json")[0], "transaction_id": "TX-COPY"}],
    )
    assert any("duplicates a row of train.json" in p for p in validate_dataset(storage, MANIFEST))


def test_loader_refuses_a_non_instant_payment(tmp_path: Path) -> None:
    path = tmp_path / "test.json"
    path.write_text(json.dumps([{**read_rows(STORAGE / "test.json")[0], "transfer_type": "standard"}]))
    with pytest.raises(DatasetError, match="instant"):
        load_partitions([path])

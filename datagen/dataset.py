"""Loading and checking the three dataset files (`storage/{train,validation,test}.json`).

`load_partitions` builds the runtime view: label columns are dropped as each row is read, and the
accounts, sign-in events, settlement ledger and security evidence are rebuilt from the rows.
`load_labels` returns the labels separately; only training and evaluation call it, never the app.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.limits import MIA_LIMITS
from datagen.generator import (
    ALL_FRAUD_TYPES,
    DAY_SECONDS,
    FRAUD_TYPES,
    LABEL_COLUMNS,
    MIA_SPEC,
    PARTITION_FILES,
    SETTLEMENT_SECONDS,
    TEST_ONLY_FRAUD_TYPES,
    row_fingerprint,
    sha256_file,
)
from datagen.records import (
    CURRENCY,
    INSTANT_TRANSFER,
    Account,
    AuthEvent,
    Counterparty,
    PaymentEvent,
    PaymentRequest,
    auth_from_session_event,
    parse_ts,
    request_from_row,
)


class DatasetError(RuntimeError):
    """A dataset file is missing, not valid JSON, or breaks the MIA contract."""


@dataclass(frozen=True)
class Dataset:
    """Runtime view of one or more partitions. Carries no label, fraud type or campaign."""

    accounts: dict[str, Account]
    counterparties: dict[str, Counterparty]
    requests: tuple[PaymentRequest, ...]
    payment_events: tuple[PaymentEvent, ...]
    auth_events: tuple[AuthEvent, ...]
    security_events: tuple[dict[str, Any], ...]
    partition_of: Mapping[str, str]  # transaction id -> train | validation | test
    context_credits: int  # incoming credits from outside SecFlow (history only, never assessed)


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Every row of one file, labels included (offline use only)."""
    if not path.exists():
        raise DatasetError(f"Missing dataset file: {path}. Run `python -m backend.cli generate`.")
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DatasetError(f"{path.name} is not valid JSON: {error}") from error
    if not isinstance(rows, list):
        raise DatasetError(f"{path.name} must contain a JSON array of transaction rows")
    return rows


def load_partitions(paths: Iterable[Path]) -> Dataset:
    """Runtime view of consecutive partition files, merged in order. Labels are dropped on read."""
    accounts: dict[str, Account] = {}
    counterparties: dict[str, Counterparty] = {}
    receivers: dict[str, Counterparty] = {}  # accounts seen only as recipients, shown like counterparties
    requests: list[PaymentRequest] = []
    payments: list[PaymentEvent] = []
    auth: dict[str, AuthEvent] = {}
    evidence: dict[str, dict[str, Any]] = {}
    partition_of: dict[str, str] = {}
    credits = 0
    for path in paths:
        for full_row in read_rows(path):
            row = {k: v for k, v in full_row.items() if k not in LABEL_COLUMNS}
            tid = row["transaction_id"]
            if row.get("transfer_type") != INSTANT_TRANSFER or row.get("currency") != CURRENCY:
                raise DatasetError(
                    f"Transaction {tid}: every payment must be an instant transfer in {CURRENCY}"
                )
            partition_of[tid] = path.stem
            ts = parse_ts(row["timestamp"])
            recipient = Counterparty(
                row["recipient_id"], row["recipient_name"], row["recipient_kind"], row["recipient_category"]
            )
            target = receivers if row["recipient_type"] == "account" else counterparties
            target[recipient.counterparty_id] = recipient
            if not row["assessed"]:  # an incoming credit from outside: history only
                credits += 1
                counterparties.setdefault(
                    row["sender_id"],
                    Counterparty(
                        row["sender_id"],
                        row["sender_name"],
                        "person" if row["sender_alias"] else "business",
                        "external_payer",
                    ),
                )
                payments.append(
                    PaymentEvent(
                        f"PE-{tid}",
                        ts,
                        "incoming_credit",
                        row["sender_id"],
                        row["recipient_id"],
                        float(row["amount"]),
                        row["currency"],
                        None,
                    )
                )
                continue
            accounts[row["sender_id"]] = Account(
                row["sender_id"],
                row["sender_name"],
                row["sender_type"],
                row["sender_home_country"],
                parse_ts(row["sender_created_at"]),
            )
            request = request_from_row(row)
            requests.append(request)
            payments.append(
                PaymentEvent(
                    f"PE-{tid}",
                    ts + SETTLEMENT_SECONDS,
                    "transfer_settled",
                    request.sender_id,
                    request.recipient_id,
                    request.amount,
                    request.currency,
                    tid,
                )
            )
            for event in row["session_events"]:
                auth.setdefault(
                    event["event_id"], auth_from_session_event(event, row["sender_id"], row["session_id"])
                )
            for item in row["security_evidence"]:
                evidence.setdefault(item["evidence_id"], item)
    for party_id, party in receivers.items():
        if party_id not in accounts:
            counterparties.setdefault(party_id, party)
    return Dataset(
        accounts=accounts,
        counterparties=counterparties,
        requests=tuple(sorted(requests, key=lambda r: (r.ts, r.request_id))),
        payment_events=tuple(sorted(payments, key=lambda e: (e.ts, e.event_id))),
        auth_events=tuple(sorted(auth.values(), key=lambda e: (e.ts, e.event_id))),
        security_events=tuple(evidence[k] for k in sorted(evidence)),
        partition_of=partition_of,
        context_credits=credits,
    )


def load_labels(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    """Offline only: the label columns of every assessed payment."""
    return {
        row["transaction_id"]: {column: row.get(column) for column in LABEL_COLUMNS}
        for path in paths
        for row in read_rows(path)
        if row.get("assessed")
    }


def read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DatasetError(f"Missing manifest: {path}. Run `python -m backend.cli generate`.")
    return json.loads(path.read_text(encoding="utf-8"))


def file_hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {path.name: sha256_file(path) for path in paths}


def validate_dataset(storage_dir: Path, manifest_path: Path) -> list[str]:
    """Every problem found in storage/ (an empty list means the dataset is valid)."""
    present = (
        {p.name for p in storage_dir.iterdir() if not p.name.startswith(".")}
        if storage_dir.exists()
        else set()
    )
    if present != set(PARTITION_FILES.values()):
        return [
            (
                f"storage/ must hold exactly {', '.join(sorted(PARTITION_FILES.values()))}; found "
                f"{', '.join(sorted(present)) or 'nothing'}"
            )
        ]
    paths = {name: storage_dir / file for name, file in PARTITION_FILES.items()}
    manifest = read_manifest(manifest_path)
    problems = [
        f"{path.name}: SHA-256 differs from the manifest"
        for name, path in paths.items()
        if sha256_file(path) != manifest["partitions"][name]["sha256"]
    ]
    limits = MIA_LIMITS.personal
    owner: dict[str, str] = {}  # row fingerprint -> partition, to find copies across files
    for name, path in paths.items():
        rows = read_rows(path)
        fraud: dict[str, int] = defaultdict(int)
        spent_day: dict[tuple[str, int], float] = defaultdict(float)
        spent_month: dict[tuple[str, str], float] = defaultdict(float)
        assessed = bad_type = over_cap = 0
        for row in rows:
            fingerprint = row_fingerprint(row)
            if owner.setdefault(fingerprint, name) != name:
                problems.append(
                    f"{name}: {row['transaction_id']} duplicates a row of {owner[fingerprint]}.json"
                )
            if row.get("transfer_type") != INSTANT_TRANSFER or row.get("currency") != CURRENCY:
                bad_type += 1
            if not row.get("assessed"):
                continue
            assessed += 1
            if row["fraud_type"]:
                fraud[row["fraud_type"]] += 1
            if row["sender_type"] == "personal":
                amount = float(row["amount"])
                over_cap += amount > limits.per_transaction + 1e-9
                spent_day[(row["sender_id"], parse_ts(row["timestamp"]) // DAY_SECONDS)] += amount
                spent_month[(row["sender_id"], row["timestamp"][:7])] += amount
        spec = next(p for p in MIA_SPEC.partitions if p.name == name)
        expected = (spec.last_day - spec.first_day + 1) * MIA_SPEC.payments_per_day
        if assessed != expected:
            problems.append(f"{name}: expected {expected} assessed payments, found {assessed}")
        if bad_type:
            problems.append(f"{name}: {bad_type} row(s) are not instant transfers in {CURRENCY}")
        if over_cap:
            problems.append(f"{name}: {over_cap} personal payment(s) above {limits.per_transaction:,.0f} MDL")
        if over := sum(v > limits.per_day + 1e-6 for v in spent_day.values()):
            problems.append(f"{name}: {over} personal sender-day(s) above the daily limit")
        if over := sum(v > limits.per_month + 1e-6 for v in spent_month.values()):
            problems.append(f"{name}: {over} personal sender-month(s) above the monthly limit")
        for fraud_type in ALL_FRAUD_TYPES:
            wanted = (
                MIA_SPEC.fraud_per_type[name]
                if fraud_type in FRAUD_TYPES
                else MIA_SPEC.test_only_fraud.get(name, {}).get(fraud_type, 0)
            )
            if fraud[fraud_type] != wanted:
                problems.append(f"{name}: {fraud[fraud_type]} {fraud_type} payments, expected {wanted}")
        if name != "test" and (leaked := [t for t in TEST_ONLY_FRAUD_TYPES if fraud[t]]):
            problems.append(f"{name}: test-only fraud types present: {', '.join(leaked)}")
        try:
            dataset = load_partitions([path])
        except DatasetError as error:
            problems.append(f"{name}: the runtime loader refuses the file ({error})")
            continue
        if any(hasattr(dataset.requests[0], column) for column in LABEL_COLUMNS):
            problems.append(f"{name}: label columns reach the runtime view")
    return problems

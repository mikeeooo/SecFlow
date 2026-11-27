"""Loading and validation of the Kaggle-style MIA partitions (`storage/{train,validation,test}.json`).

`load_partition` / `load_partitions` build the runtime view: the label columns are dropped as each
row is read, and accounts, counterparties, authentication events, the settlement ledger and the
trusted evidence stream are rebuilt from the self-contained rows. Labels have their own loader
(`load_labels`) that only offline training and evaluation call; the runtime service never does.
"""

from __future__ import annotations

import json
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.limits import MIA_LIMITS, MiaLimits
from datagen.generator import (
    ALL_FRAUD_TYPES,
    DAY_SECONDS,
    FRAUD_TYPES,
    LABEL_COLUMNS,
    MIA_SPEC,
    PARTITION_FILES,
    PARTITION_NAMES,
    ROW_COLUMNS,
    SETTLEMENT_SECONDS,
    TEST_ONLY_FRAUD_TYPES,
    DatasetSpec,
    day_start_ts,
    generate_dataset,
    row_fingerprint,
    sha256_file,
)
from datagen.records import (
    CURRENCY,
    INSTANT_TRANSFER,
    PAYMENT_SCHEMES,
    Account,
    AuthEvent,
    Counterparty,
    PaymentEvent,
    PaymentRequest,
    auth_from_session_event,
    check_instant,
    parse_ts,
    request_from_row,
)


class DatasetError(RuntimeError):
    """Raised when dataset files are missing, corrupted, or inconsistent."""


@dataclass(frozen=True)
class Dataset:
    """Runtime view of one or more partitions. Carries no label, scenario or campaign information."""

    paths: tuple[Path, ...]
    accounts: dict[str, Account]
    counterparties: dict[str, Counterparty]
    requests: tuple[PaymentRequest, ...]
    payment_events: tuple[PaymentEvent, ...]
    auth_events: tuple[AuthEvent, ...]
    security_events: tuple[dict[str, Any], ...]
    partition_of: Mapping[str, str]  # transaction id -> partition file stem (train | validation | test)
    context_credits: int


def partition_paths(storage_dir: Path) -> dict[str, Path]:
    return {name: storage_dir / PARTITION_FILES[name] for name in PARTITION_NAMES}


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Every row of one partition file, label columns included (offline use only)."""
    if not path.exists():
        raise DatasetError(f"Missing dataset file: {path}. Run `python -m backend.cli generate`.")
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DatasetError(f"{path.name} is not valid JSON: {error}") from error
    if not isinstance(rows, list):
        raise DatasetError(f"{path.name} must contain a JSON array of transaction rows")
    return rows


def runtime_rows(path: Path) -> list[dict[str, Any]]:
    """Rows with the label columns removed as soon as they are read."""
    return [{k: v for k, v in row.items() if k not in LABEL_COLUMNS} for row in read_rows(path)]


def load_partition(path: Path) -> Dataset:
    """Runtime view of one partition file (labels dropped)."""
    return load_partitions([path])


def load_partitions(paths: Iterable[Path]) -> Dataset:
    """Runtime view of several chronologically consecutive partitions, merged in order."""
    paths = tuple(paths)
    accounts: dict[str, Account] = {}
    recipient_only: dict[str, Counterparty] = {}
    counterparties: dict[str, Counterparty] = {}
    requests: list[PaymentRequest] = []
    payments: list[PaymentEvent] = []
    auth: dict[str, AuthEvent] = {}
    evidence: dict[str, dict[str, Any]] = {}
    partition_of: dict[str, str] = {}
    credits = 0
    for path in paths:
        for row in runtime_rows(path):
            try:
                check_instant(row)
                if row.get("currency") != CURRENCY:
                    raise ValueError(f"Transaction {row.get('transaction_id')}: currency must be {CURRENCY!r}")
            except ValueError as error:
                raise DatasetError(str(error)) from error
            tid = row["transaction_id"]
            partition_of[tid] = path.stem
            ts = parse_ts(row["timestamp"])
            recipient = row["recipient_id"]
            if row["recipient_type"] == "account":
                if recipient not in accounts:
                    recipient_only[recipient] = Counterparty(recipient, row["recipient_name"], row["recipient_kind"],
                                                             row["recipient_category"])
            else:
                counterparties[recipient] = Counterparty(recipient, row["recipient_name"], row["recipient_kind"],
                                                         row["recipient_category"])
            if not row["assessed"]:
                credits += 1
                counterparties.setdefault(row["sender_id"], Counterparty(
                    row["sender_id"], row["sender_name"], "person" if row["sender_alias"] else "business", "external_payer"))
                payments.append(PaymentEvent(f"PE-{tid}", ts, "incoming_credit", row["sender_id"], recipient,
                                             float(row["amount"]), row["currency"], None))
                continue
            accounts[row["sender_id"]] = Account(row["sender_id"], row["sender_name"], row["sender_type"],
                                                 row["sender_home_country"], parse_ts(row["sender_created_at"]))
            recipient_only.pop(row["sender_id"], None)
            try:
                request = request_from_row(row)
            except (KeyError, ValueError) as error:
                raise DatasetError(str(error)) from error
            requests.append(request)
            payments.append(PaymentEvent(f"PE-{tid}", ts + SETTLEMENT_SECONDS, "transfer_settled", row["sender_id"],
                                         recipient, request.amount, request.currency, tid))
            for event in row["session_events"]:
                auth.setdefault(event["event_id"], auth_from_session_event(event, row["sender_id"], row["session_id"]))
            for item in row["security_evidence"]:
                evidence.setdefault(item["evidence_id"], item)
    # An account that only receives payments in these partitions is shown by name, like a counterparty.
    for party_id, party in recipient_only.items():
        if party_id not in accounts:
            counterparties.setdefault(party_id, party)
    return Dataset(
        paths=paths,
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
    """Offline-only: label columns of the assessed payments. Never called by the runtime service."""
    labels: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_rows(path):
            if row.get("assessed"):
                labels[row["transaction_id"]] = {column: row.get(column) for column in LABEL_COLUMNS}
    return labels


def read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DatasetError(f"Missing manifest: {path}. Run `python -m backend.cli generate`.")
    return json.loads(path.read_text(encoding="utf-8"))


def file_hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {path.name: sha256_file(path) for path in paths}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def storage_listing_problems(storage_dir: Path) -> list[str]:
    """`storage/` holds exactly the three partition files and nothing else (dotfiles such as .DS_Store aside)."""
    expected = set(PARTITION_FILES.values())
    present = {p.name for p in storage_dir.iterdir() if not p.name.startswith(".")} if storage_dir.exists() else set()
    problems = []
    if missing := sorted(expected - present):
        problems.append(f"storage/ is missing {', '.join(missing)}")
    if extra := sorted(present - expected):
        problems.append(f"storage/ must contain only the three partition files; also found {', '.join(extra)}")
    return problems


def validate_dataset(storage_dir: Path, manifest_path: Path, spec: DatasetSpec = MIA_SPEC,
                     limits: MiaLimits = MIA_LIMITS, check_listing: bool = True) -> list[str]:
    """Return every validation problem (an empty list means the dataset is valid)."""
    problems = storage_listing_problems(storage_dir) if check_listing else []
    paths = partition_paths(storage_dir)
    if any(not p.exists() for p in paths.values()):
        return problems or ["partition files are missing"]
    rows_by_part = {name: read_rows(path) for name, path in paths.items()}
    manifest = read_manifest(manifest_path) if manifest_path.exists() else None
    if manifest is None:
        problems.append(f"manifest {manifest_path} is missing")
    else:
        for name, path in paths.items():
            recorded = manifest["partitions"][name]["sha256"]
            if sha256_file(path) != recorded:
                problems.append(f"{path.name}: SHA-256 differs from the manifest")
    problems += _validate_rows(rows_by_part, spec, limits)
    for name, path in paths.items():
        try:
            dataset = load_partition(path)
        except DatasetError as error:
            problems.append(f"{path.name}: the runtime loader refuses the file ({error})")
            continue
        request = dataset.requests[0]
        if any(hasattr(request, column) for column in LABEL_COLUMNS) or set(LABEL_COLUMNS) & set(runtime_rows(path)[0]):
            problems.append(f"{path.name}: label columns reach the runtime view")
    return problems


def _validate_rows(rows_by_part: dict[str, list[dict[str, Any]]], spec: DatasetSpec, limits: MiaLimits) -> list[str]:
    problems: list[str] = []
    start = day_start_ts(spec.start, 1)
    seen_ids: set[str] = set()
    previous: tuple[int, str] | None = None
    campaigns: dict[str, set[str]] = defaultdict(set)
    fingerprints: dict[str, str] = {}
    duplicate_fingerprints = 0
    senders: dict[str, tuple[Any, ...]] = {}
    for partition in spec.partitions:
        rows = rows_by_part[partition.name]
        assessed = 0
        fraud: dict[str, int] = defaultdict(int)
        anomalies = 0
        day_spent: dict[tuple[str, int], float] = defaultdict(float)
        month_spent: dict[tuple[str, str], float] = defaultdict(float)
        bad_columns = bad_instant = bad_currency = bad_scheme = over_cap = 0
        for row in rows:
            if tuple(row) != ROW_COLUMNS:
                bad_columns += 1
            tid = row.get("transaction_id")
            if tid in seen_ids:
                problems.append(f"duplicate transaction_id {tid}")
            seen_ids.add(tid)
            ts = parse_ts(row["timestamp"])
            if previous is not None and (ts, tid) < previous:
                problems.append(f"{partition.name}: rows are not in chronological order at {tid}")
            previous = (ts, tid)
            day = (ts - start) // DAY_SECONDS + 1
            if not partition.first_day <= day <= partition.last_day:
                problems.append(f"{partition.name}: {tid} falls on day {day}, outside days "
                                f"{partition.first_day}-{partition.last_day}")
            if row.get("transfer_type") != INSTANT_TRANSFER:
                bad_instant += 1
            if row.get("currency") != CURRENCY:
                bad_currency += 1
            if row.get("payment_scheme") not in PAYMENT_SCHEMES:
                bad_scheme += 1
            fingerprint = row_fingerprint(row)
            if fingerprint in fingerprints and fingerprints[fingerprint] != tid:
                duplicate_fingerprints += 1
            fingerprints[fingerprint] = tid
            if row.get("campaign_id"):
                campaigns[row["campaign_id"]].add(partition.name)
            if not row.get("assessed"):
                continue
            assessed += 1
            identity = (row["sender_type"], row["sender_name"], row["sender_home_country"], row["sender_created_at"])
            if senders.setdefault(row["sender_id"], identity) != identity:
                problems.append(f"sender {row['sender_id']} has inconsistent identity fields at {tid}")
                senders[row["sender_id"]] = identity
            if row["fraud_type"]:
                fraud[row["fraud_type"]] += 1
                if not row["is_fraud"]:
                    problems.append(f"{tid}: fraud_type without is_fraud")
            elif row["is_fraud"]:
                problems.append(f"{tid}: is_fraud without fraud_type")
            if row["anomaly_type"]:
                anomalies += 1
                if row["is_fraud"] or not row["is_anomaly"]:
                    problems.append(f"{tid}: anomaly_type rows must be legitimate and is_anomaly")
            if row["sender_type"] == "personal":
                amount = float(row["amount"])
                personal = limits.personal
                if amount > personal.per_transaction + 1e-9:
                    over_cap += 1
                day_spent[(row["sender_id"], ts // DAY_SECONDS)] += amount
                month_spent[(row["sender_id"], row["timestamp"][:7])] += amount
        expected = (partition.last_day - partition.first_day + 1) * spec.payments_per_day
        if assessed != expected:
            problems.append(f"{partition.name}: expected {expected} assessed payments, found {assessed}")
        for name, count in ((n, c) for n, c in (("columns", bad_columns), ("transfer_type", bad_instant),
                                                  ("currency", bad_currency), ("payment_scheme", bad_scheme)) if c):
            hint = {"transfer_type": f"mandatory transfer_type={INSTANT_TRANSFER!r}",
                    "currency": f"currency={CURRENCY!r}", "columns": "the documented column order",
                    "payment_scheme": "a known MIA payment scheme"}[name]
            problems.append(f"{partition.name}: {count} row(s) without {hint}")
        if over_cap:
            problems.append(f"{partition.name}: {over_cap} personal payment(s) above the "
                            f"{limits.personal.per_transaction:,.0f} MDL per-transaction cap")
        if over_day := sum(1 for v in day_spent.values() if v > limits.personal.per_day + 1e-6):
            problems.append(f"{partition.name}: {over_day} personal sender-day(s) above the daily limit")
        if limits.personal.per_month is not None and (
            over_month := sum(1 for v in month_spent.values() if v > limits.personal.per_month + 1e-6)
        ):
            problems.append(f"{partition.name}: {over_month} personal sender-month(s) above the monthly limit")
        for fraud_type in ALL_FRAUD_TYPES:
            wanted = spec.fraud_per_type[partition.name] if fraud_type in FRAUD_TYPES else \
                spec.test_only_fraud.get(partition.name, {}).get(fraud_type, 0)
            if fraud.get(fraud_type, 0) != wanted:
                problems.append(f"{partition.name}: {fraud_type} count {fraud.get(fraud_type, 0)}, expected {wanted}")
        for fraud_type in TEST_ONLY_FRAUD_TYPES:
            if partition.name != "test" and fraud.get(fraud_type):
                problems.append(f"{partition.name}: test-only fraud type {fraud_type} present")
        if anomalies != spec.anomalies[partition.name]:
            problems.append(f"{partition.name}: {anomalies} unusual-legitimate rows, expected "
                            f"{spec.anomalies[partition.name]}")
    if crossing := sorted(c for c, parts in campaigns.items() if len(parts) > 1):
        problems.append(f"{len(crossing)} campaign(s) cross a partition boundary, first {crossing[0]}")
    if duplicate_fingerprints:
        problems.append(f"{duplicate_fingerprints} duplicate row fingerprint(s) across files")
    return problems


def reproducibility_problems(storage_dir: Path, spec: DatasetSpec = MIA_SPEC) -> list[str]:
    """Regenerate from the seed into a temporary directory and compare the files byte for byte."""
    with tempfile.TemporaryDirectory(prefix="secflow-regen-") as tmp:
        out = Path(tmp)
        generate_dataset(out / "storage", out / "manifest.json", spec)
        return [
            f"{name}: regenerating from seed {spec.seed} does not reproduce the committed file byte for byte"
            for name, path in partition_paths(storage_dir).items()
            if sha256_file(path) != sha256_file(out / "storage" / PARTITION_FILES[name])
        ]


def cross_file_fingerprint_overlap(storage_dir: Path) -> int:
    seen: dict[str, str] = {}
    overlap = 0
    for name, path in partition_paths(storage_dir).items():
        for row in read_rows(path):
            fingerprint = row_fingerprint(row)
            if fingerprint in seen and seen[fingerprint] != name:
                overlap += 1
            seen.setdefault(fingerprint, name)
    return overlap

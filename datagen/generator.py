"""Seeded synthetic MIA Plăți Instant dataset generator for SecFlow (Kaggle-style partitions).

The generator simulates behaviour first: routine instant payments in MDL over the MIA payment
schemes, incoming credits (salary, client payments, diaspora remittances, mule inflows), unusual but
legitimate episodes, twelve fraud types, authentication sessions and a separate stream of simulated
trusted security evidence. Only then does it write three self-contained JSON files:

    storage/train.json        days 1-70
    storage/validation.json   days 71-85
    storage/test.json         days 86-100

Each file is a JSON array of transaction rows. Every row carries all the context of the payment
(parties, session events, trusted evidence about its entities) plus the Kaggle-style label columns
`is_fraud`, `is_anomaly`, `fraud_type`, `anomaly_type` and `campaign_id`. The runtime loader drops
the label columns. Generation metadata (seeds, versions, counts, hashes) goes to
`datagen/manifest.json`, never into `storage/`.

No generated payment breaks an MIA limit: personal payments stay within 5,000 MDL per transaction,
10,000 MDL per UTC day and 30,000 MDL per calendar month (`config/policy.json`, `mia_limits`).
Everything here is synthetic; no real person, business, phone number or fraud case is represented.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from config.limits import MIA_LIMITS, MiaLimits
from datagen.identities import (
    BILLER_CATEGORIES,
    BUSINESS_PREFIXES,
    HOME_COUNTRY,
    IDENTITY_SEED,
    PUBLIC_OFFICES,
    QR_CATEGORIES,
    AccountIdentity,
    AliasBook,
    CounterpartyIdentity,
    build_accounts,
    build_counterparties,
)
from datagen.records import CURRENCY, INSTANT_TRANSFER, QR_SCHEMES

GENERATOR_VERSION = "secflow-mia-generator-2.0.0"
SCHEMA_VERSION = "2.0"
DAY_SECONDS = 86_400
HOUR = 3_600
SETTLEMENT_SECONDS = (
    5  # MIA credits the payee in under 15 s; the runtime ledger settles every payment 5 s later
)
MIN_PAYMENT = 10.0  # smallest generated payment in MDL

FRAUD_TYPES = (
    "account_takeover_phishing",
    "sim_swap_takeover",
    "vishing_impersonation",
    "request_to_pay_scam",
    "marketplace_scam",
    "investment_scam",
    "romance_scam",
    "money_mule",
    "limit_structuring",
    "coordinated_beneficiary",
)
TEST_ONLY_FRAUD_TYPES = ("qr_code_tampering", "invoice_redirect")
ALL_FRAUD_TYPES = FRAUD_TYPES + TEST_ONLY_FRAUD_TYPES
ANOMALY_TYPES = (
    "near_limit_planned_purchase",
    "limit_increase_then_large_payment",
    "new_device_legit",
    "travel_abroad",
    "payroll_burst",
    "salary_day_spike",
    "diaspora_remittance_spend",
    "group_payment_new_payee",
    "holiday_spike",
    "night_payment",
    "new_merchant_qr",
    "legit_alias_change",
    "household_repeat_payments",
    "business_settlement",
)
ANOMALY_WEIGHTS = (0.08, 0.06, 0.09, 0.08, 0.07, 0.07, 0.06, 0.07, 0.06, 0.08, 0.08, 0.06, 0.08, 0.06)
LABEL_COLUMNS = ("is_fraud", "is_anomaly", "fraud_type", "anomaly_type", "campaign_id")
PARTITION_NAMES = ("train", "validation", "test")
PARTITION_FILES = {name: f"{name}.json" for name in PARTITION_NAMES}
ROW_COLUMNS = (
    "transaction_id",
    "timestamp",
    "transfer_type",
    "payment_scheme",
    "assessed",
    "amount",
    "currency",
    "sender_id",
    "sender_type",
    "sender_name",
    "sender_alias",
    "sender_home_country",
    "sender_created_at",
    "recipient_id",
    "recipient_type",
    "recipient_name",
    "recipient_alias",
    "recipient_kind",
    "recipient_category",
    "alias_registered_at",
    "channel",
    "device_id",
    "session_id",
    "auth_method",
    "ip_country",
    "session_events",
    "request_to_pay_id",
    "qr_payload_merchant_id",
    "security_evidence",
) + LABEL_COLUMNS

TRUSTED_SOURCES = ("bank_security_operations", "fraud_investigations_unit")
UNTRUSTED_SOURCE = "external_tip_feed"
BASE_COUNTERPARTIES = 650
POPULAR_BILLERS = 25
REMITTERS = 90
DRIFT_BILLERS = 3
DRIFT_ADOPTION_SHARE = 0.30
DRIFT_PAYMENT_PROBABILITY = 0.15
DRIFT_BILLER_KINDS = ("Broadband", "Water Services", "Streaming")
BILLER_KINDS = (
    ("Utilities", "utility"),
    ("Telecom", "utility"),
    ("Energy", "utility"),
    ("Insurance", "finance"),
    ("Property Management", "housing"),
)
FOREIGN_TRAVEL = ("RO", "IT", "DE", "FR", "GB", "ES", "PT", "IL", "TR", "UA", "IE", "CZ", "PL", "US")
VPN_COUNTRIES = ("NL", "US", "DE", "GB", "SG", "SC")
DIASPORA_COUNTRIES = ("IT", "RO", "DE", "FR", "GB", "IL", "PT", "ES", "IE", "US", "CA")
AUTH_RANK = {
    "session_revoked": 0,
    "alias_rebound": 1,
    "limit_changed": 2,
    "login_failure": 3,
    "login_success": 4,
}


@dataclass(frozen=True)
class Partition:
    name: str
    first_day: int  # 1-based, inclusive
    last_day: int  # inclusive


@dataclass(frozen=True)
class DatasetSpec:
    seed: int
    start: date
    days: int
    payments_per_day: int  # assessed outgoing payments per day (incoming context credits come on top)
    partitions: tuple[Partition, ...]
    fraud_per_type: dict[str, int]  # partition -> assessed fraud rows per type in FRAUD_TYPES
    test_only_fraud: dict[str, dict[str, int]]  # partition -> type -> rows, for TEST_ONLY_FRAUD_TYPES
    anomalies: dict[str, int]  # partition -> unusual-but-legitimate rows
    cold_start_partition: str
    cold_start_accounts: int
    drift_first_day: int
    drift_amount_factor: float
    holidays: tuple[tuple[str, date], ...] = ()

    def partition_of_day(self, day: int) -> Partition:
        for partition in self.partitions:
            if partition.first_day <= day <= partition.last_day:
                return partition
        raise ValueError(f"day {day} outside every partition")


HOLIDAYS = (
    ("orthodox_christmas", date(2026, 1, 7)),
    ("martisor", date(2026, 3, 1)),
    ("womens_day", date(2026, 3, 8)),
    ("orthodox_easter", date(2026, 4, 12)),
    ("first_of_september", date(2026, 9, 1)),
)

MIA_SPEC = DatasetSpec(
    seed=42,
    start=date(2026, 1, 5),
    days=100,
    payments_per_day=1_000,
    partitions=(Partition("train", 1, 70), Partition("validation", 71, 85), Partition("test", 86, 100)),
    fraud_per_type={"train": 150, "validation": 30, "test": 30},
    test_only_fraud={"test": {"qr_code_tampering": 30, "invoice_redirect": 30}},
    anomalies={"train": 3_500, "validation": 750, "test": 750},
    cold_start_partition="test",
    cold_start_accounts=40,
    drift_first_day=86,
    drift_amount_factor=1.12,
    holidays=HOLIDAYS,
)


def iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def day_start_ts(start: date, day: int) -> int:
    moment = datetime(start.year, start.month, start.day, tzinfo=UTC) + timedelta(days=day - 1)
    return int(moment.timestamp())


def _month_key(ts: int) -> tuple[int, int]:
    moment = datetime.fromtimestamp(ts, UTC)
    return moment.year, moment.month


# ---------------------------------------------------------------------------
# Generator-only account behaviour (never written to the dataset)
# ---------------------------------------------------------------------------


@dataclass
class AccountParams:
    account_id: str
    account_type: str
    created_at: int
    cold_start: bool
    activity_weight: float
    median_amount: float
    amount_sigma: float
    payees: list[str]
    payee_weights: list[float]
    payee_factors: list[float]
    employees: list[str]
    devices: list[str]
    hour_center: float
    hour_spread: float
    auth_mix: dict[str, float]
    paydays: tuple[int, int]
    employer: str
    salary: float
    alias_registered_at: int | None
    remitter: str | None
    remit_day: int


@dataclass
class _Universe:
    """Everything derived from the identity seed: accounts, base counterparties, behaviour parameters."""

    accounts: list[AccountIdentity]
    counterparties: list[CounterpartyIdentity]
    drift_billers: list[CounterpartyIdentity]
    params: dict[str, AccountParams]
    cp_alias_registered: dict[str, int]
    aliases: AliasBook


def _build_universe(spec: DatasetSpec) -> _Universe:
    accounts = build_accounts()
    aliases = AliasBook(np.random.default_rng(IDENTITY_SEED + 11))
    aliases.used.update(a.alias for a in accounts if a.alias)
    rng = np.random.default_rng(IDENTITY_SEED + 1000)
    reserved = {a.display_name for a in accounts}
    billers = []
    for index in range(POPULAR_BILLERS):
        label, category = BILLER_KINDS[index % len(BILLER_KINDS)]
        name = f"{BUSINESS_PREFIXES[index % len(BUSINESS_PREFIXES)]} {label}"
        if name in reserved:
            name = f"{name} Group"
        reserved.add(name)
        billers.append(CounterpartyIdentity(f"CP-{index + 1:05d}", name, "business", category))
    offices = []
    for index, name in enumerate(PUBLIC_OFFICES):
        reserved.add(name)
        offices.append(
            CounterpartyIdentity(
                f"CP-{POPULAR_BILLERS + index + 1:05d}", name, "public_institution", "government"
            )
        )
    first_other = POPULAR_BILLERS + len(PUBLIC_OFFICES) + 1
    others = build_counterparties(
        rng, BASE_COUNTERPARTIES - first_other + 1, "CP-", first_other, reserved, aliases
    )
    remitters = []
    for cp in build_counterparties(
        rng,
        REMITTERS,
        "CP-R",
        1,
        reserved | {c.display_name for c in others},
        aliases,
        business_share=0.0,
        width=4,
    ):
        country = DIASPORA_COUNTRIES[int(rng.integers(len(DIASPORA_COUNTRIES)))]
        remitters.append(
            CounterpartyIdentity(
                cp.counterparty_id, cp.display_name, "person", "individual", cp.alias, country
            )
        )
    reserved |= {c.display_name for c in others + remitters}
    drift = []
    for index in range(DRIFT_BILLERS):
        name = f"{BUSINESS_PREFIXES[(POPULAR_BILLERS + index) % len(BUSINESS_PREFIXES)]} {DRIFT_BILLER_KINDS[index]}"
        if name in reserved:
            name = f"{name} Group"
        reserved.add(name)
        drift.append(
            CounterpartyIdentity(f"CP-{BASE_COUNTERPARTIES + index + 1:05d}", name, "business", "utility")
        )
    counterparties = billers + offices + others + remitters
    start = day_start_ts(spec.start, 1)
    cp_alias_registered = {
        c.counterparty_id: start - int(rng.integers(30, 1_500)) * DAY_SECONDS
        for c in counterparties
        if c.alias
    }
    params = _build_account_params(spec, accounts, counterparties)
    return _Universe(accounts, counterparties, drift, params, cp_alias_registered, aliases)


def _build_account_params(
    spec: DatasetSpec, accounts: list[AccountIdentity], counterparties: list[CounterpartyIdentity]
) -> dict[str, AccountParams]:
    rng = np.random.default_rng(IDENTITY_SEED + 2000)
    start = day_start_ts(spec.start, 1)
    personal_ids = [a.account_id for a in accounts if a.account_type == "personal"]
    cold_ids = set(rng.choice(personal_ids, size=spec.cold_start_accounts, replace=False).tolist())
    eligible_internal = [a for a in personal_ids if a not in cold_ids]
    biller_ids = [
        c.counterparty_id for c in counterparties if c.counterparty_id <= f"CP-{POPULAR_BILLERS:05d}"
    ]
    offices = [c.counterparty_id for c in counterparties if c.kind == "public_institution"]
    business_cps = [
        c.counterparty_id
        for c in counterparties
        if c.kind == "business" and c.counterparty_id not in biller_ids
    ]
    merchants = [
        c.counterparty_id
        for c in counterparties
        if c.kind == "business" and c.category in QR_CATEGORIES and c.counterparty_id not in biller_ids
    ]
    person_cps = [
        c.counterparty_id
        for c in counterparties
        if c.kind == "person" and not c.counterparty_id.startswith("CP-R")
    ]
    remitters = [c.counterparty_id for c in counterparties if c.counterparty_id.startswith("CP-R")]
    cold_partition = next(p for p in spec.partitions if p.name == spec.cold_start_partition)
    params: dict[str, AccountParams] = {}
    for account in accounts:
        is_business = account.account_type == "business"
        cold = account.account_id in cold_ids
        if cold:
            last = max(
                cold_partition.first_day, min(cold_partition.first_day + 11, cold_partition.last_day - 2)
            )
            activation_day = int(rng.integers(cold_partition.first_day, last + 1))
            created = day_start_ts(spec.start, activation_day) - int(rng.integers(3_600, 20 * 3_600))
        else:
            created = start - int(rng.integers(120, 2_400)) * DAY_SECONDS
        payees: list[str] = []
        employees: list[str] = []
        if is_business:
            payees += rng.choice(business_cps, size=int(rng.integers(5, 11)), replace=False).tolist()
            payees += rng.choice(biller_ids, size=2, replace=False).tolist()
            for _ in range(int(rng.integers(4, 11))):
                pool = eligible_internal if rng.random() < 0.5 else person_cps
                candidate = str(rng.choice(pool))
                if candidate != account.account_id and candidate not in employees:
                    employees.append(candidate)
        else:
            payees += rng.choice(biller_ids, size=int(rng.integers(1, 4)), replace=False).tolist()
            friends = [
                f
                for f in rng.choice(eligible_internal, size=3, replace=False).tolist()
                if f != account.account_id
            ]
            payees += friends[: int(rng.integers(1, 3))]
            payees += rng.choice(merchants, size=int(rng.integers(2, 6)), replace=False).tolist()
            payees += rng.choice(person_cps, size=int(rng.integers(0, 3)), replace=False).tolist()
            if rng.random() < 0.4:
                payees.append(str(rng.choice(offices)))
        order = rng.permutation(len(payees))
        payees = [payees[i] for i in order]
        weights = [1.0 / (rank + 1) ** 1.15 for rank in range(len(payees))]
        total = sum(weights)
        factors = [float(np.exp(rng.normal(0, 0.45))) for _ in payees]
        devices = [f"DEV-{account.account_id[4:]}-A"]
        if rng.random() < 0.4:
            devices.append(f"DEV-{account.account_id[4:]}-B")
        auth_choice = rng.dirichlet([6, 3, 1, 1])
        first_payday = int(rng.integers(5, 13))
        params[account.account_id] = AccountParams(
            account_id=account.account_id,
            account_type=account.account_type,
            created_at=created,
            cold_start=cold,
            # Personal activity is bounded so that ordinary customers stay inside the MIA monthly limit;
            # businesses (not bound by the P2P caps) carry a larger share of the daily volume.
            activity_weight=float(
                np.exp(rng.normal(0, 0.85)) * 3.5
                if is_business
                else np.clip(np.exp(rng.normal(0, 0.5)), 0.35, 1.5)
            ),
            median_amount=float(np.exp(rng.normal(math.log(6_000 if is_business else 140), 0.5))),
            amount_sigma=float(rng.uniform(0.35, 0.6)),
            payees=payees,
            payee_weights=[w / total for w in weights],
            payee_factors=factors,
            employees=employees,
            devices=devices,
            hour_center=float(rng.normal(10.5, 1.5) if is_business else rng.normal(13.5, 2.8)),
            hour_spread=float(rng.uniform(1.8, 2.8) if is_business else rng.uniform(2.5, 4.2)),
            auth_mix={
                "biometric": float(auth_choice[0]),
                "password": float(auth_choice[1]),
                "otp": float(auth_choice[2]),
                "password_otp": float(auth_choice[3]),
            },
            paydays=(first_payday, int(rng.integers(24, 29))),
            employer=str(rng.choice(business_cps)),
            salary=float(np.exp(rng.normal(math.log(12_000), 0.35))),
            alias_registered_at=(created + int(rng.integers(0, 5)) * DAY_SECONDS) if account.alias else None,
            remitter=str(rng.choice(remitters)) if not is_business and rng.random() < 0.25 else None,
            remit_day=int(rng.integers(1, 29)),
        )
    return params


# ---------------------------------------------------------------------------
# Draft records
# ---------------------------------------------------------------------------


@dataclass
class Draft:
    """An assessed outgoing payment initiated by a SecFlow account."""

    ts: int
    sender: str
    recipient: str
    amount: float
    device: str
    session: str
    auth_method: str
    ip_country: str
    channel: str
    scheme: str
    sender_alias: str | None
    rtp_id: str | None
    qr_merchant: str | None
    fraud_type: str | None
    anomaly_type: str | None
    is_anomaly: bool
    campaign_id: str | None


@dataclass
class Credit:
    """A context-only incoming credit from an external payer (not assessed)."""

    ts: int
    sender: str
    recipient: str
    amount: float
    scheme: str
    campaign_id: str | None = None


@dataclass
class _Plan:
    kind: str  # fraud or anomaly type
    is_fraud: bool
    day: int
    last_day: int  # last day of the plan's partition: multi-day campaigns never cross it
    size: int
    campaign_id: str


@dataclass
class _Session:
    account: str
    events: list[dict[str, Any]] = field(default_factory=list)


class _Generator:
    def __init__(self, spec: DatasetSpec, limits: MiaLimits) -> None:
        self.spec = spec
        self.limits = limits
        self.rng = np.random.default_rng(spec.seed)
        universe = _build_universe(spec)
        self.universe = universe
        self.accounts = universe.accounts
        self.account_by_id = {a.account_id: a for a in self.accounts}
        self.params = universe.params
        self.cp: dict[str, CounterpartyIdentity] = {c.counterparty_id: c for c in universe.counterparties}
        for biller in universe.drift_billers:
            self.cp[biller.counterparty_id] = biller
        self.cp_alias_registered = dict(universe.cp_alias_registered)
        self.alias = {a.account_id: a.alias for a in self.accounts}
        self.alias_registered = {a: p.alias_registered_at for a, p in self.params.items()}
        self.aliases = universe.aliases
        self.devices = {a: list(p.devices) for a, p in self.params.items()}
        self.employees: dict[str, list[str]] = {}
        self.reserved_names = {a.display_name for a in self.accounts} | {
            c.display_name for c in self.cp.values()
        }
        self.base_ids = [
            c.counterparty_id for c in universe.counterparties if not c.counterparty_id.startswith("CP-R")
        ]
        self.biller_ids = [c for c in self.base_ids if c <= f"CP-{POPULAR_BILLERS:05d}"]
        self.merchant_ids = [
            c
            for c in self.base_ids
            if self.cp[c].kind == "business"
            and self.cp[c].category in QR_CATEGORIES
            and c not in self.biller_ids
        ]
        self.drift_adoption = self._drift_adoption()
        self.sessions: dict[str, _Session] = {}
        self.credits: list[Credit] = []
        self.evidence: list[dict[str, Any]] = []
        self.new_counterparties: list[str] = []
        self.day_spent: dict[tuple[str, int], float] = defaultdict(float)
        self.month_spent: dict[tuple[str, tuple[int, int]], float] = defaultdict(float)
        self._cp_index = 1
        self._device_counter = 0
        self._session_counter = 0
        self._evidence_counter = 0
        self._rtp_counter = 0
        self.start_ts = day_start_ts(spec.start, 1)

    # -- identities -------------------------------------------------------
    def _drift_adoption(self) -> dict[str, tuple[str, int]]:
        rng = np.random.default_rng(IDENTITY_SEED + 3000)
        billers = [c.counterparty_id for c in self.universe.drift_billers]
        adoption = {}
        for account in self.accounts:
            adopts = rng.random() < DRIFT_ADOPTION_SHARE
            biller = billers[int(rng.integers(len(billers)))]
            day = int(rng.integers(self.spec.drift_first_day, self.spec.days + 1))
            if adopts:
                adoption[account.account_id] = (biller, day)
        return adoption

    def _new_counterparty(self, ts: int, business_share: float = 0.35, recent_alias: float = 0.1) -> str:
        """A payee nobody has paid before. `recent_alias` is the chance its alias was bound days ago."""
        created = build_counterparties(
            self.rng,
            1,
            "CP-X",
            self._cp_index,
            self.reserved_names,
            self.aliases,
            business_share=business_share,
        )[0]
        self._cp_index += 1
        self.reserved_names.add(created.display_name)
        self.cp[created.counterparty_id] = created
        self.new_counterparties.append(created.counterparty_id)
        if created.alias:
            if self.rng.random() < recent_alias:
                registered = ts - int(self.rng.integers(2 * HOUR, 20 * DAY_SECONDS))
            else:
                registered = ts - int(self.rng.integers(30, 1_800)) * DAY_SECONDS
            self.cp_alias_registered[created.counterparty_id] = registered
        return created.counterparty_id

    def _new_device(self) -> str:
        self._device_counter += 1
        return f"DEV-N{self._device_counter:05d}"

    def _rtp(self) -> str:
        self._rtp_counter += 1
        return f"RTP-{self._rtp_counter:06d}"

    # -- helpers ----------------------------------------------------------
    def _choice(self, values: list[Any], weights: list[float] | None = None) -> Any:
        if weights is None:
            return values[int(self.rng.integers(len(values)))]
        return values[int(self.rng.choice(len(values), p=np.asarray(weights) / sum(weights)))]

    def _is_business(self, account: str) -> bool:
        return self.params[account].account_type == "business"

    def _active_accounts(self, ts: int) -> list[str]:
        return [a for a, p in self.params.items() if p.created_at <= ts - HOUR]

    def _room(self, sender: str, ts: int) -> float:
        """Largest single payment MIA would still accept from `sender` at `ts`."""
        limits = self.limits.for_segment(self.params[sender].account_type)
        room = min(limits.per_transaction, limits.per_day - self.day_spent[(sender, ts // DAY_SECONDS)])
        if limits.per_month is not None:
            room = min(room, limits.per_month - self.month_spent[(sender, _month_key(ts))])
        return max(0.0, room)

    def _fit(self, sender: str, ts: int, amount: float) -> float | None:
        """Clip an amount to the sender's MIA limits and book it; None when MIA would refuse any payment."""
        room = math.floor(self._room(sender, ts) * 100) / 100
        value = math.floor(min(amount, room) * 100) / 100
        if value < MIN_PAYMENT:
            return None
        self.day_spent[(sender, ts // DAY_SECONDS)] += value
        self.month_spent[(sender, _month_key(ts))] += value
        return value

    def _must_fit(self, sender: str, ts: int, amount: float) -> float:
        value = self._fit(sender, ts, amount)
        if value is None:
            value = self._fit(sender, ts, MIN_PAYMENT)
        if value is None:  # pragma: no cover - victims are chosen with room to spare
            raise RuntimeError(f"No MIA room left for scheduled payment by {sender} at {iso(ts)}")
        return value

    def _with_room(self, candidates: list[str], ts: int, needed: float) -> list[str]:
        """Accounts that can still send `needed` MDL today and this month (in one or several payments)."""

        def day_left(a: str) -> float:
            limits = self.limits.for_segment(self.params[a].account_type)
            return limits.per_day - self.day_spent[(a, ts // DAY_SECONDS)]

        return [a for a in candidates if day_left(a) >= needed and self._month_room(a, ts) >= needed]

    def _month_room(self, sender: str, ts: int) -> float:
        limits = self.limits.for_segment(self.params[sender].account_type)
        if limits.per_month is None:
            return float("inf")
        return limits.per_month - self.month_spent[(sender, _month_key(ts))]

    def _cap(self, sender: str) -> float:
        return self.limits.for_segment(self.params[sender].account_type).per_transaction

    def _amount(self, account: str, factor: float = 1.0, sigma: float | None = None) -> float:
        p = self.params[account]
        spread = p.amount_sigma if sigma is None else sigma
        value = p.median_amount * factor * math.exp(self.rng.normal(0, spread))
        return round(max(MIN_PAYMENT, value), 2)

    def _auth_method(self, account: str) -> str:
        mix = self.params[account].auth_mix
        return self._choice(list(mix.keys()), list(mix.values()))

    def _hour_offset(self, account: str) -> int:
        p = self.params[account]
        hour = self.rng.normal(p.hour_center, p.hour_spread) % 24
        return int(hour * HOUR + self.rng.integers(0, 60))

    def _clamp(self, ts: int, day_start: int) -> int:
        return max(day_start + 60, min(ts, day_start + DAY_SECONDS - 60))

    def _day_number(self, ts: int) -> int:
        return (ts - self.start_ts) // DAY_SECONDS + 1

    def _foreign_country(self, pool: tuple[str, ...] = FOREIGN_TRAVEL) -> str:
        return pool[int(self.rng.integers(len(pool)))]

    def _session(
        self,
        account: str,
        device: str,
        first_payment_ts: int,
        auth_method: str,
        failures: int,
        ip_country: str,
        extra: tuple[tuple[int, str, str], ...] = (),
    ) -> str:
        """Create a session whose sign-in succeeds shortly before `first_payment_ts`.

        `extra` holds (seconds before the successful sign-in, event type, reason) for alias rebinds,
        limit changes or an earlier administrative logout of the same session.
        """
        self._session_counter += 1
        session_id = f"SES-{self._session_counter:07d}"
        session = _Session(account)
        success = first_payment_ts - int(self.rng.integers(15, 420))
        ts = success - failures * int(self.rng.integers(20, 50))
        for _ in range(failures):
            session.events.append(
                self._auth_event(ts, "login_failure", auth_method, ip_country, device, "invalid_credentials")
            )
            ts += int(self.rng.integers(15, 45))
        for seconds_before, event_type, reason in extra:
            session.events.append(
                self._auth_event(
                    success - seconds_before, event_type, auth_method, ip_country, device, reason
                )
            )
        session.events.append(
            self._auth_event(success, "login_success", auth_method, ip_country, device, None)
        )
        self.sessions[session_id] = session
        return session_id

    @staticmethod
    def _auth_event(
        ts: int, event_type: str, method: str, ip: str, device: str, reason: str | None
    ) -> dict[str, Any]:
        return {
            "timestamp": ts,
            "event_type": event_type,
            "auth_method": method,
            "ip_country": ip,
            "device_id": device,
            "reason": reason,
        }

    def _scheme(self, sender: str, recipient: str) -> tuple[str, str | None, str | None]:
        """(payment scheme, request-to-pay id, QR merchant id) for an ordinary payment."""
        cp = self.cp.get(recipient)
        kind = "person" if cp is None or cp.kind == "person" else cp.kind
        category = cp.category if cp is not None else "individual"
        if recipient in self.account_by_id:
            kind = "business" if self._is_business(recipient) else "person"
        if self._is_business(sender):
            if kind == "public_institution":
                return "p2g", None, None
            return ("p2p_iban" if kind == "person" else "b2b"), None, None
        if kind == "public_institution":
            return "p2g", None, None
        if kind == "business" and category in BILLER_CATEGORIES:
            scheme = self._choice(["request_to_pay", "p2b_link", "p2b_qr_dynamic"], [0.25, 0.45, 0.30])
        elif kind == "business" and category in QR_CATEGORIES:
            scheme = self._choice(["p2b_qr_static", "p2b_qr_dynamic", "p2b_link"], [0.55, 0.30, 0.15])
        elif kind == "business":
            scheme = self._choice(["p2b_link", "p2p_iban"], [0.6, 0.4])
        else:
            scheme = self._choice(["p2p_alias", "p2p_iban", "request_to_pay"], [0.80, 0.17, 0.03])
        return (
            scheme,
            (self._rtp() if scheme == "request_to_pay" else None),
            (recipient if scheme in QR_SCHEMES else None),
        )

    def _draft(
        self,
        ts: int,
        sender: str,
        recipient: str,
        amount: float,
        device: str,
        session: str,
        auth_method: str,
        ip_country: str,
        *,
        fraud_type: str | None = None,
        anomaly_type: str | None = None,
        is_anomaly: bool = False,
        campaign_id: str | None = None,
        scheme: tuple[str, str | None, str | None] | None = None,
    ) -> Draft:
        scheme_id, rtp, qr = scheme if scheme is not None else self._scheme(sender, recipient)
        channel = "mobile_app" if device.endswith("-A") or self.rng.random() < 0.6 else "web"
        if scheme_id in QR_SCHEMES:
            channel = "mobile_app"  # QR codes are scanned with the banking app
        return Draft(
            ts,
            sender,
            recipient,
            amount,
            device,
            session,
            auth_method,
            ip_country,
            channel,
            scheme_id,
            self.alias[sender],
            rtp,
            qr,
            fraud_type,
            anomaly_type,
            is_anomaly,
            campaign_id,
        )

    def _add_evidence(
        self,
        evidence_type: str,
        subject_type: str,
        subject_id: str,
        recorded_at: int,
        effective_at: int,
        expires_at: int | None,
        status: str = "confirmed",
        source: str = TRUSTED_SOURCES[0],
        fixture: bool = False,
    ) -> None:
        self._evidence_counter += 1
        prefix = "SEC-FX" if fixture else "SEC"
        self.evidence.append(
            {
                "evidence_id": f"{prefix}-{self._evidence_counter:05d}",
                "evidence_type": evidence_type,
                "subject_type": subject_type,
                "subject_id": subject_id,
                "status": status,
                "source": source,
                "source_trust": "trusted" if source in TRUSTED_SOURCES else "untrusted",
                "recorded_at": recorded_at,
                "effective_at": effective_at,
                "expires_at": expires_at,
            }
        )

    def _campaign_start(self, day_start: int, duration: int, night_bias: float = 0.0) -> int:
        latest = max(7 * HOUR + 1, DAY_SECONDS - duration - 600)
        if self.rng.random() < night_bias:
            offset = int(self.rng.integers(600, min(5 * HOUR, latest)))
        else:
            offset = int(self.rng.integers(7 * HOUR, latest))
        return day_start + offset

    def _victim(
        self,
        day_start: int,
        needed: float = 0.0,
        allow_cold: bool = True,
        business: bool | None = False,
        exclude: set[str] | frozenset[str] = frozenset(),
    ) -> str:
        candidates = [
            a
            for a in self._active_accounts(day_start)
            if (allow_cold or not self.params[a].cold_start)
            and a not in exclude
            and (business is None or self._is_business(a) == business)
        ]
        roomy = self._with_room(candidates, day_start, max(needed, 1_500.0))
        return self._choice(roomy or candidates)

    def _victims(self, day_start: int, count: int, needed: float, business: bool | None = False) -> list[str]:
        candidates = [
            a
            for a in self._active_accounts(day_start)
            if business is None or self._is_business(a) == business
        ]
        roomy = self._with_room(candidates, day_start, max(needed, 1_500.0)) or candidates
        return [str(a) for a in self.rng.choice(roomy, size=min(count, len(roomy)), replace=False)]

    def _near_cap(self, sender: str, ts: int, low: float = 0.92) -> float:
        room = self._room(sender, ts)
        return room * float(self.rng.uniform(low, 0.9998))

    # -- incoming context credits ----------------------------------------
    def incoming_credits(self, day: int, day_start: int) -> None:
        current = datetime.fromtimestamp(day_start, UTC)
        for account, p in self.params.items():
            if p.created_at > day_start:
                continue
            if p.account_type == "personal":
                if current.day in p.paydays:
                    share = 0.4 if current.day == p.paydays[0] else 0.6
                    self.credits.append(
                        Credit(
                            day_start + int(self.rng.integers(6 * HOUR, 10 * HOUR)),
                            p.employer,
                            account,
                            round(p.salary * share * math.exp(self.rng.normal(0, 0.05)), 2),
                            "b2p_salary",
                        )
                    )
                if p.remitter and current.day == p.remit_day:
                    self.credits.append(
                        Credit(
                            day_start + int(self.rng.integers(8 * HOUR, 20 * HOUR)),
                            p.remitter,
                            account,
                            round(float(np.exp(self.rng.normal(math.log(3_000), 0.5))), 2),
                            "remittance_in",
                        )
                    )
            elif self.rng.random() < 0.8:
                payer = self._choice(self.base_ids)
                scheme = "b2b" if self.cp[payer].kind != "person" else "p2p_iban"
                self.credits.append(
                    Credit(
                        day_start + int(self.rng.integers(8 * HOUR, 18 * HOUR)),
                        payer,
                        account,
                        self._amount(account, 2.0),
                        scheme,
                    )
                )

    # -- routine ---------------------------------------------------------
    def routine(self, count: int, day_start: int) -> list[Draft]:
        day = self._day_number(day_start)
        factor_all = self.spec.drift_amount_factor if day >= self.spec.drift_first_day else 1.0
        active = self._active_accounts(day_start)
        weights = np.asarray([self.params[a].activity_weight for a in active])
        drafts: list[Draft] = []
        attempts = 0
        while len(drafts) < count:
            attempts += 1
            if attempts > count * 20:  # pragma: no cover - defensive
                raise RuntimeError(f"Could not place {count} routine payments on day {day}")
            sender = active[int(self.rng.choice(len(active), p=weights / weights.sum()))]
            p = self.params[sender]
            ts = self._clamp(day_start + self._hour_offset(sender), day_start)
            adoption = self.drift_adoption.get(sender)
            if adoption is not None and day >= adoption[1] and self.rng.random() < DRIFT_PAYMENT_PROBABILITY:
                recipient, factor = adoption[0], 1.0
            elif self.rng.random() < 0.06:
                # Ordinary people also pay payees nobody in the dataset has paid before.
                recipient = (
                    self._new_counterparty(ts, 0.6, recent_alias=0.12)
                    if self.rng.random() < 0.5
                    else self._choice(self.base_ids)
                )
                factor = float(np.exp(self.rng.normal(0, 0.4)))
            else:
                slot = int(self.rng.choice(len(p.payees), p=p.payee_weights))
                recipient = p.payees[slot]
                factor = p.payee_factors[slot]
            if recipient.startswith("ACC-") and self.params[recipient].created_at > ts:
                recipient = self.biller_ids[0]
            amount = self._fit(sender, ts, self._amount(sender, factor * factor_all))
            if amount is None:  # MIA would refuse: this customer has used today's or this month's limit
                continue
            if self.rng.random() < 0.0015:  # a legitimate device replacement becomes the new primary
                self.devices[sender] = [self._new_device()] + self.devices[sender][:1]
            devices = self.devices[sender]
            device = devices[0] if len(devices) == 1 or self.rng.random() < 0.88 else devices[1]
            failures = 1 if self.rng.random() < 0.03 else (2 if self.rng.random() < 0.006 else 0)
            ip = HOME_COUNTRY if self.rng.random() < 0.985 else self._foreign_country()
            method = self._auth_method(sender)
            session = self._session(sender, device, ts, method, failures, ip)
            drafts.append(self._draft(ts, sender, recipient, amount, device, session, method, ip))
        return drafts

    # -- fraud -----------------------------------------------------------
    def fraud(self, plan: _Plan, day_start: int) -> list[Draft]:
        drafts: list[Draft] = getattr(self, f"_fraud_{plan.kind}")(plan, day_start)
        if len(drafts) != plan.size:  # pragma: no cover - defensive
            raise RuntimeError(f"{plan.kind} produced {len(drafts)} instead of {plan.size}")
        return drafts

    def _span_days(self, plan: _Plan, low: int, high: int) -> int:
        return max(0, min(int(self.rng.integers(low, high + 1)), plan.last_day - plan.day))

    def _fraud_account_takeover_phishing(self, plan: _Plan, day_start: int) -> list[Draft]:
        victim = self._victim(day_start, needed=9_000)
        subtle = self.rng.random() < 0.2
        device = self._new_device()
        failures = 0 if subtle else int(min(5, 1 + self.rng.poisson(1.4)))
        ip = HOME_COUNTRY if subtle or self.rng.random() < 0.4 else self._foreign_country(VPN_COUNTRIES)
        method = self._choice(["password", "otp", "password", "password_otp"])
        start = self._campaign_start(day_start, plan.size * 1_800, night_bias=0.35)
        extra = (
            ((int(self.rng.integers(60, 300)), "limit_changed", "daily_limit_raised"),)
            if self.rng.random() < 0.3
            else ()
        )
        session = self._session(victim, device, start, method, failures, ip, extra)
        drafts, ts = [], start
        for _ in range(plan.size):
            payee = self._new_counterparty(ts, 0.1, recent_alias=0.6)
            amount = (
                self._amount(victim, float(self.rng.uniform(0.8, 1.6)), 0.15)
                if subtle
                else self._near_cap(victim, ts)
            )
            drafts.append(
                self._draft(
                    ts,
                    victim,
                    payee,
                    self._must_fit(victim, ts, amount),
                    device,
                    session,
                    method,
                    ip,
                    fraud_type=plan.kind,
                    is_anomaly=not subtle,
                    campaign_id=plan.campaign_id,
                    scheme=(self._choice(["p2p_alias", "p2p_iban"], [0.6, 0.4]), None, None),
                )
            )
            ts = self._clamp(ts + int(self.rng.integers(300, 1_800)), day_start)
        if self.rng.random() < 0.35:
            recorded = drafts[0].ts + int(self.rng.integers(90, 280))
            self._add_evidence(
                "compromised_session", "session", session, recorded, start - 400, recorded + 30 * DAY_SECONDS
            )
        return drafts

    def _fraud_sim_swap_takeover(self, plan: _Plan, day_start: int) -> list[Draft]:
        victim = self._victim(day_start, needed=9_000)
        subtle = self.rng.random() < 0.15
        device = self._new_device()
        method = self._choice(["otp", "otp", "password_otp", "password"])
        start = self._campaign_start(day_start, plan.size * 1_500 + 4 * HOUR, night_bias=0.3)
        rebound = int(self.rng.integers(20 * 60, 4 * HOUR))
        session = self._session(
            victim,
            device,
            start,
            method,
            int(self.rng.integers(0, 2)),
            HOME_COUNTRY,
            ((rebound, "alias_rebound", "sim_replaced"),),
        )
        payees = [
            self._new_counterparty(start, 0.05, recent_alias=0.5) for _ in range(int(self.rng.integers(1, 3)))
        ]
        drafts, ts = [], start
        for index in range(plan.size):
            if subtle:
                amount = self._amount(victim, float(self.rng.uniform(1.0, 2.5)), 0.2)
            else:
                # drain toward the daily limit: each payment takes most of what MIA still allows today
                remaining = plan.size - index
                daily_left = self.limits.personal.per_day - self.day_spent[(victim, ts // DAY_SECONDS)]
                amount = min(
                    self._room(victim, ts), daily_left / remaining * float(self.rng.uniform(0.9, 1.0))
                )
            drafts.append(
                self._draft(
                    ts,
                    victim,
                    payees[index % len(payees)],
                    self._must_fit(victim, ts, amount),
                    device,
                    session,
                    method,
                    HOME_COUNTRY,
                    fraud_type=plan.kind,
                    is_anomaly=not subtle,
                    campaign_id=plan.campaign_id,
                    scheme=("p2p_alias", None, None),
                )
            )
            ts = self._clamp(ts + int(self.rng.integers(120, 1_500)), day_start)
        if len(drafts) > 1 and self.rng.random() < 0.25:
            recorded = drafts[1].ts - int(self.rng.integers(5, 40))
            self._add_evidence(
                "compromised_device", "device", device, recorded, start - 600, recorded + 14 * DAY_SECONDS
            )
        return drafts

    def _fraud_vishing_impersonation(self, plan: _Plan, day_start: int) -> list[Draft]:
        victim = self._victim(day_start, needed=9_000, allow_cold=False)
        ordinary = self.rng.random() < 0.4
        device = self.devices[victim][0]
        payee = self._new_counterparty(day_start, 0.0, recent_alias=0.4)
        start = self._campaign_start(day_start, plan.size * 3_600, night_bias=0.0 if ordinary else 0.45)
        drafts, ts = [], start
        for _ in range(plan.size):
            method = self._choice(["biometric", "password_otp"])
            session = self._session(victim, device, ts, method, 0, HOME_COUNTRY)
            if ordinary:
                amount = self._amount(victim, float(self.rng.uniform(1.0, 2.5)), 0.2)
            else:
                amount = self._near_cap(victim, ts, low=0.85)
            scheme = (self._choice(["p2p_iban", "p2p_alias"], [0.6, 0.4]), None, None)
            drafts.append(
                self._draft(
                    ts,
                    victim,
                    payee,
                    self._must_fit(victim, ts, amount),
                    device,
                    session,
                    method,
                    HOME_COUNTRY,
                    fraud_type=plan.kind,
                    is_anomaly=not ordinary,
                    campaign_id=plan.campaign_id,
                    scheme=scheme,
                )
            )
            ts = self._clamp(ts + int(self.rng.integers(900, 3_600)), day_start)
        if self.rng.random() < 0.15:
            recorded = start - int(self.rng.integers(HOUR, 2 * DAY_SECONDS))
            self._add_evidence(
                "fraud_controlled_beneficiary",
                "counterparty",
                payee,
                recorded,
                recorded,
                None,
                source=TRUSTED_SOURCES[1],
            )
        return drafts

    def _fraud_request_to_pay_scam(self, plan: _Plan, day_start: int) -> list[Draft]:
        """One unknown requester sends Request-to-Pay messages disguised as 'receive money' to many victims."""
        requester = self._new_counterparty(day_start, 0.0, recent_alias=0.6)
        victims = self._victims(day_start, plan.size, 2_500)
        window = int(self.rng.integers(3 * HOUR, 9 * HOUR))
        start = self._campaign_start(day_start, window)
        quiet = self.rng.random() < 0.6
        base = float(self.rng.uniform(300, 2_500))
        drafts = []
        for victim in victims:
            ts = start + int(self.rng.integers(0, window))
            method = self._choice(["biometric", "password_otp", "password"], [0.6, 0.25, 0.15])
            session = self._session(victim, self.devices[victim][0], ts, method, 0, HOME_COUNTRY)
            amount = self._must_fit(victim, ts, base * float(self.rng.uniform(0.7, 1.3)))
            drafts.append(
                self._draft(
                    ts,
                    victim,
                    requester,
                    amount,
                    self.devices[victim][0],
                    session,
                    method,
                    HOME_COUNTRY,
                    fraud_type=plan.kind,
                    is_anomaly=not quiet,
                    campaign_id=plan.campaign_id,
                    scheme=("request_to_pay", self._rtp(), None),
                )
            )
        if self.rng.random() < 0.3 and len(drafts) > 2:
            recorded = sorted(d.ts for d in drafts)[1] + int(self.rng.integers(60, 900))
            self._add_evidence(
                "fraud_controlled_beneficiary", "counterparty", requester, recorded, recorded, None
            )
        return drafts

    def _fraud_marketplace_scam(self, plan: _Plan, day_start: int) -> list[Draft]:
        """A fake classifieds seller collects a mid-size payment from several buyers within a few days."""
        seller = self._new_counterparty(day_start, 0.0, recent_alias=0.5)
        span = self._span_days(plan, 0, 3)
        quiet = self.rng.random() < 0.6
        base = float(self.rng.uniform(500, 3_000))
        victims = self._victims(day_start, plan.size, 3_500)
        drafts = []
        for victim in victims:
            day = day_start + int(self.rng.integers(0, span + 1)) * DAY_SECONDS
            ts = self._clamp(day + self._hour_offset(victim), day)
            method = self._auth_method(victim)
            session = self._session(victim, self.devices[victim][0], ts, method, 0, HOME_COUNTRY)
            amount = self._must_fit(victim, ts, base * float(self.rng.uniform(0.6, 1.4)))
            drafts.append(
                self._draft(
                    ts,
                    victim,
                    seller,
                    amount,
                    self.devices[victim][0],
                    session,
                    method,
                    HOME_COUNTRY,
                    fraud_type=plan.kind,
                    is_anomaly=not quiet,
                    campaign_id=plan.campaign_id,
                    scheme=("p2p_alias", None, None),
                )
            )
        if self.rng.random() < 0.25 and len(drafts) > 1:
            recorded = min(d.ts for d in drafts) + int(self.rng.integers(HOUR, 6 * HOUR))
            self._add_evidence(
                "fraud_controlled_beneficiary",
                "counterparty",
                seller,
                recorded,
                recorded,
                None,
                source=TRUSTED_SOURCES[1],
            )
        return drafts

    def _escalating(
        self,
        plan: _Plan,
        day_start: int,
        low: float,
        high: float,
        span: tuple[int, int],
        payee: str,
        scheme: str,
        anomaly: bool,
    ) -> list[Draft]:
        days = max(self._span_days(plan, *span), min(plan.size - 1, plan.last_day - plan.day))
        offsets = sorted(
            int(x) for x in self.rng.choice(days + 1, size=plan.size, replace=days + 1 < plan.size)
        )
        steps = np.minimum(np.linspace(low, high, plan.size) * self.rng.uniform(0.9, 1.1, plan.size), 4_999.0)
        candidates = [
            a
            for a in self._active_accounts(day_start)
            if not self._is_business(a) and not self.params[a].cold_start
        ]
        roomy = [
            a
            for a in self._with_room(candidates, day_start, float(steps.max()) * 2)
            if self._month_room(a, day_start) >= float(steps.sum()) * 1.2
        ]
        victim = (
            self._choice(roomy)
            if roomy
            else max(candidates, key=lambda a: (self._month_room(a, day_start), a))
        )
        drafts = []
        for offset, amount in zip(offsets, steps, strict=True):
            day = day_start + offset * DAY_SECONDS
            ts = self._clamp(day + self._hour_offset(victim) + int(self.rng.integers(0, 1_800)), day)
            method = self._choice(["biometric", "password_otp"])
            session = self._session(victim, self.devices[victim][0], ts, method, 0, HOME_COUNTRY)
            drafts.append(
                self._draft(
                    ts,
                    victim,
                    payee,
                    self._must_fit(victim, ts, float(amount)),
                    self.devices[victim][0],
                    session,
                    method,
                    HOME_COUNTRY,
                    fraud_type=plan.kind,
                    is_anomaly=anomaly,
                    campaign_id=plan.campaign_id,
                    scheme=(scheme, None, None),
                )
            )
        return drafts

    def _fraud_investment_scam(self, plan: _Plan, day_start: int) -> list[Draft]:
        """Escalating repeated payments to the same new 'trading platform' payee over days or weeks."""
        platform = self._new_counterparty(day_start, 0.5, recent_alias=0.3)
        scheme = "p2p_iban" if self.cp[platform].kind == "business" else "p2p_alias"
        drafts = self._escalating(
            plan,
            day_start,
            float(self.rng.uniform(300, 900)),
            float(self.rng.uniform(3_000, 4_990)),
            (4, 12),
            platform,
            scheme,
            self.rng.random() < 0.5,
        )
        if self.rng.random() < 0.15:
            recorded = drafts[-1].ts - int(self.rng.integers(HOUR, DAY_SECONDS))
            self._add_evidence(
                "fraud_controlled_beneficiary",
                "counterparty",
                platform,
                recorded,
                recorded,
                None,
                source=TRUSTED_SOURCES[1],
            )
        return drafts

    def _fraud_romance_scam(self, plan: _Plan, day_start: int) -> list[Draft]:
        """Recurring moderate payments to one new person over weeks; looks like an ordinary relationship."""
        partner = self._new_counterparty(day_start, 0.0, recent_alias=0.3)
        base = float(self.rng.uniform(800, 2_500))
        drafts = self._escalating(
            plan,
            day_start,
            base * 0.8,
            base * 1.3,
            (6, 14),
            partner,
            self._choice(["p2p_alias", "p2p_iban"], [0.7, 0.3]),
            self.rng.random() < 0.2,
        )
        if self.rng.random() < 0.1:
            recorded = drafts[-1].ts - int(self.rng.integers(HOUR, DAY_SECONDS))
            self._add_evidence(
                "fraud_controlled_beneficiary",
                "counterparty",
                partner,
                recorded,
                recorded,
                None,
                source=TRUSTED_SOURCES[1],
            )
        return drafts

    def _fraud_money_mule(self, plan: _Plan, day_start: int) -> list[Draft]:
        """Several incoming credits, then rapid pass-through to one or two new beneficiaries."""
        mule = self._victim(day_start, needed=9_000)
        subtle = self.rng.random() < 0.2
        start = self._campaign_start(day_start, 10 * HOUR)
        total_in = float(self.rng.uniform(1_500, 4_000) if subtle else self.rng.uniform(5_000, 9_500))
        ts = start
        for share in self.rng.dirichlet(np.ones(int(self.rng.integers(2, 5)))):
            payer = self._new_counterparty(ts, 0.05, recent_alias=0.3)
            self.credits.append(
                Credit(ts, payer, mule, round(total_in * float(share), 2), "p2p_alias", plan.campaign_id)
            )
            ts += int(self.rng.integers(300, 3_600))
        beneficiaries = [
            self._new_counterparty(ts, 0.1, recent_alias=0.7) for _ in range(int(self.rng.integers(1, 3)))
        ]
        device = self.devices[mule][0] if self.rng.random() < 0.6 else self._new_device()
        method = self._auth_method(mule)
        ts += int(self.rng.integers(600, 4_000))
        session = self._session(mule, device, ts, method, 0, HOME_COUNTRY)
        per_payment = total_in * float(self.rng.uniform(0.75, 0.95)) / plan.size
        drafts = []
        for index in range(plan.size):
            amount = self._must_fit(mule, ts, per_payment * float(self.rng.uniform(0.85, 1.15)))
            drafts.append(
                self._draft(
                    self._clamp(ts, day_start),
                    mule,
                    beneficiaries[index % len(beneficiaries)],
                    amount,
                    device,
                    session,
                    method,
                    HOME_COUNTRY,
                    fraud_type=plan.kind,
                    is_anomaly=not subtle,
                    campaign_id=plan.campaign_id,
                    scheme=(self._choice(["p2p_alias", "p2p_iban"], [0.7, 0.3]), None, None),
                )
            )
            ts = self._clamp(ts + int(self.rng.integers(120, 1_500)), day_start)
        if self.rng.random() < 0.3:
            recorded = drafts[0].ts + int(self.rng.integers(60, 400))
            self._add_evidence(
                "fraud_controlled_beneficiary",
                "counterparty",
                beneficiaries[0],
                recorded,
                recorded,
                None,
                source=TRUSTED_SOURCES[1],
            )
        return drafts

    def _fraud_limit_structuring(self, plan: _Plan, day_start: int) -> list[Draft]:
        """A large sum split into payments just under the per-transaction cap, at most two per day."""
        days = max(self._span_days(plan, 1, 3), (plan.size + 1) // 2 - 1)
        days = min(days, plan.last_day - plan.day)
        slots = [d for d in range(days + 1) for _ in range(2)][: plan.size]
        if len(slots) < plan.size:  # partition edge: squeeze the remainder into the last day
            slots += [days] * (plan.size - len(slots))
        candidates = [
            a
            for a in self._active_accounts(day_start)
            if not self._is_business(a) and not self.params[a].cold_start
        ]
        needed = plan.size * self._cap(candidates[0])
        roomy = [
            a
            for a in candidates
            if self._month_room(a, day_start) >= needed
            and all(self._with_room([a], day_start + d * DAY_SECONDS, 9_990) for d in set(slots))
        ]
        sender = (
            self._choice(roomy)
            if roomy
            else max(candidates, key=lambda a: (self._month_room(a, day_start), a))
        )
        payees = [
            self._new_counterparty(day_start, 0.0, recent_alias=0.4)
            for _ in range(int(self.rng.integers(2, 4)))
        ]
        device = self.devices[sender][0]
        drafts = []
        for index, offset in enumerate(slots):
            day = day_start + offset * DAY_SECONDS
            first_of_day = index == 0 or slots[index - 1] != offset
            ts = self._clamp(
                day
                + self._hour_offset(sender)
                + (0 if first_of_day else int(self.rng.integers(600, 3 * HOUR))),
                day,
            )
            method = self._choice(["biometric", "password_otp", "password"])
            session = self._session(sender, device, ts, method, 0, HOME_COUNTRY)
            cap = min(self._cap(sender), self._room(sender, ts))
            amount = self._must_fit(sender, ts, cap * float(self.rng.uniform(0.92, 0.999)))
            drafts.append(
                self._draft(
                    ts,
                    sender,
                    payees[index % len(payees)],
                    amount,
                    device,
                    session,
                    method,
                    HOME_COUNTRY,
                    fraud_type=plan.kind,
                    is_anomaly=True,
                    campaign_id=plan.campaign_id,
                    scheme=(self._choice(["p2p_alias", "p2p_iban"]), None, None),
                )
            )
        return drafts

    def _fraud_coordinated_beneficiary(self, plan: _Plan, day_start: int) -> list[Draft]:
        """Several compromised accounts pay one new beneficiary within hours."""
        subtle = self.rng.random() < 0.25
        duration = int(self.rng.integers(*((4 * HOUR, 8 * HOUR) if subtle else (1_800, 3 * HOUR))))
        start = self._campaign_start(day_start, duration)
        base = float(self.rng.uniform(1_000, 4_500))
        senders = self._victims(day_start, plan.size, base * 1.6)
        beneficiary = self._new_counterparty(start, 0.1, recent_alias=0.6)
        drafts = []
        for sender in senders:
            ts = start + int(self.rng.integers(0, duration))
            new_device = self.rng.random() < 0.45
            device = self._new_device() if new_device else self.devices[sender][0]
            failures = int(self.rng.integers(0, 3)) if new_device else 0
            method = self._auth_method(sender)
            session = self._session(sender, device, ts, method, failures, HOME_COUNTRY)
            amount = self._must_fit(
                sender, ts, base * float(self.rng.uniform(0.5, 1.6) if subtle else self.rng.uniform(0.9, 1.1))
            )
            drafts.append(
                self._draft(
                    ts,
                    sender,
                    beneficiary,
                    amount,
                    device,
                    session,
                    method,
                    HOME_COUNTRY,
                    fraud_type=plan.kind,
                    is_anomaly=not subtle,
                    campaign_id=plan.campaign_id,
                    scheme=("p2p_alias", None, None),
                )
            )
        if self.rng.random() < 0.4:
            first = min(d.ts for d in drafts)
            recorded = (
                first - int(self.rng.integers(1_800, 2 * DAY_SECONDS))
                if self.rng.random() < 0.6
                else first + 900
            )
            self._add_evidence(
                "fraud_controlled_beneficiary", "counterparty", beneficiary, recorded, recorded, None
            )
        return drafts

    def _fraud_qr_code_tampering(self, plan: _Plan, day_start: int) -> list[Draft]:
        """Test only. A static merchant QR is replaced: customers pay a new account, the QR still names the merchant."""
        merchant = self._choice(self.merchant_ids)
        fraudster = self._new_counterparty(day_start, 0.0, recent_alias=0.6)
        window = int(self.rng.integers(4 * HOUR, 10 * HOUR))
        start = self._campaign_start(day_start, window)
        customers = self._victims(day_start, plan.size, 800)
        drafts = []
        for customer in customers:
            ts = start + int(self.rng.integers(0, window))
            method = self._choice(["biometric", "biometric", "password_otp"])
            session = self._session(customer, self.devices[customer][0], ts, method, 0, HOME_COUNTRY)
            amount = self._must_fit(customer, ts, float(np.exp(self.rng.normal(math.log(220), 0.5))))
            drafts.append(
                self._draft(
                    ts,
                    customer,
                    fraudster,
                    amount,
                    self.devices[customer][0],
                    session,
                    method,
                    HOME_COUNTRY,
                    fraud_type=plan.kind,
                    is_anomaly=False,
                    campaign_id=plan.campaign_id,
                    scheme=("p2b_qr_static", None, merchant),
                )
            )
        if self.rng.random() < 0.2 and len(drafts) > 1:
            recorded = min(d.ts for d in drafts) + int(self.rng.integers(HOUR, 3 * HOUR))
            self._add_evidence(
                "fraud_controlled_beneficiary", "counterparty", fraudster, recorded, recorded, None
            )
        return drafts

    def _fraud_invoice_redirect(self, plan: _Plan, day_start: int) -> list[Draft]:
        """Test only. Businesses pay a brand-new fake supplier an ordinary invoice amount (b2b)."""
        candidates = [
            a
            for a in self._active_accounts(day_start)
            if self._is_business(a) and not self.params[a].cold_start
        ]
        senders = [str(s) for s in self.rng.choice(candidates, size=plan.size, replace=False)]
        supplier = self._new_counterparty(day_start, 0.9, recent_alias=0.0)
        window = 6 * HOUR
        start = self._campaign_start(day_start, window)
        drafts = []
        for sender in senders:
            p = self.params[sender]
            slot = next(
                (
                    i
                    for i, payee in enumerate(p.payees)
                    if payee in self.cp and self.cp[payee].kind == "business" and payee not in self.biller_ids
                ),
                0,
            )
            ts = start + int(self.rng.integers(0, window))
            method = self._choice(["password_otp", "biometric"])
            session = self._session(sender, self.devices[sender][0], ts, method, 0, HOME_COUNTRY)
            amount = self._must_fit(sender, ts, self._amount(sender, p.payee_factors[slot], 0.1))
            drafts.append(
                self._draft(
                    ts,
                    sender,
                    supplier,
                    amount,
                    self.devices[sender][0],
                    session,
                    method,
                    HOME_COUNTRY,
                    fraud_type=plan.kind,
                    is_anomaly=False,
                    campaign_id=plan.campaign_id,
                    scheme=("b2b", None, None),
                )
            )
        return drafts

    # -- unusual but legitimate -----------------------------------------
    def unusual(self, plan: _Plan, day_start: int) -> list[Draft]:
        drafts: list[Draft] = getattr(self, f"_unusual_{plan.kind}")(plan, day_start)
        if len(drafts) != plan.size:  # pragma: no cover - defensive
            raise RuntimeError(f"{plan.kind} produced {len(drafts)} instead of {plan.size}")
        return drafts

    def _legit(self, ts: int, sender: str, recipient: str, amount: float, plan: _Plan, **kw: Any) -> Draft:
        device = kw.get("device") or self.devices[sender][0]
        method = kw.get("method") or self._auth_method(sender)
        ip = kw.get("ip") or HOME_COUNTRY
        session = kw.get("session") or self._session(
            sender, device, ts, method, kw.get("failures", 0), ip, kw.get("extra", ())
        )
        limits = self.limits.for_segment(self.params[sender].account_type)
        left = min(limits.per_day - self.day_spent[(sender, ts // DAY_SECONDS)], self._month_room(sender, ts))
        amount = min(
            amount, left * 0.75
        )  # an ordinary customer keeps a margin below the daily and monthly limits
        return self._draft(
            ts,
            sender,
            recipient,
            self._must_fit(sender, ts, amount),
            device,
            session,
            method,
            ip,
            anomaly_type=plan.kind,
            is_anomaly=True,
            campaign_id=plan.campaign_id,
            scheme=kw.get("scheme"),
        )

    def _known_payee(self, sender: str) -> str:
        p = self.params[sender]
        payee = p.payees[int(self.rng.choice(len(p.payees), p=p.payee_weights))]
        if payee.startswith("ACC-") and self.params[payee].cold_start:
            return self.biller_ids[0]
        return payee

    def _series(
        self,
        plan: _Plan,
        day_start: int,
        sender: str,
        recipients: list[str],
        amounts: list[float],
        spacing: tuple[int, int],
        night: float = 0.0,
        **kw: Any,
    ) -> list[Draft]:
        ts = self._campaign_start(day_start, len(amounts) * spacing[1], night_bias=night)
        limits = self.limits.for_segment(self.params[sender].account_type)
        room = min(limits.per_day - self.day_spent[(sender, ts // DAY_SECONDS)], self._month_room(sender, ts))
        if sum(amounts) > 0.8 * room:  # an ordinary customer stays comfortably inside the MIA limits
            amounts = [a * 0.8 * room / sum(amounts) for a in amounts]
        drafts = []
        for recipient, amount in zip(recipients, amounts, strict=True):
            drafts.append(self._legit(self._clamp(ts, day_start), sender, recipient, amount, plan, **kw))
            ts += int(self.rng.integers(*spacing))
        return drafts

    def _unusual_near_limit_planned_purchase(self, plan: _Plan, day_start: int) -> list[Draft]:
        sender = self._victim(day_start, needed=4_999)
        ts = self._clamp(day_start + self._hour_offset(sender), day_start)
        recipient = self._known_payee(sender) if self.rng.random() < 0.4 else self._new_counterparty(ts, 0.9)
        amount = min(self._cap(sender), self._room(sender, ts)) * float(self.rng.uniform(0.88, 0.9995))
        return [
            self._legit(
                ts, sender, recipient, amount, plan, method=self._choice(["password_otp", "biometric"])
            )
        ]

    def _unusual_limit_increase_then_large_payment(self, plan: _Plan, day_start: int) -> list[Draft]:
        sender = self._victim(day_start, needed=4_999)
        ts = self._clamp(day_start + self._hour_offset(sender), day_start)
        method = self._choice(["password_otp", "biometric"])
        extra = ((int(self.rng.integers(120, 1_800)), "limit_changed", "daily_limit_raised"),)
        session = self._session(sender, self.devices[sender][0], ts, method, 0, HOME_COUNTRY, extra)
        drafts = []
        for index in range(plan.size):
            recipient = (
                self._known_payee(sender) if self.rng.random() < 0.5 else self._new_counterparty(ts, 0.8)
            )
            amount = min(self._cap(sender), self._room(sender, ts)) * float(self.rng.uniform(0.8, 0.9995))
            drafts.append(self._legit(ts, sender, recipient, amount, plan, method=method, session=session))
            ts = self._clamp(ts + int(self.rng.integers(600, 3_600)), day_start)
        return drafts

    def _unusual_new_device_legit(self, plan: _Plan, day_start: int) -> list[Draft]:
        sender = self._victim(day_start)
        device = self._new_device()
        self.devices[sender] = [device] + self.devices[sender][:1]
        ts = self._clamp(day_start + self._hour_offset(sender), day_start)
        failures = int(self.rng.integers(1, 3)) if self.rng.random() < 0.3 else 0
        method = self._choice(["password_otp", "otp", "password"])
        session = self._session(sender, device, ts, method, failures, HOME_COUNTRY)
        drafts = []
        for _ in range(plan.size):
            drafts.append(
                self._legit(
                    ts,
                    sender,
                    self._known_payee(sender),
                    self._amount(sender),
                    plan,
                    device=device,
                    method=method,
                    session=session,
                )
            )
            ts = self._clamp(ts + int(self.rng.integers(120, 2_400)), day_start)
        return drafts

    def _unusual_travel_abroad(self, plan: _Plan, day_start: int) -> list[Draft]:
        sender = self._victim(day_start)
        ip = self._foreign_country()
        device = self.devices[sender][0] if self.rng.random() < 0.7 else self._new_device()
        recipients = [
            self._new_counterparty(day_start, 0.95) if self.rng.random() < 0.5 else self._known_payee(sender)
            for _ in range(plan.size)
        ]
        amounts = [self._amount(sender, float(self.rng.uniform(0.6, 3.0))) for _ in range(plan.size)]
        return self._series(
            plan, day_start, sender, recipients, amounts, (1_200, 5_400), night=0.3, device=device, ip=ip
        )

    def _unusual_payroll_burst(self, plan: _Plan, day_start: int) -> list[Draft]:
        businesses = [a for a in self._active_accounts(day_start) if self._is_business(a)]
        sender = self._choice(businesses)
        employees = self.employees.setdefault(sender, list(self.params[sender].employees))
        while len(employees) < plan.size:  # a new hire stays on the payroll for later runs
            employees.append(self._new_counterparty(day_start, 0.0))
        start = self._campaign_start(day_start, HOUR)
        method = self._auth_method(sender)
        session = self._session(sender, self.devices[sender][0], start, method, 0, HOME_COUNTRY)
        drafts, ts = [], start
        for index in range(plan.size):
            recipient = employees[index]
            if recipient.startswith("ACC-") and self.params[recipient].created_at > ts:
                recipient = employees[index] = self._new_counterparty(ts, 0.0)
            drafts.append(
                self._legit(
                    self._clamp(ts, day_start),
                    sender,
                    recipient,
                    round(float(self.rng.uniform(6_000, 20_000)), 2),
                    plan,
                    method=method,
                    session=session,
                    scheme=("b2p_salary", None, None),
                )
            )
            ts += int(self.rng.integers(20, 240))
        return drafts

    def _unusual_salary_day_spike(self, plan: _Plan, day_start: int) -> list[Draft]:
        dom = datetime.fromtimestamp(day_start, UTC).day
        paid = [
            a
            for a in self._active_accounts(day_start)
            if not self._is_business(a) and dom in self.params[a].paydays
        ]
        paid = self._with_room(paid, day_start, 3_000.0)
        sender = self._choice(paid) if paid else self._victim(day_start)
        recipients = [self._known_payee(sender) for _ in range(plan.size)]
        amounts = [self._amount(sender, float(self.rng.uniform(1.5, 4.0))) for _ in range(plan.size)]
        return self._series(plan, day_start, sender, recipients, amounts, (600, 3_600))

    def _unusual_diaspora_remittance_spend(self, plan: _Plan, day_start: int) -> list[Draft]:
        sender = self._victim(day_start, needed=3_000)
        p = self.params[sender]
        remitter = p.remitter or self._choice([c for c in self.cp if c.startswith("CP-R")])
        start = self._campaign_start(day_start, 6 * HOUR)
        received = round(float(np.exp(self.rng.normal(math.log(6_000), 0.4))), 2)
        self.credits.append(
            Credit(
                start - int(self.rng.integers(HOUR, 4 * HOUR)),
                remitter,
                sender,
                received,
                "remittance_in",
                plan.campaign_id,
            )
        )
        recipients = [
            self._known_payee(sender) if self.rng.random() < 0.5 else self._new_counterparty(start, 0.7)
            for _ in range(plan.size)
        ]
        amounts = [received * float(self.rng.uniform(0.15, 0.4)) for _ in range(plan.size)]
        return self._series(plan, day_start, sender, recipients, amounts, (900, 5_400))

    def _unusual_group_payment_new_payee(self, plan: _Plan, day_start: int) -> list[Draft]:
        recipient = self._new_counterparty(day_start, 0.0, recent_alias=0.1)
        senders = self._victims(day_start, plan.size, 600)
        start = self._campaign_start(day_start, 5 * HOUR)
        amount = float(self.rng.uniform(100, 500))
        return [
            self._legit(
                self._clamp(start + int(self.rng.integers(0, 5 * HOUR)), day_start),
                sender,
                recipient,
                round(amount * float(self.rng.uniform(0.95, 1.05)), 2),
                plan,
            )
            for sender in senders
        ]

    def _unusual_holiday_spike(self, plan: _Plan, day_start: int) -> list[Draft]:
        sender = self._victim(day_start, needed=2_000)
        recipients = [
            self._new_counterparty(day_start, 0.6) if self.rng.random() < 0.5 else self._known_payee(sender)
            for _ in range(plan.size)
        ]
        amounts = [self._amount(sender, float(self.rng.uniform(2.0, 6.0))) for _ in range(plan.size)]
        return self._series(plan, day_start, sender, recipients, amounts, (900, 4_800))

    def _unusual_night_payment(self, plan: _Plan, day_start: int) -> list[Draft]:
        sender = self._victim(day_start)
        ts = day_start + int(self.rng.integers(600, 4 * HOUR))
        return [self._legit(ts, sender, self._known_payee(sender), self._amount(sender), plan)]

    def _unusual_new_merchant_qr(self, plan: _Plan, day_start: int) -> list[Draft]:
        sender = self._victim(day_start)
        ts = self._clamp(day_start + self._hour_offset(sender), day_start)
        merchant = self._new_counterparty(ts, 1.0)
        while self.cp[merchant].category not in QR_CATEGORIES:
            merchant = self._new_counterparty(ts, 1.0)
        scheme = self._choice(["p2b_qr_static", "p2b_qr_dynamic"], [0.6, 0.4])
        return [
            self._legit(
                ts,
                sender,
                merchant,
                self._amount(sender, float(self.rng.uniform(0.5, 2.5))),
                plan,
                scheme=(scheme, None, merchant),
            )
        ]

    def _unusual_legit_alias_change(self, plan: _Plan, day_start: int) -> list[Draft]:
        """A customer moves MIA to a new phone number; half also start on a new phone."""
        candidates = [a for a in self._active_accounts(day_start) if self.alias[a]]
        sender = self._choice(self._with_room(candidates, day_start, 1_500.0) or candidates)
        ts = self._clamp(day_start + self._hour_offset(sender), day_start)
        rebound = int(self.rng.integers(10 * 60, 6 * HOUR))
        self.alias[sender] = self.aliases.new()
        self.alias_registered[sender] = ts - rebound
        device = self.devices[sender][0]
        if self.rng.random() < 0.5:
            device = self._new_device()
            self.devices[sender] = [device] + self.devices[sender][:1]
        method = self._choice(["otp", "password_otp", "biometric"])
        session = self._session(
            sender, device, ts, method, 0, HOME_COUNTRY, ((rebound, "alias_rebound", "number_changed"),)
        )
        drafts = []
        for _ in range(plan.size):
            drafts.append(
                self._legit(
                    ts,
                    sender,
                    self._known_payee(sender),
                    self._amount(sender),
                    plan,
                    device=device,
                    method=method,
                    session=session,
                )
            )
            ts = self._clamp(ts + int(self.rng.integers(300, 3_600)), day_start)
        return drafts

    def _unusual_household_repeat_payments(self, plan: _Plan, day_start: int) -> list[Draft]:
        sender = self._victim(day_start)
        recipient = self._known_payee(sender)
        amounts = [self._amount(sender, float(self.rng.uniform(0.3, 1.0))) for _ in range(plan.size)]
        return self._series(plan, day_start, sender, [recipient] * plan.size, amounts, (600, 4_800))

    def _unusual_business_settlement(self, plan: _Plan, day_start: int) -> list[Draft]:
        businesses = [a for a in self._active_accounts(day_start) if self._is_business(a)]
        sender = self._choice(businesses)
        recipient = self._known_payee(sender)
        amounts = [self._amount(sender, float(self.rng.uniform(3, 10)), 0.2) for _ in range(plan.size)]
        return self._series(
            plan, day_start, sender, [recipient] * plan.size, amounts, (300, 3_600), method="password_otp"
        )

    # -- never-block fixtures and administrative events -----------------
    def evidence_fixtures(self, first_ts: int, last_ts: int) -> dict[str, int]:
        """Ten records on popular billers that must never block: two each of five inactive kinds."""
        counts: dict[str, int] = defaultdict(int)
        billers = self.biller_ids
        for index in range(2):
            self._add_evidence(
                "fraud_controlled_beneficiary",
                "counterparty",
                billers[index],
                first_ts - 40 * DAY_SECONDS,
                first_ts - 40 * DAY_SECONDS,
                first_ts - 5 * DAY_SECONDS,
                fixture=True,
            )
            counts["expired"] += 1
            self._add_evidence(
                "fraud_controlled_beneficiary",
                "counterparty",
                billers[index + 2],
                first_ts - 20 * DAY_SECONDS,
                first_ts - 20 * DAY_SECONDS,
                None,
                status="retracted",
                source=TRUSTED_SOURCES[1],
                fixture=True,
            )
            counts["retracted"] += 1
            self._add_evidence(
                "fraud_controlled_beneficiary",
                "counterparty",
                billers[index + 4],
                last_ts + 3 * DAY_SECONDS,
                first_ts,
                None,
                fixture=True,
            )
            counts["future_recorded"] += 1
            self._add_evidence(
                "fraud_controlled_beneficiary",
                "counterparty",
                billers[index + 6],
                first_ts - DAY_SECONDS,
                first_ts - DAY_SECONDS,
                None,
                status="pending_review",
                fixture=True,
            )
            counts["pending_review"] += 1
            self._add_evidence(
                "fraud_controlled_beneficiary",
                "counterparty",
                billers[index + 8],
                first_ts - DAY_SECONDS,
                first_ts - DAY_SECONDS,
                None,
                source=UNTRUSTED_SOURCE,
                fixture=True,
            )
            counts["untrusted_source"] += 1
        return dict(counts)

    def administrative_revocations(self, drafts: list[Draft]) -> int:
        """25 legitimate sessions that were logged out administratively before a fresh sign-in (not fraud evidence)."""
        legit = [d for d in drafts if d.fraud_type is None]
        picks = self.rng.choice(len(legit), size=min(25, len(legit)), replace=False)
        for index in sorted(int(i) for i in picks):
            draft = legit[index]
            session = self.sessions[draft.session]
            success = next(e["timestamp"] for e in session.events if e["event_type"] == "login_success")
            session.events.append(
                self._auth_event(
                    success - int(self.rng.integers(30, 600)),
                    "session_revoked",
                    draft.auth_method,
                    draft.ip_country,
                    draft.device,
                    "administrative_logout",
                )
            )
        return len(picks)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

FRAUD_SIZES = {
    "account_takeover_phishing": (1, 2),
    "sim_swap_takeover": (2, 3),
    "vishing_impersonation": (1, 3),
    "request_to_pay_scam": (3, 6),
    "marketplace_scam": (2, 5),
    "investment_scam": (3, 6),
    "romance_scam": (3, 5),
    "money_mule": (2, 5),
    "limit_structuring": (3, 5),
    "coordinated_beneficiary": (3, 7),
    "qr_code_tampering": (2, 5),
    "invoice_redirect": (1, 3),
}
ANOMALY_SIZES = {
    "near_limit_planned_purchase": (1, 1),
    "limit_increase_then_large_payment": (1, 1),
    "new_device_legit": (1, 2),
    "travel_abroad": (2, 4),
    "payroll_burst": (5, 10),
    "salary_day_spike": (3, 5),
    "diaspora_remittance_spend": (1, 3),
    "group_payment_new_payee": (3, 6),
    "holiday_spike": (2, 4),
    "night_payment": (1, 1),
    "new_merchant_qr": (1, 1),
    "legit_alias_change": (1, 2),
    "household_repeat_payments": (3, 5),
    "business_settlement": (1, 2),
}
MULTI_ONLY = {"payroll_burst", "group_payment_new_payee", "household_repeat_payments", "salary_day_spike"}


def _split_total(rng: np.random.Generator, total: int, low: int, high: int) -> list[int]:
    sizes: list[int] = []
    remaining = total
    while remaining > 0:
        size = min(int(rng.integers(low, high + 1)), remaining)
        sizes.append(size)
        remaining -= size
    return sizes


def holiday_days(spec: DatasetSpec) -> set[int]:
    """Day numbers inside a holiday window (the holiday and the two days before it)."""
    days = set()
    for _, holiday in spec.holidays:
        for back in range(3):
            day = (holiday - timedelta(days=back) - spec.start).days + 1
            if 1 <= day <= spec.days:
                days.add(day)
    return days


def _plan(spec: DatasetSpec, rng: np.random.Generator) -> list[_Plan]:
    plans: list[_Plan] = []
    counter = 0
    holidays = holiday_days(spec)
    for partition in spec.partitions:
        days = list(range(partition.first_day, partition.last_day + 1))
        festive = [d for d in days if d in holidays]
        fraud = [(t, spec.fraud_per_type[partition.name]) for t in FRAUD_TYPES]
        fraud += list(spec.test_only_fraud.get(partition.name, {}).items())
        for kind, total in fraud:
            for size in _split_total(rng, total, *FRAUD_SIZES[kind]):
                counter += 1
                # structuring sends at most two near-cap payments per day, so it needs ceil(size / 2) days
                need = {
                    "limit_structuring": (size + 1) // 2,
                    "investment_scam": size,
                    "romance_scam": size,
                }.get(kind, 1)
                usable = days[: max(1, len(days) - need + 1)]
                plans.append(
                    _Plan(kind, True, int(rng.choice(usable)), partition.last_day, size, f"CMP-{counter:04d}")
                )
        remaining = spec.anomalies[partition.name]
        while remaining > 0:
            kind = ANOMALY_TYPES[int(rng.choice(len(ANOMALY_TYPES), p=ANOMALY_WEIGHTS))]
            low, high = ANOMALY_SIZES[kind]
            size = min(int(rng.integers(low, high + 1)), remaining)
            if kind in MULTI_ONLY and size < 3:
                kind = "night_payment" if size == 1 else "travel_abroad"
            if kind == "holiday_spike" and not festive:
                kind = "new_merchant_qr" if size == 1 else "travel_abroad"
            day = int(rng.choice(festive)) if kind == "holiday_spike" else int(rng.choice(days))
            counter += 1
            plans.append(_Plan(kind, False, day, partition.last_day, size, f"EP-{counter:05d}"))
            remaining -= size
    return plans


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_fingerprint(row: dict[str, Any]) -> str:
    material = (
        f"{row['timestamp']}|{row['sender_id']}|{row['recipient_id']}|{float(row['amount']):.2f}|"
        f"{row.get('device_id')}|{row['payment_scheme']}"
    )
    return hashlib.sha256(material.encode()).hexdigest()


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """A JSON array with one compact row per line: valid JSON, diff-friendly, a third of the indented size."""
    with path.open("w", encoding="utf-8") as handle:
        handle.write("[\n")
        for index, row in enumerate(rows):
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write(",\n" if index < len(rows) - 1 else "\n")
        handle.write("]\n")


def generate_dataset(
    storage_dir: Path, manifest_path: Path, spec: DatasetSpec = MIA_SPEC, limits: MiaLimits = MIA_LIMITS
) -> dict[str, Any]:
    """Generate train/validation/test JSON files into `storage_dir` and the manifest; return the manifest."""
    storage_dir.mkdir(parents=True, exist_ok=True)
    gen = _Generator(spec, limits)
    plans = _plan(spec, gen.rng)
    by_day: dict[int, list[_Plan]] = defaultdict(list)
    for plan in plans:
        by_day[plan.day].append(plan)

    pending: dict[int, list[Draft]] = defaultdict(list)
    drafts: list[Draft] = []
    for day in range(1, spec.days + 1):
        start = day_start_ts(spec.start, day)
        gen.incoming_credits(day, start)
        for plan in by_day.get(day, []):
            for draft in gen.fraud(plan, start) if plan.is_fraud else gen.unusual(plan, start):
                pending[gen._day_number(draft.ts)].append(draft)
        scheduled = pending.pop(day, [])
        routine_count = spec.payments_per_day - len(scheduled)
        if routine_count < 0:
            raise RuntimeError(f"Day {day} is over capacity ({len(scheduled)} scheduled payments)")
        drafts += scheduled + gen.routine(routine_count, start)
    if pending:  # pragma: no cover - defensive
        raise RuntimeError(f"Campaign payments scheduled after the last day: {sorted(pending)}")

    first_ts = day_start_ts(spec.start, 1)
    last_ts = day_start_ts(spec.start, spec.days) + DAY_SECONDS - 1
    fixture_counts = gen.evidence_fixtures(first_ts, last_ts)
    revocations = gen.administrative_revocations(drafts)
    rows_by_partition = _rows(spec, gen, drafts)
    names = {}
    for name in PARTITION_NAMES:
        path = storage_dir / PARTITION_FILES[name]
        write_rows(path, rows_by_partition[name])
        names[name] = path
    manifest = _manifest(spec, limits, gen, rows_by_partition, names, fixture_counts, revocations)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def _party(gen: _Generator, party_id: str, ts: int) -> dict[str, Any]:
    if party_id in gen.account_by_id:
        account = gen.account_by_id[party_id]
        return {
            "type": "account",
            "name": account.display_name,
            "kind": "business" if gen._is_business(party_id) else "person",
            "category": "business_account" if gen._is_business(party_id) else "individual",
        }
    cp = gen.cp[party_id]
    return {"type": "counterparty", "name": cp.display_name, "kind": cp.kind, "category": cp.category}


def _rows(spec: DatasetSpec, gen: _Generator, drafts: list[Draft]) -> dict[str, list[dict[str, Any]]]:
    # Stable global numbering of authentication events.
    all_events = [
        (e["timestamp"], sid, AUTH_RANK[e["event_type"]], e)
        for sid, s in gen.sessions.items()
        for e in s.events
    ]
    all_events.sort(key=lambda item: (item[0], item[1], item[2]))
    for number, (_, _, _, event) in enumerate(all_events, start=1):
        event["event_id"] = f"AE-{number:07d}"
    evidence_by_subject: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in gen.evidence:
        evidence_by_subject[(item["subject_type"], item["subject_id"])].append(item)

    def evidence_rows(draft: Draft) -> list[dict[str, Any]]:
        subjects = (
            ("session", draft.session),
            ("device", draft.device),
            ("account", draft.sender),
            ("counterparty", draft.recipient),
            ("account", draft.recipient),
        )
        found = {e["evidence_id"]: e for s in subjects for e in evidence_by_subject.get(s, ())}
        return [_evidence_row(found[k]) for k in sorted(found)]

    def alias_info(party_id: str) -> tuple[str | None, str | None]:
        if party_id in gen.account_by_id:
            alias, registered = gen.alias[party_id], gen.alias_registered[party_id]
        else:
            alias, registered = gen.cp[party_id].alias, gen.cp_alias_registered.get(party_id)
        return alias, (iso(registered) if registered is not None else None)

    entries: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for d in drafts:
        recipient = _party(gen, d.recipient, d.ts)
        account = gen.account_by_id[d.sender]
        alias, registered = alias_info(d.recipient)
        events = sorted(
            (e for e in gen.sessions[d.session].events if e["timestamp"] < d.ts),
            key=lambda e: (e["timestamp"], AUTH_RANK[e["event_type"]]),
        )
        row = {
            "timestamp": iso(d.ts),
            "transfer_type": INSTANT_TRANSFER,
            "payment_scheme": d.scheme,
            "assessed": True,
            "amount": d.amount,
            "currency": CURRENCY,
            "sender_id": d.sender,
            "sender_type": account.account_type,
            "sender_name": account.display_name,
            "sender_alias": d.sender_alias,
            "sender_home_country": account.home_country,
            "sender_created_at": iso(gen.params[d.sender].created_at),
            "recipient_id": d.recipient,
            "recipient_type": recipient["type"],
            "recipient_name": recipient["name"],
            "recipient_alias": alias,
            "recipient_kind": recipient["kind"],
            "recipient_category": recipient["category"],
            "alias_registered_at": registered,
            "channel": d.channel,
            "device_id": d.device,
            "session_id": d.session,
            "auth_method": d.auth_method,
            "ip_country": d.ip_country,
            "session_events": [
                {
                    "event_id": e["event_id"],
                    "timestamp": iso(e["timestamp"]),
                    "event_type": e["event_type"],
                    "auth_method": e["auth_method"],
                    "ip_country": e["ip_country"],
                    "device_id": e["device_id"],
                    "reason": e["reason"],
                }
                for e in events
            ],
            "request_to_pay_id": d.rtp_id,
            "qr_payload_merchant_id": d.qr_merchant,
            "security_evidence": evidence_rows(d),
            "is_fraud": d.fraud_type is not None,
            "is_anomaly": d.is_anomaly,
            "fraud_type": d.fraud_type,
            "anomaly_type": d.anomaly_type,
            "campaign_id": d.campaign_id,
        }
        entries.append(((d.ts, 1, d.sender, d.recipient, d.amount, d.session), row))
    for c in gen.credits:
        payer = gen.cp[c.sender]
        account = gen.account_by_id[c.recipient]
        alias, registered = alias_info(c.recipient)
        row = {
            "timestamp": iso(c.ts),
            "transfer_type": INSTANT_TRANSFER,
            "payment_scheme": c.scheme,
            "assessed": False,
            "amount": c.amount,
            "currency": CURRENCY,
            "sender_id": c.sender,
            "sender_type": "external",
            "sender_name": payer.display_name,
            "sender_alias": payer.alias,
            "sender_home_country": payer.home_country,
            "sender_created_at": None,
            "recipient_id": c.recipient,
            "recipient_type": "account",
            "recipient_name": account.display_name,
            "recipient_alias": alias,
            "recipient_kind": "business" if gen._is_business(c.recipient) else "person",
            "recipient_category": "business_account" if gen._is_business(c.recipient) else "individual",
            "alias_registered_at": registered,
            "channel": None,
            "device_id": None,
            "session_id": None,
            "auth_method": None,
            "ip_country": None,
            "session_events": [],
            "request_to_pay_id": None,
            "qr_payload_merchant_id": None,
            "security_evidence": [],
            "is_fraud": False,
            "is_anomaly": False,
            "fraud_type": None,
            "anomaly_type": None,
            "campaign_id": c.campaign_id,
        }
        entries.append(((c.ts, 0, c.sender, c.recipient, c.amount, ""), row))
    entries.sort(key=lambda item: item[0])
    by_partition: dict[str, list[dict[str, Any]]] = {name: [] for name in PARTITION_NAMES}
    for number, (key, row) in enumerate(entries, start=1):
        ordered = {"transaction_id": f"TX-{number:07d}", **row}
        ordered = {column: ordered[column] for column in ROW_COLUMNS}
        day = (key[0] - gen.start_ts) // DAY_SECONDS + 1
        by_partition[spec.partition_of_day(int(day)).name].append(ordered)
    return by_partition


def _evidence_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_id": item["evidence_id"],
        "evidence_type": item["evidence_type"],
        "subject_type": item["subject_type"],
        "subject_id": item["subject_id"],
        "status": item["status"],
        "source": item["source"],
        "source_trust": item["source_trust"],
        "recorded_at": iso(item["recorded_at"]),
        "effective_at": iso(item["effective_at"]),
        "expires_at": iso(item["expires_at"]) if item["expires_at"] is not None else None,
    }


def _manifest(
    spec: DatasetSpec,
    limits: MiaLimits,
    gen: _Generator,
    rows: dict[str, list[dict[str, Any]]],
    paths: dict[str, Path],
    fixtures: dict[str, int],
    revocations: int,
) -> dict[str, Any]:
    partitions = {}
    for partition in spec.partitions:
        part_rows = rows[partition.name]
        assessed = [r for r in part_rows if r["assessed"]]
        fraud_by_type: dict[str, int] = defaultdict(int)
        anomaly_by_type: dict[str, int] = defaultdict(int)
        schemes: dict[str, int] = defaultdict(int)
        for r in part_rows:
            schemes[r["payment_scheme"]] += 1
            if r["fraud_type"]:
                fraud_by_type[r["fraud_type"]] += 1
            if r["anomaly_type"]:
                anomaly_by_type[r["anomaly_type"]] += 1
        partitions[partition.name] = {
            "file": f"storage/{PARTITION_FILES[partition.name]}",
            "days": [partition.first_day, partition.last_day],
            "date_range": {"start": part_rows[0]["timestamp"], "end": part_rows[-1]["timestamp"]},
            "rows": len(part_rows),
            "assessed_payments": len(assessed),
            "context_credits": len(part_rows) - len(assessed),
            "fraud": sum(fraud_by_type.values()),
            "fraud_is_anomaly_false": sum(1 for r in assessed if r["is_fraud"] and not r["is_anomaly"]),
            "unusual_legitimate": sum(anomaly_by_type.values()),
            "fraud_by_type": dict(sorted(fraud_by_type.items())),
            "anomaly_by_type": dict(sorted(anomaly_by_type.items())),
            "payment_schemes": dict(sorted(schemes.items())),
            "senders": len({r["sender_id"] for r in assessed}),
            "sha256": sha256_file(paths[partition.name]),
            "bytes": paths[partition.name].stat().st_size,
        }
    cold = sorted(a for a, p in gen.params.items() if p.cold_start)
    campaign_evidence = [e for e in gen.evidence if not e["evidence_id"].startswith("SEC-FX")]
    return {
        "dataset": "secflow-mia-instant",
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "seed": spec.seed,
        "identity_seed": IDENTITY_SEED,
        "provenance": (
            "Fully synthetic data produced by datagen/generator.py, modelled on MIA Plati Instant (National Bank of "
            "Moldova). Accounts, counterparties, phone aliases, payments, sessions and security evidence are "
            "generated; they describe no real people, businesses, phone numbers or confirmed fraud cases."
        ),
        "currency": CURRENCY,
        "mia_limits": asdict(limits),
        "start_date": spec.start.isoformat(),
        "days": spec.days,
        "assessed_payments_per_day": spec.payments_per_day,
        "settlement_seconds": SETTLEMENT_SECONDS,
        "partitions": partitions,
        "fraud_types": list(FRAUD_TYPES),
        "test_only_fraud_types": list(TEST_ONLY_FRAUD_TYPES),
        "anomaly_types": list(ANOMALY_TYPES),
        "holidays": [{"name": n, "date": d.isoformat()} for n, d in spec.holidays],
        "drift": {
            "first_day": spec.drift_first_day,
            "amount_factor": spec.drift_amount_factor,
            "adoption_share": DRIFT_ADOPTION_SHARE,
            "biller_ids": [c.counterparty_id for c in gen.universe.drift_billers],
        },
        "cold_start": {"partition": spec.cold_start_partition, "account_ids": cold},
        "security_evidence": {
            "campaign_evidence_records": len(campaign_evidence),
            "fixtures": fixtures,
            "administrative_session_revocations": revocations,
            "trusted_sources": list(TRUSTED_SOURCES),
            "note": "All evidence is simulated. Fixtures (SEC-FX-*) must never trigger a confirmed-fraud block.",
        },
        "label_columns": list(LABEL_COLUMNS),
        "feature_assumptions": [
            "Runtime features use only events ordered strictly before the payment (timestamp, type rank, id).",
            "Label columns are offline-only; the runtime loader drops them.",
            f"Every assessed payment settles {SETTLEMENT_SECONDS} s after its timestamp in the runtime ledger.",
        ],
    }

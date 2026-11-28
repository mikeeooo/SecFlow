"""Typed runtime records shared by the loader, feature pipeline, and service.

None of these records carries a ground-truth label, scenario, or campaign identifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

# Events that share a timestamp are ordered auth -> payment -> request, then by identifier.
EVENT_RANK = {"auth": 0, "payment": 1, "request": 2}
KEY_SCALE = 1_000_000  # ordering key = epoch_seconds * KEY_SCALE + global sequence number
# SecFlow assesses instant transfers only; every dataset row must declare this type.
INSTANT_TRANSFER = "instant"
CURRENCY = "MDL"  # MIA Plati Instant settles in Moldovan lei only
QR_SCHEMES = frozenset({"p2b_qr_static", "p2b_qr_dynamic"})


def parse_ts(value: str) -> int:
    return int(datetime.fromisoformat(value).timestamp())


def format_ts(ts: float) -> str:
    return datetime.fromtimestamp(int(ts), UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_key(ts: int, seq: int) -> int:
    if seq >= KEY_SCALE:
        raise ValueError("sequence number exceeds key capacity")
    return ts * KEY_SCALE + seq


@dataclass(frozen=True, slots=True)
class PaymentRequest:
    request_id: str
    ts: int
    sender_id: str
    recipient_id: str
    recipient_type: str
    amount: float
    currency: str
    channel: str
    device_id: str
    session_id: str
    auth_method: str
    ip_country: str
    transfer_type: str = INSTANT_TRANSFER
    payment_scheme: str = "p2p_alias"
    request_to_pay_id: str | None = None
    qr_payload_merchant_id: str | None = None  # QR schemes: the merchant the QR names
    recipient_alias: str | None = None
    alias_registered_at: int | None = None  # when the recipient's alias was last bound


@dataclass(frozen=True, slots=True)
class AuthEvent:
    event_id: str
    ts: int
    account_id: str
    session_id: str
    device_id: str
    event_type: str  # login_failure | login_success | session_revoked | alias_rebound | limit_changed
    auth_method: str
    ip_country: str
    reason: str | None


@dataclass(frozen=True, slots=True)
class PaymentEvent:
    event_id: str
    ts: int
    event_type: str  # transfer_settled | incoming_credit
    from_id: str
    to_id: str
    amount: float
    currency: str
    request_id: str | None


@dataclass(frozen=True, slots=True)
class Account:
    account_id: str
    display_name: str
    account_type: str
    home_country: str
    created_at: int


@dataclass(frozen=True, slots=True)
class Counterparty:
    counterparty_id: str
    display_name: str
    kind: str
    category: str


def request_from_row(row: dict[str, Any]) -> PaymentRequest:
    """Runtime request from one assessed dataset row (label columns are never read)."""
    registered = row.get("alias_registered_at")
    return PaymentRequest(
        request_id=row["transaction_id"],
        ts=parse_ts(row["timestamp"]),
        sender_id=row["sender_id"],
        recipient_id=row["recipient_id"],
        recipient_type=row["recipient_type"],
        amount=float(row["amount"]),
        currency=row["currency"],
        channel=row["channel"],
        device_id=row["device_id"],
        session_id=row["session_id"],
        auth_method=row["auth_method"],
        ip_country=row["ip_country"],
        transfer_type=row["transfer_type"],
        payment_scheme=row["payment_scheme"],
        request_to_pay_id=row.get("request_to_pay_id"),
        qr_payload_merchant_id=row.get("qr_payload_merchant_id"),
        recipient_alias=row.get("recipient_alias"),
        alias_registered_at=parse_ts(registered) if registered else None,
    )


def auth_from_session_event(event: dict[str, Any], account_id: str, session_id: str) -> AuthEvent:
    return AuthEvent(
        event_id=event["event_id"],
        ts=parse_ts(event["timestamp"]),
        account_id=account_id,
        session_id=session_id,
        device_id=event["device_id"],
        event_type=event["event_type"],
        auth_method=event["auth_method"],
        ip_country=event["ip_country"],
        reason=event.get("reason"),
    )

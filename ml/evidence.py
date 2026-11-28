"""Trusted security evidence (for example a confirmed compromised session) and when it applies.

Evidence comes only from the dataset rows' `security_evidence` lists, never from a client payload
or a label. A record is active for a payment at time `t` when all of these hold:

* it was recorded at or before `t`, so the bank already knew about it;
* `effective_from <= t` and it has not expired;
* its latest status at or before `t` is `confirmed` (a later retraction does not rewrite the past);
* its source is a trusted source and its type is a blocking type (both from `config/policy.json`).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from datagen.records import PaymentRequest, parse_ts


@dataclass(frozen=True)
class SecurityEvidence:
    evidence_id: str
    evidence_type: str
    entity_type: str  # session | counterparty | device | account
    entity_id: str
    source: str
    recorded_at: int
    effective_from: int
    effective_until: int | None
    status_history: tuple[tuple[int, str], ...]  # (at, status), oldest first
    note: str

    def status_at(self, ts: int) -> str | None:
        current = None
        for at, status in self.status_history:
            if at <= ts:
                current = status
        return current


def evidence_from_row(row: dict) -> SecurityEvidence:
    """Parse one `security_evidence` entry. Without a status history it is effective from `recorded_at`."""
    recorded = parse_ts(row["recorded_at"])
    history = row.get("status_history") or [{"at": row["recorded_at"], "status": row["status"]}]
    return SecurityEvidence(
        evidence_id=row["evidence_id"],
        evidence_type=row["evidence_type"],
        entity_type=row["subject_type"],
        entity_id=row["subject_id"],
        source=row["source"],
        recorded_at=recorded,
        effective_from=parse_ts(row["effective_at"]),
        effective_until=parse_ts(row["expires_at"]) if row.get("expires_at") else None,
        status_history=tuple((parse_ts(h["at"]), h["status"]) for h in history),
        note=row.get("note", ""),
    )


class EvidenceStore:
    def __init__(self, evidence: Iterable[SecurityEvidence], trusted_sources: Iterable[str],
                 blocking_types: Iterable[str]) -> None:
        self.trusted = frozenset(trusted_sources)
        self.blocking = frozenset(blocking_types)
        self.records = tuple(evidence)
        self.by_entity: dict[tuple[str, str], list[SecurityEvidence]] = {}
        for item in self.records:
            self.by_entity.setdefault((item.entity_type, item.entity_id), []).append(item)

    def is_active(self, item: SecurityEvidence, ts: int) -> bool:
        return (
            item.source in self.trusted
            and item.evidence_type in self.blocking
            and item.recorded_at <= ts
            and item.effective_from <= ts
            and (item.effective_until is None or ts < item.effective_until)
            and item.status_at(ts) == "confirmed"
        )

    def active_for(self, request: PaymentRequest) -> list[SecurityEvidence]:
        """Active evidence on the request's session, device, sender or recipient, sorted by ID."""
        entities = (("session", request.session_id), ("device", request.device_id), ("account", request.sender_id),
                    ("counterparty", request.recipient_id), ("account", request.recipient_id))
        found = {item.evidence_id: item for entity in entities for item in self.by_entity.get(entity, ())
                 if self.is_active(item, request.ts)}
        return [found[k] for k in sorted(found)]

    def active_entities(self, entity_type: str, ts: int) -> list[SecurityEvidence]:
        return [item for item in self.records if item.entity_type == entity_type and self.is_active(item, ts)]

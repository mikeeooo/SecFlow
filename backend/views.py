"""Turn in-memory assessment records into the API response models."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from backend.schemas import (
    AssessmentDetail,
    AssessmentSummary,
    ComponentBrief,
    ComponentView,
    EvidenceView,
    PartyRef,
    ReasonView,
    ReviewView,
    Thresholds,
    TimelineEntry,
)
from datagen.records import format_ts
from ml.policy import COMPONENTS, OperatingPoint, Policy

if TYPE_CHECKING:
    from backend.service import AssessmentRecord, ReviewRecord

PartyLookup = Callable[[str], PartyRef]
COMPLETED_OUTCOMES = frozenset({"permitted", "approved_after_review"})
OUTCOME_FOR_DECISION = {"Permit": "permitted", "Verify": "pending_verification", "Block": "blocked_attempt"}
COMPONENT_LABELS = {
    "isolation_forest": "Isolation Forest",
    "lof": "Local Outlier Factor",
    "dbscan": "DBSCAN account context",
    "rules": "Behaviour rules",
}
SESSION_TITLES = {
    "login_failure": "Failed sign-in attempt",
    "login_success": "Signed in",
    "session_revoked": "Session ended",
    "alias_rebound": "Phone alias rebound to a new SIM or device",
    "limit_changed": "Payment limit changed",
}
OUTCOME_TEXT = {
    "permitted": ("Recorded as permitted", "Simulated outcome only: no funds moved through a bank."),
    "pending_verification": ("Held for verification", "Not counted as a completed transfer until an analyst approves it."),
    "blocked_attempt": ("Recorded as a blocked attempt", "Not executed and excluded from completed-payment totals."),
    "approved_after_review": ("Approved after review", "Simulated outcome only: no funds moved through a bank."),
    "rejected_after_review": ("Rejected after review", "Not executed and excluded from completed-payment totals."),
}


def review_status(record: AssessmentRecord) -> str:
    if record.result.policy.decision != "Verify":
        return "not_applicable"
    if not record.reviews:
        return "pending"
    return "approved" if record.reviews[-1].action == "approve" else "rejected"


def _scores(record: AssessmentRecord) -> dict[str, float]:
    c = record.result.components
    return {"isolation_forest": c.if_score, "lof": c.lof_score, "dbscan": c.context.score,
            "rules": record.result.rules.score}


def summary(record: AssessmentRecord, party: PartyLookup) -> AssessmentSummary:
    result, request = record.result, record.request
    contributions = result.risk.contributions
    return AssessmentSummary(
        cursor=record.seq,
        request_id=request.request_id,
        origin=record.origin,
        producer=record.producer,
        event_time=format_ts(request.ts),
        sender=party(request.sender_id),
        recipient=party(request.recipient_id),
        amount=request.amount,
        currency=request.currency,
        payment_scheme=request.payment_scheme,
        risk_score=round(result.risk.total, 2),
        decision=result.policy.decision,
        decision_source=result.policy.decision_source,
        decision_label=result.policy.label,
        assessment_status=result.policy.assessment_status,
        outcome=record.outcome,
        review_status=review_status(record),
        segment=result.segment,
        components=ComponentBrief(**{name: round(v, 4) for name, v in _scores(record).items()}),
        contributions=ComponentBrief(**{name: round(contributions[name], 3) for name in COMPONENTS}),
        evidence_ids=list(result.policy.evidence_ids),
        corroborating_signals=list(result.policy.corroborating_signals),
    )


def _context_detail(record: AssessmentRecord) -> str:
    ctx = record.result.components.context
    if ctx.status == "within_context":
        return (f"The trailing 24-hour account profile lies within a frozen DBSCAN core neighbourhood "
                f"(distance {ctx.distance:.2f}, cluster {ctx.cluster}).")
    if ctx.status == "outside_context":
        return (f"The trailing 24-hour account profile lies outside every frozen core neighbourhood "
                f"(nearest distance {ctx.distance:.2f}). This is context, not a fraud verdict.")
    return "Fewer than three earlier payments: the context score is 0 and marked as insufficient history."


def _component_details(record: AssessmentRecord, policy: Policy) -> list[ComponentView]:
    c, result = record.result.components, record.result
    scores, contributions = _scores(record), result.risk.contributions
    triggered = len(result.rules.triggered)
    raw = {"isolation_forest": c.if_raw, "lof": c.lof_raw, "dbscan": c.context.distance, "rules": None}
    status = {"isolation_forest": "scored", "lof": "scored", "dbscan": c.context.status, "rules": "scored"}
    detail = {
        "isolation_forest": f"More isolated than {c.if_score:.0%} of reference-period requests.",
        "lof": f"Local density lower than for {c.lof_score:.0%} of reference-period requests.",
        "dbscan": _context_detail(record),
        "rules": f"{triggered} rule{'s' if triggered != 1 else ''} triggered; the rules score is the capped sum of "
                 "their weights.",
    }
    return [ComponentView(key=name, label=COMPONENT_LABELS[name], score=round(scores[name], 4),
                          weight=policy.weights[name], contribution=round(contributions[name], 3),
                          raw=None if raw[name] is None else round(raw[name], 4), status=status[name],
                          detail=detail[name]) for name in COMPONENTS]


def detail(record: AssessmentRecord, party: PartyLookup, policy: Policy, point: OperatingPoint, model_version: str,
           replay: bool = False) -> AssessmentDetail:
    result, request = record.result, record.request
    return AssessmentDetail(
        **summary(record, party).model_dump(),
        explanation=result.policy.explanation,
        status_notes=list(result.status_notes),
        component_details=_component_details(record, policy),
        reasons=[ReasonView(code=t.code, family=t.family, description=t.description, detail=t.detail,
                            weight=t.weight, corroborating=t.corroborating) for t in result.rules.triggered],
        evidence=[EvidenceView(evidence_id=e.evidence_id, evidence_type=e.evidence_type, entity_type=e.entity_type,
                               entity_id=e.entity_id, source=e.source, recorded_at=format_ts(e.recorded_at),
                               effective_from=format_ts(e.effective_from),
                               effective_until=format_ts(e.effective_until) if e.effective_until else None,
                               note=e.note) for e in result.evidence],
        thresholds=Thresholds(verify=point.verify, block=point.block, min_corroborating_families=policy.min_families,
                              source=point.source, segment=result.segment),
        model_version=model_version,
        policy_version=policy.version,
        operating_point_version=point.version,
        assessed_at=record.assessed_at,
        device_id=request.device_id,
        session_id=request.session_id,
        auth_method=request.auth_method,
        ip_country=request.ip_country,
        channel=request.channel,
        request_to_pay_id=request.request_to_pay_id,
        qr_payload_merchant_id=request.qr_payload_merchant_id,
        recipient_alias=request.recipient_alias,
        facts={k: round(v, 4) if isinstance(v, float) else v for k, v in result.snapshot.facts.items()},
        idempotent_replay=replay,
    )


def review_view(review: ReviewRecord) -> ReviewView:
    return ReviewView(review_id=review.review_id, action=review.action, reason=review.reason, actor=review.actor,
                      created_at=review.created_at, original_decision=review.original_decision,
                      original_risk_score=review.original_risk_score, resulting_outcome=review.resulting_outcome)


def timeline(record: AssessmentRecord) -> list[TimelineEntry]:
    """Sign-ins, evidence, the request, the automated decision and any review, in time order."""
    request, policy = record.request, record.result.policy
    entries = [
        TimelineEntry(at=format_ts(e.ts), kind="authentication", title=SESSION_TITLES.get(e.event_type, e.event_type),
                      detail=f"{e.auth_method} on {e.device_id} from {e.ip_country}"
                      + (f" ({e.reason.replace('_', ' ')})" if e.reason else ""))
        for e in record.session_events
    ]
    entries += [TimelineEntry(at=format_ts(e.recorded_at), kind="evidence",
                              title=f"Security evidence recorded: {e.evidence_type.replace('_', ' ')}",
                              detail=f"{e.evidence_id} from {e.source.replace('_', ' ')}") for e in record.result.evidence]
    at = format_ts(request.ts)
    title, text = OUTCOME_TEXT[OUTCOME_FOR_DECISION[policy.decision]]
    entries += [
        TimelineEntry(at=at, kind="request", title="Payment request received",
                      detail=f"{request.amount:,.2f} {request.currency} by MIA {request.payment_scheme.replace('_', ' ')} "
                      f"via {request.channel.replace('_', ' ')}"),
        TimelineEntry(at=at, kind="decision", title=f"Automated decision: {policy.label}", detail=policy.explanation),
        TimelineEntry(at=at, kind="outcome", title=title, detail=text),
    ]
    for review in record.reviews:
        title, text = OUTCOME_TEXT[review.resulting_outcome]
        entries.append(TimelineEntry(at=format_ts(review.simulated_time), kind="review",
                                     title=f"{title} by {review.actor}", detail=f"“{review.reason}” — {text}"))
    return sorted(entries, key=lambda e: e.at)

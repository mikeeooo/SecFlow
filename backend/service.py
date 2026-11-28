"""The app's state: test.json as history, every assessment, simulations and reviews, all in memory.

At startup the service loads test.json without its labels, the model bundle, the policy and the
thresholds, and assesses every test payment as of its own time. Simulations append to the same
history index, so a later simulation sees the earlier ones. Nothing is written to disk: a restart
starts again from test.json. One lock serialises simulations and reviews, which is all a single
process needs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from backend import views
from backend.analytics import ClusterGraphBuilder, dashboard
from backend.schemas import (
    OUTGOING_SCHEMES,
    AccountPage,
    AccountSummary,
    AssessmentDetail,
    Investigation,
    PartyRef,
    ReviewHistoryItem,
    ReviewHistoryPage,
    ReviewRequest,
    ScoreBand,
    SimulationRequest,
    TransactionPage,
    TransactionQuery,
)
from backend.views import COMPLETED_OUTCOMES, OUTCOME_FOR_DECISION
from config.settings import Settings
from datagen.dataset import load_partitions, read_manifest
from datagen.generator import sha256_file
from datagen.records import (
    KEY_SCALE,
    AuthEvent,
    Counterparty,
    PaymentEvent,
    PaymentRequest,
    format_ts,
    make_key,
)
from ml.assess import AssessmentResult, Assessor, evidence_store
from ml.history import HistoryIndex
from ml.model import load_bundle
from ml.policy import load_operating_point, load_policy

log = logging.getLogger("secflow.service")
HISTOGRAM_BINS = 20
FOREIGN_COUNTRIES = ("RO", "IT", "DE", "FR", "GB", "UA", "TR", "US", "NL")


class NotFoundError(KeyError):
    """Unknown request, account or other identifier (404)."""


class ConflictError(RuntimeError):
    """A request ID reused with another payload, or a review that is not allowed (409)."""


class MiaLimitError(ValueError):
    """The payment breaks the MIA per-transaction limit of the sender's account type (422)."""


def utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-4] + "Z"


@dataclass(frozen=True)
class ReviewRecord:
    review_id: str
    request_id: str
    action: str  # approve | reject
    reason: str
    actor: str
    created_at: str
    simulated_time: int
    original_decision: str
    original_risk_score: float | None
    resulting_outcome: str


@dataclass
class AssessmentRecord:
    """A frozen automated assessment plus its payment outcome, which only a review changes."""

    seq: int
    origin: str  # baseline | simulation
    producer: str | None  # manual | stream | None for baseline
    request: PaymentRequest
    key: int
    result: AssessmentResult
    assessed_at: str
    session_events: tuple[AuthEvent, ...]
    outcome: str
    fingerprint: str | None = None  # hash of the simulation payload, to answer retries
    reviews: list[ReviewRecord] = field(default_factory=list)
    latency_ms: float | None = None


class SecFlowService:
    def __init__(self, settings: Settings) -> None:
        started = time.perf_counter()
        self.settings = settings
        self.started_at = utc_now_iso()
        self.policy = load_policy(settings.policy_path)
        self.limits = self.policy.limits
        self.bundle = load_bundle(settings.models_dir)
        self.point = load_operating_point(settings.models_dir, self.bundle.version)
        # validation.json is history only: the 15 days before the test period give each account and
        # beneficiary its past, as in the evaluation. Only test.json payments are assessed and shown.
        self.dataset = load_partitions([settings.validation_path, settings.test_path])
        self.shown = tuple(r for r in self.dataset.requests if self.dataset.partition_of[r.request_id] == "test")
        self.manifest = read_manifest(settings.manifest_path) if settings.manifest_path.exists() else {}
        recorded = self.manifest.get("partitions", {})
        self.hashes_verified = all(
            recorded.get(path.stem, {}).get("sha256") == sha256_file(path)
            for path in (settings.validation_path, settings.test_path)
        )
        self.evidence = evidence_store(self.dataset.security_events, self.policy)
        self.assessor = Assessor(self.bundle, self.evidence, self.policy, self.point)
        self.accounts = self.dataset.accounts
        self.counterparties: dict[str, Counterparty] = dict(self.dataset.counterparties)
        self.history, keys = HistoryIndex.from_events(self.dataset.requests, self.dataset.auth_events,
                                                      self.dataset.payment_events)
        self.lock = threading.RLock()
        self.records: dict[str, AssessmentRecord] = {}  # insertion order = time order
        self.reviews: list[ReviewHistoryItem] = []
        self._seq = 0
        self._assess_baseline(keys)
        self.evaluation = self._read_json(settings.artifacts_dir / "evaluation.json")
        self.training = self._read_json(settings.models_dir / "training_report.json")
        self._sim_epoch = max(self.history.max_key // KEY_SCALE, self.dataset.requests[-1].ts) + 3_600
        self._wall_epoch = time.monotonic()
        self._last_ts = self._sim_epoch
        self._graph = ClusterGraphBuilder(self.bundle.dbscan)
        self.startup_seconds = round(time.perf_counter() - started, 2)
        log.info("SecFlow service ready in %.2fs", self.startup_seconds)

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _assess_baseline(self, keys: dict[str, int]) -> None:
        sessions: dict[str, list[AuthEvent]] = defaultdict(list)
        for event in self.dataset.auth_events:
            sessions[event.session_id].append(event)
        ordered = sorted(self.shown, key=lambda r: keys[r.request_id])
        now = utc_now_iso()
        for request, result in zip(ordered, self.assessor.assess_batch(ordered, keys, self.history, self.accounts),
                                   strict=True):
            self.records[request.request_id] = AssessmentRecord(
                self._next_seq(), "baseline", None, request, keys[request.request_id], result, now,
                tuple(sessions.get(request.session_id, ())), OUTCOME_FOR_DECISION[result.policy.decision])

    @staticmethod
    def _read_json(path) -> dict[str, Any] | None:
        return json.loads(path.read_text()) if path.exists() else None

    # -- helpers -------------------------------------------------------------
    def simulation_now(self) -> int:
        """The simulated clock: one hour after the last test event, then real time."""
        return int(self._sim_epoch + (time.monotonic() - self._wall_epoch))

    def party(self, party_id: str) -> PartyRef:
        if account := self.accounts.get(party_id):
            return PartyRef(id=party_id, name=account.display_name, kind="account")
        counterparty = self.counterparties.get(party_id)
        return PartyRef(id=party_id, name=counterparty.display_name if counterparty else party_id, kind="counterparty")

    def _record(self, request_id: str) -> AssessmentRecord:
        if (record := self.records.get(request_id)) is None:
            raise NotFoundError(request_id)
        return record

    def _all(self) -> list[AssessmentRecord]:
        with self.lock:
            return list(self.records.values())

    def _next_key(self, ts: int) -> int:
        """An ordering key after every event already in the history."""
        self.history.max_seq += 1
        return make_key(max(ts, self.history.max_key // KEY_SCALE + 1), self.history.max_seq)

    def _clock(self) -> int:
        self._last_ts = max(self.simulation_now(), self._last_ts)
        return self._last_ts

    def last_device(self, account_id: str) -> str | None:
        series = self.history.auth.get(account_id)
        return series.devices[-1] if series and series.devices else None

    def score_band(self, score: float) -> ScoreBand | None:
        """The fraud rate measured on test for the score band this risk score falls in."""
        report = self.evaluation
        if not report or report.get("model_version") != self.bundle.version:
            return None  # the bands describe another model
        bands = report["score_bands"]
        for i, band in enumerate(bands["bands"]):
            last = i == len(bands["bands"]) - 1
            if band["low"] <= score and (score <= band["high"] if last else score < band["high"]):
                return ScoreBand(**{k: band[k] for k in ScoreBand.model_fields if k in band},
                                 base_fraud_rate=bands["base_fraud_rate"], evaluated_requests=bands["requests"])
        return None

    def _detail(self, record: AssessmentRecord, replay: bool = False) -> AssessmentDetail:
        detail = views.detail(record, self.party, self.policy, self.point, self.bundle.version, replay)
        detail.score_band = self.score_band(record.result.risk.total)
        return detail

    # -- simulations ---------------------------------------------------------
    def simulate(self, payload: SimulationRequest, producer: str = "manual") -> tuple[AssessmentDetail, bool]:
        """Assess one simulated payment. Returns (detail, replayed): a retry with the same ID and payload
        gets the stored answer; the same ID with another payload is a conflict."""
        if payload.sender_id not in self.accounts:
            raise NotFoundError(f"Unknown sender {payload.sender_id}")
        if payload.recipient_id and not (payload.recipient_id in self.accounts or payload.recipient_id in self.counterparties):
            raise NotFoundError(f"Unknown recipient {payload.recipient_id}")
        if payload.recipient_id == payload.sender_id:
            raise ConflictError("Sender and recipient must differ")
        segment = self.accounts[payload.sender_id].account_type
        cap = self.limits.for_segment(segment).per_transaction
        if payload.amount > cap:
            raise MiaLimitError(f"MIA Plati Instant accepts at most {cap:,.0f} MDL per transaction from a {segment} "
                                f"account; {payload.amount:,.2f} MDL was requested")
        fingerprint = hashlib.sha256(json.dumps(payload.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()
        with self.lock:
            if (existing := self.records.get(payload.request_id)) is not None:
                if existing.fingerprint != fingerprint:
                    raise ConflictError(f"Request ID {payload.request_id} was already used with a different payload")
                return self._detail(existing, replay=True), True
            record = self._assess_and_commit(payload, producer, fingerprint)
            self.records[payload.request_id] = record
            return self._detail(record), False

    def _assess_and_commit(self, payload: SimulationRequest, producer: str, fingerprint: str) -> AssessmentRecord:
        """Append the sign-in events, assess the payment as of now, then record the attempt."""
        suffix = f"{self._seq + 1:06d}"
        recipient_id = payload.recipient_id
        if recipient_id is None:
            recipient_id = f"CP-SIM-{suffix}"
            self.counterparties[recipient_id] = Counterparty(recipient_id, payload.new_recipient_name or recipient_id,
                                                             "person", "simulated")
        device = payload.device_id or f"DEV-SIM-{suffix}"
        session = f"SES-SIM-{suffix}"
        ts = self._clock()
        events = [AuthEvent(f"AE-SIM-{suffix}-{i}", ts + 20 * i, payload.sender_id, session, device, "login_failure",
                            payload.auth_method, payload.ip_country, "invalid_credentials")
                  for i in range(payload.failed_login_attempts)]
        signed_in = ts + 20 * payload.failed_login_attempts
        events.append(AuthEvent(f"AE-SIM-{suffix}-ok", signed_in, payload.sender_id, session, device, "login_success",
                                payload.auth_method, payload.ip_country, None))
        for event in events:
            self.history.add_auth(event, self._next_key(event.ts))
        qr = payload.payment_scheme in ("p2b_qr_static", "p2b_qr_dynamic")
        request = PaymentRequest(
            request_id=payload.request_id, ts=signed_in + 8, sender_id=payload.sender_id, recipient_id=recipient_id,
            recipient_type="account" if recipient_id in self.accounts else "counterparty", amount=payload.amount,
            currency=payload.currency, channel=payload.channel, device_id=device, session_id=session,
            auth_method=payload.auth_method, ip_country=payload.ip_country, transfer_type=payload.transfer_type,
            payment_scheme=payload.payment_scheme,
            request_to_pay_id=f"RTP-SIM-{suffix}" if payload.payment_scheme == "request_to_pay" else None,
            qr_payload_merchant_id=(payload.qr_payload_merchant_id or recipient_id) if qr else None,
            recipient_alias=payload.recipient_alias)
        key = self._next_key(request.ts)
        started = time.perf_counter()
        [result] = self.assessor.assess_batch([request], {request.request_id: key}, self.history, self.accounts)
        latency_ms = round((time.perf_counter() - started) * 1_000, 3)
        self.history.add_request(request, key)  # an attempt counts as activity whatever the decision
        outcome = OUTCOME_FOR_DECISION[result.policy.decision]
        if outcome == "permitted":
            self._settle(request, request.ts + 2)
        self._last_ts = request.ts + 2
        return AssessmentRecord(self._next_seq(), "simulation", producer, request, key, result, utc_now_iso(),
                                tuple(events), outcome, fingerprint, latency_ms=latency_ms)

    def _settle(self, request: PaymentRequest, ts: int) -> None:
        """A completed payment credits the recipient, which feeds their incoming-funds features."""
        self.history.add_payment(PaymentEvent(f"PE-SIM-{request.request_id}", ts, "transfer_settled",
                                              request.sender_id, request.recipient_id, request.amount,
                                              request.currency, request.request_id), self._next_key(ts))

    # -- reviews -------------------------------------------------------------
    def review(self, request_id: str, payload: ReviewRequest) -> Investigation:
        """Approve or reject a Verify decision. The automated decision itself never changes."""
        with self.lock:
            record = self._record(request_id)
            policy = record.result.policy
            if policy.decision != "Verify":
                raise ConflictError("Only requests whose automated decision is Verify can be reviewed")
            if record.reviews:
                raise ConflictError("This request already has a final review outcome")
            approved = payload.action == "approve"
            now = self._clock()
            request = record.request
            review = ReviewRecord(
                review_id=f"REV-{len(self.reviews) + 1:05d}", request_id=request_id, action=payload.action,
                reason=payload.reason, actor=payload.actor, created_at=utc_now_iso(), simulated_time=now,
                original_decision=policy.decision, original_risk_score=round(record.result.risk.total, 2),
                resulting_outcome="approved_after_review" if approved else "rejected_after_review")
            record.reviews.append(review)
            record.outcome = review.resulting_outcome
            if approved and record.origin == "simulation":
                self._settle(request, now)
            self.reviews.append(ReviewHistoryItem(
                review_id=review.review_id, request_id=request_id, action=review.action, reason=review.reason,
                actor=review.actor, created_at=review.created_at, original_decision=review.original_decision,
                original_decision_label=policy.label, original_risk_score=review.original_risk_score,
                resulting_outcome=review.resulting_outcome, origin=record.origin, event_time=format_ts(request.ts),
                sender_id=request.sender_id, sender_name=self.party(request.sender_id).name,
                recipient_id=request.recipient_id, recipient_name=self.party(request.recipient_id).name,
                amount=request.amount, currency=request.currency, investigation_available=True))
            return self.investigation(request_id)

    def review_history(self, action: str | None, q: str | None, page: int, page_size: int,
                       sort: str) -> ReviewHistoryPage:
        with self.lock:
            entries = list(self.reviews)
        query = (q or "").strip().lower()
        selected = [e for e in entries if (not action or e.action == action) and (not query or query in f"{e.request_id} {e.sender_name} {e.recipient_name} {e.sender_id} {e.recipient_id} {e.actor} {e.reason}".lower())]
        if sort != "oldest":
            selected.reverse()
        counts = Counter(e.action for e in entries)
        start = (page - 1) * page_size
        return ReviewHistoryPage(items=selected[start:start + page_size], total=len(selected), page=page,
                                 page_size=page_size, approved=counts["approve"], rejected=counts["reject"],
                                 persisted=False, storage_note="Reviews live in memory; a restart clears them.")

    # -- queries -------------------------------------------------------------
    def investigation(self, request_id: str) -> Investigation:
        with self.lock:
            record = self._record(request_id)
            return Investigation(assessment=self._detail(record), timeline=views.timeline(record),
                                 reviews=[views.review_view(r) for r in record.reviews],
                                 can_review=record.result.policy.decision == "Verify" and not record.reviews)

    def transactions(self, f: TransactionQuery) -> TransactionPage:
        query = (f.q or "").strip().lower()

        def matches(r: AssessmentRecord) -> bool:
            policy, score, day = r.result.policy, r.result.risk.total, format_ts(r.request.ts)[:10]
            checks = (
                not f.decision or policy.decision == f.decision,
                not f.decision_source or policy.decision_source == f.decision_source,
                not f.origin or r.origin == f.origin,
                not f.review_status or views.review_status(r) == f.review_status,
                not f.date_from or day >= f.date_from,
                not f.date_to or day <= f.date_to,
                f.min_score is None or score >= f.min_score,
                f.max_score is None or score <= f.max_score,
            )
            if not all(checks):
                return False
            if not query:
                return True
            request = r.request
            return query in f"{request.request_id} {request.sender_id} {request.recipient_id} {self.party(request.sender_id).name} {self.party(request.recipient_id).name}".lower()

        selected = [r for r in self._all() if matches(r)]
        if f.sort == "risk_desc":
            selected.sort(key=lambda r: (r.result.risk.total, r.key), reverse=True)
        else:
            selected.sort(key=lambda r: r.key, reverse=f.sort != "time_asc")
        start = (f.page - 1) * f.page_size
        return TransactionPage(items=[views.summary(r, self.party) for r in selected[start:start + f.page_size]],
                               total=len(selected), page=f.page, page_size=f.page_size)

    def live_feed(self, after: int, limit: int) -> list[Any]:
        simulated = [r for r in self._all() if r.origin == "simulation" and r.seq > after]
        return [views.summary(r, self.party) for r in simulated[-limit:]]

    def overview(self) -> dict[str, Any]:
        records = self._all()
        labels = Counter(r.result.policy.label for r in records)
        origins = Counter(r.origin for r in records)
        volume: defaultdict[str, float] = defaultdict(float)
        days: dict[str, dict[str, Any]] = {}
        for r in records:
            volume[r.outcome] += r.request.amount
            day = format_ts(r.request.ts)[:10]
            bucket = days.setdefault(day, {"date": day, "requests": 0, "risk": [], "Permit": 0, "Verify": 0, "Block": 0})
            bucket["requests"] += 1
            bucket[r.result.policy.decision] += 1
            bucket["risk"].append(r.result.risk.total)
        pending = sorted((r for r in records if views.review_status(r) == "pending"),
                         key=lambda r: r.result.risk.total, reverse=True)
        return {
            "totals": {
                "assessed": len(records),
                "baseline": origins["baseline"],
                "simulations": origins["simulation"],
                "permitted": labels["Permitted"],
                "needs_verification": labels["Needs verification"],
                "preventive_blocks": labels["Preventive risk block"],
                "confirmed_evidence_blocks": labels["Confirmed fraud evidence"],
                "awaiting_review": len(pending),
                "reviewed": len(self.reviews),
            },
            "outcomes": dict(Counter(r.outcome for r in records)),
            "volumes": {
                "completed": round(sum(volume[o] for o in COMPLETED_OUTCOMES), 2),
                "pending_verification": round(volume["pending_verification"], 2),
                "blocked_attempts": round(volume["blocked_attempt"] + volume["rejected_after_review"], 2),
                "currency": "MDL",
                "note": "Completed volume counts only permitted or approved simulated outcomes. "
                        "Blocked and pending attempts are excluded; no funds move through a bank.",
            },
            "trend": [{"date": b["date"], "requests": b["requests"], "permit": b["Permit"], "verify": b["Verify"],
                       "block": b["Block"], "mean_risk": round(float(np.mean(b["risk"])), 2)}
                      for b in sorted(days.values(), key=lambda b: b["date"])],
            "recent": [views.summary(r, self.party) for r in sorted(records, key=lambda r: r.key, reverse=True)[:8]],
            "review_queue": [views.summary(r, self.party) for r in pending[:8]],
            "simulation_time": format_ts(self.simulation_now()),
        }

    def accounts_page(self, q: str | None, page: int, page_size: int, account_type: str | None) -> AccountPage:
        stats: dict[str, dict[str, Any]] = {}
        for r in self._all():
            s = stats.setdefault(r.request.sender_id, {"requests": 0, "flagged": 0, "rejected": 0, "last": None})
            s["requests"] += 1
            s["flagged"] += r.result.policy.decision != "Permit"
            s["rejected"] += r.outcome == "rejected_after_review"
            if s["last"] is None or r.key > s["last"].key:
                s["last"] = r
        query = (q or "").strip().lower()
        accounts = sorted((a for a in self.accounts.values()
                           if (not query or query in a.display_name.lower() or query in a.account_id.lower())
                           and (not account_type or a.account_type == account_type)), key=lambda a: a.display_name)
        items = []
        for a in accounts[(page - 1) * page_size:page * page_size]:
            s = stats.get(a.account_id, {"requests": 0, "flagged": 0, "rejected": 0, "last": None})
            last = s["last"]
            items.append(AccountSummary(
                account_id=a.account_id, display_name=a.display_name, account_type=a.account_type,
                home_country=a.home_country, created_at=format_ts(a.created_at), requests=s["requests"],
                flagged=s["flagged"], rejected=s["rejected"], last_activity=format_ts(last.request.ts) if last else None,
                last_decision=last.result.policy.decision if last else None))
        types = Counter(a.account_type for a in self.accounts.values())
        return AccountPage(items=items, total=len(accounts), page=page, page_size=page_size,
                           all_accounts=len(self.accounts), personal=types["personal"], business=types["business"],
                           active=sum(1 for a in self.accounts if a in stats))

    def account_detail(self, account_id: str) -> dict[str, Any]:
        if (account := self.accounts.get(account_id)) is None:
            raise NotFoundError(account_id)
        records = self._all()
        sent = [r for r in records if r.request.sender_id == account_id]
        received = [r for r in records if r.request.recipient_id == account_id]
        amounts = sorted(r.request.amount for r in sent)
        devices: dict[str, dict[str, Any]] = {}
        if auth := self.history.auth.get(account_id):
            for key, device in zip(auth.keys, auth.devices, strict=True):
                entry = devices.setdefault(device, {"device_id": device, "first_seen": format_ts(key // KEY_SCALE),
                                                    "sign_ins": 0})
                entry["sign_ins"] += 1
                entry["last_seen"] = format_ts(key // KEY_SCALE)
        latest = max(sent, key=lambda r: r.key) if sent else None
        context = None
        if latest:
            ctx = latest.result.components.context
            context = {"request_id": latest.request.request_id, "status": ctx.status, "score": ctx.score,
                       "distance": ctx.distance, "cluster": ctx.cluster, "at": format_ts(latest.request.ts)}
        return {
            "account": {"account_id": account.account_id, "display_name": account.display_name,
                        "account_type": account.account_type, "home_country": account.home_country,
                        "created_at": format_ts(account.created_at)},
            "profile": {
                "requests_sent": len(sent),
                "requests_received": len(received),
                "median_amount": amounts[len(amounts) // 2] if amounts else None,
                "p90_amount": amounts[int(len(amounts) * 0.9)] if amounts else None,
                "typical_hour": round(float(np.median([(r.request.ts % 86_400) / 3_600 for r in sent])), 1)
                if sent else None,
                "decisions": dict(Counter(r.result.policy.decision for r in sent)),
                "top_recipients": [{**self.party(rid).model_dump(), "payments": n}
                                   for rid, n in Counter(r.request.recipient_id for r in sent).most_common(6)],
                "devices": sorted(devices.values(), key=lambda d: d["first_seen"]),
            },
            "context": {"latest": context, "counts": dict(Counter(r.result.components.context.status for r in sent))},
            "activity": [views.summary(r, self.party)
                         for r in sorted(sent + received, key=lambda r: r.key, reverse=True)[:30]],
        }

    def parties(self, q: str, limit: int) -> list[dict[str, Any]]:
        query = q.strip().lower()
        found = [{"id": a.account_id, "name": a.display_name, "kind": "account", "detail": a.account_type}
                 for a in self.accounts.values()
                 if not query or query in a.display_name.lower() or query in a.account_id.lower()]
        found += [{"id": c.counterparty_id, "name": c.display_name, "kind": "counterparty", "detail": c.category}
                  for c in self.counterparties.values()
                  if query and (query in c.display_name.lower() or query in c.counterparty_id.lower())]
        return found[:limit]

    def simulator_options(self, sender_id: str | None) -> dict[str, Any]:
        now = self.simulation_now()
        sender = None
        if sender_id in self.accounts:
            series = self.history.sender.get(sender_id)
            recent = list(dict.fromkeys(reversed(series.recipients[-40:]))) if series else []
            amounts = sorted(series.amounts[-30:]) if series else []
            account = self.accounts[sender_id]
            auth = self.history.auth.get(sender_id)
            sender = {
                "account_id": sender_id,
                "account_type": account.account_type,
                "home_country": account.home_country,
                "max_amount": self.limits.for_segment(account.account_type).per_transaction,
                "known_devices": list(dict.fromkeys(reversed(auth.devices)))[:4] if auth else [],
                "recent_recipients": [self.party(r).model_dump() for r in recent[:8]],
                "median_amount": amounts[len(amounts) // 2] if amounts else None,
            }
        return {
            "simulation_time": format_ts(now),
            "currencies": ["MDL"],
            "payment_schemes": list(OUTGOING_SCHEMES),
            "limits": {segment: {"per_transaction": lim.per_transaction, "per_day": lim.per_day,
                                 "per_month": lim.per_month}
                       for segment, lim in (("personal", self.limits.personal), ("business", self.limits.business))},
            "auth_methods": ["biometric", "password_otp", "otp", "password"],
            "countries": sorted({a.home_country for a in self.accounts.values()} | set(FOREIGN_COUNTRIES)),
            "flagged_beneficiaries": [
                {"evidence_id": e.evidence_id, "id": e.entity_id, "name": self.party(e.entity_id).name,
                 "evidence_type": e.evidence_type} for e in self.evidence.active_entities("counterparty", now)],
            "sender": sender,
        }

    # -- algorithms and health -----------------------------------------------
    def _thresholds(self) -> dict[str, dict[str, float]]:
        return {name: {"verify": v, "block": b} for name, (v, b) in self.point.segments.items()}

    def algorithms(self) -> dict[str, Any]:
        records = self._all()

        def histogram(values: list[float], top: float = 1.0) -> list[int]:
            return np.histogram(values, bins=HISTOGRAM_BINS, range=(0.0, top))[0].tolist()

        daily: dict[str, Counter] = defaultdict(Counter)
        for r in records:
            daily[format_ts(r.request.ts)[:10]][r.result.components.context.status] += 1
        training = self.training
        point = self.point
        return {
            "model": self.bundle.metadata,
            "model_error": None,
            "policy": {
                "version": self.policy.version,
                "weights": dict(self.policy.weights),
                "weights_source": "policy",
                "segments": self._thresholds(),
                "operating_point": {"verify": point.verify, "block": point.block, "source": point.source,
                                    "version": point.version},
                "min_corroborating_families": self.policy.min_families,
                "families": dict(self.policy.families),
                "rules": [{"code": r.code, "family": r.family, "weight": r.weight, "corroborating": r.corroborating,
                           "description": r.description, "params": dict(r.params)} for r in self.policy.rules],
                "cold_start": {"max_amount_without_history": self.policy.cold_start_max_amount,
                               "strong_auth_methods": sorted(self.policy.strong_auth_methods)},
                "trusted_sources": list(self.policy.trusted_sources),
            },
            "distributions": {
                "isolation_forest": histogram([r.result.components.if_score for r in records]),
                "lof": histogram([r.result.components.lof_score for r in records]),
                "rules": histogram([r.result.rules.score for r in records]),
                "dbscan": histogram([r.result.components.context.score for r in records]),
                "risk_score": histogram([r.result.risk.total for r in records], 100.0),
            },
            "histogram_bins": HISTOGRAM_BINS,
            "dbscan_context": {
                "status_counts": dict(Counter(r.result.components.context.status for r in records)),
                "daily": [{"date": d, **dict(c)} for d, c in sorted(daily.items())],
            },
            "decision_sources": dict(Counter(f"{r.result.policy.decision}/{r.result.policy.decision_source}"
                                             for r in records)),
            "training": training and {
                "validation_component_average_precision": training["validation_component_average_precision"],
                "threshold_selection": training["threshold_selection"],
                "search": training["search"],
                "weights_search": training["weights_search"],
                "segment_thresholds": training["segment_thresholds"],
                "holdout_accounts": 0,
            },
        }

    def dashboard(self, window: str) -> dict[str, Any]:
        return dashboard(self._all(), window, self.simulation_now(), self.point, dict(self.policy.goals))

    def cluster_graph(self, max_anomalies: int) -> dict[str, Any]:
        return self._graph.build(self._all(), self.party, max_anomalies)

    def health(self, stream: dict[str, Any]) -> dict[str, Any]:
        requests = self.shown
        simulations = sum(1 for r in self._all() if r.origin == "simulation")
        return {
            "status": "ok",
            "started_at": self.started_at,
            "startup_seconds": self.startup_seconds,
            "data": {
                "source": "local_files",
                "dataset": self.manifest.get("dataset", "secflow-mia-instant"),
                "file": "storage/test.json",
                "generator_version": self.manifest.get("generator_version"),
                "seed": self.manifest.get("seed"),
                "currency": "MDL",
                "requests": len(requests),
                "context_credits": self.dataset.context_credits,
                "accounts": len(self.accounts),
                "hashes_verified": self.hashes_verified,
                "date_range": {"start": format_ts(requests[0].ts), "end": format_ts(requests[-1].ts)},
            },
            "model": {"loaded": True, "version": self.bundle.version, "error": None},
            "policy": {
                "version": self.policy.version,
                "operating_point": {"verify": self.point.verify, "block": self.point.block, "source": self.point.source,
                                    "weights": dict(self.policy.weights), "segments": self._thresholds()},
            },
            "state": {
                "kind": "process_memory",
                "simulations": simulations,
                "reviews": len(self.reviews),
                "persisted": False,
                "shared_across_workers": False,
                "review_history": {"persisted": False, "path": None, "error": None},
                "note": "Simulations and reviews live in this process only; a restart discards them. "
                        "Run a single backend process.",
            },
            "stream": stream,
            "simulation_time": format_ts(self.simulation_now()),
        }

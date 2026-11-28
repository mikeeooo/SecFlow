"""One path from a payment to a decision, shared by the app, the training report and the evaluation.

features (only past events) -> IsolationForest, LOF, DBSCAN context -> rules -> risk score ->
active trusted evidence -> decision table.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from datagen.records import Account, PaymentRequest
from ml.evidence import EvidenceStore, SecurityEvidence, evidence_from_row
from ml.features import FeatureSnapshot, build_features
from ml.history import HistoryIndex
from ml.model import Components, ModelBundle
from ml.policy import (
    OperatingPoint,
    Policy,
    PolicyResult,
    RiskBreakdown,
    RuleResult,
    assessment_status,
    combine_scores,
    decide_action,
    evaluate_rules,
    segment_of,
)


@dataclass(frozen=True)
class AssessmentResult:
    request: PaymentRequest
    snapshot: FeatureSnapshot
    components: Components
    rules: RuleResult
    risk: RiskBreakdown
    evidence: tuple[SecurityEvidence, ...]
    policy: PolicyResult
    status_notes: tuple[str, ...]
    segment: str


@dataclass(frozen=True)
class Assessor:
    """Everything a decision needs that does not change between payments."""

    bundle: ModelBundle
    evidence: EvidenceStore
    policy: Policy
    point: OperatingPoint

    def assess_batch(self, requests: Sequence[PaymentRequest], keys: Mapping[str, int], history: HistoryIndex,
                     accounts: Mapping[str, Account]) -> list[AssessmentResult]:
        """Assess payments; each one sees only history ordered before its own key."""
        snapshots = [build_features(r, keys[r.request_id], history, accounts.get(r.sender_id)) for r in requests]
        if not snapshots:
            return []
        components = self.bundle.score_batch(np.array([s.payment_vector for s in snapshots]),
                                             np.array([s.context_vector for s in snapshots]),
                                             [s.context_status for s in snapshots])
        return [self._decide(r, s, c, segment_of(accounts.get(r.sender_id)))
                for r, s, c in zip(requests, snapshots, components, strict=True)]

    def _decide(self, request: PaymentRequest, snapshot: FeatureSnapshot, components: Components,
                segment: str) -> AssessmentResult:
        rules = evaluate_rules(snapshot.facts, snapshot.history_status, self.policy)
        status, notes = assessment_status(snapshot.history_status, request.amount, request.auth_method, self.policy)
        risk = combine_scores({"isolation_forest": components.if_score, "lof": components.lof_score,
                               "dbscan": components.context.score, "rules": rules.score}, self.policy.weights)
        evidence = tuple(self.evidence.active_for(request))
        decision = decide_action(risk, rules, evidence, status, self.point, self.policy, segment)
        return AssessmentResult(request, snapshot, components, rules, risk, evidence, decision, tuple(notes), segment)


def evidence_store(rows: Sequence[dict], policy: Policy) -> EvidenceStore:
    return EvidenceStore((evidence_from_row(r) for r in rows), policy.trusted_sources, policy.blocking_types)

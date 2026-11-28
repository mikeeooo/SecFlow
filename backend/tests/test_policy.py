"""The risk score, the rules and the decision table."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backend.tests.conftest import ROOT
from backend.tests.test_features import T0, request
from ml.evidence import EvidenceStore, SecurityEvidence
from ml.policy import (
    OperatingPoint,
    RiskBreakdown,
    RuleResult,
    assessment_status,
    combine_scores,
    decide_action,
    decision_codes,
    evaluate_rules,
    load_policy,
)

POINT = OperatingPoint(60.0, 85.0, "validated", "op-test")
NO_RULES = RuleResult(0.0, (), ())
TWO_FAMILIES = RuleResult(0.6, (), ("compromised_access", "velocity"))


def risk(total: float) -> RiskBreakdown:
    return RiskBreakdown(total, {"isolation_forest": total, "lof": 0.0, "dbscan": 0.0, "rules": 0.0})


def decide(
    total: float, rules: RuleResult = NO_RULES, evidence=(), status: str = "assessed", *, policy
) -> str:
    return decide_action(risk(total), rules, evidence, status, POINT, policy).decision


def evidence(**changes) -> SecurityEvidence:
    base = {
        "evidence_id": "EV-1",
        "evidence_type": "compromised_session",
        "entity_type": "session",
        "entity_id": "S-R1",
        "source": "bank_security_operations",
        "recorded_at": T0 - 60,
        "effective_from": T0 - 60,
        "effective_until": None,
        "status_history": ((T0 - 60, "confirmed"),),
        "note": "",
    }
    return SecurityEvidence(**{**base, **changes})


def test_risk_is_the_weighted_sum_of_the_components(policy) -> None:
    breakdown = combine_scores(
        {"isolation_forest": 0.8, "lof": 0.4, "dbscan": 1.0, "rules": 0.5}, policy.weights
    )
    assert breakdown.total == pytest.approx(100 * (0.65 * 0.8 + 0.05 * 0.4 + 0.0 * 1.0 + 0.30 * 0.5))
    assert sum(breakdown.contributions.values()) == pytest.approx(breakdown.total)


def test_decision_follows_the_thresholds(policy) -> None:
    assert decide(59.9, policy=policy) == "Permit"
    assert decide(60.0, policy=policy) == "Verify"
    assert decide(90.0, TWO_FAMILIES, policy=policy) == "Block"


def test_a_high_score_without_two_corroborating_families_is_only_verified(policy) -> None:
    one_family = RuleResult(0.6, (), ("velocity",))
    assert decide(99.0, one_family, policy=policy) == "Verify"


def test_trusted_evidence_blocks_even_a_low_score(policy) -> None:
    store = EvidenceStore([evidence()], policy.trusted_sources, policy.blocking_types)
    active = store.active_for(request("R1", T0))
    result = decide_action(risk(5.0), NO_RULES, active, "assessed", POINT, policy)
    assert (result.decision, result.decision_source) == ("Block", "confirmed_fraud_evidence")


@pytest.mark.parametrize(
    "item",
    [
        evidence(source="external_tip_feed"),
        evidence(recorded_at=T0 + 60),
        evidence(effective_until=T0 - 1),
        evidence(status_history=((T0 - 60, "confirmed"), (T0 - 30, "retracted"))),
        evidence(status_history=((T0 - 60, "pending_review"),)),
    ],
    ids=["untrusted", "recorded_later", "expired", "retracted", "pending"],
)
def test_inactive_or_untrusted_evidence_never_blocks(policy, item: SecurityEvidence) -> None:
    store = EvidenceStore([item], policy.trusted_sources, policy.blocking_types)
    assert store.active_for(request("R1", T0)) == []


def test_a_sender_without_history_is_verified_unless_small_and_strongly_authenticated(policy) -> None:
    assert assessment_status("none", 2_000, "biometric", policy)[0] == "insufficient_history"
    assert assessment_status("none", 200, "password", policy)[0] == "insufficient_history"
    assert assessment_status("none", 200, "biometric", policy)[0] == "limited_history"
    assert decide(1.0, status="insufficient_history", policy=policy) == "Verify"


def test_rules_of_one_family_corroborate_once(policy) -> None:
    facts = {
        "amount": 100.0,
        "auth_failures_24h": 4,
        "device_change": True,
        "device_age_hours": 1.0,
        "foreign_ip": False,
        "requests_1h": 0,
        "requests_24h": 0,
        "unique_recipients_24h": 0,
        "new_recipient": False,
        "recipient_senders_24h": 0,
    }
    result = evaluate_rules(facts, "sufficient", policy)
    assert {"NEW_DEVICE_AFTER_FAILED_LOGINS", "REPEATED_FAILED_LOGINS"} <= {t.code for t in result.triggered}
    assert result.corroborating_families == ("compromised_access",)


def test_history_based_rules_stay_silent_without_history(policy) -> None:
    facts = {
        "amount": 4_000.0,
        "amount_zscore_30d": None,
        "auth_failures_24h": 0,
        "device_change": True,
        "foreign_ip": False,
        "requests_1h": 0,
        "requests_24h": 0,
        "unique_recipients_24h": 0,
        "new_recipient": True,
        "recipient_senders_24h": 0,
    }
    codes = {t.code for t in evaluate_rules(facts, "none", policy).triggered}
    assert not codes & {"NEW_DEVICE", "NEW_BENEFICIARY", "AMOUNT_SPIKE", "NEW_BENEFICIARY_HIGH_AMOUNT"}


def test_array_decisions_match_the_decision_table(policy) -> None:
    rng = np.random.default_rng(0)
    totals, families = rng.uniform(0, 100, 300), rng.integers(0, 4, 300)
    unsafe = rng.random(300) < 0.1
    codes = decision_codes(totals, families, unsafe, POINT.verify, POINT.block, policy.min_families)
    for total, n, bad, code in zip(totals, families, unsafe, codes, strict=True):
        rules = RuleResult(0.0, (), tuple(f"f{i}" for i in range(n)))
        status = "insufficient_history" if bad else "assessed"
        assert {"Permit": 0, "Verify": 1, "Block": 2}[
            decide(total, rules, status=status, policy=policy)
        ] == code


def test_policy_weights_must_sum_to_one(tmp_path: Path) -> None:
    raw = json.loads((ROOT / "config" / "policy.json").read_text())
    raw["weights"]["rules"] = 0.9
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="sum to 1"):
        load_policy(path)

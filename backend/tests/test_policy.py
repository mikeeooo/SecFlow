from __future__ import annotations

import itertools
from dataclasses import replace
from unittest import mock

import numpy as np
import pytest

from config.settings import load_settings
from ml.policy import (
    SEGMENTS,
    ComponentScores,
    OperatingPoint,
    RuleResult,
    TriggeredRule,
    assessment_status,
    combine_arrays,
    combine_scores,
    decide_action,
    decide_arrays,
    evaluate_rules,
    load_policy,
    segment_of,
)
from storage.evidence import EvidenceStore, SecurityEvidence, StatusChange
from storage.records import Account, PaymentRequest

POLICY = load_policy(load_settings().policy_path)
OP = OperatingPoint(verify=40.0, block=70.0, source="test", version="op-test")
T = 1_790_000_000


def components(i: float, l: float, d: float, r: float) -> ComponentScores:
    return ComponentScores(i, l, d, r)


def rules_with(*families: str) -> RuleResult:
    triggered = tuple(TriggeredRule(f"R_{f}", f, 0.3, True, "x", "y") for f in families)
    return RuleResult(min(1.0, 0.3 * len(families)), triggered, tuple(sorted(set(families))))


def test_numeric_example_yields_42() -> None:
    risk = combine_scores(components(0.2, 0.4, 1.0, 0.5), POLICY.weights)
    assert risk.total == pytest.approx(42.0)
    assert sum(risk.contributions.values()) == pytest.approx(risk.total)
    assert risk.contributions == pytest.approx({"isolation_forest": 7.0, "lof": 10.0, "dbscan": 10.0, "rules": 15.0})


def test_vectorised_combination_matches_scalar() -> None:
    values = np.random.default_rng(1).random((50, 4))
    scalar = [combine_scores(components(*row), POLICY.weights).total for row in values]
    vector = combine_arrays(values[:, 0], values[:, 1], values[:, 2], values[:, 3], POLICY.weights)
    np.testing.assert_allclose(vector, scalar)


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf")])
def test_components_must_be_finite_and_in_range(bad: float) -> None:
    with pytest.raises(ValueError):
        components(bad, 0.0, 0.0, 0.0)


def test_policy_combination_never_calls_an_estimator() -> None:
    with mock.patch("sklearn.ensemble.IsolationForest.score_samples") as if_scores, mock.patch(
        "sklearn.neighbors.LocalOutlierFactor.score_samples"
    ) as lof_scores:
        risk = combine_scores(components(0.9, 0.9, 1.0, 1.0), POLICY.weights)
        decide_action(risk, rules_with("velocity", "amount_deviation"), [], "assessed", OP, POLICY)
    if_scores.assert_not_called()
    lof_scores.assert_not_called()


def decide(total_components: ComponentScores, rules: RuleResult, evidence=(), status: str = "assessed"):
    return decide_action(combine_scores(total_components, POLICY.weights), rules, list(evidence), status, OP, POLICY)


def test_low_risk_is_permitted() -> None:
    result = decide(components(0.2, 0.2, 0.0, 0.0), rules_with())
    assert (result.decision, result.decision_source, result.label) == ("Permit", "risk_policy", "Permitted")


def test_ambiguous_risk_goes_to_verify() -> None:
    result = decide(components(0.8, 0.6, 0.0, 0.2), rules_with("velocity"))
    assert result.decision == "Verify" and result.label == "Needs verification"


def test_high_score_without_corroboration_is_verify_not_block() -> None:
    result = decide(components(1.0, 1.0, 1.0, 1.0), rules_with("velocity"))
    assert result.decision == "Verify"
    assert "without enough independent corroborating" in result.explanation


def test_corroborated_high_risk_is_a_preventive_block() -> None:
    result = decide(components(1.0, 1.0, 1.0, 0.6), rules_with("velocity", "compromised_access"))
    assert (result.decision, result.decision_source, result.label) == ("Block", "risk_policy", "Preventive risk block")
    assert result.corroborating_signals == ("compromised_access", "velocity")
    assert "not confirmed fraud" in result.explanation


def test_threshold_boundaries_use_unrounded_values() -> None:
    at_verify = replace(OP, verify=combine_scores(components(0.5, 0.5, 0.0, 0.5), POLICY.weights).total)
    risk = combine_scores(components(0.5, 0.5, 0.0, 0.5), POLICY.weights)
    assert decide_action(risk, rules_with(), [], "assessed", at_verify, POLICY).decision == "Verify"
    just_below = replace(at_verify, verify=at_verify.verify + 1e-9)
    assert decide_action(risk, rules_with(), [], "assessed", just_below, POLICY).decision == "Permit"


def evidence(**overrides) -> SecurityEvidence:
    base = {
        "evidence_id": "SEC-1", "evidence_type": "fraud_controlled_beneficiary", "entity_type": "counterparty",
        "entity_id": "CP-BAD", "source": "bank_security_operations", "recorded_at": T - 100, "effective_from": T - 100,
        "effective_until": None, "status_history": (StatusChange("confirmed", T - 100, None),), "note": "",
    }
    return SecurityEvidence(**{**base, **overrides})


def request_to(recipient: str, ts: int = T) -> PaymentRequest:
    return PaymentRequest("R1", ts, "ACC-0001", recipient, "counterparty", 20.0, "EUR", "mobile_app", "DEV-1",
                          "SES-1", "biometric", "DE")


def store(*items: SecurityEvidence) -> EvidenceStore:
    return EvidenceStore(items, POLICY.trusted_sources, POLICY.blocking_types)


def test_confirmed_evidence_blocks_even_with_low_score() -> None:
    active = store(evidence()).active_for(request_to("CP-BAD"))
    result = decide(components(0.0, 0.0, 0.0, 0.0), rules_with(), active)
    assert (result.decision, result.decision_source, result.label) == (
        "Block", "confirmed_fraud_evidence", "Confirmed fraud evidence")
    assert result.evidence_ids == ("SEC-1",)


@pytest.mark.parametrize(
    "item",
    [
        evidence(effective_until=T - 10),  # expired
        evidence(status_history=(StatusChange("confirmed", T - 100, None), StatusChange("retracted", T - 50, "fp"))),
        evidence(recorded_at=T + 10),  # future knowledge
        evidence(effective_from=T + 10),  # not yet effective
        evidence(status_history=(StatusChange("pending_review", T - 100, None),)),
        evidence(source="external_tip_feed"),  # untrusted
        evidence(evidence_type="session_revoked_administratively"),  # not a blocking type
    ],
)
def test_inactive_or_untrusted_evidence_never_blocks(item: SecurityEvidence) -> None:
    active = store(item).active_for(request_to("CP-BAD"))
    assert active == []
    assert decide(components(0.1, 0.1, 0.0, 0.0), rules_with(), active).decision == "Permit"


def test_retraction_after_request_time_does_not_rewrite_the_past() -> None:
    item = evidence(status_history=(StatusChange("confirmed", T - 100, None), StatusChange("retracted", T + 50, "fp")))
    assert [e.evidence_id for e in store(item).active_for(request_to("CP-BAD"))] == ["SEC-1"]
    assert store(item).active_for(request_to("CP-BAD", ts=T + 60)) == []


def test_unavailable_model_or_missing_history_requires_verification() -> None:
    status, notes = assessment_status("sufficient", 50, "biometric", False, True, POLICY)
    result = decide_action(None, rules_with(), [], status, OP, POLICY)
    assert (status, result.decision, result.decision_source) == ("model_unavailable", "Verify", "assessment_safeguard")
    assert notes
    cold_status, _ = assessment_status("none", 5_000, "password", True, True, POLICY)
    assert cold_status == "insufficient_history"
    assert decide(components(0.1, 0.1, 0.0, 0.0), rules_with(), status=cold_status).decision == "Verify"


def test_cold_start_with_strong_auth_and_small_amount_is_assessed() -> None:
    status, _ = assessment_status("none", 40, "biometric", True, True, POLICY)
    assert status == "limited_history"
    assert decide(components(0.3, 0.3, 0.0, 0.0), rules_with(), status=status).decision == "Permit"


BASE_FACTS = {
    "amount": 100.0, "auth_method": "password", "ip_country": "DE", "home_country": "DE", "prior_requests_total": 40,
    "prior_requests_30d": 20, "median_amount_30d": 40.0, "max_amount_30d": 90.0, "amount_zscore_30d": 0.5,
    "requests_1h": 0, "requests_24h": 1, "sum_24h": 40.0, "velocity_ratio_24h": 1.0, "unique_recipients_24h": 1,
    "new_recipient": False, "recipient_relationship_hours": 900.0, "recipient_senders_24h": 0,
    "recipient_history_days": 80.0, "device_age_hours": 2_000.0, "new_device": False, "device_change": False, "auth_failures_24h": 0,
    "hours_since_last_request": 5.0, "account_age_days": 700.0, "incoming_24h": 0.0, "foreign_ip": False,
}


def test_rules_from_same_fact_family_count_once() -> None:
    facts = {**BASE_FACTS, "new_device": True, "device_change": True, "device_age_hours": 0.1, "auth_failures_24h": 4, "foreign_ip": True,
             "ip_country": "US"}
    result = evaluate_rules(facts, "sufficient", POLICY)
    codes = {t.code for t in result.triggered}
    assert {"NEW_DEVICE_AFTER_FAILED_LOGINS", "REPEATED_FAILED_LOGINS", "NEW_DEVICE", "FOREIGN_SESSION"} <= codes
    assert result.corroborating_families == ("compromised_access",)


def test_rules_have_stable_codes_and_bounded_score() -> None:
    facts = {**BASE_FACTS, "new_device": True, "device_change": True, "device_age_hours": 0.1, "auth_failures_24h": 4, "requests_1h": 6,
             "requests_24h": 9, "velocity_ratio_24h": 9.0, "amount": 5_000.0, "amount_zscore_30d": 6.0,
             "new_recipient": True, "recipient_relationship_hours": None, "recipient_history_days": None,
             "recipient_senders_24h": 3, "incoming_24h": 4_000.0, "sum_24h": 1_000.0, "unique_recipients_24h": 6}
    first = evaluate_rules(facts, "sufficient", POLICY)
    second = evaluate_rules(facts, "sufficient", POLICY)
    assert first == second
    assert first.score == 1.0
    assert len(first.corroborating_families) >= 3


def test_history_dependent_rules_do_not_fire_without_history() -> None:
    facts = {**BASE_FACTS, "new_recipient": True, "new_device": True, "amount_zscore_30d": None}
    codes = {t.code for t in evaluate_rules(facts, "none", POLICY).triggered}
    assert not codes & {"NEW_BENEFICIARY", "NEW_DEVICE", "AMOUNT_SPIKE", "NEW_BENEFICIARY_HIGH_AMOUNT"}


def test_vectorised_decisions_match_the_action_table() -> None:
    rng = np.random.default_rng(3)
    risk = rng.uniform(0, 100, 400)
    fams = rng.integers(0, 4, 400)
    unsafe = rng.random(400) < 0.1
    has_evidence = rng.random(400) < 0.05
    vector = decide_arrays(risk, fams, unsafe, has_evidence, OP.verify, OP.block, POLICY.min_families)
    codes = {"Permit": 0, "Verify": 1, "Block": 2}
    families = ["velocity", "amount_deviation", "compromised_access"]
    for i in range(400):
        breakdown = mock.Mock(total=float(risk[i]))
        rules = RuleResult(0.0, (), tuple(itertools.islice(families, int(fams[i]))))
        ev = [evidence()] if has_evidence[i] else []
        status = "insufficient_history" if unsafe[i] else "assessed"
        assert codes[decide_action(breakdown, rules, ev, status, OP, POLICY).decision] == vector[i]


def test_weights_must_sum_to_one(tmp_path) -> None:
    import json

    raw = json.loads(load_settings().policy_path.read_text())
    raw["weights"]["rules"] = 0.5
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_policy(path)


def _codes(facts: dict, status: str = "sufficient") -> set[str]:
    return {t.code for t in evaluate_rules(facts, status, POLICY).triggered}


def test_repeat_new_beneficiary_fires_on_second_payment() -> None:
    facts = {**BASE_FACTS, "new_recipient": True, "pair_requests_24h": 1}
    hit = [t for t in evaluate_rules(facts, "sufficient", POLICY).triggered if t.code == "REPEAT_NEW_BENEFICIARY"]
    assert len(hit) == 1
    assert (hit[0].family, hit[0].weight, hit[0].corroborating) == ("beneficiary_risk", 0.15, False)
    assert "2 payments to this new beneficiary in 24 hours" in hit[0].detail


def test_repeat_new_beneficiary_silent_on_first_or_established() -> None:
    assert "REPEAT_NEW_BENEFICIARY" not in _codes({**BASE_FACTS, "new_recipient": True, "pair_requests_24h": 0})
    assert "REPEAT_NEW_BENEFICIARY" not in _codes({**BASE_FACTS, "new_recipient": False, "pair_requests_24h": 3})
    assert "REPEAT_NEW_BENEFICIARY" not in _codes({**BASE_FACTS, "new_recipient": True, "pair_requests_24h": 2}, "none")


def test_repeat_new_beneficiary_does_not_corroborate() -> None:
    result = evaluate_rules({**BASE_FACTS, "new_recipient": True, "pair_requests_24h": 2}, "sufficient", POLICY)
    assert "REPEAT_NEW_BENEFICIARY" in {t.code for t in result.triggered}
    assert "beneficiary_risk" not in result.corroborating_families


# -- learned weights and per-segment thresholds ------------------------------------------------

SEGMENTED = OperatingPoint(verify=40.0, block=70.0, source="test", version="op-seg",
                           weights={"isolation_forest": 0.5, "lof": 0.05, "dbscan": 0.05, "rules": 0.4},
                           segments={"personal": (60.0, 90.0), "business": (70.0, 95.0)})


def _risk_at(total: float):
    # all four components equal -> risk == 100 * value whatever the weights
    return combine_scores(components(total / 100, total / 100, total / 100, total / 100), POLICY.weights)


def test_thresholds_for_falls_back_to_global() -> None:
    assert OP.thresholds_for("business") == (40.0, 70.0)
    assert SEGMENTED.thresholds_for("business") == (70.0, 95.0)
    assert SEGMENTED.thresholds_for("unknown") == (40.0, 70.0)


def test_segment_of_unknown_or_missing_account_is_personal() -> None:
    assert segment_of(None) == "personal"
    assert segment_of(Account("A", "n", "personal", "DE", 0)) == "personal"
    assert segment_of(Account("B", "n", "business", "DE", 0)) == "business"
    assert segment_of(Account("C", "n", "charity", "DE", 0)) == "personal"
    assert SEGMENTS == ("personal", "business")


def test_decide_action_uses_segment_thresholds() -> None:
    rules = rules_with()
    personal = decide_action(_risk_at(65.0), rules, [], "assessed", SEGMENTED, POLICY, segment="personal")
    business = decide_action(_risk_at(65.0), rules, [], "assessed", SEGMENTED, POLICY, segment="business")
    assert (personal.decision, business.decision) == ("Verify", "Permit")
    assert "personal" in personal.explanation and "business" in business.explanation


def test_decide_arrays_matches_decide_action_with_per_row_thresholds() -> None:
    rng = np.random.default_rng(3)
    segments = np.array(["personal", "business"] * 50)
    risk = rng.uniform(0, 100, 100)
    risk[:4] = [60.0, 70.0, 90.0, 95.0]  # exactly at segment thresholds
    families = rng.integers(0, 3, 100)
    verify = np.array([SEGMENTED.thresholds_for(s)[0] for s in segments])
    block = np.array([SEGMENTED.thresholds_for(s)[1] for s in segments])
    no = np.zeros(100, dtype=bool)
    vector = decide_arrays(risk, families, no, no, verify, block, POLICY.min_families)
    codes = {"Permit": 0, "Verify": 1, "Block": 2}
    for i in range(100):
        fam = tuple(f"f{j}" for j in range(families[i]))
        rules = RuleResult(0.0, (), fam)
        breakdown = _risk_at(float(risk[i]))
        got = decide_action(breakdown, rules, [], "assessed", SEGMENTED, POLICY, segment=str(segments[i])).decision
        assert codes[got] == vector[i], i


def test_combination_uses_learned_weights_and_contributions_sum() -> None:
    from ml.algorithms.dbscan_context import ContextResult
    from ml.assess import finalize
    from ml.features import FeatureSnapshot
    from ml.model import ModelComponents
    from storage.evidence import EvidenceStore

    request = PaymentRequest("R1", T, "ACC-1", "CP-1", "counterparty", 50.0, "EUR", "mobile_app", "D", "S",
                             "biometric", "DE")
    snapshot = FeatureSnapshot((0.0,) * 24, (0.0,) * 8, "sufficient", "ok", {**BASE_FACTS})
    comps = ModelComponents(0.0, 0.8, 0.0, 0.2, ContextResult(0.5, "outside_context", 1.0, None, 0))
    result = finalize(request, snapshot, comps, EvidenceStore([], POLICY.trusted_sources, POLICY.blocking_types),
                      POLICY, SEGMENTED, "business")
    assert result.segment == "business"
    expected = {"isolation_forest": 100 * 0.5 * 0.8, "lof": 100 * 0.05 * 0.2, "dbscan": 100 * 0.05 * 0.5}
    for name, value in expected.items():
        assert result.risk.contributions[name] == pytest.approx(value)
    assert result.risk.total == pytest.approx(sum(result.risk.contributions.values()))

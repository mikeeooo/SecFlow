"""Features use only events before the payment, and never identifiers or labels."""

from __future__ import annotations

import math

from datagen.records import Account, AuthEvent, PaymentRequest
from ml.features import CONTEXT_FEATURES, PAYMENT_FEATURES, build_features
from ml.history import HistoryIndex

T0 = 1_780_000_000
ACCOUNT = Account("ACC-0001", "Ion Rusu", "personal", "MD", T0 - 400 * 86_400)


def request(rid: str, ts: int, amount: float = 50.0, recipient: str = "CP-00001", device: str = "DEV-1") -> PaymentRequest:
    return PaymentRequest(rid, ts, "ACC-0001", recipient, "counterparty", amount, "MDL", "mobile_app", device,
                          f"S-{rid}", "biometric", "MD")


def features(events: list[PaymentRequest], target: PaymentRequest, auth: list[AuthEvent] | None = None):
    index, keys = HistoryIndex.from_events(events, auth or [], [])
    return build_features(target, keys[target.request_id], index, ACCOUNT)


def test_vectors_have_the_documented_layout_and_are_finite() -> None:
    snap = features([request("R1", T0)], request("R1", T0))
    assert len(snap.payment_vector) == len(PAYMENT_FEATURES) == 31
    assert len(snap.context_vector) == len(CONTEXT_FEATURES) == 8
    assert all(math.isfinite(v) for v in snap.payment_vector + snap.context_vector)


def test_no_feature_is_an_identifier_or_a_label() -> None:
    for name in PAYMENT_FEATURES + CONTEXT_FEATURES:
        assert not any(word in name for word in ("_id", "name", "fraud", "label", "campaign", "anomaly_type"))


def test_a_first_payment_has_no_history_rather_than_a_suspicious_one() -> None:
    snap = features([request("R1", T0)], request("R1", T0))
    assert snap.history_status == "none"
    assert snap.context_status == "insufficient_history"
    assert snap.facts["prior_requests_total"] == snap.facts["requests_24h"] == 0


def test_the_payment_is_not_part_of_its_own_history() -> None:
    events = [request(f"R{i}", T0 + i * 60) for i in range(5)]
    assert features(events, events[-1]).facts["requests_1h"] == 4


def test_future_events_do_not_change_past_features() -> None:
    past = [request(f"R{i}", T0 + i * 3_600, amount=40 + i) for i in range(10)]
    future = [request(f"F{i}", T0 + 20 * 3_600 + i, amount=4_999, recipient="CP-NEW", device="DEV-X") for i in range(5)]
    before, after = features(past, past[6]), features(past + future, past[6])
    assert before.payment_vector == after.payment_vector
    assert before.context_vector == after.context_vector


def test_equal_timestamps_resolve_by_identifier() -> None:
    first, second = request("R-A", T0), request("R-B", T0)
    assert features([second, first], first).facts["prior_requests_total"] == 0
    assert features([second, first], second).facts["prior_requests_total"] == 1


def test_the_24_hour_window_includes_its_start_and_excludes_older_payments() -> None:
    current = request("R2", T0)
    events = [request("R0", T0 - 86_401), request("R1", T0 - 86_400), current]
    assert features(events, current).facts["requests_24h"] == 1


def test_failed_sign_ins_after_the_payment_are_ignored() -> None:
    current = request("R1", T0)
    auth = [AuthEvent(f"A{i}", ts, "ACC-0001", "S", "DEV-1", "login_failure", "password", "MD", None)
            for i, ts in enumerate((T0 - 100, T0 + 100))]
    assert features([current], current, auth).facts["auth_failures_24h"] == 1


def test_a_beneficiary_stays_new_for_24_hours() -> None:
    events = [request("R1", T0, recipient="CP-9"), request("R2", T0 + 60, recipient="CP-9"),
              request("R3", T0 + 2 * 86_400, recipient="CP-9")]
    assert features(events, events[1]).facts["new_recipient"] is True
    assert features(events, events[2]).facts["new_recipient"] is False

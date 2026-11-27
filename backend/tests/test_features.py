from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from datagen.dataset import load_partition, read_rows
from datagen.generator import write_rows
from datagen.records import Account, AuthEvent, PaymentRequest
from ml.features import CONTEXT_FEATURES, FEATURE_VERSION, PAYMENT_FEATURES, build_features
from ml.replay import replay
from runtime.history import HistoryIndex

T0 = 1_780_000_000
ACCOUNT = Account("ACC-0001", "Ion Rusu", "personal", "MD", T0 - 400 * 86_400)
BUSINESS = Account("ACC-0002", "Valea Verde Grocers", "business", "MD", T0 - 400 * 86_400)


def request(rid: str, ts: int, amount: float = 50.0, recipient: str = "CP-00001", device: str = "DEV-1",
            sender: str = "ACC-0001", scheme: str = "p2p_alias", qr: str | None = None) -> PaymentRequest:
    return PaymentRequest(rid, ts, sender, recipient, "counterparty", amount, "MDL", "mobile_app", device,
                          f"S-{rid}", "biometric", "MD", payment_scheme=scheme, qr_payload_merchant_id=qr)


def build(events: list[PaymentRequest]) -> tuple[HistoryIndex, dict[str, int]]:
    return HistoryIndex.from_events(events, [], [])


def test_feature_vectors_have_stable_order_and_are_finite() -> None:
    index, keys = build([request("R1", T0)])
    snap = build_features(request("R1", T0), keys["R1"], index, ACCOUNT)
    assert len(snap.payment_vector) == len(PAYMENT_FEATURES) == 31
    assert len(snap.context_vector) == len(CONTEXT_FEATURES) == 8
    assert all(math.isfinite(v) for v in snap.payment_vector + snap.context_vector)


def test_empty_history_is_explicit_not_fraud() -> None:
    index, keys = build([request("R1", T0)])
    snap = build_features(request("R1", T0), keys["R1"], index, ACCOUNT)
    assert snap.history_status == "none"
    assert snap.context_status == "insufficient_history"
    assert snap.facts["prior_requests_total"] == 0
    assert snap.facts["requests_24h"] == 0


def test_current_request_is_excluded_from_its_own_history() -> None:
    events = [request(f"R{i}", T0 + i * 60) for i in range(5)]
    index, keys = build(events)
    snap = build_features(events[-1], keys["R4"], index, ACCOUNT)
    assert snap.facts["requests_1h"] == 4
    assert snap.facts["prior_requests_total"] == 4


def test_future_events_do_not_change_past_features() -> None:
    past = [request(f"R{i}", T0 + i * 3_600, amount=40 + i) for i in range(10)]
    future = [request(f"F{i}", T0 + 20 * 3_600 + i, amount=9_999, recipient="CP-NEW", device="DEV-X") for i in range(5)]
    index_a, keys_a = build(past)
    index_b, keys_b = build(past + future)
    target = past[6]
    a = build_features(target, keys_a[target.request_id], index_a, ACCOUNT)
    b = build_features(target, keys_b[target.request_id], index_b, ACCOUNT)
    assert a.payment_vector == b.payment_vector
    assert a.context_vector == b.context_vector


def test_equal_timestamps_use_stable_identifier_order() -> None:
    first, second = request("R-A", T0), request("R-B", T0)
    index, keys = build([second, first])
    assert keys["R-A"] < keys["R-B"]
    assert build_features(first, keys["R-A"], index, ACCOUNT).facts["prior_requests_total"] == 0
    assert build_features(second, keys["R-B"], index, ACCOUNT).facts["prior_requests_total"] == 1


def test_trailing_window_boundaries() -> None:
    at_boundary = request("R1", T0 - 86_400)
    just_outside = request("R0", T0 - 86_401)
    current = request("R2", T0)
    index, keys = build([just_outside, at_boundary, current])
    snap = build_features(current, keys["R2"], index, ACCOUNT)
    assert snap.facts["requests_24h"] == 1


def test_repeated_counterparty_and_new_device() -> None:
    events = [
        request("R1", T0, recipient="CP-9"),
        request("R2", T0 + 60, recipient="CP-9", device="DEV-NEW"),
        request("R3", T0 + 2 * 86_400, recipient="CP-9"),
    ]
    auth = [AuthEvent("A1", T0 - 10 * 86_400, "ACC-0001", "S", "DEV-1", "login_success", "biometric", "DE", None)]
    index, keys = HistoryIndex.from_events(events, auth, [])
    first = build_features(events[0], keys["R1"], index, ACCOUNT)
    second = build_features(events[1], keys["R2"], index, ACCOUNT)
    third = build_features(events[2], keys["R3"], index, ACCOUNT)
    # a beneficiary first paid a minute ago is still new; after 24 hours the relationship is established
    assert first.facts["new_recipient"] is True and second.facts["new_recipient"] is True
    assert third.facts["new_recipient"] is False
    assert first.facts["new_device"] is False
    assert second.facts["new_device"] is True


def test_auth_failures_count_only_before_request() -> None:
    current = request("R1", T0)
    auth = [
        AuthEvent("A1", T0 - 100, "ACC-0001", "S", "DEV-1", "login_failure", "password", "DE", None),
        AuthEvent("A2", T0 + 100, "ACC-0001", "S", "DEV-1", "login_failure", "password", "DE", None),
    ]
    index, keys = HistoryIndex.from_events([current], auth, [])
    assert build_features(current, keys["R1"], index, ACCOUNT).facts["auth_failures_24h"] == 1


def test_label_changes_do_not_alter_features(generated: dict[str, Path], tmp_path: Path) -> None:
    before = replay(load_partition(generated["test"]))
    rows = read_rows(generated["test"])
    path = tmp_path / "test.json"
    write_rows(path, [{**r, "is_fraud": True, "is_anomaly": True, "fraud_type": "money_mule"} for r in rows])
    after = replay(load_partition(path))
    assert np.array_equal(before.payment, after.payment)
    assert np.array_equal(before.context, after.context)


def test_replay_matches_truncated_history(generated: dict[str, Path]) -> None:
    dataset = load_partition(generated["test"])
    full = replay(dataset, keep_facts_for=())
    cutoff = dataset.requests[5_000].ts
    from dataclasses import replace

    truncated = replace(
        dataset,
        requests=tuple(r for r in dataset.requests if r.ts <= cutoff),
        auth_events=tuple(e for e in dataset.auth_events if e.ts <= cutoff),
        payment_events=tuple(e for e in dataset.payment_events if e.ts <= cutoff),
    )
    partial = replay(truncated, keep_facts_for=())
    position = {rid: i for i, rid in enumerate(full.request_ids)}
    sample = [rid for rid in partial.request_ids[-200:]]
    for i, rid in enumerate(partial.request_ids[-200:]):
        j = position[rid]
        np.testing.assert_allclose(partial.payment[len(partial.request_ids) - 200 + i], full.payment[j])
    assert sample


@pytest.mark.parametrize("name", ["requests_24h", "new_recipient", "auth_failures_24h"])
def test_features_never_use_identifiers_or_labels(name: str) -> None:
    assert not any(token in " ".join(PAYMENT_FEATURES + CONTEXT_FEATURES) for token in ("fraud", "scenario", "label", "id"))
    assert name in " ".join(PAYMENT_FEATURES) or name in {"new_recipient"}


# -- beneficiary-side features (since features-1.1.0) -------------------------------------------

HOUR, DAY = 3_600, 86_400
TARGET_TS = T0 + 30 * DAY
BENEFICIARY = ("log_recipient_unique_senders_7d", "log_recipient_inbound_24h", "pair_requests_24h", "recipient_new_global")


def _snapshot(events: list[PaymentRequest], target: PaymentRequest):
    index, keys = build(events + [target])
    return build_features(target, keys[target.request_id], index, ACCOUNT)


def _beneficiary_values(snap) -> list[float]:
    return [snap.payment_vector[PAYMENT_FEATURES.index(name)] for name in BENEFICIARY]


def test_feature_vector_layout_and_version() -> None:
    assert len(PAYMENT_FEATURES) == 31 and FEATURE_VERSION == "features-2.0.0"
    assert PAYMENT_FEATURES[20:24] == BENEFICIARY
    assert PAYMENT_FEATURES[24:] == MIA


def test_beneficiary_features_ignore_future_events() -> None:
    past = [request("P1", TARGET_TS - 5 * HOUR, recipient="CP-X", sender="ACC-0002"),
            request("P2", TARGET_TS - 2 * HOUR, recipient="CP-X", sender="ACC-0001")]
    target = request("T", TARGET_TS, recipient="CP-X")
    future = [request(f"F{i}", TARGET_TS + 60 + i, amount=900, recipient="CP-X", sender=f"ACC-01{i:02d}")
              for i in range(5)]
    index_a, keys_a = build(past + [target])
    index_b, keys_b = build(past + [target] + future)
    a = build_features(target, keys_a["T"], index_a, ACCOUNT)
    b = build_features(target, keys_b["T"], index_b, ACCOUNT)
    assert _beneficiary_values(a) == _beneficiary_values(b)


def test_recipient_unique_senders_7d_excludes_events_older_than_7_days_and_self() -> None:
    events = [request("O", TARGET_TS - 8 * DAY, recipient="CP-X", sender="ACC-0002"),
              request("A", TARGET_TS - 2 * DAY, recipient="CP-X", sender="ACC-0003"),
              request("B", TARGET_TS - 1 * DAY, recipient="CP-X", sender="ACC-0004"),
              request("C", TARGET_TS - 3 * HOUR, recipient="CP-X", sender="ACC-0004"),
              request("S", TARGET_TS - 1 * HOUR, recipient="CP-X", sender="ACC-0001")]
    snap = _snapshot(events, request("T", TARGET_TS, recipient="CP-X"))
    assert snap.facts["recipient_unique_senders_7d"] == 2
    assert snap.payment_vector[PAYMENT_FEATURES.index("log_recipient_unique_senders_7d")] == pytest.approx(math.log1p(2))


def test_recipient_inbound_24h_sums_amounts_received_in_window() -> None:
    events = [request("A", TARGET_TS - 30 * HOUR, amount=100, recipient="CP-X", sender="ACC-0002"),
              request("B", TARGET_TS - 5 * HOUR, amount=50, recipient="CP-X", sender="ACC-0003"),
              request("C", TARGET_TS - 1 * HOUR, amount=70, recipient="CP-X", sender="ACC-0001")]
    snap = _snapshot(events, request("T", TARGET_TS, amount=999, recipient="CP-X"))
    assert snap.facts["recipient_inbound_24h"] == pytest.approx(120.0)


def test_pair_requests_24h_counts_only_while_relationship_new() -> None:
    new_pair = [request("A", TARGET_TS - 3 * HOUR, recipient="CP-A"), request("B", TARGET_TS - 2 * HOUR, recipient="CP-A")]
    assert _snapshot(new_pair, request("T", TARGET_TS, recipient="CP-A")).facts["pair_requests_24h"] == 2
    established = [request("A", TARGET_TS - 30 * HOUR, recipient="CP-B"), request("B", TARGET_TS - 2 * HOUR, recipient="CP-B")]
    assert _snapshot(established, request("T", TARGET_TS, recipient="CP-B")).facts["pair_requests_24h"] == 0


def test_recipient_new_global_true_until_24h_after_first_payment_by_anyone() -> None:
    recent = [request("A", TARGET_TS - 10 * HOUR, recipient="CP-Z", sender="ACC-0002")]
    assert _snapshot(recent, request("T", TARGET_TS, recipient="CP-Z")).facts["recipient_new_global"] is True
    old = [request("A", TARGET_TS - 30 * HOUR, recipient="CP-Y", sender="ACC-0002")]
    snap = _snapshot(old, request("T", TARGET_TS, recipient="CP-Y"))
    assert snap.facts["recipient_new_global"] is False
    assert snap.payment_vector[PAYMENT_FEATURES.index("recipient_new_global")] == 0.0
    never = _snapshot([request("A", TARGET_TS - HOUR, recipient="CP-1")], request("T", TARGET_TS, recipient="CP-NEW"))
    assert never.facts["recipient_new_global"] is True


# -- MIA Plati Instant features (features-2.0.0) -------------------------------------------------

MIA = ("amount_to_tx_cap_ratio", "daily_cumulative_to_cap_ratio", "near_cap_count_24h", "alias_rebound_age_hours",
       "is_request_to_pay", "qr_merchant_mismatch", "scheme_novelty")


def _value(snap, name: str) -> float:
    return snap.payment_vector[PAYMENT_FEATURES.index(name)]


def test_cap_ratios_use_the_segment_limits() -> None:
    day = TARGET_TS - TARGET_TS % DAY
    events = [request("Y", day - 2 * HOUR, amount=4_000), request("A", day + HOUR, amount=3_000)]
    target = request("T", day + 5 * HOUR, amount=4_500)
    snap = _snapshot(events, target)
    assert _value(snap, "amount_to_tx_cap_ratio") == pytest.approx(4_500 / 5_000)
    # only today's (UTC calendar day) earlier payments count toward the daily limit
    assert _value(snap, "daily_cumulative_to_cap_ratio") == pytest.approx((3_000 + 4_500) / 10_000)
    business = [request("B", TARGET_TS, amount=4_500, sender="ACC-0002")]
    index, keys = build(business)
    snap_b = build_features(business[0], keys["B"], index, BUSINESS)
    assert _value(snap_b, "amount_to_tx_cap_ratio") == pytest.approx(4_500 / 100_000)
    assert snap_b.facts["segment"] == "business" and snap_b.facts["tx_cap"] == 100_000


def test_near_cap_count_counts_earlier_near_limit_payments_in_24h() -> None:
    events = [request("A", TARGET_TS - 30 * HOUR, amount=4_900), request("B", TARGET_TS - 3 * HOUR, amount=4_800),
              request("C", TARGET_TS - 2 * HOUR, amount=4_400), request("D", TARGET_TS - HOUR, amount=4_999)]
    snap = _snapshot(events, request("T", TARGET_TS, amount=4_950))
    assert snap.facts["near_cap_count_24h"] == 2  # B and D; A is older than 24 h, C is below 90% of the cap


def test_alias_rebound_age_uses_only_earlier_rebinds() -> None:
    target = request("T", TARGET_TS)
    auth = [AuthEvent("A1", TARGET_TS - 5 * HOUR, "ACC-0001", "S", "DEV-9", "alias_rebound", "otp", "MD", "sim_replaced"),
            AuthEvent("A2", TARGET_TS + 60, "ACC-0001", "S", "DEV-9", "alias_rebound", "otp", "MD", "sim_replaced")]
    index, keys = HistoryIndex.from_events([target], auth, [])
    snap = build_features(target, keys["T"], index, ACCOUNT)
    assert snap.facts["alias_rebound_age_hours"] == pytest.approx(5.0)
    assert _value(snap, "alias_rebound_age_hours") == pytest.approx(5.0)
    none = _snapshot([], request("U", TARGET_TS))
    assert none.facts["alias_rebound_age_hours"] is None and _value(none, "alias_rebound_age_hours") == 720.0


def test_request_to_pay_qr_mismatch_and_scheme_novelty() -> None:
    history = [request("A", TARGET_TS - 2 * DAY, scheme="p2p_alias")]
    rtp = _snapshot(history, request("T", TARGET_TS, scheme="request_to_pay"))
    assert _value(rtp, "is_request_to_pay") == 1.0 and _value(rtp, "scheme_novelty") == 1.0
    again = _snapshot(history, request("T", TARGET_TS, scheme="p2p_alias"))
    assert _value(again, "is_request_to_pay") == 0.0 and _value(again, "scheme_novelty") == 0.0
    first_ever = _snapshot([], request("T", TARGET_TS, scheme="request_to_pay"))
    assert _value(first_ever, "scheme_novelty") == 0.0  # no history: novelty is undefined, not a signal
    tampered = _snapshot(history, request("T", TARGET_TS, recipient="CP-FAKE", scheme="p2b_qr_static", qr="CP-00343"))
    assert _value(tampered, "qr_merchant_mismatch") == 1.0 and tampered.facts["qr_merchant_mismatch"] is True
    honest = _snapshot(history, request("T", TARGET_TS, recipient="CP-00343", scheme="p2b_qr_static", qr="CP-00343"))
    assert _value(honest, "qr_merchant_mismatch") == 0.0


def test_mia_features_ignore_future_events() -> None:
    past = [request("P", TARGET_TS - HOUR, amount=4_900)]
    target = request("T", TARGET_TS, amount=4_800)
    future = [request(f"F{i}", TARGET_TS + 60 + i, amount=4_990, scheme="request_to_pay") for i in range(3)]
    auth_future = [AuthEvent("A", TARGET_TS + 30, "ACC-0001", "S", "D", "alias_rebound", "otp", "MD", None)]
    index_a, keys_a = HistoryIndex.from_events(past + [target], [], [])
    index_b, keys_b = HistoryIndex.from_events(past + [target] + future, auth_future, [])
    a = build_features(target, keys_a["T"], index_a, ACCOUNT)
    b = build_features(target, keys_b["T"], index_b, ACCOUNT)
    assert [_value(a, n) for n in MIA] == [_value(b, n) for n in MIA]

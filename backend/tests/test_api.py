"""The HTTP API as the frontend uses it: contract, validation errors, simulations, reviews and the stream."""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.tests.conftest import ROOT


def sender_and_recipient(client: TestClient) -> tuple[str, str]:
    """An established personal account and one of its usual recipients."""
    accounts = client.get("/api/accounts", params={"page_size": 1000}).json()["items"]
    sender = next(a for a in accounts if a["account_type"] == "personal" and a["requests"] > 20)["account_id"]
    options = client.get("/api/simulator/options", params={"sender_id": sender}).json()["sender"]
    return sender, options["recent_recipients"][0]["id"]


def payment(client: TestClient, request_id: str = "SIM-000001", **changes: Any) -> dict[str, Any]:
    sender, recipient = sender_and_recipient(client)
    options = client.get("/api/simulator/options", params={"sender_id": sender}).json()["sender"]
    return {
        "request_id": request_id,
        "sender_id": sender,
        "recipient_id": recipient,
        "amount": options["median_amount"],
        "device_id": options["known_devices"][0],
        **changes,
    }


def test_responses_keep_the_shape_the_frontend_was_built_against() -> None:
    spec = importlib.util.spec_from_file_location("api_contract", ROOT / "scripts" / "api_contract.py")
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    problems, _ = contract.check(contract.capture())
    assert problems == []


def test_health_reports_the_loaded_model_and_memory_state(client: TestClient) -> None:
    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    assert health["data"]["requests"] == 15_000 and health["data"]["hashes_verified"] is True
    assert health["state"]["kind"] == "process_memory" and health["state"]["persisted"] is False


def test_the_app_decides_like_the_evaluation(client: TestClient) -> None:
    """The app replays test.json on top of the validation days, so its Verify rate matches the report."""
    totals = client.get("/api/overview").json()["totals"]
    assert totals["assessed"] == 15_000
    assert totals["needs_verification"] / totals["assessed"] < 0.06


def test_transactions_filter_sort_and_paginate(client: TestClient) -> None:
    page = client.get(
        "/api/transactions", params={"decision": "Verify", "sort": "risk_desc", "page_size": 10}
    ).json()
    assert page["total"] > 10 and len(page["items"]) == 10
    assert {i["decision"] for i in page["items"]} == {"Verify"}
    scores = [i["risk_score"] for i in page["items"]]
    assert scores == sorted(scores, reverse=True)


def test_unknown_identifiers_return_404(client: TestClient) -> None:
    for path in ("/api/investigations/NOPE", "/api/accounts/NOPE"):
        response = client.get(path)
        assert response.status_code == 404 and response.json()["error"]["code"] == "not_found"


@pytest.mark.parametrize(
    "change",
    [
        {"currency": "EUR"},
        {"transfer_type": "standard"},
        {"is_fraud": True},
        {"amount": 0},
        {"recipient_alias": "+40712345678"},
        {"new_recipient_name": "Both Recipients"},
    ],
    ids=["currency", "transfer_type", "label_field", "amount", "alias", "two_recipients"],
)
def test_invalid_simulations_are_rejected_with_422(client: TestClient, change: dict[str, Any]) -> None:
    response = client.post("/api/simulations", json={**payment(client), **change})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_a_personal_payment_above_the_mia_cap_is_rejected(client: TestClient) -> None:
    response = client.post("/api/simulations", json=payment(client, amount=5_000.01))
    assert response.status_code == 422
    assert response.json()["error"]["details"] == {"limit": "mia_per_transaction"}


def test_an_ordinary_payment_is_permitted_and_a_retry_is_answered_once(client: TestClient) -> None:
    body = payment(client)
    first = client.post("/api/simulations", json=body)
    assert first.status_code == 201 and first.json()["decision"] == "Permit"
    retry = client.post("/api/simulations", json=body)
    assert retry.status_code == 200 and retry.json()["idempotent_replay"] is True
    assert client.post("/api/simulations", json={**body, "amount": body["amount"] + 1}).status_code == 409
    assert client.get("/api/live-transfers").json()["items"][0]["request_id"] == body["request_id"]


def test_a_takeover_like_payment_is_stopped(client: TestClient) -> None:
    body = payment(
        client,
        recipient_id=None,
        new_recipient_name="Unknown Payee",
        amount=4_900.0,
        device_id=None,
        auth_method="password",
        failed_login_attempts=4,
        ip_country="NL",
    )
    detail = client.post("/api/simulations", json=body).json()
    assert detail["decision"] in {"Verify", "Block"}
    assert {"NEW_DEVICE_AFTER_FAILED_LOGINS", "FOREIGN_SESSION"} <= {r["code"] for r in detail["reasons"]}


def test_a_payment_to_a_beneficiary_with_trusted_evidence_is_blocked(client: TestClient) -> None:
    flagged = client.get("/api/simulator/options").json()["flagged_beneficiaries"][0]
    detail = client.post("/api/simulations", json=payment(client, recipient_id=flagged["id"])).json()
    assert (detail["decision"], detail["decision_source"]) == ("Block", "confirmed_fraud_evidence")
    assert detail["evidence_ids"] == [flagged["evidence_id"]]


def test_later_simulations_see_earlier_ones(client: TestClient) -> None:
    body = payment(client, recipient_id=None, new_recipient_name="Burst Payee")
    first = client.post("/api/simulations", json=body).json()
    for i in range(2, 6):
        last = client.post(
            "/api/simulations",
            json={
                **body,
                "request_id": f"SIM-00000{i}",
                "new_recipient_name": None,
                "recipient_id": first["recipient"]["id"],
            },
        ).json()
    assert last["facts"]["requests_1h"] >= 4
    assert "REPEAT_NEW_BENEFICIARY" in {r["code"] for r in last["reasons"]}


def test_a_verify_decision_is_reviewed_once_and_kept_in_the_history(client: TestClient) -> None:
    target = client.get("/api/transactions", params={"decision": "Verify", "review_status": "pending"}).json()
    request_id = target["items"][0]["request_id"]
    path = f"/api/investigations/{request_id}/reviews"
    assert client.post(path, json={"action": "reject", "reason": "short"}).status_code == 422
    review = client.post(path, json={"action": "reject", "reason": "Customer confirmed a scam call"})
    assert review.status_code == 201
    investigation = review.json()
    assert investigation["assessment"]["decision"] == "Verify"  # the automated decision never changes
    assert (
        investigation["assessment"]["outcome"] == "rejected_after_review" and not investigation["can_review"]
    )
    assert client.post(path, json={"action": "approve", "reason": "Second opinion here"}).status_code == 409
    history = client.get("/api/reviews").json()
    assert (history["total"], history["rejected"], history["persisted"]) == (1, 1, False)
    assert history["items"][0]["request_id"] == request_id


def test_only_verify_decisions_can_be_reviewed(client: TestClient) -> None:
    permitted = client.get("/api/transactions", params={"decision": "Permit", "page_size": 1}).json()[
        "items"
    ][0]
    response = client.post(
        f"/api/investigations/{permitted['request_id']}/reviews",
        json={"action": "approve", "reason": "Looks fine to me"},
    )
    assert response.status_code == 409


def test_the_stream_is_started_and_stopped_by_the_backend(client: TestClient) -> None:
    assert client.get("/api/simulation-stream").json()["running"] is False
    assert client.post("/api/simulation-stream/mode", json={"mode": "batch_10"}).json()["mode"] == "batch_10"
    assert client.post("/api/simulation-stream/mode", json={"mode": "batch_99"}).status_code == 422
    assert client.post("/api/simulation-stream/start").json()["running"] is True
    status = client.post("/api/simulation-stream/stop").json()
    assert status["running"] is False and status["owner"] == "backend"


def test_one_stream_tick_records_assessed_payments(client: TestClient) -> None:
    stream = client.app.state.stream
    stream.running, stream.mode = True, "batch_20"
    assert client.portal.call(stream.tick) == 20
    stream.running = False
    feed = client.get("/api/live-transfers", params={"limit": 100}).json()
    assert len(feed["items"]) == 20 and feed["stream"]["produced"] == 20
    assert {i["currency"] for i in feed["items"]} == {"MDL"}

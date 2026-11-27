"""Durable review history: every approve / reject is saved and survives a restart."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api import create_app
from config.settings import load_settings
from runtime.review_log import ReviewHistoryEntry, ReviewLog


def _settings(state_dir: Path | None):
    return replace(load_settings(), state_dir=state_dir, autostart_stream=False)


def _pending_baseline(client: TestClient, offset: int = 0) -> str:
    page = client.get("/api/transactions",
                      params={"origin": "baseline", "review_status": "pending", "page_size": 5}).json()
    return page["items"][offset]["request_id"]


def test_reject_is_saved_in_history_and_survives_restart(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        rid = _pending_baseline(client)
        before = client.get(f"/api/investigations/{rid}").json()["assessment"]
        done = client.post(f"/api/investigations/{rid}/reviews",
                           json={"action": "reject", "reason": "Beneficiary could not be verified", "actor": "Ana"})
        assert done.status_code == 201
        history = client.get("/api/reviews").json()
        assert history["total"] == 1 and history["rejected"] == 1 and history["approved"] == 0
        assert history["persisted"] is True
        item = history["items"][0]
        assert item["request_id"] == rid and item["action"] == "reject" and item["actor"] == "Ana"
        assert item["resulting_outcome"] == "rejected_after_review"
        assert item["original_risk_score"] == pytest.approx(before["risk_score"], abs=0.01)
        assert item["sender_name"] == before["sender"]["name"] and item["amount"] == before["amount"]
        assert item["investigation_available"] is True
        assert client.get("/api/reviews", params={"action": "approve"}).json()["total"] == 0
        assert client.get("/api/reviews", params={"q": before["sender"]["name"][:4]}).json()["total"] == 1

    assert (tmp_path / "review_history.sqlite3").exists()
    with TestClient(create_app(_settings(tmp_path))) as restarted:
        history = restarted.get("/api/reviews").json()
        assert history["total"] == 1 and history["items"][0]["review_id"] == "REV-00001"
        investigation = restarted.get(f"/api/investigations/{rid}").json()
        assert investigation["assessment"]["outcome"] == "rejected_after_review"
        assert investigation["can_review"] is False
        assert restarted.get("/api/transactions", params={"review_status": "rejected"}).json()["total"] == 1
        # A second review after the restart is still refused, and new IDs continue the sequence.
        again = restarted.post(f"/api/investigations/{rid}/reviews", json={"action": "approve", "reason": "Second try"})
        assert again.status_code == 409
        other = _pending_baseline(restarted)
        restarted.post(f"/api/investigations/{other}/reviews", json={"action": "approve", "reason": "Customer confirmed"})
        assert restarted.get("/api/reviews").json()["items"][0]["review_id"] == "REV-00002"


def test_history_works_without_a_state_directory() -> None:
    with TestClient(create_app(_settings(None))) as client:
        rid = _pending_baseline(client)
        client.post(f"/api/investigations/{rid}/reviews", json={"action": "reject", "reason": "Suspicious payee"})
        history = client.get("/api/reviews").json()
        assert history["total"] == 1 and history["persisted"] is False


def test_review_log_falls_back_to_memory_when_the_file_breaks(tmp_path: Path) -> None:
    log = ReviewLog(tmp_path)
    entry = ReviewHistoryEntry("REV-00001", 1, "R1", "reject", "reason text", "Ana", "2026-01-01T00:00:00.00Z", 0,
                               "Verify", "Needs verification", 70.0, "rejected_after_review", "baseline",
                               "2026-01-01T00:00:00Z", "ACC-1", "A", "CP-1", "B", 10.0, "EUR")
    log.append(entry)
    log._db.close()  # simulate the file becoming unusable; the next write must not fail
    log.append(replace(entry, review_id="REV-00002", seq=2))
    assert log.persisted is False and [e.review_id for e in log.entries()] == ["REV-00002"]
    assert log.status()["error"]


def test_review_log_rejects_duplicate_ids(tmp_path: Path) -> None:
    log = ReviewLog(tmp_path)
    entry = ReviewHistoryEntry("REV-00001", 1, "R1", "approve", "reason text", "Ana", "t", 0, "Verify",
                               "Needs verification", None, "approved_after_review", "baseline", "t", "A", "A", "B",
                               "B", 1.0, "EUR")
    log.append(entry)
    with pytest.raises(sqlite3.IntegrityError):
        log.append(entry)
    assert log.next_seq() == 2


def test_accounts_can_list_every_account_on_one_page(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        page = client.get("/api/accounts", params={"page_size": 1000}).json()
        assert page["total"] == page["all_accounts"] == 400 and len(page["items"]) == 400
        assert page["personal"] + page["business"] == 400 and 0 < page["active"] <= 400
        assert all("rejected" in a for a in page["items"])


def test_assessment_explains_score_with_observed_fraud_rate(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        rid = client.get("/api/transactions", params={"sort": "risk_desc", "page_size": 1}).json()["items"][0]["request_id"]
        assessment = client.get(f"/api/investigations/{rid}").json()["assessment"]
        band = assessment["score_band"]
        assert band is not None and band["low"] <= assessment["risk_score"] <= band["high"]
        assert 0 <= band["fraud_rate"] <= 1 and band["evaluated_requests"] == 20_000

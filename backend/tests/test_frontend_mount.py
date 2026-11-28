"""The backend serves the frontend build: files as they are, the single-page app for routes, and 404 for
a missing file."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import _mount_frontend
from backend.tests.conftest import ROOT


@pytest.fixture
def site(tmp_path) -> TestClient:
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<!doctype html><title>SecFlow</title>")
    (tmp_path / "robots.txt").write_text((ROOT / "frontend" / "public" / "robots.txt").read_text())
    app = FastAPI()
    _mount_frontend(app, tmp_path)
    return TestClient(app)


def test_robots_txt_is_served_as_text(site: TestClient) -> None:
    response = site.get("/robots.txt")
    assert response.status_code == 200
    assert response.text.startswith("User-agent: *")


def test_a_route_gets_the_single_page_app(site: TestClient) -> None:
    response = site.get("/accounts/AC-00001")
    assert response.status_code == 200
    assert "<title>SecFlow</title>" in response.text


def test_a_missing_file_is_404_not_the_app(site: TestClient) -> None:
    assert site.get("/missing.png").status_code == 404
    assert site.get("/api/unknown").status_code == 404

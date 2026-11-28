"""Shared fixtures. Tests read the committed storage/ and ml/models/ and write only to temporary
directories, so they never change the dataset, the models or the evaluation report."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest

os.environ["SECFLOW_AUTOSTART_STREAM"] = "false"

from fastapi.testclient import TestClient

from backend.api import create_app
from config.settings import load_settings
from ml.policy import load_policy

ROOT = Path(__file__).resolve().parents[2]
STORAGE = ROOT / "storage"


def load_script(name: str) -> ModuleType:
    """Import scripts/<name>.py, which is not part of a package."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def policy():
    return load_policy(load_settings().policy_path)


@pytest.fixture
def client():
    """A fresh app over the committed data and models (simulations and reviews start empty)."""
    with TestClient(create_app()) as test_client:
        yield test_client

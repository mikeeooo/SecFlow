"""Shared fixtures. Generated datasets are written to temporary directories so tests never
modify the committed data or model bundle."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("SECFLOW_AUTOSTART_STREAM", "false")
# Review history goes to a throwaway directory so tests never touch runtime/state/.
os.environ.setdefault("SECFLOW_STATE_DIR", tempfile.mkdtemp(prefix="secflow-state-"))

from datagen.generator import DEMO_SPEC, TRAINING_SPEC, generate_dataset

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def generated(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("datasets")
    paths = {}
    for spec in (TRAINING_SPEC, DEMO_SPEC):
        generate_dataset(base / spec.name, spec)
        paths[spec.name] = base / spec.name
    return paths


@pytest.fixture(scope="session")
def trained(generated: dict[str, Path], tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Train on the freshly generated corpus into a temporary models directory."""
    from config.settings import load_settings
    from ml.policy import load_policy
    from ml.training import train_models

    models_dir = tmp_path_factory.mktemp("models")
    report = train_models(generated["training"], models_dir, load_policy(load_settings().policy_path))
    return {"models_dir": models_dir, "report": report, "training_dir": generated["training"]}

"""Shared fixtures. Generated datasets are written to temporary directories so tests never
modify the committed data (storage/*.json, datagen/manifest.json) or the model bundle."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("SECFLOW_AUTOSTART_STREAM", "false")
# Review history goes to a throwaway directory so tests never touch runtime/state/.
os.environ.setdefault("SECFLOW_STATE_DIR", tempfile.mkdtemp(prefix="secflow-state-"))

from datagen.dataset import partition_paths
from datagen.generator import MIA_SPEC, generate_dataset

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def generated(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """The full MIA dataset generated from the seed: {"storage", "manifest", "train", "validation", "test"}."""
    base = tmp_path_factory.mktemp("dataset")
    storage, manifest = base / "storage", base / "manifest.json"
    generate_dataset(storage, manifest, MIA_SPEC)
    return {"storage": storage, "manifest": manifest, **partition_paths(storage)}


@pytest.fixture(scope="session")
def trained(generated: dict[str, Path], tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Train on the freshly generated train/validation files into a temporary models directory."""
    from config.settings import load_settings
    from ml.policy import load_policy
    from ml.training import train_models

    models_dir = tmp_path_factory.mktemp("models")
    report = train_models(generated["train"], generated["validation"], models_dir,
                          load_policy(load_settings().policy_path))
    return {"models_dir": models_dir, "report": report, **generated}

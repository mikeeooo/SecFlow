"""File locations and stream settings. Environment variables override them, so tests can point the
app at temporary copies without touching the committed data or models."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).resolve() if value else default


@dataclass(frozen=True)
class Settings:
    storage_dir: Path  # exactly train.json, validation.json and test.json
    manifest_path: Path
    models_dir: Path
    artifacts_dir: Path
    policy_path: Path
    autostart_stream: bool
    stream_interval_seconds: float
    batch_interval_seconds: float

    @property
    def train_path(self) -> Path:
        return self.storage_dir / "train.json"

    @property
    def validation_path(self) -> Path:
        return self.storage_dir / "validation.json"

    @property
    def test_path(self) -> Path:  # the final evaluation and the history the app serves
        return self.storage_dir / "test.json"


def load_settings() -> Settings:
    return Settings(
        storage_dir=_path("SECFLOW_STORAGE_DIR", ROOT / "storage"),
        manifest_path=_path("SECFLOW_MANIFEST_PATH", ROOT / "datagen" / "manifest.json"),
        models_dir=_path("SECFLOW_MODELS_DIR", ROOT / "ml" / "models"),
        artifacts_dir=_path("SECFLOW_ARTIFACTS_DIR", ROOT / "artifacts"),
        policy_path=_path("SECFLOW_POLICY_PATH", ROOT / "config" / "policy.json"),
        autostart_stream=os.environ.get("SECFLOW_AUTOSTART_STREAM", "true").lower() in {"1", "true", "yes", "on"},
        stream_interval_seconds=float(os.environ.get("SECFLOW_STREAM_INTERVAL", "2")),
        batch_interval_seconds=float(os.environ.get("SECFLOW_BATCH_INTERVAL", "4")),
    )

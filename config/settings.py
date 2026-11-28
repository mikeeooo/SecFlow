"""File locations and stream settings. SECFLOW_* environment variables and a `.env` file in the repo root
override them, so tests can point the app at temporary copies without touching the committed data or
models. `.env.example` lists every variable."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field, PositiveFloat, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
PREFIX = "SECFLOW_"


class Settings(BaseSettings):
    """Each field reads the SECFLOW_* variable in its alias, so errors name the variable."""

    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_ignore_empty=True, extra="forbid", frozen=True, populate_by_name=True
    )

    storage_dir: Path = Field(ROOT / "storage", alias="SECFLOW_STORAGE_DIR")  # train, validation, test
    manifest_path: Path = Field(ROOT / "datagen" / "manifest.json", alias="SECFLOW_MANIFEST_PATH")
    models_dir: Path = Field(ROOT / "ml" / "models", alias="SECFLOW_MODELS_DIR")
    artifacts_dir: Path = Field(ROOT / "artifacts", alias="SECFLOW_ARTIFACTS_DIR")
    policy_path: Path = Field(ROOT / "config" / "policy.json", alias="SECFLOW_POLICY_PATH")
    autostart_stream: bool = Field(True, alias="SECFLOW_AUTOSTART_STREAM")
    stream_interval_seconds: PositiveFloat = Field(2.0, alias="SECFLOW_STREAM_INTERVAL")
    batch_interval_seconds: PositiveFloat = Field(4.0, alias="SECFLOW_BATCH_INTERVAL")

    @field_validator("storage_dir", "manifest_path", "models_dir", "artifacts_dir", "policy_path")
    @classmethod
    def _absolute(cls, path: Path) -> Path:
        return path.resolve()

    @property
    def train_path(self) -> Path:
        return self.storage_dir / "train.json"

    @property
    def validation_path(self) -> Path:
        return self.storage_dir / "validation.json"

    @property
    def test_path(self) -> Path:  # the final evaluation and the history the app serves
        return self.storage_dir / "test.json"


def variable_names() -> set[str]:
    """Every SECFLOW_* variable that Settings reads."""
    return {str(field.alias) for field in Settings.model_fields.values()}


def load_settings() -> Settings:
    """Settings from the environment. An unknown SECFLOW_* variable is an error, so a typo cannot pass
    silently. A value that does not parse raises pydantic's ValidationError, which names the variable."""
    unknown = sorted(name for name in os.environ if name.startswith(PREFIX) and name not in variable_names())
    if unknown:
        raise ValueError(
            f"unknown setting(s) {', '.join(unknown)}; known: {', '.join(sorted(variable_names()))}"
        )
    return Settings()

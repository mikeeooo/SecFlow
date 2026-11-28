"""Settings come from SECFLOW_* variables and .env. A typo in a name or a value stops the process with
an error that names the variable."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.tests.conftest import ROOT
from config.settings import Settings, load_settings


def test_defaults_point_at_the_committed_files(monkeypatch) -> None:
    monkeypatch.delenv("SECFLOW_AUTOSTART_STREAM")
    settings = load_settings()
    assert settings.test_path == ROOT / "storage" / "test.json"
    assert settings.models_dir == ROOT / "ml" / "models"
    assert (settings.autostart_stream, settings.stream_interval_seconds) == (True, 2.0)


def test_variables_override_and_relative_paths_become_absolute(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SECFLOW_STORAGE_DIR", "data")
    monkeypatch.setenv("SECFLOW_BATCH_INTERVAL", "0.5")
    settings = load_settings()
    assert settings.storage_dir == tmp_path.resolve() / "data"
    assert settings.batch_interval_seconds == 0.5


def test_a_misspelled_variable_name_is_an_error(monkeypatch) -> None:
    monkeypatch.setenv("SECFLOW_STREAM_INTERVL", "3")
    with pytest.raises(ValueError, match="SECFLOW_STREAM_INTERVL"):
        load_settings()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SECFLOW_STREAM_INTERVAL", "two"),
        ("SECFLOW_STREAM_INTERVAL", "0"),
        ("SECFLOW_BATCH_INTERVAL", "-4"),
        ("SECFLOW_AUTOSTART_STREAM", "ture"),
    ],
)
def test_an_invalid_value_is_an_error_that_names_the_variable(monkeypatch, name, value) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError, match=name):
        load_settings()


def test_a_misspelled_name_in_the_env_file_is_an_error(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("SECFLOW_AUTOSTART_STREM=false\n")
    with pytest.raises(ValidationError, match="secflow_autostart_strem"):
        Settings(_env_file=env_file)

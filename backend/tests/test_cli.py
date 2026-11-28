"""The command line runs the whole pipeline into the directories the SECFLOW_* variables name."""

from __future__ import annotations

import json

from backend import cli


def test_generate_train_evaluate_write_only_to_the_configured_directories(tmp_path, monkeypatch) -> None:
    paths = {
        "SECFLOW_STORAGE_DIR": tmp_path / "storage",
        "SECFLOW_MANIFEST_PATH": tmp_path / "manifest.json",
        "SECFLOW_MODELS_DIR": tmp_path / "models",
        "SECFLOW_ARTIFACTS_DIR": tmp_path / "artifacts",
    }
    for name, path in paths.items():
        monkeypatch.setenv(name, str(path))

    for command in ("generate", "validate", "train", "evaluate"):
        assert cli.main([command]) == 0, command

    assert sorted(p.name for p in paths["SECFLOW_STORAGE_DIR"].iterdir()) == [
        "test.json",
        "train.json",
        "validation.json",
    ]
    assert (paths["SECFLOW_MODELS_DIR"] / "secflow_model.joblib").is_file()
    headline = json.loads((paths["SECFLOW_ARTIFACTS_DIR"] / "evaluation.json").read_text())["headline"]
    assert headline["test_requests"] == 15_000


def test_validate_fails_when_the_dataset_is_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SECFLOW_STORAGE_DIR", str(tmp_path))
    monkeypatch.setenv("SECFLOW_MANIFEST_PATH", str(tmp_path / "manifest.json"))
    assert cli.main(["validate"]) == 1
